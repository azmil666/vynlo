#!/usr/bin/env python3
"""Video Clipping Pipeline Evaluation Harness.

Runs the real pipeline on configured local videos across 8 representative categories:
1. single-speaker talk
2. two-person podcast
3. screen share / slides
4. gameplay with webcam
5. vertical source
6. long (60+ min)
7. non-English
8. noisy / music

Executes 10 automated quality checks per clip:
1. output spec (ffprobe: 1080x1920, H.264, AAC stereo, CFR, yuv420p, faststart, trailing frames)
2. starts/ends on sentence boundary
3. no clipping into neighbouring words
4. caption/audio sync
5. crop jumps (smooth pan velocity <= 320 px/s)
6. face-in-frame ratio (YuNet >= 85%)
7. non-black-bar ratio
8. loudness within tolerance (-14 LUFS +- 2.0, peak <= -1.0 dBTP)
9. duration bounds (5-60s)
10. warnings present when degraded

Generates:
- eval_results/report.json (machine readable results)
- eval_results/summary.md (human readable markdown summary)
- eval_results/contact_sheets/*.jpg (visual contact sheets for start/mid/end frames)

Exit code: 0 if all hard checks pass, 1 if any hard check fails.
"""

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple
import yaml

# Ensure project root and backend are on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Ensure host-compatible directories before app config loads
os.environ.setdefault("STATIC_DIR", str((BACKEND_DIR / "static").resolve()))
os.environ.setdefault("TEMP_DIR", "/tmp/clipper")

import fakeredis

from app.config import settings
from app.schemas import SourceInfo, TranscriptValidation
from app.services.analyzer import find_viral_clips
from app.services.captions import generate_ass_subtitles, render_clip_single_pass
from app.services.cropper import compute_crop_filter
from app.services.transcriber import TranscriptWordList
import app.main as app_main
import app.worker as app_worker

from scripts.eval.checks import (
    CheckResult,
    check_caption_audio_sync,
    check_crop_jumps,
    check_duration_bounds,
    check_face_in_frame_ratio,
    check_loudness_tolerance,
    check_no_word_clipping,
    check_non_black_bar_ratio,
    check_output_spec,
    check_sentence_boundary,
    check_warnings_when_degraded,
)
from scripts.eval.contact_sheet import generate_contact_sheet
from scripts.eval.dataset_generator import generate_synthetic_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval.runner")


def load_transcript(transcript_path: Path) -> Tuple[List[Dict], TranscriptWordList]:
    """Load transcript JSON into words list and TranscriptWordList object."""
    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    words = data if isinstance(data, list) else data.get("words", [])

    speech_dur = float(words[-1]["end"]) - float(words[0]["start"]) if words else 0.0
    val = TranscriptValidation(
        word_count=len(words),
        speech_duration=speech_dur,
        speech_coverage_ratio=0.85,
        words_per_minute=140.0,
        language="en",
        language_probability=0.99,
        model="fixture",
    )
    return words, TranscriptWordList(words, val)


def evaluate_category(
    category_cfg: Dict[str, Any],
    defaults: Dict[str, Any],
    output_dir: Path,
    generate_missing: bool = False,
    fail_fast: bool = False,
) -> Dict[str, Any]:
    """Run pipeline and quality checks on a single video category."""
    cat_id = category_cfg["id"]
    cat_name = category_cfg.get("name", cat_id)
    logger.info(f"--- Evaluating Category: {cat_name} ({cat_id}) ---")

    source_path = Path(category_cfg["source_path"])
    transcript_path = Path(category_cfg["transcript_path"])

    if not source_path.is_absolute():
        source_path = ROOT_DIR / source_path
    if not transcript_path.is_absolute():
        transcript_path = ROOT_DIR / transcript_path

    # If missing and requested, generate offline synthetic fixtures
    if not source_path.exists() or not transcript_path.exists():
        if generate_missing:
            logger.info(f"Generating synthetic fixture for {cat_id}...")
            source_path, transcript_path = generate_synthetic_video(cat_id, source_path, transcript_path)
        else:
            raise FileNotFoundError(
                f"Source file {source_path} or transcript {transcript_path} does not exist. "
                "Use --generate-missing to generate offline fixtures."
            )

    words_raw, word_list = load_transcript(transcript_path)

    # Prepare clip slices
    slices = category_cfg.get("clip_slices")
    if not slices:
        # Heuristic slice selection from transcript
        slices = [{"start": 0.5, "end": min(10.0, float(words_raw[-1]["end"])), "title": f"{cat_name} Clip"}]

    layout_mode = category_cfg.get("expected_layout", "speaker")
    has_face = category_cfg.get("has_face", True)
    expected_degradation = category_cfg.get("expected_degradation")

    work_dir = output_dir / "work" / cat_id
    work_dir.mkdir(parents=True, exist_ok=True)
    contact_dir = output_dir / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)

    evaluated_clips = []
    category_passed = True
    hard_check_failures = 0

    for idx, sl in enumerate(slices):
        clip_start = float(sl["start"])
        clip_end = float(sl["end"])
        clip_title = sl.get("title", f"Clip {idx + 1}")
        logger.info(f"Processing Clip {idx + 1}: {clip_title} [{clip_start:.2f}s -> {clip_end:.2f}s]")

        clip_warnings: List[str] = []
        ass_path = work_dir / f"clip_{idx}.ass"
        out_clip_path = work_dir / f"final_{cat_id}_clip_{idx}.mp4"
        contact_path = contact_dir / f"{cat_id}_clip_{idx}_contact.jpg"

        # 1. Compute crop filter
        crop_filter = compute_crop_filter(
            str(source_path),
            clip_start,
            clip_end,
            mode=layout_mode,
            words=word_list,
        )
        actual_layout = getattr(crop_filter, "layout_used", layout_mode)
        if actual_layout != layout_mode:
            clip_warnings.append(f"LAYOUT_FALLBACK: requested {layout_mode}, used {actual_layout}")

        # 2. Generate subtitles
        generate_ass_subtitles(
            word_list,
            clip_start,
            clip_end,
            str(ass_path),
            preset="clean_white",
            layout=actual_layout,
            warnings=clip_warnings,
        )

        # 3. Render clip single-pass
        t_render_start = time.time()
        render_clip_single_pass(
            video_path=str(source_path),
            start_time=clip_start,
            end_time=clip_end,
            crop_filter=crop_filter,
            ass_path=str(ass_path),
            output_path=str(out_clip_path),
            threads=2,
        )
        render_duration = time.time() - t_render_start

        # 4. Generate contact sheet
        generate_contact_sheet(
            clip_path=out_clip_path,
            output_image_path=contact_path,
            title=f"{cat_name} - {clip_title}",
        )

        # 5. Run all 10 quality checks
        checks: List[CheckResult] = []

        # 1. Output spec
        checks.append(check_output_spec(out_clip_path))

        # 2. Sentence boundary
        checks.append(check_sentence_boundary(clip_start, clip_end, words_raw))

        # 3. No word clipping
        checks.append(check_no_word_clipping(clip_start, clip_end, words_raw))

        # 4. Caption / audio sync
        checks.append(check_caption_audio_sync(ass_path, clip_start, clip_end, words_raw))

        # 5. Crop jumps
        checks.append(check_crop_jumps(crop_filter, max_speed_px_per_sec=defaults.get("max_crop_speed_px_per_sec", 320.0)))

        # 6. Face in frame ratio
        checks.append(check_face_in_frame_ratio(
            out_clip_path,
            min_ratio=defaults.get("min_face_in_frame_ratio", 0.85),
            is_face_expected=has_face,
        ))

        # 7. Non-black-bar ratio
        checks.append(check_non_black_bar_ratio(
            out_clip_path,
            max_black_bar_ratio=defaults.get("max_black_bar_ratio", 0.15),
        ))

        # 8. Loudness within tolerance
        checks.append(check_loudness_tolerance(
            out_clip_path,
            target_lufs=defaults.get("target_lufs", -14.0),
            lufs_tol=defaults.get("lufs_tolerance", 2.0),
        ))

        # 9. Duration bounds
        checks.append(check_duration_bounds(
            out_clip_path,
            min_dur=defaults.get("min_duration", 5.0),
            max_dur=defaults.get("max_duration", 60.0),
            requested_start=clip_start,
            requested_end=clip_end,
        ))

        # 10. Warnings present when degraded
        checks.append(check_warnings_when_degraded(
            clip_warnings=clip_warnings,
            expected_degradation=expected_degradation,
        ))

        clip_hard_fails = sum(1 for c in checks if c.is_hard_check and not c.passed)
        hard_check_failures += clip_hard_fails
        if clip_hard_fails > 0:
            category_passed = False

        avg_score = sum(c.score for c in checks) / len(checks) if checks else 1.0

        evaluated_clips.append({
            "clip_index": idx,
            "title": clip_title,
            "interval": [clip_start, clip_end],
            "clip_path": str(out_clip_path),
            "contact_sheet": str(contact_path),
            "render_duration_sec": round(render_duration, 2),
            "layout_used": actual_layout,
            "warnings": clip_warnings,
            "hard_check_failures": clip_hard_fails,
            "avg_score": round(avg_score, 3),
            "checks": [c.to_dict() for c in checks],
        })

        if fail_fast and clip_hard_fails > 0:
            logger.error(f"Fail-fast triggered on {cat_id} Clip {idx + 1}")
            break

    return {
        "id": cat_id,
        "name": cat_name,
        "description": category_cfg.get("description", ""),
        "passed": category_passed,
        "hard_check_failures": hard_check_failures,
        "clips": evaluated_clips,
    }


def emit_reports(results: List[Dict[str, Any]], output_dir: Path) -> Tuple[Path, Path]:
    """Generate eval_results/report.json and eval_results/summary.md."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    md_path = output_dir / "summary.md"

    total_categories = len(results)
    passed_categories = sum(1 for r in results if r["passed"])
    total_clips = sum(len(r["clips"]) for r in results)
    total_hard_failures = sum(r["hard_check_failures"] for r in results)

    # 1. Write report.json
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_categories": total_categories,
            "passed_categories": passed_categories,
            "failed_categories": total_categories - passed_categories,
            "total_clips": total_clips,
            "total_hard_failures": total_hard_failures,
            "overall_status": "PASSED" if total_hard_failures == 0 else "FAILED",
        },
        "categories": results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    # 2. Write summary.md
    md_lines = [
        "# Video Clipping Pipeline Evaluation Report",
        f"\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Status:** {'🟩 **ALL CHECKS PASSED**' if total_hard_failures == 0 else '🟥 **HARD CHECK FAILURES DETECTED**'}  ",
        f"**Categories Evaluated:** {passed_categories}/{total_categories} passed  ",
        f"**Total Clips:** {total_clips} | **Hard Check Failures:** {total_hard_failures}\n",
        "## Category Results Overview\n",
        "| Category | Clips | Hard Fails | Quality Score | Status | Contact Sheet |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
    ]

    for cat in results:
        status_badge = "✅ PASS" if cat["passed"] else "❌ FAIL"
        clip_count = len(cat["clips"])
        hard_fails = cat["hard_check_failures"]
        avg_score = sum(c["avg_score"] for c in cat["clips"]) / clip_count if clip_count else 0.0

        sheet_links = []
        for c in cat["clips"]:
            sheet_p = Path(c["contact_sheet"])
            rel_path = f"contact_sheets/{sheet_p.name}"
            sheet_links.append(f"[{c['title']}]({rel_path})")

        md_lines.append(
            f"| **{cat['name']}** | {clip_count} | {hard_fails} | {avg_score:.1%} | {status_badge} | {', '.join(sheet_links)} |"
        )

    md_lines.append("\n## Detailed Quality Check Matrix\n")
    md_lines.append(
        "| Category | Clip | Check Name | Hard? | Passed? | Score | Message |\n"
        "| :--- | :--- | :--- | :---: | :---: | :---: | :--- |"
    )

    for cat in results:
        for c in cat["clips"]:
            for ch in c["checks"]:
                chk_icon = "✅" if ch["passed"] else ("❌" if ch["is_hard_check"] else "⚠️")
                hard_label = "Yes" if ch["is_hard_check"] else "No"
                md_lines.append(
                    f"| {cat['name']} | {c['title']} | `{ch['name']}` | {hard_label} | {chk_icon} | {ch['score']:.2f} | {ch['message']} |"
                )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline Quality Evaluation Harness")
    parser.add_argument(
        "--config",
        type=Path,
        default=SCRIPT_DIR / "eval_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "eval_results",
        help="Directory to store evaluation reports and contact sheets",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="",
        help="Comma-separated category IDs to run (empty runs all)",
    )
    parser.add_argument(
        "--generate-missing",
        action="store_true",
        help="Generate synthetic video and transcript fixtures if missing",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort run immediately upon first hard check failure",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print final JSON report to stdout",
    )

    args = parser.parse_args()

    # Isolate Redis with FakeRedis for offline eval run
    fake_r = fakeredis.FakeRedis(decode_responses=True)
    app_main.set_redis_client(fake_r)
    app_worker.set_redis_client(fake_r)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    defaults = config.get("defaults", {})
    categories = config.get("categories", [])

    if args.categories:
        target_ids = {c.strip() for c in args.categories.split(",") if c.strip()}
        categories = [c for c in categories if c["id"] in target_ids]

    results = []
    total_hard_fails = 0

    for cat in categories:
        res = evaluate_category(
            category_cfg=cat,
            defaults=defaults,
            output_dir=args.output_dir,
            generate_missing=args.generate_missing,
            fail_fast=args.fail_fast,
        )
        results.append(res)
        total_hard_fails += res["hard_check_failures"]
        if args.fail_fast and res["hard_check_failures"] > 0:
            break

    json_path, md_path = emit_reports(results, args.output_dir)

    print("\n" + "=" * 60)
    print("EVALUATION RUN COMPLETE")
    print("=" * 60)
    print(f"Report (JSON): {json_path}")
    print(f"Summary (MD):  {md_path}")
    print(f"Total Hard Failures: {total_hard_fails}")

    if args.json:
        with open(json_path, "r", encoding="utf-8") as f:
            print(f.read())

    if total_hard_fails > 0:
        print("\n❌ RUN FAILED: One or more hard quality checks failed.")
        return 1
    else:
        print("\n✅ RUN PASSED: All quality checks satisfied.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
