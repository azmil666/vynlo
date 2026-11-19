"""Unit tests for the evaluation harness (scripts/eval/).

Tests all 10 quality checks, contact sheet generator, and report outputs:
1. output spec (ffprobe)
2. sentence boundary
3. no word clipping
4. caption / audio sync
5. crop jumps
6. face-in-frame ratio
7. non-black-bar ratio
8. loudness within tolerance
9. duration bounds
10. warnings present when degraded
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import pytest

# Ensure repository root is on sys.path so scripts.eval is importable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.checks import (
    CheckResult,
    check_caption_audio_sync,
    check_crop_jumps,
    check_duration_bounds,
    check_face_in_frame_ratio,
    check_faststart,
    check_loudness_tolerance,
    check_no_word_clipping,
    check_non_black_bar_ratio,
    check_output_spec,
    check_sentence_boundary,
    check_trailing_frame_decodable,
    check_warnings_when_degraded,
)
from scripts.eval.contact_sheet import extract_frame_at_timestamp, format_timestamp, generate_contact_sheet
from scripts.eval.dataset_generator import _generate_synthetic_transcript, generate_synthetic_video
from scripts.eval.run_eval import emit_reports


@pytest.fixture
def vertical_clip(tmp_path: Path) -> Path:
    """Generate a 1080x1920 6-second vertical test clip with faststart and stereo aac."""
    out = tmp_path / "valid_vertical.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=1080x1920:rate=30",
        "-f",
        "lavfi",
        "-i",
        "sine=f=1000:r=48000",
        "-t",
        "6",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        str(out),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return out


@pytest.fixture
def horizontal_clip(tmp_path: Path) -> Path:
    """Generate a 1920x1080 3-second horizontal test clip (invalid spec for 9:16 delivery)."""
    out = tmp_path / "invalid_horizontal.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=1920x1080:rate=30",
        "-f",
        "lavfi",
        "-i",
        "sine=f=1000:r=48000",
        "-t",
        "3",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(out),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return out


# ============================================================================
# Check 1: Output Spec
# ============================================================================


def test_check_output_spec_valid(vertical_clip: Path):
    result = check_output_spec(vertical_clip)
    assert result.passed is True
    assert result.score == 1.0
    assert result.is_hard_check is True
    assert "Output spec validated" in result.message
    assert result.details["video"]["width"] == 1080
    assert result.details["video"]["height"] == 1920
    assert result.details["video"]["codec"] in ("h264", "libx264")
    assert result.details["audio"]["codec"] == "aac"
    assert result.details["audio"]["channels"] == 2
    assert result.details["faststart"] is True


def test_check_output_spec_invalid_dimensions(horizontal_clip: Path):
    result = check_output_spec(horizontal_clip)
    assert result.passed is False
    assert result.is_hard_check is True
    assert "Invalid dimensions: 1920x1080" in result.message


def test_check_output_spec_nonexistent():
    result = check_output_spec("/tmp/does_not_exist_abc123.mp4")
    assert result.passed is False
    assert "does not exist" in result.message


# ============================================================================
# Check 2: Sentence Boundary
# ============================================================================


def test_check_sentence_boundary_clean():
    words = [
        {"word": "Hello.", "start": 0.5, "end": 1.0},
        {"word": "Welcome", "start": 1.8, "end": 2.2},  # 0.8s pause before
        {"word": "everyone.", "start": 2.3, "end": 3.0},
    ]
    result = check_sentence_boundary(clip_start=1.8, clip_end=3.0, words=words)
    assert result.passed is True
    assert result.score == 1.0
    assert result.details["start_clean"] is True
    assert result.details["end_clean"] is True


def test_check_sentence_boundary_mid_sentence():
    words = [
        {"word": "Hello", "start": 0.5, "end": 0.9},
        {"word": "there", "start": 1.0, "end": 1.4},
        {"word": "friend", "start": 1.5, "end": 2.0},
    ]
    # Slicing at 1.0 (no pause, no punctuation) and ending at 1.4 (no punctuation, no pause)
    result = check_sentence_boundary(clip_start=1.0, clip_end=1.4, words=words)
    assert result.details["start_clean"] is False
    assert result.details["end_clean"] is False
    assert result.score == 0.0


# ============================================================================
# Check 3: No Word Clipping
# ============================================================================


def test_check_no_word_clipping_clean():
    words = [
        {"word": "First", "start": 1.0, "end": 1.8},
        {"word": "Second", "start": 2.2, "end": 3.0},
    ]
    # Boundaries fall in the 1.8 - 2.2 gap
    result = check_no_word_clipping(clip_start=2.0, clip_end=3.5, words=words)
    assert result.passed is True
    assert result.is_hard_check is True
    assert result.score == 1.0
    assert len(result.details["clipped_words"]) == 0


def test_check_no_word_clipping_sliced():
    words = [
        {"word": "Crucial", "start": 1.0, "end": 2.0},
    ]
    # Boundary cuts right into the middle of 'Crucial' at 1.5s
    result = check_no_word_clipping(clip_start=1.5, clip_end=3.0, words=words)
    assert result.passed is False
    assert result.score == 0.0
    assert len(result.details["clipped_words"]) == 1
    assert result.details["clipped_words"][0]["word"] == "Crucial"


# ============================================================================
# Check 4: Caption / Audio Sync
# ============================================================================


def test_check_caption_audio_sync_valid(tmp_path: Path):
    ass_file = tmp_path / "test.ass"
    ass_file.write_text(
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:00.50,0:00:01.20,Default,,0,0,0,,Hello world\n"
        "Dialogue: 0,0:00:01.50,0:00:02.50,Default,,0,0,0,,Next phrase\n"
    )
    words = [
        {"word": "Hello", "start": 10.5, "end": 11.2},
        {"word": "Next", "start": 11.5, "end": 12.5},
    ]
    result = check_caption_audio_sync(
        ass_path=ass_file,
        clip_start=10.0,
        clip_end=15.0,
        words=words,
    )
    assert result.passed is True
    assert result.score == 1.0


def test_check_caption_audio_sync_out_of_bounds(tmp_path: Path):
    ass_file = tmp_path / "bad.ass"
    ass_file.write_text(
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:35.00,0:00:40.00,Default,,0,0,0,,Too late\n"
    )
    result = check_caption_audio_sync(
        ass_path=ass_file,
        clip_start=0.0,
        clip_end=10.0,
        words=[],
    )
    assert result.passed is False
    assert "outside clip bounds" in result.message


# ============================================================================
# Check 5: Crop Jumps
# ============================================================================


def test_check_crop_jumps_static():
    filter_str = "crop=1080:1920:420:0"
    result = check_crop_jumps(filter_str)
    assert result.passed is True
    assert result.score == 1.0
    assert result.details["max_speed"] == 0.0


def test_check_crop_jumps_smooth_pan():
    # 200 px transition over 1.0s = 200 px/s (<= 320 px/s)
    filter_str = "crop=1080:1920:between(t,1.0,2.0)*(100 + (200)*(t-1.0)/1.0):0"
    result = check_crop_jumps(filter_str, max_speed_px_per_sec=320.0)
    assert result.passed is True
    assert result.details["max_observed_speed"] == 200.0


def test_check_crop_jumps_teleport_violation():
    # 500 px transition over 0.5s = 1000 px/s (> 320 px/s)
    filter_str = "crop=1080:1920:between(t,1.0,1.5)*(100 + (500)*(t-1.0)/0.5):0"
    result = check_crop_jumps(filter_str, max_speed_px_per_sec=320.0)
    assert result.passed is False
    assert len(result.details["violations"]) == 1


# ============================================================================
# Check 6: Face in Frame Ratio
# ============================================================================


def test_check_face_in_frame_ratio_not_expected(vertical_clip: Path):
    result = check_face_in_frame_ratio(vertical_clip, is_face_expected=False)
    assert result.passed is True
    assert result.details["is_face_expected"] is False


# ============================================================================
# Check 7: Non-Black-Bar Ratio
# ============================================================================


def test_check_non_black_bar_ratio_clean(vertical_clip: Path):
    result = check_non_black_bar_ratio(vertical_clip)
    assert result.passed is True
    assert result.details["max_black_ratio"] < 0.15


# ============================================================================
# Check 8: Loudness within Tolerance
# ============================================================================


def test_check_loudness_tolerance(vertical_clip: Path):
    result = check_loudness_tolerance(vertical_clip, target_lufs=-14.0, lufs_tol=30.0)
    assert result.details["input_i"] is not None
    assert result.details["input_tp"] is not None


# ============================================================================
# Check 9: Duration Bounds
# ============================================================================


def test_check_duration_bounds(vertical_clip: Path):
    # Probed 6s vertical clip
    result = check_duration_bounds(vertical_clip, min_dur=5.0, max_dur=10.0, requested_start=0.0, requested_end=6.0)
    assert result.passed is True
    assert result.details["duration"] == pytest.approx(6.0, abs=0.2)


def test_check_duration_bounds_too_short(vertical_clip: Path):
    result = check_duration_bounds(vertical_clip, min_dur=10.0, max_dur=60.0)
    assert result.passed is False
    assert "below minimum" in result.message


# ============================================================================
# Check 10: Warnings When Degraded
# ============================================================================


def test_check_warnings_when_degraded_matched():
    clip_warnings = ["LAYOUT_FALLBACK: face not detected"]
    result = check_warnings_when_degraded(clip_warnings, expected_degradation="LAYOUT_FALLBACK")
    assert result.passed is True
    assert result.score == 1.0


def test_check_warnings_when_degraded_missing():
    clip_warnings = ["FONT_FALLBACK"]
    result = check_warnings_when_degraded(clip_warnings, expected_degradation="LAYOUT_FALLBACK")
    assert result.passed is False
    assert result.score == 0.0


# ============================================================================
# Contact Sheet & Reports
# ============================================================================


def test_contact_sheet_generation(vertical_clip: Path, tmp_path: Path):
    out_img = tmp_path / "contact.jpg"
    generate_contact_sheet(
        clip_path=vertical_clip,
        output_image_path=out_img,
        title="Test Sheet",
    )
    assert out_img.exists()
    assert out_img.stat().st_size > 1000


def test_emit_reports(tmp_path: Path):
    results = [
        {
            "id": "test_cat",
            "name": "Test Category",
            "description": "Category for unit test",
            "passed": True,
            "hard_check_failures": 0,
            "clips": [
                {
                    "clip_index": 0,
                    "title": "Clip 1",
                    "contact_sheet": str(tmp_path / "sheet.jpg"),
                    "avg_score": 1.0,
                    "checks": [
                        {
                            "name": "output_spec",
                            "passed": True,
                            "is_hard_check": True,
                            "score": 1.0,
                            "message": "Validated",
                        }
                    ],
                }
            ],
        }
    ]
    json_path, md_path = emit_reports(results, tmp_path)
    assert json_path.exists()
    assert md_path.exists()

    with open(json_path) as f:
        data = json.load(f)
    assert data["summary"]["overall_status"] == "PASSED"
    assert data["summary"]["total_hard_failures"] == 0

    md_content = md_path.read_text()
    assert "ALL CHECKS PASSED" in md_content
    assert "Test Category" in md_content
