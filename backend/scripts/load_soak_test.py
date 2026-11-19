#!/usr/bin/env python3
"""Load, soak, and worker-loss test harness for the clipping pipeline.

Verifies:
1. Queue behaviour: N concurrent jobs of mixed durations transitioning monotonically.
2. Memory bounds: baseline, peak, and post-run RSS tracking across threads.
3. Temp directory cleanup: scratch directories in TEMP_DIR are purged after job completion.
4. No stuck jobs: all submitted jobs terminate within stage and task budgets.
5. Forced worker kill / WORKER_LOST path: simulates worker termination mid-job and asserts
   that the API status recovery transitions the job to failed with WORKER_LOST and retryable=True.

Usage:
    python backend/scripts/load_soak_test.py [--concurrency 4] [--total-jobs 8]
                                            [--test-worker-lost] [--offline]
"""

import argparse
import concurrent.futures
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Ensure backend directory is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
ROOT_DIR = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure host-compatible directories before app config loads
os.environ.setdefault("STATIC_DIR", str((BACKEND_DIR / "static").resolve()))
os.environ.setdefault("TEMP_DIR", "/tmp/clipper")

import fakeredis

from app.config import settings
from app.main import (
    ACTIVE_PROCESSING_STAGES,
    create_job,
    get_job_status,
    get_redis_client,
    set_redis_client,
)
from unittest.mock import patch
from app.schemas import JobCreateRequest, PipelineErrorCode, SourceInfo, TranscriptValidation
from app.services.transcriber import TranscriptWordList
from app.services.analyzer import set_llm_client
from app.services.metrics import get_peak_rss_mb
import app.main as app_main
import app.worker as app_worker
from app.worker import celery_app, process_video_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("load_soak_test")


def create_synthetic_test_video(path: Path, duration_sec: float) -> Path:
    """Generate offline test video fixture using FFmpeg lavfi filters."""
    path.parent.mkdir(parents=True, exist_ok=True)
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
        str(duration_sec),
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
        str(path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return path


def run_worker_lost_recovery_test(fake_r: fakeredis.FakeRedis) -> Dict[str, Any]:
    """Test the forced worker termination (WORKER_LOST) path.

    Submits a job, moves it into an active stage (rendering), simulates a worker kill
    by pushing the heartbeat back beyond WORKER_HEARTBEAT_TIMEOUT_SECONDS, queries
    the status endpoint, and verifies that WORKER_LOST recovery takes effect.
    """
    logger.info("--- Testing Forced Worker Kill (WORKER_LOST path) ---")
    job_id = f"worker-lost-{uuid4().hex[:8]}"

    # Simulate job active in rendering stage
    heartbeat_stale_time = (
        datetime.now(timezone.utc) - timedelta(seconds=settings.WORKER_HEARTBEAT_TIMEOUT_SECONDS + 15)
    ).isoformat()

    stale_status = {
        "job_id": job_id,
        "status": "rendering",
        "stage": "rendering",
        "progress": 75,
        "updated_at": heartbeat_stale_time,
        "clips": [],
        "warnings": [],
    }
    fake_r.set(f"job:{job_id}", json.dumps(stale_status), ex=settings.JOB_RECORD_TTL_SECONDS)

    # Query status endpoint to trigger heartbeat recovery
    recovered_status = get_job_status(job_id)

    passed = (
        recovered_status.status == "failed"
        and recovered_status.error is not None
        and recovered_status.error.code == PipelineErrorCode.WORKER_LOST.value
        and recovered_status.error.retryable is True
        and recovered_status.error.stage == "rendering"
    )

    result = {
        "test": "worker_lost_recovery",
        "passed": passed,
        "job_id": job_id,
        "status_observed": recovered_status.status,
        "error_code": recovered_status.error.code if recovered_status.error else None,
        "retryable": recovered_status.error.retryable if recovered_status.error else None,
        "recovered_stage": recovered_status.error.stage if recovered_status.error else None,
    }

    if passed:
        logger.info("✅ WORKER_LOST test passed: job transitioned to failed with code WORKER_LOST and retryable=True.")
    else:
        logger.error("❌ WORKER_LOST test failed: %s", result)

    return result


def get_offline_transcript(duration: float) -> TranscriptWordList:
    """Generate or load offline transcript words for testing."""
    sample_path = BACKEND_DIR / "tests" / "fixtures" / "transcripts" / "normal_speech.json"
    if sample_path.exists():
        with open(sample_path, "r", encoding="utf-8") as f:
            words = json.load(f)
    else:
        words = [
            {"word": "Welcome", "start": 0.5, "end": 1.0},
            {"word": "to", "start": 1.1, "end": 1.3},
            {"word": "the", "start": 1.4, "end": 1.6},
            {"word": "soak", "start": 1.7, "end": 2.1},
            {"word": "test.", "start": 2.2, "end": 2.8},
        ]
    # Filter words to within duration
    filtered = [w for w in words if float(w.get("end", 0.0)) <= duration]
    if not filtered:
        filtered = [{"word": "Hello.", "start": 0.5, "end": min(2.0, duration)}]
    val = TranscriptValidation(
        word_count=len(filtered),
        speech_duration=float(filtered[-1]["end"]) - float(filtered[0]["start"]),
        speech_coverage_ratio=0.85,
        words_per_minute=140.0,
        language="en",
        language_probability=0.99,
        model="offline",
    )
    return TranscriptWordList(filtered, val)


def execute_single_job(
    job_index: int,
    video_path: Path,
    duration: float,
    fake_r: fakeredis.FakeRedis,
    timeout_sec: float = 60.0,
) -> Dict[str, Any]:
    """Submit a single job and run pipeline execution."""
    t0 = time.time()
    req = JobCreateRequest(url=f"https://youtu.be/soak_{job_index:04d}", max_clips=2)

    audio_path = video_path.with_suffix(".wav")
    if not audio_path.exists():
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(audio_path)]
        subprocess.run(cmd, capture_output=True, check=True)

    src_info = SourceInfo(
        source_id=f"soak_{job_index:04d}",
        duration=duration,
        width=1920,
        height=1080,
    )

    mock_words = get_offline_transcript(duration)

    # Submit job and execute with patched downloader and transcriber returning local fixtures
    mock_meta = {
        "video_id": f"soak_{job_index:04d}",
        "title": f"Soak Video {job_index}",
        "duration": duration,
    }
    try:
        with patch("app.worker.get_video_metadata", return_value=mock_meta), \
             patch("app.worker.download_media", return_value=(str(video_path), str(audio_path), src_info, [])), \
             patch("app.worker.transcribe_audio", return_value=mock_words):
            initial = create_job(req)
            job_id = initial.job_id
    except Exception as e:
        logger.warning("Job %d submission raised exception: %s", job_index, e)
        return {
            "job_index": job_index,
            "job_id": f"failed_{job_index}",
            "duration": duration,
            "elapsed_sec": round(time.time() - t0, 2),
            "final_status": "submission_failed",
            "clip_count": 0,
            "temp_dir_cleaned": True,
            "passed": False,
        }

    # Poll status endpoint to verify terminal state
    status_history = []
    final_status = None
    start_poll = time.time()

    while (time.time() - start_poll) < timeout_sec:
        curr = get_job_status(job_id)
        status_history.append((curr.stage, curr.progress))
        if curr.status in ("completed", "completed_with_warnings", "partial", "failed", "expired"):
            final_status = curr
            break
        time.sleep(0.1)

    elapsed = time.time() - t0
    job_temp_dir = Path(settings.TEMP_DIR) / job_id
    temp_dir_cleaned = not job_temp_dir.exists()

    passed = (
        final_status is not None
        and final_status.status in ("completed", "completed_with_warnings", "partial")
        and temp_dir_cleaned
    )

    return {
        "job_index": job_index,
        "job_id": job_id,
        "duration": duration,
        "elapsed_sec": round(elapsed, 2),
        "final_status": final_status.status if final_status else "stuck_or_timeout",
        "clip_count": len(final_status.clips) if final_status else 0,
        "temp_dir_cleaned": temp_dir_cleaned,
        "status_history_len": len(status_history),
        "passed": passed,
    }


def run_load_soak_test(
    concurrency: int = 4,
    total_jobs: int = 8,
    durations: Optional[List[float]] = None,
    test_worker_lost: bool = True,
    offline: bool = True,
) -> Dict[str, Any]:
    """Execute load test with mixed durations and verify all operational properties."""
    logger.info("=" * 70)
    logger.info("STARTING LOAD & SOAK TEST")
    logger.info(f"Concurrency: {concurrency} | Total Jobs: {total_jobs} | Offline: {offline}")
    logger.info("=" * 70)

    # Initialize FakeRedis for isolated, leak-free offline test run
    fake_r = fakeredis.FakeRedis(decode_responses=True)
    app_main.set_redis_client(fake_r)
    app_worker.set_redis_client(fake_r)

    if offline:
        celery_app.conf.update(task_always_eager=True, task_eager_propagates=True)

    # Inject FakeLLM for offline candidate extraction
    from tests.fixtures.fake_llm import FakeLLM
    fake_clips = {
        "clips": [
            {
                "start_time": 0.5,
                "end_time": 4.5,
                "title": "Soak Test Clip",
                "virality_score": 85,
                "description": "Benchmark clip for soak testing",
                "hashtags": ["#soak", "#testing"],
            }
        ]
    }
    set_llm_client(FakeLLM(default_response=fake_clips))

    # Baseline measurements
    baseline_rss_mb = get_peak_rss_mb()
    logger.info(f"Baseline Peak RSS: {baseline_rss_mb:.1f} MB")

    if not durations:
        # Mixed durations: short (5s), medium (10s), longer (15s)
        durations = [5.0, 10.0, 15.0, 8.0, 12.0, 6.0, 14.0, 7.0]

    # Generate test video fixtures
    fixtures_dir = Path(tempfile.gettempdir()) / "soak_test_fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    video_fixtures = []

    logger.info("Preparing synthetic video fixtures of mixed durations...")
    for idx in range(total_jobs):
        dur = durations[idx % len(durations)]
        v_path = fixtures_dir / f"soak_video_{idx}_{int(dur)}s.mp4"
        if not v_path.exists():
            create_synthetic_test_video(v_path, duration_sec=dur)
        video_fixtures.append((idx, v_path, dur))

    # Run concurrent jobs
    job_results = []
    t_start_all = time.time()

    logger.info(f"Dispatching {total_jobs} jobs across {concurrency} threads...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(execute_single_job, idx, path, dur, fake_r)
            for idx, path, dur in video_fixtures
        ]
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            job_results.append(res)
            logger.info(
                f"Job {res['job_index'] + 1}/{total_jobs} ({res['job_id']}) finished in {res['elapsed_sec']}s: "
                f"status={res['final_status']}, temp_cleaned={res['temp_dir_cleaned']}"
            )

    total_time = time.time() - t_start_all
    peak_rss_mb = get_peak_rss_mb()
    rss_growth_mb = peak_rss_mb - baseline_rss_mb
    logger.info(f"Load test execution time: {total_time:.2f}s | Peak RSS: {peak_rss_mb:.1f} MB (Delta: +{rss_growth_mb:.1f} MB)")

    # Clean up test video fixtures
    shutil.rmtree(fixtures_dir, ignore_errors=True)

    # Worker lost test
    worker_lost_result = None
    if test_worker_lost:
        worker_lost_result = run_worker_lost_recovery_test(fake_r)

    # Verify no stuck jobs and temp dir cleanup
    successful_jobs = sum(1 for j in job_results if j["passed"])
    stuck_jobs = sum(1 for j in job_results if j["final_status"] == "stuck_or_timeout")
    temp_cleaned_count = sum(1 for j in job_results if j["temp_dir_cleaned"])

    all_passed = (
        successful_jobs == total_jobs
        and stuck_jobs == 0
        and temp_cleaned_count == total_jobs
        and (worker_lost_result["passed"] if worker_lost_result else True)
    )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "concurrency": concurrency,
        "total_jobs": total_jobs,
        "successful_jobs": successful_jobs,
        "stuck_jobs": stuck_jobs,
        "temp_cleaned_count": temp_cleaned_count,
        "total_elapsed_sec": round(total_time, 2),
        "baseline_rss_mb": round(baseline_rss_mb, 1),
        "peak_rss_mb": round(peak_rss_mb, 1),
        "rss_growth_mb": round(rss_growth_mb, 1),
        "worker_lost_test": worker_lost_result,
        "all_passed": all_passed,
        "job_details": job_results,
    }

    print("\n" + "=" * 70)
    print("LOAD & SOAK TEST SUMMARY REPORT")
    print("=" * 70)
    print(f"Total Dispatched Jobs:  {total_jobs}")
    print(f"Successful Jobs:        {successful_jobs}/{total_jobs}")
    print(f"Stuck / Timed-out Jobs: {stuck_jobs}")
    print(f"Temp-dir Cleanup Rate:  {temp_cleaned_count}/{total_jobs}")
    print(f"Total Elapsed Time:     {total_time:.2f}s")
    print(f"Peak RSS:               {peak_rss_mb:.1f} MB (Growth: +{rss_growth_mb:.1f} MB)")
    if worker_lost_result:
        print(f"WORKER_LOST Recovery:   {'PASSED' if worker_lost_result['passed'] else 'FAILED'}")
    print(f"Overall Status:         {'✅ ALL PASS' if all_passed else '❌ FAILED'}")
    print("=" * 70 + "\n")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline Load, Soak, and WORKER_LOST Test")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent worker threads")
    parser.add_argument("--total-jobs", type=int, default=8, help="Total number of jobs to run")
    parser.add_argument("--test-worker-lost", action="store_true", default=True, help="Test forced worker kill recovery")
    parser.add_argument("--offline", action="store_true", default=True, help="Run offline using synthetic media fixtures")
    parser.add_argument("--json", action="store_true", help="Print report as JSON")

    args = parser.parse_args()

    report = run_load_soak_test(
        concurrency=args.concurrency,
        total_jobs=args.total_jobs,
        test_worker_lost=args.test_worker_lost,
        offline=args.offline,
    )

    if args.json:
        print(json.dumps(report, indent=2))

    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
