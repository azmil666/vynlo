"""Tests for face-center timeline generation, bounded gap-filling, easing to neutral, and layout fallback to fit_blur."""

import os
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from app.config import settings
from app.schemas import ClipResult, JobWarning
from app.services.cropper import (
    CropFilterResult,
    DetectedFace,
    FaceSample,
    FIT_BLUR_FILTER,
    SPLIT_FILTER,
    compute_crop_filter,
    compute_detection_coverage,
    fill_face_timeline,
)
from app.worker import process_video_task


# ============================================================================
# 1. Timeline Gap Filling & Easing Unit Tests
# ============================================================================

def test_fill_timeline_all_empty():
    """When no faces are detected in any sample, timeline defaults to neutral position (0.5)."""
    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    raw = [None, None, None, None, None]
    filled = fill_face_timeline(times, raw, max_gap=1.5, ease_duration=1.0, neutral_position=0.5)
    assert len(filled) == 5
    assert all(c == 0.5 for c in filled)


def test_fill_timeline_gap_shorter_than_max():
    """Short gaps (<= max_gap) are interpolated between detections without easing to neutral."""
    # Gap between t=0.0 and t=1.0 is 1.0s (<= max_gap=1.5s)
    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    raw = [0.2, None, None, None, 0.4]
    filled = fill_face_timeline(times, raw, max_gap=1.5, ease_duration=1.0, neutral_position=0.5)
    
    assert filled[0] == 0.2
    assert filled[-1] == 0.4
    # Linearly interpolated values should increase smoothly from 0.2 to 0.4 without dropping to neutral
    assert filled[1] == pytest.approx(0.25, abs=0.01)
    assert filled[2] == pytest.approx(0.30, abs=0.01)
    assert filled[3] == pytest.approx(0.35, abs=0.01)


def test_fill_timeline_gap_longer_than_max():
    """Long gaps (> max_gap) hold up to max_gap, ease toward neutral (0.5), stay neutral, then ease into next face."""
    # Face at t=0.0 (0.1), next face at t=6.0 (0.9). max_gap=1.5s, ease=1.0s.
    times = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0, 6.0]
    raw = [0.1, None, None, None, None, None, None, None, None, 0.9]
    filled = fill_face_timeline(times, raw, max_gap=1.5, ease_duration=1.0, neutral_position=0.5)

    # At t=0.0, exactly detected face
    assert filled[0] == 0.1
    # At t=1.0 and t=1.5 (dt <= 1.5s), bounded forward fill holds 0.1
    assert filled[1] == 0.1
    assert filled[2] == 0.1
    # At t=2.0 (dt=2.0s, half-way through 1.0s ease), eases half-way from 0.1 to 0.5 -> 0.3
    assert filled[3] == pytest.approx(0.3, abs=0.01)
    # At t=2.5 and t=3.0 (dt >= 2.5s), reached and holds neutral position 0.5
    assert filled[4] == pytest.approx(0.5, abs=0.01)
    assert filled[5] == pytest.approx(0.5, abs=0.01)
    # Approaching t=6.0 from t=4.5 to t=5.0: eases toward 0.9
    # At t=5.0 (dt_to_next=1.0s <= max_gap), bounded backward fill holds 0.9
    assert filled[8] == 0.9
    assert filled[9] == 0.9


def test_fill_timeline_single_detection_in_long_clip():
    """A single isolated detection in a 10s clip is bounded, easing to neutral before and after."""
    # Single face detected at t=5.0s (center=0.85)
    times = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    raw = [None, None, None, None, 0.85, None, None, None, None]
    filled = fill_face_timeline(times, raw, max_gap=1.5, ease_duration=1.0, neutral_position=0.5)

    # Far before detection (t=0.0, t=2.0): neutral 0.5
    assert filled[0] == pytest.approx(0.5, abs=0.01)
    assert filled[1] == pytest.approx(0.5, abs=0.01)
    # Leading edge (t=4.0 is dt=1.0s <= 1.5s from t=5.0): bounded backward fill = 0.85
    assert filled[3] == 0.85
    # Exact detection at t=5.0
    assert filled[4] == 0.85
    # Trailing edge (t=6.0 is dt=1.0s <= 1.5s from t=5.0): bounded forward fill = 0.85
    assert filled[5] == 0.85
    # Far after detection (t=8.0, t=10.0): neutral 0.5
    assert filled[7] == pytest.approx(0.5, abs=0.01)
    assert filled[8] == pytest.approx(0.5, abs=0.01)


def test_fill_timeline_bounded_leading_and_trailing():
    """Bounded leading gap <= max_gap backward-fills; trailing gap <= max_gap forward-fills."""
    # First face at t=1.0s (gap=1.0s <= max_gap=1.5s)
    # Last face at t=4.0s (clip ends at 5.0s, gap=1.0s <= max_gap=1.5s)
    times = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 4.5, 5.0]
    raw = [None, None, 0.3, 0.3, 0.3, 0.7, None, None]
    filled = fill_face_timeline(times, raw, max_gap=1.5, ease_duration=1.0, neutral_position=0.5)

    # Leading samples bounded backward-fill to 0.3
    assert filled[0] == 0.3
    assert filled[1] == 0.3
    assert filled[2] == 0.3
    # Trailing samples bounded forward-fill to 0.7
    assert filled[5] == 0.7
    assert filled[6] == 0.7
    assert filled[7] == 0.7


# ============================================================================
# 2. Coverage Calculation Tests
# ============================================================================

def test_compute_detection_coverage():
    """Verify detection coverage computation for empty, full, and partial sequences."""
    # Empty sequence
    assert compute_detection_coverage([]) == 0.0

    dummy_face = DetectedFace(box=(100, 100, 200, 200), score=0.8)
    sample_with_face = FaceSample(timestamp=0.0, faces=[dummy_face])
    sample_no_face = FaceSample(timestamp=1.0, faces=[])

    # 100% coverage
    assert compute_detection_coverage([sample_with_face, sample_with_face]) == 1.0

    # 0% coverage
    assert compute_detection_coverage([sample_no_face, sample_no_face]) == 0.0

    # 25% coverage (1 face in 4 samples)
    assert compute_detection_coverage([
        sample_no_face,
        sample_with_face,
        sample_no_face,
        sample_no_face,
    ]) == pytest.approx(0.25)


# ============================================================================
# 3. Layout Selection & CropFilterResult Tests
# ============================================================================

def test_crop_filter_result_string_compatibility():
    """CropFilterResult acts as a drop-in str while exposing layout_used and coverage."""
    res = CropFilterResult("crop=606:1080:656:0", layout_used="speaker", coverage=0.75)
    assert isinstance(res, str)
    assert res == "crop=606:1080:656:0"
    assert res.startswith("crop=")
    assert res.layout_used == "speaker"
    assert res.coverage == 0.75
    assert f"-vf {res}" == "-vf crop=606:1080:656:0"


def test_compute_crop_filter_explicit_fit_blur_and_split():
    """Explicitly requested fit_blur and split layouts return their respective filter strings."""
    split_res = compute_crop_filter("dummy.mp4", 0.0, 5.0, mode="split")
    assert split_res.layout_used == "split"
    assert split_res == SPLIT_FILTER

    blur_res = compute_crop_filter("dummy.mp4", 0.0, 5.0, mode="fit_blur")
    assert blur_res.layout_used == "fit_blur"
    assert blur_res == FIT_BLUR_FILTER
    assert "boxblur" in blur_res


def test_compute_crop_filter_low_coverage_falls_back_to_fit_blur(video_16_9_30s: str):
    """When face detection coverage is below threshold (or 0 faces), layout_used is fit_blur."""
    res = compute_crop_filter(video_16_9_30s, 0.0, 3.0, mode="speaker", coverage_threshold=0.40)
    assert res.layout_used == "fit_blur"
    assert res.coverage < 0.40
    assert "boxblur=20:20" in res
    assert "overlay=" in res


def test_compute_crop_filter_force_crop_overrides_low_coverage(video_16_9_30s: str):
    """When force_crop=True is set, center crop filter is returned even with 0% face coverage."""
    res = compute_crop_filter(
        video_16_9_30s,
        0.0,
        3.0,
        mode="speaker",
        force_crop=True,
    )
    assert res.layout_used == "speaker"
    assert res.startswith("crop=606:1080:")


def test_compute_crop_filter_high_coverage_uses_speaker_crop(video_16_9_30s: str):
    """When face detection coverage is >= threshold, layout_used remains speaker with dynamic crop."""
    dummy_face = DetectedFace(box=(400, 200, 250, 300), score=0.9)
    mock_samples = [
        FaceSample(timestamp=float(t), faces=[dummy_face])
        for t in [0.0, 0.5, 1.0, 1.5, 2.0]
    ]

    with patch("app.services.cropper.sample_faces_sequentially", return_value=mock_samples):
        res = compute_crop_filter(video_16_9_30s, 0.0, 2.0, mode="speaker", coverage_threshold=0.40)
        assert res.layout_used == "speaker"
        assert res.coverage == 1.0
        assert res.startswith("crop=606:1080:")


# ============================================================================
# 4. Worker Integration: Per-Clip Info & Warning Generation
# ============================================================================

def test_worker_records_layout_fallback_warning_and_coverage(fake_redis):
    """When face coverage is below threshold, worker emits LAYOUT_FALLBACK warning and records layout_used."""
    job_id = "test_layout_fallback_job"
    
    # Mock video metadata, audio, transcript, and LLM candidate
    mock_meta = {"video_id": "test_vid", "duration": 15.0, "title": "Test Vid", "width": 1920, "height": 1080}
    from app.schemas import SourceInfo
    mock_source = SourceInfo(width=1920, height=1080, duration=15.0)
    mock_words = [
        {"word": "Hello", "start": 0.5, "end": 1.0},
        {"word": "world", "start": 1.0, "end": 1.5},
    ]
    mock_candidate = {
        "start_time": 0.0,
        "end_time": 5.0,
        "title": "Low Face Coverage Clip",
        "virality_score": 85,
        "description": "Test clip with low face coverage",
        "hashtags": ["#test"],
        "is_fallback": False,
        "selection_method": "llm",
    }

    # Low coverage samples: 0 faces
    mock_samples = [
        FaceSample(timestamp=float(t), faces=[])
        for t in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    ]

    with patch("app.worker.get_video_metadata", return_value=mock_meta), \
         patch("app.worker.download_media", return_value=("/tmp/dummy.mp4", "/tmp/dummy.wav", mock_source, [])), \
         patch("app.worker.transcribe_audio", return_value=mock_words), \
         patch("app.worker.find_viral_clips", return_value=[mock_candidate]), \
         patch("app.services.cropper.sample_faces_sequentially", return_value=mock_samples), \
         patch("app.worker.generate_ass_subtitles", return_value="/tmp/dummy.ass"), \
         patch("app.worker.render_clip_single_pass", return_value="/tmp/dummy_out.mp4"), \
         patch("app.worker.upload_clip", return_value="http://storage/clip_1.mp4"), \
         patch("cv2.VideoCapture") as mock_cap_cls:

        # Mock VideoCapture to report 1920x1080
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 1920 if prop == cv2.CAP_PROP_FRAME_WIDTH else 1080
        mock_cap_cls.return_value = mock_cap

        # Execute Celery task
        process_video_task(job_id, {"url": "https://www.youtube.com/watch?v=12345678901", "layout_mode": "speaker"})

    # Check job record in Redis
    raw_job = fake_redis.get(f"job:{job_id}")
    assert raw_job is not None

    import json
    job_data = json.loads(raw_job)
    assert job_data["status"] == "completed"
    assert len(job_data["clips"]) == 1

    clip_info = job_data["clips"][0]
    # layout_used must be fit_blur because coverage was 0.0
    assert clip_info["layout_used"] == "fit_blur"
    assert clip_info["coverage"] == 0.0
    assert clip_info["detection_coverage"] == 0.0

    # Warnings must contain LAYOUT_FALLBACK indicating layout changed from requested speaker
    warnings = job_data["warnings"]
    fallback_warnings = [w for w in warnings if w.get("code") == "LAYOUT_FALLBACK"]
    assert len(fallback_warnings) == 1
    assert fallback_warnings[0]["clip_index"] == 1
    assert "fit_blur" in fallback_warnings[0]["message"]
    assert "speaker" in fallback_warnings[0]["message"]
