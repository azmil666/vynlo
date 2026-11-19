"""Tests for active speaker detection, multi-face tracking, hysteresis switching,
dual-speaker framing, and unreliable landmark fallbacks.
"""

import math
from typing import Optional
import pytest

from app.config import settings
from app.services.cropper import (
    CropFilterResult,
    DetectedFace,
    FaceLandmarks,
    FaceSample,
    FaceTrack,
    assess_speaker_tracking_reliability,
    calculate_box_iou,
    compute_normalized_mouth_distance,
    compute_track_speaking_score,
    is_speech_active_at,
    select_active_speaker_trajectory,
    track_faces_across_samples,
)


def _make_face(
    x: float,
    y: float,
    w: float = 160.0,
    h: float = 160.0,
    mouth_w: float = 40.0,
    score: float = 0.95,
) -> DetectedFace:
    """Helper to create a DetectedFace with specified coordinates and mouth width."""
    cx = x + w / 2.0
    cy = y + h / 2.0
    half_m = mouth_w / 2.0
    landmarks = FaceLandmarks(
        right_eye=(cx - 30.0, cy - 25.0),
        left_eye=(cx + 30.0, cy - 25.0),
        nose_tip=(cx, cy),
        mouth_right=(cx - half_m, cy + 40.0),
        mouth_left=(cx + half_m, cy + 40.0),
    )
    return DetectedFace(
        box=(float(x), float(y), float(w), float(h)),
        score=score,
        landmarks=landmarks,
    )


# ============================================================================
# 1. Multi-Face Tracking & Spatial Association Tests
# ============================================================================

def test_track_faces_across_samples_association():
    """Verify that multiple faces in frame are correctly tracked into persistent tracks across time."""
    samples: list[FaceSample] = []
    # 8 samples over 2 seconds: Person A at x=300, Person B at x=1200
    for idx in range(8):
        t = idx * 0.25
        # Slight jitter / natural motion in box position (+- 3px)
        jitter = 3.0 if idx % 2 == 0 else -3.0
        face_a = _make_face(x=300.0 + jitter, y=300.0, w=150.0, h=150.0)
        face_b = _make_face(x=1200.0 - jitter, y=300.0, w=150.0, h=150.0)
        samples.append(FaceSample(timestamp=t, faces=[face_a, face_b]))

    tracks = track_faces_across_samples(samples)
    assert len(tracks) == 2, f"Expected exactly 2 tracks, got {len(tracks)}"
    assert tracks[0].count == 8
    assert tracks[1].count == 8

    # Verify track centers separate correctly
    t0_x = tracks[0].detections[0].face.center_x
    t1_x = tracks[1].detections[0].face.center_x
    assert abs(t0_x - 375.0) < 10.0 or abs(t0_x - 1275.0) < 10.0
    assert abs(t1_x - 375.0) < 10.0 or abs(t1_x - 1275.0) < 10.0
    assert abs(t0_x - t1_x) > 500.0


def test_calculate_box_iou():
    """IoU computation correctness on overlapping, identical, and disjoint boxes."""
    b1 = (100.0, 100.0, 100.0, 100.0)
    # Identical
    assert calculate_box_iou(b1, b1) == pytest.approx(1.0)
    # Disjoint
    b2 = (300.0, 300.0, 100.0, 100.0)
    assert calculate_box_iou(b1, b2) == pytest.approx(0.0)
    # 50% horizontal overlap, identical y
    b3 = (150.0, 100.0, 100.0, 100.0)
    # Inter: 50 * 100 = 5000; Union: 10000 + 10000 - 5000 = 15000 -> 1/3
    assert calculate_box_iou(b1, b3) == pytest.approx(5000.0 / 15000.0)


# ============================================================================
# 2. Speaking Score & Mouth Motion Tests
# ============================================================================

def test_compute_normalized_mouth_distance_and_variance():
    """Normalized mouth distance and variance distinguish moving mouths from still mouths."""
    face_still = _make_face(x=100.0, y=100.0, w=200.0, h=200.0, mouth_w=40.0)
    d_norm = compute_normalized_mouth_distance(face_still)
    assert d_norm == pytest.approx(40.0 / 200.0)

    # Build tracks: one still, one speaking with varying mouth width
    still_samples: list[FaceSample] = []
    speaking_samples: list[FaceSample] = []
    for i in range(8):
        t = i * 0.25
        # Still face: constant 40px mouth
        still_samples.append(FaceSample(timestamp=t, faces=[_make_face(100.0, 100.0, mouth_w=40.0)]))
        # Speaking face: mouth oscillates between 30px and 60px
        osc_w = 45.0 + 15.0 * math.sin(i * 1.5)
        speaking_samples.append(FaceSample(timestamp=t, faces=[_make_face(500.0, 100.0, mouth_w=osc_w)]))

    still_track = track_faces_across_samples(still_samples)[0]
    speaking_track = track_faces_across_samples(speaking_samples)[0]

    still_score = compute_track_speaking_score(still_track, t=1.0, window_sec=1.0, noise_floor=1e-4)
    speaking_score = compute_track_speaking_score(speaking_track, t=1.0, window_sec=1.0, noise_floor=1e-4)

    assert still_score == 0.0, f"Still face should have 0 speaking score, got {still_score}"
    assert speaking_score > 5e-4, f"Speaking face should have significant score, got {speaking_score}"


# ============================================================================
# 3. Two Synthetic Faces Alternating Mouth Motion (Active Speaker Switching)
# ============================================================================

def test_two_synthetic_faces_alternating_mouth_motion():
    """Two distant faces where Speaker A speaks first, then Speaker B speaks.

    Camera must stay on Speaker A during Speaker A's speech, and switch to Speaker B
    only after minimum dwell time is satisfied.
    """
    frame_w, frame_h = 1920, 1080
    crop_w = int(frame_h * 9 / 16)  # 607 px
    samples: list[FaceSample] = []

    # Total 6 seconds at 4 Hz (25 samples)
    # Face A at x=300 (norm center ~0.20), Face B at x=1400 (norm center ~0.77)
    # t in [0.0, 2.5]: Face A speaks (mouth varies 30-55px), Face B is still (35px)
    # t in [2.5, 6.0]: Face A is still (35px), Face B speaks (mouth varies 30-55px)
    words = [
        {"word": "hello", "start": 0.1, "end": 2.4},
        {"word": "there", "start": 2.6, "end": 5.9},
    ]

    for i in range(25):
        t = i * 0.25
        if t <= 2.5:
            mouth_a = 42.0 + 12.0 * math.sin(i * 1.8)
            mouth_b = 35.0
        else:
            mouth_a = 35.0
            mouth_b = 42.0 + 12.0 * math.sin(i * 1.8)

        face_a = _make_face(x=300.0, y=300.0, mouth_w=mouth_a)
        face_b = _make_face(x=1400.0, y=300.0, mouth_w=mouth_b)
        samples.append(FaceSample(timestamp=t, faces=[face_a, face_b]))

    raw_centers, warnings = select_active_speaker_trajectory(
        samples=samples,
        frame_w=frame_w,
        frame_h=frame_h,
        crop_w=crop_w,
        words=words,
        min_dwell_sec=2.0,
        switch_margin=1.3,
    )

    assert not warnings, f"Expected no warnings, got {warnings}"
    assert len(raw_centers) == 25

    # Early samples (t < 2.0s): must track Face A (center_x ~ 380px / 1920 ~ 0.198)
    for idx in range(0, 8):
        c = raw_centers[idx]
        assert c is not None
        assert abs(c - (380.0 / 1920.0)) < 0.05, f"Sample {idx} at t={samples[idx].timestamp} was {c}, expected Face A (~0.20)"

    # Later samples (t >= 4.5s): dwell time has passed and Face B was speaking -> must track Face B (center_x ~ 1480px / 1920 ~ 0.771)
    for idx in range(18, 25):
        c = raw_centers[idx]
        assert c is not None
        assert abs(c - (1480.0 / 1920.0)) < 0.05, f"Sample {idx} at t={samples[idx].timestamp} was {c}, expected Face B (~0.77)"


# ============================================================================
# 4. Same-Size Faces (No Flip-Flopping / Hysteresis Stability)
# ============================================================================

def test_same_size_faces_no_flip_flopping():
    """When two identical-size faces are both silent or equally active, hysteresis prevents flip-flopping."""
    frame_w, frame_h = 1920, 1080
    crop_w = int(frame_h * 9 / 16)
    samples: list[FaceSample] = []

    # 16 samples over 4 seconds: two silent faces at x=300 and x=1400
    for i in range(16):
        t = i * 0.25
        face_a = _make_face(x=300.0, y=300.0, w=160.0, h=160.0, mouth_w=40.0)
        face_b = _make_face(x=1400.0, y=300.0, w=160.0, h=160.0, mouth_w=40.0)
        samples.append(FaceSample(timestamp=t, faces=[face_a, face_b]))

    raw_centers, warnings = select_active_speaker_trajectory(
        samples=samples,
        frame_w=frame_w,
        frame_h=frame_h,
        crop_w=crop_w,
        words=None,
    )

    assert not warnings
    # All centers should remain on whichever face was initially selected
    initial_center = raw_centers[0]
    assert initial_center is not None

    for idx, c in enumerate(raw_centers):
        assert c is not None
        assert abs(c - initial_center) < 0.01, f"Frame {idx} flipped from {initial_center} to {c}!"


# ============================================================================
# 5. Dual-Speaker Framing When Both Fit Inside 9:16 Crop
# ============================================================================

def test_dual_speaker_framing_when_close():
    """When two speakers are close enough that a 9:16 crop fits both, both are framed simultaneously."""
    frame_w, frame_h = 1920, 1080
    crop_w = int(frame_h * 9 / 16)  # ~607 px
    # Max dual span = 607 * 0.85 = 516 px
    # Place Face A at x=750 (w=150) -> [750, 900]
    # Place Face B at x=980 (w=150) -> [980, 1130]
    # Total span = 1130 - 750 = 380 px <= 516 px!
    samples: list[FaceSample] = []
    for i in range(10):
        t = i * 0.25
        # Even with one person speaking and one silent
        mouth_a = 45.0 + 10.0 * math.sin(i * 1.5)
        mouth_b = 38.0
        face_a = _make_face(x=750.0, y=300.0, w=150.0, h=150.0, mouth_w=mouth_a)
        face_b = _make_face(x=980.0, y=300.0, w=150.0, h=150.0, mouth_w=mouth_b)
        samples.append(FaceSample(timestamp=t, faces=[face_a, face_b]))

    raw_centers, warnings = select_active_speaker_trajectory(
        samples=samples,
        frame_w=frame_w,
        frame_h=frame_h,
        crop_w=crop_w,
    )

    assert not warnings
    # Expected midpoint: (750 + 1130) / 2 = 940 px -> 940 / 1920 ~ 0.4896
    expected_dual_center = 940.0 / 1920.0
    for idx, c in enumerate(raw_centers):
        assert c is not None
        assert abs(c - expected_dual_center) < 0.02, f"Sample {idx} center {c} differed from dual center {expected_dual_center}"


# ============================================================================
# 6. Single-Face Clips Unchanged
# ============================================================================

def test_single_face_clips_unchanged():
    """Clips with a single detected face track follow that face directly with zero warnings."""
    frame_w, frame_h = 1920, 1080
    crop_w = int(frame_h * 9 / 16)
    samples: list[FaceSample] = []

    # Single face moving across frame
    for i in range(12):
        t = i * 0.25
        x = 500.0 + i * 20.0
        face = _make_face(x=x, y=200.0, w=180.0, h=180.0)
        samples.append(FaceSample(timestamp=t, faces=[face]))

    raw_centers, warnings = select_active_speaker_trajectory(
        samples=samples,
        frame_w=frame_w,
        frame_h=frame_h,
        crop_w=crop_w,
    )

    assert not warnings
    for i, c in enumerate(raw_centers):
        expected_c = (500.0 + i * 20.0 + 90.0) / 1920.0
        assert c == pytest.approx(expected_c, abs=1e-3)


# ============================================================================
# 7. Unreliable Signal Fallback & Warnings
# ============================================================================

def test_unreliable_signal_fallback_low_resolution():
    """Tiny faces (<35px) trigger SPEAKER_TRACKING_FALLBACK and fall back to largest stable track."""
    frame_w, frame_h = 1920, 1080
    crop_w = int(frame_h * 9 / 16)
    samples: list[FaceSample] = []

    # Two very small distant faces (20x20 pixels)
    for i in range(8):
        t = i * 0.25
        face_a = _make_face(x=200.0, y=200.0, w=20.0, h=20.0, mouth_w=4.0)
        face_b = _make_face(x=800.0, y=200.0, w=25.0, h=25.0, mouth_w=5.0)
        samples.append(FaceSample(timestamp=t, faces=[face_a, face_b]))

    raw_centers, warnings = select_active_speaker_trajectory(
        samples=samples,
        frame_w=frame_w,
        frame_h=frame_h,
        crop_w=crop_w,
    )

    assert len(warnings) == 1
    assert "Face landmarks or resolution unreliable" in warnings[0]
    assert "fell back to largest stable track" in warnings[0]

    # Face B is 25x25 (larger than 20x20), so it should track Face B (center_x = 800 + 12.5 = 812.5 / 1920 ~ 0.423)
    expected_b = 812.5 / 1920.0
    for c in raw_centers:
        assert c == pytest.approx(expected_b, abs=1e-3)


def test_unreliable_signal_fallback_collapsed_mouth_landmarks():
    """Collapsed mouth landmarks (mouth width < 3px, e.g. severe profile) trigger fallback warning."""
    frame_w, frame_h = 1920, 1080
    crop_w = int(frame_h * 9 / 16)
    samples: list[FaceSample] = []

    for i in range(8):
        t = i * 0.25
        # Normal size faces (150px), but mouth landmarks collapsed to 1px
        face_a = _make_face(x=300.0, y=300.0, w=150.0, h=150.0, mouth_w=1.0)
        face_b = _make_face(x=1400.0, y=300.0, w=160.0, h=160.0, mouth_w=1.0)
        samples.append(FaceSample(timestamp=t, faces=[face_a, face_b]))

    raw_centers, warnings = select_active_speaker_trajectory(
        samples=samples,
        frame_w=frame_w,
        frame_h=frame_h,
        crop_w=crop_w,
    )

    assert len(warnings) == 1
    assert "Degenerate mouth landmarks" in warnings[0]
    assert "fell back to largest stable track" in warnings[0]


# ============================================================================
# 8. Speech Activity Gating Holds Previous Speaker Through Pauses
# ============================================================================

def test_speech_activity_gating_holds_speaker_through_pauses():
    """When speech is inactive (pause), camera does not switch to challenger even if mouth twitches."""
    frame_w, frame_h = 1920, 1080
    crop_w = int(frame_h * 9 / 16)
    samples: list[FaceSample] = []

    # 12 samples over 3 seconds
    # Words only active in [0.0, 1.0]. Silence in [1.1, 3.0].
    words = [{"word": "speech", "start": 0.0, "end": 1.0}]

    for i in range(12):
        t = i * 0.25
        if t <= 1.0:
            # Face A speaks
            mouth_a = 45.0 + 10.0 * math.sin(i * 1.5)
            mouth_b = 35.0
        else:
            # Face A stops, Face B twitches mouth during silence
            mouth_a = 35.0
            mouth_b = 45.0 + 10.0 * math.sin(i * 1.5)

        face_a = _make_face(x=300.0, y=300.0, mouth_w=mouth_a)
        face_b = _make_face(x=1400.0, y=300.0, mouth_w=mouth_b)
        samples.append(FaceSample(timestamp=t, faces=[face_a, face_b]))

    raw_centers, warnings = select_active_speaker_trajectory(
        samples=samples,
        frame_w=frame_w,
        frame_h=frame_h,
        crop_w=crop_w,
        words=words,
        min_dwell_sec=1.5,
    )

    assert not warnings
    # Face A must be held throughout because speech is inactive after t=1.0s
    face_a_center = 380.0 / 1920.0
    for idx, c in enumerate(raw_centers):
        assert c is not None
        assert abs(c - face_a_center) < 0.02, f"Sample {idx} at t={samples[idx].timestamp} switched to {c} during silence!"


# ============================================================================
# 9. CropFilterResult Warnings Integration
# ============================================================================

def test_crop_filter_result_warnings_roundtrip():
    """CropFilterResult preserves layout_used, coverage, and warnings list."""
    res = CropFilterResult(
        "crop=608:1080:100:0",
        layout_used="speaker",
        coverage=0.85,
        warnings=["Face landmarks or resolution unreliable; fell back to largest stable track."],
    )

    assert str(res) == "crop=608:1080:100:0"
    assert res.layout_used == "speaker"
    assert res.coverage == 0.85
    assert len(res.warnings) == 1
    assert "fell back to largest stable track" in res.warnings[0]
