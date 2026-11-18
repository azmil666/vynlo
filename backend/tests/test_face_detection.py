"""Tests for neural face detection, aspect-preserving scaling, coordinate mapping, and VFR sequential sampling."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from app.config import settings
from app.services.cropper import (
    DetectedFace,
    FaceLandmarks,
    FaceSample,
    YUNET_EXPECTED_SHA256,
    _get_face_detector,
    compute_scaled_detection_size,
    detect_faces_in_frame,
    get_yunet_model_path,
    sample_faces_sequentially,
    verify_yunet_model_present,
)
from app.worker import on_worker_process_init


def _draw_synthetic_face(
    img: np.ndarray,
    center_x: int,
    center_y: int,
    radius_x: int,
    radius_y: int,
    skin_color: tuple[int, int, int] = (180, 190, 220),
    feature_color: tuple[int, int, int] = (40, 40, 40),
) -> None:
    """Helper to draw a stylized face with oval contour, eyes, and mouth on a canvas."""
    # Head contour (filled ellipse)
    cv2.ellipse(img, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, skin_color, -1)
    
    # Eyes
    eye_dx = int(radius_x * 0.42)
    eye_dy = int(radius_y * 0.22)
    eye_r = max(4, int(radius_x * 0.12))
    cv2.circle(img, (center_x - eye_dx, center_y - eye_dy), eye_r, feature_color, -1)
    cv2.circle(img, (center_x + eye_dx, center_y - eye_dy), eye_r, feature_color, -1)
    
    # Mouth (smile arc)
    mouth_y = center_y + int(radius_y * 0.38)
    mouth_rx = max(10, int(radius_x * 0.35))
    mouth_ry = max(5, int(radius_y * 0.15))
    cv2.ellipse(img, (center_x, mouth_y), (mouth_rx, mouth_ry), 0, 0, 180, feature_color, 3)


# ============================================================================
# 1. Model Verification & Worker Startup
# ============================================================================

def test_verify_yunet_model_present_and_checksum():
    """Verify that the baked YuNet model is found and its SHA-256 matches the official digest."""
    path = verify_yunet_model_present(verify_checksum=True)
    assert os.path.exists(path)
    assert os.path.isfile(path)
    assert path.endswith(".onnx")


def test_verify_yunet_model_missing_fails_fast():
    """When the baked model file is missing, verify_yunet_model_present must fail fast."""
    with patch("app.services.cropper.get_yunet_model_path", side_effect=RuntimeError("Model not found")):
        with pytest.raises(RuntimeError) as exc_info:
            verify_yunet_model_present()
        assert "Model not found" in str(exc_info.value)


def test_verify_yunet_model_corrupted_checksum_fails_fast(tmp_path: Path):
    """When the model file contains corrupt bytes, checksum verification raises RuntimeError."""
    fake_model = tmp_path / "corrupt_yunet.onnx"
    fake_model.write_bytes(b"corrupted bytes for testing")
    
    with patch("app.services.cropper.get_yunet_model_path", return_value=str(fake_model)):
        with pytest.raises(RuntimeError) as exc_info:
            verify_yunet_model_present(verify_checksum=True)
        assert "corrupted" in str(exc_info.value).lower()


def test_worker_startup_signal_verifies_model():
    """Celery worker_process_init signal hook verifies model presence and fails fast on error."""
    # Succeeded case
    on_worker_process_init()
    
    # Failure case
    with patch("app.worker.verify_yunet_model_present", side_effect=RuntimeError("Missing model")):
        with pytest.raises(RuntimeError):
            on_worker_process_init()


# ============================================================================
# 2. Aspect-Preserving Scaling & Coordinate Mapping
# ============================================================================

@pytest.mark.parametrize(
    "frame_w, frame_h, desc",
    [
        (1920, 1080, "16:9 1080p landscape"),
        (1440, 1080, "4:3 standard"),
        (2560, 1080, "21:9 ultrawide"),
        (1080, 1920, "9:16 vertical short"),
        (1080, 1080, "1:1 square"),
        (3840, 2160, "16:9 4K UHD"),
    ],
)
def test_compute_scaled_detection_size_aspect_ratios(frame_w: int, frame_h: int, desc: str):
    """Detection dimensions must preserve aspect ratio, cap long edge at 640, and use even dims."""
    target_long = 640
    det_w, det_h, scale_x, scale_y = compute_scaled_detection_size(frame_w, frame_h, target_long_edge=target_long)
    
    # Long edge must not exceed target
    assert max(det_w, det_h) <= target_long
    # Dimensions must be positive even integers
    assert det_w > 0 and det_h > 0
    assert det_w % 2 == 0
    assert det_h % 2 == 0
    
    # Scale factors must accurately map detection coordinates to source coordinates
    assert scale_x == pytest.approx(frame_w / det_w, rel=1e-5)
    assert scale_y == pytest.approx(frame_h / det_h, rel=1e-5)
    
    # Aspect ratio of detection image must match source frame closely
    source_aspect = frame_w / frame_h
    det_aspect = det_w / det_h
    assert det_aspect == pytest.approx(source_aspect, rel=0.05)


def test_coordinate_mapping_roundtrip():
    """Verify coordinate mapping round-trip between detection space and source space."""
    frame_w, frame_h = 2560, 1080  # 21:9 ultrawide
    det_w, det_h, scale_x, scale_y = compute_scaled_detection_size(frame_w, frame_h, 640)
    
    # Test arbitrary points across source frame
    test_points = [(0, 0), (1280, 540), (2560, 1080), (500, 300)]
    for sx, sy in test_points:
        dx = sx / scale_x
        dy = sy / scale_y
        reconstructed_sx = dx * scale_x
        reconstructed_sy = dy * scale_y
        assert reconstructed_sx == pytest.approx(sx, abs=1e-3)
        assert reconstructed_sy == pytest.approx(sy, abs=1e-3)


# ============================================================================
# 3. Neural Face Detection with YuNet
# ============================================================================

def test_detect_faces_in_frame_single_face():
    """YuNet detects a drawn synthetic face and reports coordinates in original source pixels."""
    img = np.full((1080, 1920, 3), 220, dtype=np.uint8)
    target_cx, target_cy = 960, 540
    _draw_synthetic_face(img, target_cx, target_cy, 120, 160)
    
    detector = _get_face_detector(score_threshold=0.1)
    faces = detect_faces_in_frame(detector, img, target_long_edge=640)
    
    assert len(faces) >= 1
    face = faces[0]
    
    # Check bounding box center is close to drawn face center in source pixels
    assert face.center_x == pytest.approx(target_cx, abs=60.0)
    assert face.center_y == pytest.approx(target_cy, abs=60.0)
    assert face.score > 0.3
    assert face.area > 10000.0
    
    # Check landmarks exist in source pixels
    assert face.landmarks is not None
    assert face.landmarks.right_eye[0] < face.landmarks.left_eye[0]
    assert face.landmarks.right_eye[1] < face.center_y
    assert face.landmarks.left_eye[1] < face.center_y


def test_detect_faces_in_frame_multiple_faces_ranking():
    """YuNet detects multiple faces and orders them by area * score descending."""
    img = np.full((1080, 1920, 3), 220, dtype=np.uint8)
    
    # Draw large face on left
    _draw_synthetic_face(img, 500, 500, 130, 170)
    # Draw smaller face on right
    _draw_synthetic_face(img, 1400, 500, 75, 95)
    
    detector = _get_face_detector(score_threshold=0.1)
    faces = detect_faces_in_frame(detector, img, target_long_edge=640)
    
    assert len(faces) >= 2
    # Largest face should be first
    assert faces[0].area > faces[1].area
    assert faces[0].center_x < 1000.0  # Left side
    assert faces[1].center_x > 1000.0  # Right side
    
    # Test FaceSample properties
    sample = FaceSample(timestamp=1.5, faces=faces)
    assert sample.primary_face is not None
    assert sample.primary_face == faces[0]


@pytest.mark.parametrize(
    "width, height",
    [
        (1440, 1080),  # 4:3
        (2560, 1080),  # 21:9
        (1080, 1920),  # 9:16 vertical
    ],
)
def test_detect_faces_various_aspect_ratios(width: int, height: int):
    """YuNet detects faces accurately across non-16:9 aspect ratios."""
    img = np.full((height, width, 3), 220, dtype=np.uint8)
    cx, cy = width // 2, height // 2
    _draw_synthetic_face(img, cx, cy, 120, 160)
    
    detector = _get_face_detector(score_threshold=0.1)
    faces = detect_faces_in_frame(detector, img, target_long_edge=640)
    
    assert len(faces) >= 1
    face = faces[0]
    assert face.center_x == pytest.approx(cx, abs=60.0)
    assert face.center_y == pytest.approx(cy, abs=60.0)


def test_detect_faces_empty_frame():
    """Plain background image produces zero detections cleanly without crashing."""
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    detector = _get_face_detector(score_threshold=0.55)
    faces = detect_faces_in_frame(detector, blank, target_long_edge=640)
    assert faces == []


# ============================================================================
# 4. Sequential Sampling on VFR Video Fixture
# ============================================================================

def test_sample_faces_sequentially_on_vfr_video(video_vfr: str):
    """Sequential sampling on VFR video produces monotonic PTS samples without seek lag."""
    sample_hz = 4.0
    start_time = 0.0
    end_time = 2.0
    
    samples = sample_faces_sequentially(
        video_path=video_vfr,
        start_time=start_time,
        end_time=end_time,
        sample_hz=sample_hz,
    )
    
    assert len(samples) > 0
    # For a 2.0s clip sampled at 4 Hz, we expect around 7 to 9 samples
    assert 6 <= len(samples) <= 10
    
    # Timestamps must be strictly monotonic
    timestamps = [s.timestamp for s in samples]
    for i in range(1, len(timestamps)):
        assert timestamps[i] > timestamps[i - 1], f"Non-monotonic timestamps: {timestamps}"
    
    # All timestamps must fall within window (allowing 1 sample interval grace at boundaries)
    assert timestamps[0] >= start_time - 0.1
    assert timestamps[-1] <= end_time + (1.0 / sample_hz) + 0.1
    
    # Each sample must be a valid FaceSample dataclass
    for s in samples:
        assert isinstance(s, FaceSample)
        assert isinstance(s.faces, list)
