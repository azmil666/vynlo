"""Computer vision and face detection service for vertical 9:16 dynamic reframing.

Uses OpenCV's neural FaceDetectorYN (YuNet 2023mar ONNX model) with:
- Long-edge aspect-preserving scaling
- Per-frame setInputSize
- Dual x/y coordinate re-mapping
- Sequential decoding (grab/retrieve) with PTS tracking for CFR and VFR sources
- Multi-face extraction (boxes, scores, landmarks)
"""

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

YUNET_EXPECTED_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


# ============================================================================
# Data Models for Face Detections and Samples
# ============================================================================

@dataclass
class FaceLandmarks:
    """5-point facial landmarks in source pixel coordinates."""
    right_eye: tuple[float, float]
    left_eye: tuple[float, float]
    nose_tip: tuple[float, float]
    mouth_right: tuple[float, float]
    mouth_left: tuple[float, float]


@dataclass
class DetectedFace:
    """Detected face with bounding box, confidence score, and landmarks in original source pixels."""
    box: tuple[float, float, float, float]  # (x, y, w, h) in source pixels
    score: float
    landmarks: Optional[FaceLandmarks] = None

    @property
    def center_x(self) -> float:
        """Horizontal center in source pixels."""
        return self.box[0] + self.box[2] / 2.0

    @property
    def center_y(self) -> float:
        """Vertical center in source pixels."""
        return self.box[1] + self.box[3] / 2.0

    @property
    def area(self) -> float:
        """Bounding box pixel area."""
        return max(0.0, self.box[2]) * max(0.0, self.box[3])


@dataclass
class FaceSample:
    """Face detection sample at a specific video presentation timestamp (PTS)."""
    timestamp: float
    faces: list[DetectedFace]

    @property
    def primary_face(self) -> Optional[DetectedFace]:
        """Return the primary face (largest area * score) in this sample."""
        if not self.faces:
            return None
        return self.faces[0]  # Pre-sorted by area * score descending


# ============================================================================
# YuNet Model Loading & Verification
# ============================================================================

_cached_detector: Optional[cv2.FaceDetectorYN] = None


def get_yunet_model_path() -> str:
    """Locate the baked YuNet ONNX model path or raise RuntimeError."""
    candidate_paths = [
        getattr(settings, "YUNET_MODEL_PATH", ""),
        "/app/models/face_detection_yunet_2023mar.onnx",
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "models",
                "face_detection_yunet_2023mar.onnx",
            )
        ),
        "/tmp/face_detection_yunet_2023mar.onnx",
    ]

    for p in candidate_paths:
        if p and os.path.exists(p):
            return p

    raise RuntimeError(
        "YuNet face detection model 'face_detection_yunet_2023mar.onnx' not found. "
        "The model must be baked into the Docker image; runtime downloads are disabled."
    )


def verify_yunet_model_present(verify_checksum: bool = False) -> str:
    """Verify at startup that YuNet model is present and optionally check its SHA-256 digest."""
    path = get_yunet_model_path()
    if verify_checksum:
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        if digest != YUNET_EXPECTED_SHA256:
            raise RuntimeError(
                f"YuNet ONNX model corrupted: expected {YUNET_EXPECTED_SHA256}, got {digest}"
            )
    return path


def _get_face_detector(
    score_threshold: Optional[float] = None,
    nms_threshold: Optional[float] = None,
    top_k: Optional[int] = None,
) -> cv2.FaceDetectorYN:
    """Load and return the YuNet neural face detector."""
    global _cached_detector
    s_thresh = (
        score_threshold
        if score_threshold is not None
        else getattr(settings, "FACE_DETECTION_SCORE_THRESHOLD", 0.55)
    )
    n_thresh = (
        nms_threshold
        if nms_threshold is not None
        else getattr(settings, "FACE_DETECTION_NMS_THRESHOLD", 0.3)
    )
    k = top_k if top_k is not None else getattr(settings, "FACE_DETECTION_TOP_K", 10)

    if _cached_detector is not None:
        _cached_detector.setScoreThreshold(s_thresh)
        _cached_detector.setNMSThreshold(n_thresh)
        _cached_detector.setTopK(k)
        return _cached_detector

    model_path = get_yunet_model_path()
    try:
        detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            score_threshold=s_thresh,
            nms_threshold=n_thresh,
            top_k=k,
        )
        if detector is None:
            raise RuntimeError(f"Failed to create cv2.FaceDetectorYN from {model_path}")
        _cached_detector = detector
        return detector
    except Exception as e:
        logger.error("Failed loading YuNet face detector from %s: %s", model_path, e)
        raise RuntimeError(f"YuNet detector initialization failed: {e}") from e


# ============================================================================
# Aspect-Preserving Scaling & Face Detection
# ============================================================================

def compute_scaled_detection_size(
    frame_w: int,
    frame_h: int,
    target_long_edge: Optional[int] = None,
) -> tuple[int, int, float, float]:
    """Compute aspect-preserving detection dimensions and scale factors back to source.

    Returns:
        (det_w, det_h, scale_x, scale_y)
        where source_coord = det_coord * scale
    """
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError(f"Invalid frame dimensions: {frame_w}x{frame_h}")

    edge = (
        target_long_edge
        if target_long_edge is not None
        else getattr(settings, "FACE_DETECTION_LONG_EDGE", 640)
    )
    long_edge = max(frame_w, frame_h)
    scale = min(1.0, float(edge) / float(long_edge))

    det_w = int(round(frame_w * scale))
    det_h = int(round(frame_h * scale))

    # Keep dimensions even for codec/CNN stability
    det_w = max(2, det_w - (det_w % 2))
    det_h = max(2, det_h - (det_h % 2))

    scale_x = float(frame_w) / float(det_w)
    scale_y = float(frame_h) / float(det_h)
    return det_w, det_h, scale_x, scale_y


def detect_faces_in_frame(
    detector: cv2.FaceDetectorYN,
    frame: np.ndarray,
    target_long_edge: Optional[int] = None,
) -> list[DetectedFace]:
    """Detect all faces in a frame, scaling to target_long_edge and mapping coordinates back.

    Returns:
        List of DetectedFace instances with boxes and landmarks in original source pixels,
        sorted descending by area * score.
    """
    if detector is None or frame is None:
        return []

    frame_h, frame_w = frame.shape[:2]
    if frame_w <= 0 or frame_h <= 0:
        return []

    det_w, det_h, scale_x, scale_y = compute_scaled_detection_size(
        frame_w, frame_h, target_long_edge=target_long_edge
    )

    scaled_frame = cv2.resize(frame, (det_w, det_h))
    detector.setInputSize((det_w, det_h))

    _, raw_faces = detector.detect(scaled_frame)
    if raw_faces is None or len(raw_faces) == 0:
        return []

    results: list[DetectedFace] = []
    for f in raw_faces:
        x = float(f[0]) * scale_x
        y = float(f[1]) * scale_y
        w = float(f[2]) * scale_x
        h = float(f[3]) * scale_y
        score = float(f[14])

        landmarks = FaceLandmarks(
            right_eye=(float(f[4]) * scale_x, float(f[5]) * scale_y),
            left_eye=(float(f[6]) * scale_x, float(f[7]) * scale_y),
            nose_tip=(float(f[8]) * scale_x, float(f[9]) * scale_y),
            mouth_right=(float(f[10]) * scale_x, float(f[11]) * scale_y),
            mouth_left=(float(f[12]) * scale_x, float(f[13]) * scale_y),
        )

        results.append(
            DetectedFace(
                box=(round(x, 2), round(y, 2), round(w, 2), round(h, 2)),
                score=round(score, 4),
                landmarks=landmarks,
            )
        )

    # Sort descending by area * score
    results.sort(key=lambda item: item.area * item.score, reverse=True)
    return results


def _detect_face_center(detector: Any, frame: np.ndarray, frame_w: int) -> Optional[float]:
    """Detect normalized horizontal face center [0.0, 1.0] of primary face (backward compatibility)."""
    if detector is None or frame is None or frame_w <= 0:
        return None
    faces = detect_faces_in_frame(detector, frame)
    if not faces:
        return None
    return float(faces[0].center_x / float(frame_w))


# ============================================================================
# Sequential Decoding Frame Sampling
# ============================================================================

def sample_faces_sequentially(
    video_path: str,
    start_time: float,
    end_time: float,
    sample_hz: Optional[float] = None,
    detector: Optional[cv2.FaceDetectorYN] = None,
    source_info: Optional[Any] = None,
) -> list[FaceSample]:
    """Sample video frames sequentially using grab/retrieve at configurable Hz.

    Verifies timestamps with PTS where available (including VFR sources).
    Avoids per-sample seek operations.
    """
    if not os.path.exists(video_path):
        raise ValueError(f"Failed to open video file: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    active_detector = detector or _get_face_detector()
    hz = sample_hz if sample_hz is not None else getattr(settings, "FACE_DETECTION_SAMPLE_HZ", 4.0)
    step = 1.0 / max(0.1, hz)
    rotation = getattr(source_info, "rotation", 0) if source_info else 0

    samples: list[FaceSample] = []
    duration = max(0.0, end_time - start_time)
    if duration <= 0.0:
        cap.release()
        return samples

    try:
        # Seek once to initial start_time
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_time) * 1000.0)

        next_target_time = start_time
        max_time = end_time + 0.05
        frame_idx = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        while True:
            grabbed = cap.grab()
            if not grabbed:
                break

            # Read presentation timestamp (PTS) in milliseconds
            pts_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pts_msec > 0:
                current_time = pts_msec / 1000.0
            else:
                current_time = start_time + (frame_idx / fps)
            frame_idx += 1

            if current_time < (start_time - 0.02):
                continue
            if current_time > max_time:
                break

            if current_time >= (next_target_time - 0.02):
                ret, frame = cap.retrieve()
                if ret and frame is not None:
                    # Apply rotation if present in metadata
                    if rotation == 90:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif rotation == 180:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    elif rotation == 270:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    faces = detect_faces_in_frame(active_detector, frame)
                    samples.append(FaceSample(timestamp=round(current_time, 3), faces=faces))
                else:
                    samples.append(FaceSample(timestamp=round(current_time, 3), faces=[]))

                next_target_time = current_time + step
                if next_target_time > max_time:
                    break

        return samples
    finally:
        cap.release()


# ============================================================================
# Crop Filter Computation & Single-Pass Vertical Reframing
# ============================================================================

def compute_crop_filter(
    video_path: str,
    start_time: float,
    end_time: float,
    mode: str = "speaker",
    source_info: Optional[Any] = None,
) -> str:
    """Compute crop/scale filter string for FFmpeg single-pass rendering using YuNet neural face detection."""
    if mode == "split":
        # Split screen layout: top gameplay, bottom webcam
        return (
            "[0:v]crop=ih*9/8:ih:(iw-ih*9/8)/2:0,scale=1080:960:flags=lanczos,setsar=1[top];"
            "[0:v]crop=iw*0.30:ih*0.30:iw*0.70:ih*0.70,scale=1080:960:flags=lanczos,setsar=1[bottom];"
            "[top][bottom]vstack=inputs=2[v]"
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    try:
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Normalize dimensions if video stream contains rotation metadata
        if source_info and getattr(source_info, "rotation", 0) in (90, 270):
            frame_w, frame_h = frame_h, frame_w

        if frame_w <= 0 or frame_h <= 0:
            raise ValueError(f"Failed to read valid video dimensions from {video_path}")

        crop_w = int(frame_h * 9 / 16)
        crop_w = crop_w - (crop_w % 2)

        # Handle Already-Vertical Source Videos
        if frame_w <= frame_h * 0.75 or crop_w >= frame_w:
            return (
                "scale=1080:1920:force_original_aspect_ratio=decrease,"
                "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1"
            )

        duration = max(0.1, end_time - start_time)
        samples = sample_faces_sequentially(
            video_path,
            start_time,
            end_time,
            source_info=source_info,
        )

        raw_centers: list[Optional[float]] = []
        sample_times: list[float] = []

        if samples:
            for s in samples:
                sample_times.append(s.timestamp)
                primary = s.primary_face
                if primary is not None:
                    raw_centers.append(float(primary.center_x / float(frame_w)))
                else:
                    raw_centers.append(None)
        else:
            sample_times = [start_time]
            raw_centers = [0.5]

        # Carry forward last known valid face center_x (defaulting to 0.5 if no face was ever seen)
        filled_centers: list[Optional[float]] = []
        last_valid: Optional[float] = None

        for c in raw_centers:
            if c is not None:
                last_valid = c
                filled_centers.append(c)
            elif last_valid is not None:
                filled_centers.append(last_valid)
            else:
                filled_centers.append(None)

        first_valid = next((c for c in raw_centers if c is not None), 0.5)
        clean_centers = [c if c is not None else first_valid for c in filled_centers]

        # 1.5-second rolling median filter
        sample_hz = getattr(settings, "FACE_DETECTION_SAMPLE_HZ", 4.0)
        half_window = max(1, int(round(1.5 * sample_hz / 2)))
        n_samples = len(clean_centers)
        smoothed_centers: list[float] = []

        for i in range(n_samples):
            w_start = max(0, i - half_window)
            w_end = min(n_samples, i + half_window + 1)
            smoothed_centers.append(float(np.median(clean_centers[w_start:w_end])))

        # Segment into 1-second chunks and clamp within [0, frame_w - crop_w]
        total_seconds = max(1, int(np.ceil(duration)))
        raw_segments: list[tuple[float, int]] = []

        for s in range(total_seconds):
            sample_indices = [
                idx
                for idx, t in enumerate(sample_times)
                if s <= (t - start_time) < (s + 1.0)
            ]
            if sample_indices:
                seg_center = float(np.median([smoothed_centers[idx] for idx in sample_indices]))
            else:
                nearest_idx = min(
                    range(len(sample_times)),
                    key=lambda idx: abs((sample_times[idx] - start_time) - (s + 0.5)),
                )
                seg_center = smoothed_centers[nearest_idx]

            crop_x = int(round(seg_center * frame_w - crop_w / 2.0))
            crop_x = max(0, min(frame_w - crop_w, crop_x))
            crop_x = crop_x - (crop_x % 2)
            raw_segments.append((float(s + 1.0), crop_x))

        # Merge consecutive identical segments
        merged_segments: list[tuple[float, int]] = []
        for t_cut, val in raw_segments:
            if merged_segments and merged_segments[-1][1] == val:
                merged_segments[-1] = (t_cut, val)
            else:
                merged_segments.append((t_cut, val))

        if len(merged_segments) <= 1:
            static_x = (
                merged_segments[0][1]
                if merged_segments
                else max(0, (frame_w - crop_w) // 2)
            )
            static_x = static_x - (static_x % 2)
            return f"crop={crop_w}:{frame_h}:{static_x}:0,scale=1080:1920:flags=lanczos,setsar=1"

        # Build dynamic piecewise if(lt(t, ...), ...) expression
        expr = str(merged_segments[-1][1])
        for t_cutoff, val in reversed(merged_segments[:-1]):
            expr = f"if(lt(t,{t_cutoff:.1f}),{val},{expr})"

        return f"crop={crop_w}:{frame_h}:x='{expr}':y=0,scale=1080:1920:flags=lanczos,setsar=1"

    finally:
        cap.release()


def crop_to_vertical(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: str,
    mode: str = "speaker",
) -> str:
    """Crop video segment [start_time, end_time] to vertical 9:16 (1080x1920)."""
    crop_filter = compute_crop_filter(video_path, start_time, end_time, mode=mode)
    from app.services.captions import render_clip_single_pass

    return render_clip_single_pass(
        video_path=video_path,
        start_time=start_time,
        end_time=end_time,
        crop_filter=crop_filter,
        ass_path="",
        output_path=output_path,
    )
