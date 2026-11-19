"""Computer vision and face detection service for vertical 9:16 dynamic reframing.

Uses OpenCV's neural FaceDetectorYN (YuNet 2023mar ONNX model) with:
- Long-edge aspect-preserving scaling
- Per-frame setInputSize
- Dual x/y coordinate re-mapping
- Sequential decoding (grab/retrieve) with PTS tracking for CFR and VFR sources
- Multi-face extraction (boxes, scores, landmarks)
"""

import hashlib
import json
import logging
import math
import os
import subprocess
from dataclasses import dataclass, field
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


@dataclass
class TrackedDetection:
    """Face detection associated with a track at timestamp t."""
    timestamp: float
    face: DetectedFace
    norm_mouth_dist: float


@dataclass
class FaceTrack:
    """Continuous temporal track of a detected face across samples."""
    track_id: int
    detections: list[TrackedDetection] = field(default_factory=list)

    @property
    def start_time(self) -> float:
        return self.detections[0].timestamp if self.detections else 0.0

    @property
    def end_time(self) -> float:
        return self.detections[-1].timestamp if self.detections else 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def count(self) -> int:
        return len(self.detections)

    @property
    def mean_area(self) -> float:
        if not self.detections:
            return 0.0
        return sum(d.face.area for d in self.detections) / len(self.detections)

    def get_detection_at_time(self, t: float, max_dt: float = 0.5) -> Optional[TrackedDetection]:
        """Find the detection closest to time t within max_dt."""
        best: Optional[TrackedDetection] = None
        best_dt = max_dt
        for d in self.detections:
            dt = abs(d.timestamp - t)
            if dt < best_dt:
                best_dt = dt
                best = d
        return best


def compute_normalized_mouth_distance(face: DetectedFace) -> float:
    """Compute Euclidean distance between mouth corners normalized by face box size."""
    if not face.landmarks:
        return 0.0
    lm = face.landmarks
    dx = lm.mouth_right[0] - lm.mouth_left[0]
    dy = lm.mouth_right[1] - lm.mouth_left[1]
    dist = math.sqrt(dx * dx + dy * dy)
    face_size = max(face.box[2], face.box[3], 1.0)
    return dist / face_size


def calculate_box_iou(
    box1: tuple[float, float, float, float],
    box2: tuple[float, float, float, float],
) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes (x, y, w, h)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2]) * max(0.0, box1[3])
    area2 = max(0.0, box2[2]) * max(0.0, box2[3])
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def track_faces_across_samples(
    samples: list[FaceSample],
    iou_threshold: Optional[float] = None,
    max_gap: Optional[float] = None,
) -> list[FaceTrack]:
    """Track faces across consecutive samples using IoU and spatial proximity association."""
    iou_thresh = (
        iou_threshold
        if iou_threshold is not None
        else getattr(settings, "SPEAKER_IOU_THRESHOLD", 0.25)
    )
    gap_limit = (
        max_gap
        if max_gap is not None
        else getattr(settings, "SPEAKER_MAX_TRACK_GAP_SECONDS", 1.5)
    )

    tracks: list[FaceTrack] = []
    next_track_id = 1

    for sample in samples:
        t = sample.timestamp
        faces = sorted(sample.faces, key=lambda f: f.area * f.score, reverse=True)
        active_tracks = [tr for tr in tracks if (t - tr.end_time) <= gap_limit]

        pair_scores: list[tuple[float, int, int]] = []
        for tr_idx, tr in enumerate(active_tracks):
            last_det = tr.detections[-1]
            last_box = last_det.face.box
            for f_idx, face in enumerate(faces):
                iou = calculate_box_iou(last_box, face.box)
                dx = last_det.face.center_x - face.center_x
                dy = last_det.face.center_y - face.center_y
                center_dist = math.sqrt(dx * dx + dy * dy)
                max_dim = max(last_box[2], last_box[3], face.box[2], face.box[3], 1.0)
                norm_dist = center_dist / max_dim

                score = 0.0
                if iou >= iou_thresh:
                    score = iou
                elif norm_dist < 0.6:
                    score = 1.0 / (1.0 + norm_dist)

                if score > 0.0:
                    pair_scores.append((score, tr_idx, f_idx))

        pair_scores.sort(key=lambda x: x[0], reverse=True)

        matched_tracks: set[int] = set()
        matched_faces: set[int] = set()

        for score, tr_idx, f_idx in pair_scores:
            tr = active_tracks[tr_idx]
            if tr.track_id in matched_tracks or f_idx in matched_faces:
                continue

            matched_tracks.add(tr.track_id)
            matched_faces.add(f_idx)
            face = faces[f_idx]
            d_norm = compute_normalized_mouth_distance(face)
            tr.detections.append(TrackedDetection(timestamp=t, face=face, norm_mouth_dist=d_norm))

        for f_idx, face in enumerate(faces):
            if f_idx not in matched_faces:
                d_norm = compute_normalized_mouth_distance(face)
                new_track = FaceTrack(
                    track_id=next_track_id,
                    detections=[TrackedDetection(timestamp=t, face=face, norm_mouth_dist=d_norm)],
                )
                next_track_id += 1
                tracks.append(new_track)

    return tracks


def compute_track_speaking_score(
    track: FaceTrack,
    t: float,
    window_sec: float = 1.0,
    noise_floor: float = 1e-4,
) -> float:
    """Compute speaking score as normalized mouth-distance variance over rolling window around t."""
    half_w = window_sec / 2.0
    window_dists = [
        d.norm_mouth_dist
        for d in track.detections
        if (t - half_w) <= d.timestamp <= (t + half_w)
    ]
    if len(window_dists) < 2:
        return 0.0
    var = float(np.var(window_dists))
    return max(0.0, var - noise_floor)


def is_speech_active_at(
    t: float,
    speech_intervals: list[tuple[float, float]],
    margin: float = 0.3,
) -> bool:
    """Return whether speech is active at time t within margin.
    If speech_intervals is empty, returns True (unconstrained).
    """
    if not speech_intervals:
        return True
    return any(start - margin <= t <= end + margin for start, end in speech_intervals)


def assess_speaker_tracking_reliability(
    tracks: list[FaceTrack],
    min_face_size: float = 35.0,
    min_mouth_width: float = 3.0,
) -> tuple[bool, Optional[str]]:
    """Assess whether facial landmarks and resolution are reliable for active speaker tracking."""
    if not tracks:
        return False, "No face tracks available"

    all_detections = [d for tr in tracks for d in tr.detections]
    if not all_detections:
        return False, "No face detections found"

    face_sizes = [max(d.face.box[2], d.face.box[3]) for d in all_detections]
    med_size = float(np.median(face_sizes)) if face_sizes else 0.0
    if med_size < min_face_size:
        return False, f"Low face resolution (median dimension {med_size:.1f}px < {min_face_size:.0f}px)"

    mouth_widths: list[float] = []
    for d in all_detections:
        if d.face.landmarks:
            dx = d.face.landmarks.mouth_right[0] - d.face.landmarks.mouth_left[0]
            dy = d.face.landmarks.mouth_right[1] - d.face.landmarks.mouth_left[1]
            mouth_widths.append(math.sqrt(dx * dx + dy * dy))

    if not mouth_widths:
        return False, "No facial landmarks detected"

    med_mouth = float(np.median(mouth_widths))
    if med_mouth < min_mouth_width:
        return False, f"Degenerate mouth landmarks (median mouth width {med_mouth:.1f}px < {min_mouth_width:.0f}px)"

    return True, None


def select_active_speaker_trajectory(
    samples: list[FaceSample],
    frame_w: int,
    frame_h: int,
    crop_w: int,
    words: Optional[list[dict]] = None,
    min_dwell_sec: Optional[float] = None,
    switch_margin: Optional[float] = None,
    mouth_window_sec: Optional[float] = None,
    noise_floor: Optional[float] = None,
    dual_frame_ratio: Optional[float] = None,
) -> tuple[list[Optional[float]], list[str]]:
    """Select horizontal crop centers [0.0, 1.0] across samples following the active speaker.

    Features:
    1. Per-face tracking across samples via IoU and nearest-box association.
    2. Per-track speaking score based on normalized mouth-corner distance variance.
    3. Hysteresis switching: requires minimum dwell time and score margin above current speaker;
       holds current speaker through short pauses.
    4. Dual-speaker framing: if two active speakers fit within crop_w with margin, frames both.
    5. Fallback: if landmarks or resolution are unreliable, falls back to largest stable track and records warning.

    Returns:
        (raw_centers, warnings)
    """
    if not samples:
        return [0.5], []

    dwell_time = (
        min_dwell_sec
        if min_dwell_sec is not None
        else getattr(settings, "SPEAKER_MIN_DWELL_SECONDS", 2.0)
    )
    margin = (
        switch_margin
        if switch_margin is not None
        else getattr(settings, "SPEAKER_SWITCH_MARGIN", 1.3)
    )
    window_sec = (
        mouth_window_sec
        if mouth_window_sec is not None
        else getattr(settings, "SPEAKER_MOUTH_WINDOW_SECONDS", 1.0)
    )
    floor = (
        noise_floor
        if noise_floor is not None
        else getattr(settings, "SPEAKER_MOUTH_NOISE_FLOOR", 1e-4)
    )
    dual_ratio = (
        dual_frame_ratio
        if dual_frame_ratio is not None
        else getattr(settings, "SPEAKER_DUAL_FRAME_MAX_RATIO", 0.85)
    )

    speech_intervals: list[tuple[float, float]] = []
    if words:
        for w in words:
            if isinstance(w, dict) and "start" in w and "end" in w:
                try:
                    speech_intervals.append((float(w["start"]), float(w["end"])))
                except (ValueError, TypeError):
                    pass

    tracks = track_faces_across_samples(samples)
    if not tracks:
        return [None] * len(samples), []

    # Single-face optimization: track directly without multi-face switching
    if len(tracks) == 1:
        single_track = tracks[0]
        raw_centers = [
            float(det.face.center_x / float(frame_w))
            if (det := single_track.get_detection_at_time(s.timestamp, max_dt=0.5))
            else None
            for s in samples
        ]
        return raw_centers, []

    # Assess reliability of facial landmarks & resolution across tracks
    min_face_size = getattr(settings, "SPEAKER_MIN_FACE_SIZE_PX", 35.0)
    reliable, unreliability_reason = assess_speaker_tracking_reliability(
        tracks, min_face_size=min_face_size
    )

    warnings: list[str] = []
    if not reliable:
        warn_msg = (
            f"Face landmarks or resolution unreliable for active speaker tracking "
            f"({unreliability_reason}); fell back to largest stable track."
        )
        warnings.append(warn_msg)
        logger.warning(warn_msg)

        largest_track = max(tracks, key=lambda tr: tr.mean_area * tr.count)
        fallback_centers: list[Optional[float]] = []
        for s in samples:
            det = largest_track.get_detection_at_time(s.timestamp, max_dt=0.5)
            if det:
                fallback_centers.append(float(det.face.center_x / float(frame_w)))
            else:
                fallback_centers.append(None)
        return fallback_centers, warnings

    # Multi-speaker tracking with dual-framing & hysteresis
    raw_centers: list[Optional[float]] = []
    current_speaker_track: Optional[FaceTrack] = None
    speaker_switch_time: float = -999.0

    for s in samples:
        t = s.timestamp
        active_pairs: list[tuple[FaceTrack, TrackedDetection]] = []
        for tr in tracks:
            det = tr.get_detection_at_time(t, max_dt=0.5)
            if det is not None:
                active_pairs.append((tr, det))

        if not active_pairs:
            raw_centers.append(None)
            continue

        active_pairs.sort(key=lambda p: p[1].face.area * p[1].face.score, reverse=True)

        # -------------------------------------------------------------
        # 1. Dual-Speaker Framing Check
        # -------------------------------------------------------------
        dual_framed = False
        if len(active_pairs) >= 2:
            p1, p2 = active_pairs[0], active_pairs[1]
            area1, area2 = p1[1].face.area, p2[1].face.area
            if area2 >= 0.25 * area1:
                box1, box2 = p1[1].face.box, p2[1].face.box
                x_min = min(box1[0], box2[0])
                x_max = max(box1[0] + box1[2], box2[0] + box2[2])
                span = x_max - x_min
                if span <= crop_w * dual_ratio:
                    both_center = (x_min + x_max) / 2.0
                    raw_centers.append(float(both_center / float(frame_w)))
                    dual_framed = True

        if dual_framed:
            continue

        # -------------------------------------------------------------
        # 2. Active Speaker Selection with Hysteresis
        # -------------------------------------------------------------
        if current_speaker_track is None:
            best_init_track = active_pairs[0][0]
            best_init_score = -1.0
            for tr, _ in active_pairs:
                sc = compute_track_speaking_score(tr, t, window_sec, floor)
                if sc > best_init_score:
                    best_init_score = sc
                    best_init_track = tr
            current_speaker_track = best_init_track
            speaker_switch_time = t

        # If previous speaker hasn't been seen for > 1.5s, transfer focus to primary visible track
        if (t - current_speaker_track.end_time) > getattr(settings, "FACE_MAX_GAP_FILL_SECONDS", 1.5):
            current_speaker_track = active_pairs[0][0]
            speaker_switch_time = t

        track_scores = {
            tr.track_id: compute_track_speaking_score(tr, t, window_sec, floor)
            for tr, _ in active_pairs
        }

        speech_on = is_speech_active_at(t, speech_intervals)
        dwell = t - speaker_switch_time

        # Switch only if speech is active, dwell time satisfied, and challenger beats margin
        if speech_on and dwell >= dwell_time:
            curr_score = track_scores.get(current_speaker_track.track_id, 0.0)
            best_challenger_track: Optional[FaceTrack] = None
            best_challenger_score = 0.0
            for tr, _ in active_pairs:
                if tr.track_id == current_speaker_track.track_id:
                    continue
                sc = track_scores.get(tr.track_id, 0.0)
                if sc > best_challenger_score:
                    best_challenger_score = sc
                    best_challenger_track = tr

            if (
                best_challenger_track is not None
                and best_challenger_score > floor
                and best_challenger_score > (curr_score * margin)
            ):
                current_speaker_track = best_challenger_track
                speaker_switch_time = t

        det = current_speaker_track.get_detection_at_time(t, max_dt=0.5)
        if det is not None:
            raw_centers.append(float(det.face.center_x / float(frame_w)))
        else:
            raw_centers.append(None)

    return raw_centers, warnings


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
# Geometry Normalization (SAR & Rotation)
# ============================================================================

def normalize_geometry(
    raw_w: int,
    raw_h: int,
    source_info: Optional[Any] = None,
) -> tuple[int, int, float, int]:
    """Compute normalized square-pixel display dimensions accounting for SAR and rotation.

    1. Applies SAR (Sample Aspect Ratio) to compute square-pixel dimensions: W_sq = round(raw_w * sar).
    2. Applies rotation: 90° or 270° swaps display width and height.

    Returns:
        (norm_w, norm_h, sar_ratio, rotation)
    """
    if raw_w <= 0 or raw_h <= 0:
        raise ValueError(f"Invalid raw video dimensions: {raw_w}x{raw_h}")

    rotation = getattr(source_info, "rotation", 0) if source_info else 0
    sar_str = getattr(source_info, "sar", "1:1") if source_info else "1:1"
    sar_ratio = 1.0

    if sar_str and sar_str not in ("1:1", "0:1", "1", "0", ""):
        try:
            num, den = sar_str.replace("/", ":").split(":", 1)
            num_f, den_f = float(num), float(den)
            if den_f > 0 and num_f > 0:
                sar_ratio = num_f / den_f
        except (ValueError, ZeroDivisionError):
            sar_ratio = 1.0

    # Square pixel dimensions
    w_sq = int(round(raw_w * sar_ratio))
    w_sq = max(2, w_sq - (w_sq % 2))
    h_sq = max(2, raw_h - (raw_h % 2))

    # Rotation (90 / 270 swaps display width and height)
    if rotation in (90, 270):
        norm_w, norm_h = h_sq, w_sq
    else:
        norm_w, norm_h = w_sq, h_sq

    return norm_w, norm_h, sar_ratio, rotation


def probe_geometry(video_path: str, source_info: Optional[Any] = None) -> tuple[int, int, float, int]:
    """Probe geometry directly if source_info is not provided or incomplete."""
    if source_info and getattr(source_info, "width", 0) > 0 and getattr(source_info, "height", 0) > 0:
        return normalize_geometry(source_info.width, source_info.height, source_info=source_info)

    raw_w, raw_h = 0, 0
    rotation = 0
    sar_str = "1:1"

    if os.path.exists(video_path):
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,sample_aspect_ratio,side_data_list:stream_tags=rotate",
                "-of",
                "json",
                video_path,
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            if res.returncode == 0 and res.stdout:
                data = json.loads(res.stdout)
                streams = data.get("streams", [])
                if streams:
                    v = streams[0]
                    raw_w = int(v.get("width", 0) or 0)
                    raw_h = int(v.get("height", 0) or 0)
                    sar_str = str(v.get("sample_aspect_ratio", "1:1") or "1:1")
                    tags = v.get("tags", {})
                    if "rotate" in tags:
                        try:
                            rotation = abs(int(float(tags["rotate"]))) % 360
                        except (ValueError, TypeError):
                            pass
                    if rotation == 0:
                        for sd in v.get("side_data_list", []):
                            if "rotation" in sd:
                                try:
                                    rotation = abs(int(float(sd["rotation"]))) % 360
                                except (ValueError, TypeError):
                                    pass
        except Exception:
            pass

    if raw_w <= 0 or raw_h <= 0:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

    if raw_w <= 0 or raw_h <= 0:
        raise ValueError(f"Failed to read valid video dimensions from {video_path}")

    from app.schemas import SourceInfo
    info = source_info or SourceInfo(
        width=raw_w,
        height=raw_h,
        duration=0.0,
        rotation=rotation,
        sar=sar_str,
    )
    return normalize_geometry(raw_w, raw_h, source_info=info)


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
    sar_str = getattr(source_info, "sar", "1:1") if source_info else "1:1"
    sar_ratio = 1.0
    if sar_str and sar_str not in ("1:1", "0:1", "1", "0", ""):
        try:
            num, den = sar_str.replace("/", ":").split(":", 1)
            num_f, den_f = float(num), float(den)
            if den_f > 0 and num_f > 0:
                sar_ratio = num_f / den_f
        except (ValueError, ZeroDivisionError):
            sar_ratio = 1.0

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

                    # Normalize SAR to square pixels if non-square
                    if abs(sar_ratio - 1.0) > 0.01 and sar_ratio > 0:
                        new_w = max(2, int(round(frame.shape[1] * sar_ratio)))
                        new_w = new_w - (new_w % 2)
                        frame = cv2.resize(frame, (new_w, frame.shape[0]))

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

FIT_BLUR_FILTER = (
    "[0:v]scale=trunc(iw*sar/2)*2:ih,setsar=1[norm];"
    "[norm]split=2[bg_in][fg_in];"
    "[bg_in]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:20,drawbox=c=black@0.25:t=fill,setsar=1[bg];"
    "[fg_in]scale=1080:1920:force_original_aspect_ratio=decrease,setsar=1[fg];"
    "[bg][fg]overlay=(W-w)/2:(H-h)/2[v]"
)

SPLIT_FILTER = (
    "[0:v]crop=ih*9/8:ih:(iw-ih*9/8)/2:0,scale=1080:960:flags=lanczos,setsar=1[top];"
    "[0:v]crop=iw*0.30:ih*0.30:iw*0.70:ih*0.70,scale=1080:960:flags=lanczos,setsar=1[bottom];"
    "[top][bottom]vstack=inputs=2[v]"
)


class CropFilterResult(str):
    """FFmpeg filter string enriched with computed layout metadata and detection coverage.

    Subclasses str so it can be passed directly to FFmpeg command builders, format strings,
    and existing legacy string-expecting functions without breaking backward compatibility.
    """

    layout_used: str
    coverage: float
    filter_str: str
    warnings: list[str]

    def __new__(
        cls,
        filter_str: str,
        layout_used: str = "speaker",
        coverage: float = 0.0,
        warnings: Optional[list[str]] = None,
    ):
        instance = super().__new__(cls, filter_str)
        instance.layout_used = layout_used
        instance.coverage = float(coverage)
        instance.filter_str = filter_str
        instance.warnings = list(warnings or [])
        return instance

    def __repr__(self) -> str:
        warn_str = f", warnings={len(self.warnings)}" if self.warnings else ""
        return f"CropFilterResult('{self.filter_str}', layout_used='{self.layout_used}', coverage={self.coverage:.3f}{warn_str})"


def compute_detection_coverage(samples: list[FaceSample]) -> float:
    """Calculate the ratio of samples containing at least one valid detected face."""
    if not samples:
        return 0.0
    valid_count = sum(1 for s in samples if s.primary_face is not None)
    return float(valid_count / len(samples))


def fill_face_timeline(
    sample_times: list[float],
    raw_centers: list[Optional[float]],
    max_gap: Optional[float] = None,
    ease_duration: Optional[float] = None,
    neutral_position: Optional[float] = None,
) -> list[float]:
    """Fill face center timeline with bounded gaps and smooth easing toward neutral position.

    1. Bounded backward-fill for leading gaps: fills up to max_gap; longer gaps ease from neutral.
    2. Bounded forward-fill for trailing gaps: fills up to max_gap; longer gaps ease to neutral.
    3. Intermediate gaps: gaps <= max_gap are linearly interpolated; gaps > max_gap ease to/from neutral.
    4. Clips with zero detected faces default to neutral_position throughout.
    """
    n = len(sample_times)
    if n == 0 or len(raw_centers) == 0:
        return []

    gap_limit = max_gap if max_gap is not None else getattr(settings, "FACE_MAX_GAP_FILL_SECONDS", 1.5)
    ease_dur = ease_duration if ease_duration is not None else getattr(settings, "FACE_EASE_DURATION_SECONDS", 1.0)
    neutral = neutral_position if neutral_position is not None else getattr(settings, "FACE_NEUTRAL_POSITION", 0.5)

    valid_indices = [idx for idx, c in enumerate(raw_centers) if c is not None]
    if not valid_indices:
        return [neutral] * n

    clean_centers: list[float] = []

    for i in range(n):
        c = raw_centers[i]
        if c is not None:
            clean_centers.append(c)
            continue

        t = sample_times[i]
        prev_idx = max((j for j in valid_indices if j < i), default=None)
        next_idx = min((j for j in valid_indices if j > i), default=None)

        if prev_idx is None and next_idx is not None:
            # Case 1: Leading gap before first face
            t_next = sample_times[next_idx]
            c_next = raw_centers[next_idx]
            dt = t_next - t
            if dt <= gap_limit:
                clean_centers.append(c_next)
            elif dt >= gap_limit + ease_dur:
                clean_centers.append(neutral)
            else:
                progress = (gap_limit + ease_dur - dt) / ease_dur
                val = neutral + progress * (c_next - neutral)
                clean_centers.append(float(val))

        elif next_idx is None and prev_idx is not None:
            # Case 2: Trailing gap after last face
            t_prev = sample_times[prev_idx]
            c_prev = raw_centers[prev_idx]
            dt = t - t_prev
            if dt <= gap_limit:
                clean_centers.append(c_prev)
            elif dt >= gap_limit + ease_dur:
                clean_centers.append(neutral)
            else:
                progress = (dt - gap_limit) / ease_dur
                val = c_prev + progress * (neutral - c_prev)
                clean_centers.append(float(val))

        elif prev_idx is not None and next_idx is not None:
            # Case 3: Intermediate gap between two detected faces
            t_prev = sample_times[prev_idx]
            c_prev = raw_centers[prev_idx]
            t_next = sample_times[next_idx]
            c_next = raw_centers[next_idx]
            gap_dur = t_next - t_prev

            if gap_dur <= gap_limit:
                # Short gap: linear interpolation between previous and next face
                ratio = (t - t_prev) / gap_dur if gap_dur > 0 else 0.5
                val = c_prev + ratio * (c_next - c_prev)
                clean_centers.append(float(val))
            else:
                # Long gap: forward-fill / ease on left, backward-fill / ease on right
                dt_from_prev = t - t_prev
                dt_to_next = t_next - t

                if dt_from_prev <= dt_to_next:
                    if dt_from_prev <= gap_limit:
                        clean_centers.append(c_prev)
                    elif dt_from_prev >= gap_limit + ease_dur:
                        clean_centers.append(neutral)
                    else:
                        progress = (dt_from_prev - gap_limit) / ease_dur
                        val = c_prev + progress * (neutral - c_prev)
                        clean_centers.append(float(val))
                else:
                    if dt_to_next <= gap_limit:
                        clean_centers.append(c_next)
                    elif dt_to_next >= gap_limit + ease_dur:
                        clean_centers.append(neutral)
                    else:
                        progress = (dt_to_next - gap_limit) / ease_dur
                        val = c_next + progress * (neutral - c_next)
                        clean_centers.append(float(val))
        else:
            clean_centers.append(neutral)

    return clean_centers


def douglas_peucker_keyframes(
    points: list[tuple[float, float]],
    epsilon: float = 2.0,
    max_keyframes: int = 40,
) -> list[tuple[float, float]]:
    """Simplify 2D keyframes (t, x) using vertical distance Douglas-Peucker algorithm.

    Iteratively increases epsilon if keyframe count exceeds max_keyframes to cap expression complexity.
    """
    if len(points) <= 2:
        return points

    def _rdp(pts: list[tuple[float, float]], eps: float) -> list[tuple[float, float]]:
        if len(pts) <= 2:
            return pts
        first, last = pts[0], pts[-1]
        dt = last[0] - first[0]
        dx = last[1] - first[1]
        if dt <= 1e-6:
            return [first, last]

        max_err = 0.0
        max_idx = 0
        for i in range(1, len(pts) - 1):
            pt = pts[i]
            interp_x = first[1] + (pt[0] - first[0]) / dt * dx
            err = abs(pt[1] - interp_x)
            if err > max_err:
                max_err = err
                max_idx = i

        if max_err > eps:
            left = _rdp(pts[: max_idx + 1], eps)
            right = _rdp(pts[max_idx:], eps)
            return left[:-1] + right
        return [first, last]

    current_eps = max(0.1, epsilon)
    simplified = _rdp(points, current_eps)
    while len(simplified) > max_keyframes and current_eps < 50.0:
        current_eps *= 1.5
        simplified = _rdp(points, current_eps)

    return simplified


def generate_smooth_camera_path(
    sample_times: list[float],
    clean_centers: list[float],
    start_time: float,
    duration: float,
    frame_w: int,
    crop_w: int,
    dead_zone_ratio: Optional[float] = None,
    max_pan_speed: Optional[float] = None,
    omega: Optional[float] = None,
    dt: float = 0.02,
) -> list[tuple[float, float]]:
    """Build a continuous, smoothed camera crop-x trajectory with dead zone and ease-in/out pan limits."""
    max_crop_x = max(0, frame_w - crop_w)
    dz_ratio = (
        dead_zone_ratio
        if dead_zone_ratio is not None
        else getattr(settings, "CROP_DEAD_ZONE_RATIO", 0.15)
    )
    pan_speed = (
        max_pan_speed
        if max_pan_speed is not None
        else getattr(settings, "CROP_MAX_PAN_SPEED", 300.0)
    )
    freq = (
        omega
        if omega is not None
        else getattr(settings, "CROP_EASE_OMEGA", 3.0)
    )

    dead_zone_px = dz_ratio * crop_w

    if not sample_times or not clean_centers:
        static_x = max(0.0, min(float(max_crop_x), (frame_w - crop_w) / 2.0))
        return [(0.0, static_x), (max(0.1, duration), static_x)]

    rel_times = [float(t - start_time) for t in sample_times]
    centers = [float(c) for c in clean_centers]

    valid_pts = [(max(0.0, min(duration, rt)), c) for rt, c in zip(rel_times, centers)]
    valid_pts.sort(key=lambda p: p[0])

    deduped_r: list[float] = []
    deduped_c: list[float] = []
    for rt, c in valid_pts:
        if not deduped_r or rt > deduped_r[-1] + 1e-5:
            deduped_r.append(rt)
            deduped_c.append(c)
        else:
            deduped_c[-1] = (deduped_c[-1] + c) / 2.0

    if not deduped_r:
        static_x = max(0.0, min(float(max_crop_x), (frame_w - crop_w) / 2.0))
        return [(0.0, static_x), (max(0.1, duration), static_x)]

    if deduped_r[0] > 0.0:
        deduped_r.insert(0, 0.0)
        deduped_c.insert(0, deduped_c[0])
    if deduped_r[-1] < duration:
        deduped_r.append(duration)
        deduped_c.append(deduped_c[-1])

    t_grid = np.arange(0.0, max(0.1, duration) + dt / 2.0, dt)
    face_px_grid = np.interp(t_grid, deduped_r, [c * frame_w for c in deduped_c])

    x = float(np.clip(face_px_grid[0] - crop_w / 2.0, 0.0, float(max_crop_x)))
    v = 0.0
    cam_x = x
    path: list[tuple[float, float]] = []

    for t_step, face_px in zip(t_grid, face_px_grid):
        cam_center = cam_x + crop_w / 2.0
        # Dead zone: hold still while subject center stays within safe margin
        if face_px > cam_center + dead_zone_px:
            x_target = face_px - dead_zone_px - crop_w / 2.0
        elif face_px < cam_center - dead_zone_px:
            x_target = face_px + dead_zone_px - crop_w / 2.0
        else:
            x_target = cam_x

        x_target = float(np.clip(x_target, 0.0, float(max_crop_x)))

        # Critically damped spring integration with velocity capping
        acc = freq * freq * (x_target - x) - 2.0 * freq * v
        v = float(np.clip(v + acc * dt, -pan_speed, pan_speed))
        x = float(np.clip(x + v * dt, 0.0, float(max_crop_x)))
        cam_x = x
        path.append((float(t_step), x))

    return path


def build_piecewise_linear_crop_expr(keyframes: list[tuple[float, float]], max_crop_x: int) -> str:
    """Emit piecewise-linear FFmpeg x expression between keyframes (time-clamped, even-pixel result)."""
    if not keyframes:
        static_x = max(0, max_crop_x // 2)
        static_x -= (static_x % 2)
        return str(static_x)

    x_values = [pt[1] for pt in keyframes]
    if len(keyframes) <= 1 or (max(x_values) - min(x_values) < 1.0):
        static_x = int(round(x_values[0]))
        static_x = max(0, min(max_crop_x, static_x))
        static_x -= (static_x % 2)
        return str(static_x)

    segments: list[tuple[float, float, str]] = []
    for i in range(len(keyframes) - 1):
        t0, x0 = keyframes[i]
        t1, x1 = keyframes[i + 1]
        dt = max(1e-4, t1 - t0)
        dx = x1 - x0
        if abs(dx) < 0.2:
            seg_expr = f"{x0:.1f}"
        else:
            slope = dx / dt
            seg_expr = f"({x0:.1f}+{slope:.2f}*(t-{t0:.2f}))"
        segments.append((t0, t1, seg_expr))

    def _build_tree(segs: list[tuple[float, float, str]]) -> str:
        if len(segs) == 1:
            return segs[0][2]
        mid = len(segs) // 2
        t_split = segs[mid][0]
        left = _build_tree(segs[:mid])
        right = _build_tree(segs[mid:])
        return f"if(lt(t,{t_split:.2f}),{left},{right})"

    tree_expr = _build_tree(segments)
    return f"2*round(clip({tree_expr},0,{max_crop_x})/2)"


def compute_crop_filter(
    video_path: str,
    start_time: float,
    end_time: float,
    mode: str = "speaker",
    source_info: Optional[Any] = None,
    force_crop: bool = False,
    coverage_threshold: Optional[float] = None,
    words: Optional[list[dict]] = None,
) -> CropFilterResult:
    """Compute crop/scale filter string for FFmpeg single-pass rendering using YuNet neural face detection."""
    if mode == "split":
        # Split screen layout: top gameplay, bottom webcam
        return CropFilterResult(SPLIT_FILTER, layout_used="split", coverage=1.0)

    if mode == "fit_blur":
        # Explicit fit with blurred background bars
        return CropFilterResult(FIT_BLUR_FILTER, layout_used="fit_blur", coverage=0.0)

    if not (source_info and getattr(source_info, "width", 0) > 0 and getattr(source_info, "height", 0) > 0):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        cap.release()

    # 1. Normalize geometry first: apply rotation and non-square SAR before computing crops
    norm_w, norm_h, sar_ratio, rotation = probe_geometry(video_path, source_info=source_info)

    crop_w = int(norm_h * 9 / 16)
    crop_w = crop_w - (crop_w % 2)

    dar = float(norm_w) / float(norm_h)
    target_dar = 9.0 / 16.0

    # 2. Exact 9:16 source passes through scaled with no distortion or black padding
    if abs(dar - target_dar) < 0.01 or crop_w == norm_w:
        sar_prefix = "scale=trunc(iw*sar/2)*2:ih,setsar=1," if abs(sar_ratio - 1.0) > 0.01 else ""
        exact_filter = f"{sar_prefix}scale=1080:1920:flags=lanczos,setsar=1"
        return CropFilterResult(exact_filter, layout_used="speaker", coverage=1.0)

    # 3. Source is narrower than 9:16 (Wcrop > norm_w, e.g. 1:2)
    # A 9:16 crop cannot fit horizontally without cutting top/bottom or out of bounds -> use fit_blur
    if norm_w < crop_w:
        return CropFilterResult(FIT_BLUR_FILTER, layout_used="fit_blur", coverage=0.0)

    # 4. Source is wider than 9:16 (Wcrop < norm_w): crop fits!
    duration = max(0.1, end_time - start_time)
    samples = sample_faces_sequentially(
        video_path,
        start_time,
        end_time,
        source_info=source_info,
    )

    coverage = compute_detection_coverage(samples)
    cov_thresh = (
        coverage_threshold
        if coverage_threshold is not None
        else getattr(settings, "FACE_COVERAGE_THRESHOLD", 0.40)
    )

    # Fall back to fit_blur if coverage is below threshold or zero faces detected
    if (coverage < cov_thresh or coverage == 0.0) and not force_crop:
        return CropFilterResult(FIT_BLUR_FILTER, layout_used="fit_blur", coverage=coverage)

    sample_times = [s.timestamp for s in samples] if samples else [start_time]

    filter_warnings: list[str] = []
    if getattr(settings, "SPEAKER_TRACKING_ENABLED", True) and samples:
        raw_centers, spk_warnings = select_active_speaker_trajectory(
            samples=samples,
            frame_w=norm_w,
            frame_h=norm_h,
            crop_w=crop_w,
            words=words,
        )
        filter_warnings.extend(spk_warnings)
    else:
        raw_centers = (
            [
                float(s.primary_face.center_x / float(norm_w))
                if s.primary_face
                else None
                for s in samples
            ]
            if samples
            else [0.5]
        )

    clean_centers = fill_face_timeline(sample_times, raw_centers)

    # Generate smooth continuous camera path with dead zone and ease-in/out pan limits
    path = generate_smooth_camera_path(
        sample_times=sample_times,
        clean_centers=clean_centers,
        start_time=start_time,
        duration=duration,
        frame_w=norm_w,
        crop_w=crop_w,
    )

    simplify_eps = getattr(settings, "CROP_SIMPLIFY_EPSILON", 2.0)
    max_kf = getattr(settings, "CROP_MAX_KEYFRAMES", 40)
    keyframes = douglas_peucker_keyframes(path, epsilon=simplify_eps, max_keyframes=max_kf)

    max_crop_x = max(0, norm_w - crop_w)
    crop_expr = build_piecewise_linear_crop_expr(keyframes, max_crop_x)

    sar_prefix = "scale=trunc(iw*sar/2)*2:ih,setsar=1," if abs(sar_ratio - 1.0) > 0.01 else ""

    if crop_expr.isdigit() or (crop_expr.replace("-", "").isdigit()):
        res_str = f"{sar_prefix}crop={crop_w}:{norm_h}:{crop_expr}:0,scale=1080:1920:flags=lanczos,setsar=1"
    else:
        res_str = f"{sar_prefix}crop={crop_w}:{norm_h}:x='{crop_expr}':y=0,scale=1080:1920:flags=lanczos,setsar=1"

    return CropFilterResult(
        res_str,
        layout_used="speaker",
        coverage=coverage,
        warnings=filter_warnings,
    )


def crop_to_vertical(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: str,
    mode: str = "speaker",
    words: Optional[list[dict]] = None,
) -> str:
    """Crop video segment [start_time, end_time] to vertical 9:16 (1080x1920)."""
    crop_filter = compute_crop_filter(video_path, start_time, end_time, mode=mode, words=words)
    from app.services.captions import render_clip_single_pass

    return render_clip_single_pass(
        video_path=video_path,
        start_time=start_time,
        end_time=end_time,
        crop_filter=crop_filter,
        ass_path="",
        output_path=output_path,
    )
