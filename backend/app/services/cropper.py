import os
import urllib.request
from typing import Any, Optional

import cv2
import numpy as np


def _get_face_detector():
    """Load YuNet neural face detector, with fallback to Haar cascade if model is unavailable."""
    # Try loading YuNet ONNX model first
    candidate_paths = [
        "/app/models/face_detection_yunet_2023mar.onnx",
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "models",
            "face_detection_yunet_2023mar.onnx",
        ),
        "/tmp/face_detection_yunet_2023mar.onnx",
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            try:
                detector = cv2.FaceDetectorYN.create(
                    path,
                    "",
                    (320, 180),
                    score_threshold=0.55,
                    nms_threshold=0.3,
                    top_k=5,
                )
                if detector is not None:
                    return detector
            except Exception:
                pass

    tmp_model = "/tmp/face_detection_yunet_2023mar.onnx"
    if not os.path.exists(tmp_model):
        try:
            urllib.request.urlretrieve(
                "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                tmp_model,
            )
        except Exception:
            pass

    if os.path.exists(tmp_model):
        try:
            detector = cv2.FaceDetectorYN.create(
                tmp_model,
                "",
                (320, 180),
                score_threshold=0.55,
                nms_threshold=0.3,
                top_k=5,
            )
            if detector is not None:
                return detector
        except Exception:
            pass

    # Secondary fallback to Haar cascade if YuNet is not present
    haar_paths = [
        "/app/models/haarcascade_frontalface_default.xml",
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "models",
            "haarcascade_frontalface_default.xml",
        ),
    ]
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        haar_paths.append(
            os.path.join(
                cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
            )
        )
    haar_paths.append("/tmp/haarcascade_frontalface_default.xml")

    for hpath in haar_paths:
        if os.path.exists(hpath):
            try:
                cascade = cv2.CascadeClassifier(hpath)
                if not cascade.empty():
                    return cascade
            except Exception:
                pass

    return None


def _detect_face_center(detector, frame, frame_w: int) -> Optional[float]:
    """Detect normalized horizontal face center [0.0, 1.0]."""
    if detector is None or frame is None:
        return None

    try:
        if isinstance(detector, cv2.FaceDetectorYN):
            # Resize frame to (320, 180) for ~2ms fast CPU detection
            small_frame = cv2.resize(frame, (320, 180))
            _, faces = detector.detect(small_frame)
            if faces is not None and len(faces) > 0:
                largest_face = max(
                    faces, key=lambda f: float(f[2]) * float(f[3])
                )
                x, _, w, _ = largest_face[:4]
                return float((float(x) + float(w) / 2.0) / 320.0)
        elif isinstance(detector, cv2.CascadeClassifier):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )
            if len(faces) > 0:
                largest_face = max(faces, key=lambda f: float(f[2]) * float(f[3]))
                x, y, w, h = largest_face
                return float((float(x) + float(w) / 2.0) / float(frame_w))
    except Exception:
        pass

    return None


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
            raise ValueError(
                f"Failed to read valid video dimensions from {video_path}"
            )

        crop_w = int(frame_h * 9 / 16)
        crop_w = crop_w - (crop_w % 2)

        # 2. Handle Already-Vertical Source Videos (frame_w <= frame_h * 0.75 or crop_w >= frame_w)
        if frame_w <= frame_h * 0.75 or crop_w >= frame_w:
            return "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1"

        # 3. Smooth Dynamic Tracking
        duration = max(0.1, end_time - start_time)
        step = 0.25  # 4 samples per second
        num_samples = int(duration / step) + 1
        sample_times = [
            start_time + i * step
            for i in range(num_samples)
            if (start_time + i * step) <= end_time
        ]
        if not sample_times:
            sample_times = [start_time]

        detector = _get_face_detector()
        raw_centers: list[Optional[float]] = []

        for t in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ret, frame = cap.read()
            if not ret or frame is None:
                raw_centers.append(None)
                continue

            center_x = _detect_face_center(detector, frame, frame_w)
            raw_centers.append(center_x)

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

        # Backward fill any leading None values from first valid face
        first_valid = next((c for c in raw_centers if c is not None), 0.5)
        clean_centers = [
            c if c is not None else first_valid for c in filled_centers
        ]

        # 1.5-second rolling median filter (at 4 samples/sec, 1.5s is 6 samples -> window radius 3)
        half_window = max(1, int(round(1.5 * 4 / 2)))
        n_samples = len(clean_centers)
        smoothed_centers: list[float] = []

        for i in range(n_samples):
            w_start = max(0, i - half_window)
            w_end = min(n_samples, i + half_window + 1)
            smoothed_centers.append(
                float(np.median(clean_centers[w_start:w_end]))
            )

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
                seg_center = float(
                    np.median([smoothed_centers[idx] for idx in sample_indices])
                )
            else:
                nearest_idx = min(
                    range(len(sample_times)),
                    key=lambda idx: abs(
                        (sample_times[idx] - start_time) - (s + 0.5)
                    ),
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
    crop_filter = compute_crop_filter(
        video_path, start_time, end_time, mode=mode
    )
    from app.services.captions import render_clip_single_pass

    return render_clip_single_pass(
        video_path=video_path,
        start_time=start_time,
        end_time=end_time,
        crop_filter=crop_filter,
        ass_path="",
        output_path=output_path,
    )
