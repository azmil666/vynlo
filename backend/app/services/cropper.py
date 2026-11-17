import os
import urllib.request
from typing import Optional

import cv2
import numpy as np


def _get_face_cascade() -> Optional[cv2.CascadeClassifier]:
    """Load Haar cascade XML for face detection."""
    candidate_paths = [
        "/app/models/haarcascade_frontalface_default.xml",
        os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "haarcascade_frontalface_default.xml"
        ),
    ]

    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        candidate_paths.append(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        )

    candidate_paths.append("/tmp/haarcascade_frontalface_default.xml")

    for path in candidate_paths:
        if os.path.exists(path):
            try:
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    return cascade
            except Exception:
                pass

    tmp_cascade = "/tmp/haarcascade_frontalface_default.xml"
    if not os.path.exists(tmp_cascade):
        try:
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml",
                tmp_cascade,
            )
        except Exception:
            pass

    if os.path.exists(tmp_cascade):
        try:
            cascade = cv2.CascadeClassifier(tmp_cascade)
            if not cascade.empty():
                return cascade
        except Exception:
            pass

    return None


def compute_crop_filter(
    video_path: str,
    start_time: float,
    end_time: float,
    mode: str = "speaker",
) -> str:
    """Compute crop/scale filter string for FFmpeg single-pass rendering without pre-slicing video."""
    if mode == "split":
        # Split screen layout: top gameplay, bottom webcam
        return (
            "[0:v]crop=ih*9/8:ih:(iw-ih*9/8)/2:0,scale=1080:960:flags=lanczos,setsar=1[top];"
            "[0:v]crop=iw*0.30:ih*0.30:iw*0.70:ih*0.70,scale=1080:960:flags=lanczos,setsar=1[bottom];"
            "[top][bottom]vstack=inputs=2[v]"
        )

    # Speaker mode: Face detection with OpenCV Haar Cascade and smart horizontal crop
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    try:
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_w <= 0 or frame_h <= 0:
            raise ValueError(f"Failed to read valid video dimensions from {video_path}")

        crop_w = int(frame_h * 9 / 16)
        crop_w = crop_w - (crop_w % 2)

        # Edge case: already vertical or narrower than 9:16
        if crop_w >= frame_w:
            return "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1"

        face_centers_x: list[float] = []
        face_cascade = _get_face_cascade()

        if face_cascade is not None:
            # Sample 4 timestamps per second across [start_time, end_time]
            duration = max(0.0, end_time - start_time)
            step = 0.25  # 4 samples per second
            num_samples = int(duration / step) + 1
            sample_times = [
                start_time + i * step
                for i in range(num_samples)
                if start_time + i * step <= end_time
            ]

            for t in sample_times:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(40, 40),
                )

                if len(faces) > 0:
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest_face
                    center_x = (float(x) + float(w) / 2.0) / frame_w
                    face_centers_x.append(center_x)

        median_center_x = (
            float(np.median(face_centers_x)) if face_centers_x else 0.5
        )
        crop_x = int(median_center_x * frame_w - crop_w / 2)
        crop_x = max(0, min(frame_w - crop_w, crop_x))
        crop_x = crop_x - (crop_x % 2)

        return f"crop={crop_w}:{frame_h}:{crop_x}:0,scale=1080:1920:flags=lanczos,setsar=1"

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
