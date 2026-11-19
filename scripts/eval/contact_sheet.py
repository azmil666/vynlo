"""Contact sheet generation for human review of rendered clips.

Extracts representative frames at start, mid, and end of each rendered clip,
annotates with timestamp overlays, and tiles them side-by-side into a single
contact-sheet image.
"""

import logging
import os
from pathlib import Path
import subprocess
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("eval.contact_sheet")


def extract_frame_at_timestamp(video_path: str, timestamp_sec: float) -> Optional[np.ndarray]:
    """Extract a single frame from video at timestamp using OpenCV with ffmpeg fallback."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_target = int(timestamp_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_target)
    ret, frame = cap.read()
    cap.release()

    if ret and frame is not None:
        return frame

    # Fallback to ffmpeg single-frame extraction
    try:
        cmd = [
            "ffmpeg",
            "-ss",
            f"{timestamp_sec:.3f}",
            "-i",
            video_path,
            "-vframes",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, check=True)
        img_array = np.frombuffer(proc.stdout, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.warning(f"Failed to extract frame at {timestamp_sec:.2f}s: {e}")
        return None


def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS.cc string."""
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:04.1f}"


def generate_contact_sheet(
    clip_path: str | Path,
    output_image_path: str | Path,
    title: Optional[str] = None,
    timestamps: Optional[List[float]] = None,
    target_frame_height: int = 640,
) -> str:
    """Generate a 3-frame contact sheet (start, mid, end) for visual inspection."""
    clip_str = str(clip_path)
    out_str = str(output_image_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_str)), exist_ok=True)

    # Determine clip duration
    cap = cv2.VideoCapture(clip_str)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {clip_str}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = total_frames / fps if fps > 0 else 0.0
    cap.release()

    if duration <= 0.0:
        duration = 5.0  # fallback duration

    # Default timestamps: 10% (start), 50% (mid), 90% (end)
    if not timestamps or len(timestamps) < 3:
        t_start = min(1.0, duration * 0.1)
        t_mid = duration * 0.5
        t_end = max(t_mid + 0.5, duration * 0.9)
        timestamps = [t_start, t_mid, t_end]

    labels = ["Start", "Mid", "End"]
    extracted_frames = []

    for t, label in zip(timestamps, labels):
        frame = extract_frame_at_timestamp(clip_str, t)
        if frame is None:
            # Create a placeholder frame
            frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                f"Missing frame at {t:.1f}s",
                (100, 960),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3,
            )

        # Resize keeping 9:16 aspect ratio
        fh, fw = frame.shape[:2]
        target_width = int(target_frame_height * (fw / fh))
        resized = cv2.resize(frame, (target_width, target_frame_height), interpolation=cv2.INTER_AREA)

        # Draw semi-transparent banner at the bottom for timestamp
        banner_height = 48
        overlay = resized.copy()
        cv2.rectangle(
            overlay,
            (0, target_frame_height - banner_height),
            (target_width, target_frame_height),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.65, resized, 0.35, 0, resized)

        # Text overlay: "Start: 00:01.0"
        ts_text = f"{label}: {format_timestamp(t)}"
        cv2.putText(
            resized,
            ts_text,
            (16, target_frame_height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        extracted_frames.append(resized)

    # Combine frames horizontally with a small 4px border between them
    border_px = 6
    border = np.zeros((target_frame_height, border_px, 3), dtype=np.uint8)
    combined = np.hstack([
        extracted_frames[0],
        border,
        extracted_frames[1],
        border,
        extracted_frames[2],
    ])

    # If title provided, add a header banner
    if title:
        header_height = 42
        header = np.zeros((header_height, combined.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            header,
            title,
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 230, 255),
            2,
            cv2.LINE_AA,
        )
        combined = np.vstack([header, combined])

    # Save image
    cv2.imwrite(out_str, combined, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return out_str
