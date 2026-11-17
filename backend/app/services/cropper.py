import os
import subprocess
import urllib.request
from typing import Optional

import cv2
import numpy as np


def _get_face_cascade() -> Optional[cv2.CascadeClassifier]:
    """Load or download Haar cascade XML for face detection."""
    xml_path = None

    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        candidate = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        if os.path.exists(candidate):
            xml_path = candidate

    if not xml_path:
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
            xml_path = tmp_cascade

    if not xml_path:
        return None

    try:
        cascade = cv2.CascadeClassifier(xml_path)
        if cascade.empty():
            return None
        return cascade
    except Exception:
        return None


def crop_to_vertical(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: str,
    mode: str = "speaker",
) -> str:
    """Crop video segment [start_time, end_time] to vertical 9:16 (1080x1920)."""
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(output_path))[0]
    temp_slice = os.path.join(out_dir, f"temp_{base_name}.mp4")

    # Step 1: Slice source segment
    duration = max(0.1, end_time - start_time)
    slice_cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_time),
        "-t",
        str(duration),
        "-i",
        video_path,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        temp_slice,
    ]
    subprocess.run(
        slice_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    try:
        # Step 2: Inspect resolution
        cap = cv2.VideoCapture(temp_slice)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if frame_w <= 0 or frame_h <= 0:
            raise ValueError(
                f"Failed to read valid video dimensions from {temp_slice}"
            )

        if mode == "split":
            # Split screen layout: top gameplay, bottom webcam
            filter_complex = (
                "[0:v]crop=ih*9/8:ih:(iw-ih*9/8)/2:0,scale=1080:960:flags=lanczos,setsar=1[top];"
                "[0:v]crop=iw*0.30:ih*0.30:iw*0.70:ih*0.70,scale=1080:960:flags=lanczos,setsar=1[bottom];"
                "[top][bottom]vstack=inputs=2[v]"
            )
            render_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                temp_slice,
                "-filter_complex",
                filter_complex,
                "-map",
                "[v]",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                output_path,
            ]
            subprocess.run(
                render_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        else:
            # Speaker mode: Face detection with OpenCV Haar Cascade and smart horizontal crop
            crop_w = int(frame_h * 9 / 16)
            crop_w = crop_w - (crop_w % 2)

            # Edge case: already vertical or narrower than 9:16
            if crop_w >= frame_w:
                render_cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    temp_slice,
                    "-vf",
                    "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-c:a",
                    "copy",
                    output_path,
                ]
                subprocess.run(
                    render_cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            else:
                face_centers_x: list[float] = []
                face_cascade = _get_face_cascade()

                if face_cascade is not None:
                    cap = cv2.VideoCapture(temp_slice)
                    frame_count = 0

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_count += 1
                        if frame_count % 5 != 0:
                            continue

                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(
                            gray,
                            scaleFactor=1.1,
                            minNeighbors=5,
                            minSize=(40, 40),
                        )

                        if len(faces) > 0:
                            # Pick the largest face by area (w * h)
                            largest_face = max(faces, key=lambda f: f[2] * f[3])
                            x, y, w, h = largest_face
                            center_x = (float(x) + float(w) / 2.0) / frame_w
                            face_centers_x.append(center_x)

                    cap.release()

                median_center_x = (
                    float(np.median(face_centers_x)) if face_centers_x else 0.5
                )
                crop_x = int(median_center_x * frame_w - crop_w / 2)
                crop_x = max(0, min(frame_w - crop_w, crop_x))
                crop_x = crop_x - (crop_x % 2)

                render_cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    temp_slice,
                    "-vf",
                    f"crop={crop_w}:{frame_h}:{crop_x}:0,scale=1080:1920:flags=lanczos,setsar=1",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-c:a",
                    "copy",
                    output_path,
                ]
                subprocess.run(
                    render_cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

    finally:
        if os.path.exists(temp_slice):
            try:
                os.remove(temp_slice)
            except OSError:
                pass

    return output_path
