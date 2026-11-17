import os
import subprocess
from typing import Any

import cv2
import mediapipe as mp
import numpy as np


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
            # Speaker mode: Face detection with MediaPipe and smart horizontal crop
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
                cap = cv2.VideoCapture(temp_slice)
                frame_count = 0
                face_centers_x: list[float] = []

                mp_face_detection = mp.solutions.face_detection
                with mp_face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5
                ) as face_detector:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_count += 1
                        if frame_count % 5 != 0:
                            continue

                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results: Any = face_detector.process(rgb_frame)

                        if results.detections:
                            # Select highest confidence or largest face detection
                            best_detection = max(
                                results.detections,
                                key=lambda d: d.score[0] if d.score else 0.0,
                            )
                            bbox = best_detection.location_data.relative_bounding_box
                            center_x = bbox.xmin + bbox.width / 2.0
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
