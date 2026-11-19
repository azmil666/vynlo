"""Tests for undistorted, non-duplicated split screen layout with dynamic webcam detection."""

import os
import subprocess
import cv2
import numpy as np
import pytest

from app.schemas import WebcamRegion
from app.services.captions import render_clip_single_pass
from app.services.cropper import (
    compute_crop_filter,
    compute_split_pane_boxes,
    find_webcam_region,
    get_default_webcam_region,
    SPLIT_FILTER,
)
from tests.test_face_detection import _draw_synthetic_face


@pytest.fixture
def synthetic_split_gameplay_video(tmp_path) -> str:
    """Generate a 1920x1080 video with gameplay graphics and a static bottom-right facecam.

    Includes:
    - A YuNet-detectable synthetic face in the bottom-right corner (cx=1600, cy=800).
    - A perfect black test circle (radius 35) drawn inside the webcam area at (1500, 950).
    - A unique bright green square (50x50) at (1800, 750) exclusively in the webcam area.
    """
    video_path = str(tmp_path / "gameplay_with_facecam.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 10
    duration = 1.0
    total_frames = int(fps * duration)
    w, h = 1920, 1080
    out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    for _ in range(total_frames):
        # Dark gameplay background
        frame = np.full((h, w, 3), 40, dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (1200, 900), (80, 60, 40), -1)
        cv2.putText(
            frame,
            "GAMEPLAY AREA",
            (400, 500),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (200, 200, 200),
            4,
        )

        # Light webcam box in bottom-right corner
        cv2.rectangle(frame, (1350, 600), (1920, 1080), (200, 200, 200), -1)

        # Face in bottom-right webcam
        _draw_synthetic_face(frame, 1600, 800, 110, 150)

        # Perfect test circle inside webcam area: cx=1500, cy=950, r=35 (pure black on light gray)
        cv2.circle(frame, (1500, 950), 35, (0, 0, 0), -1)

        # Unique green marker in webcam area: (0, 255, 0)
        cv2.rectangle(frame, (1800, 750), (1850, 800), (0, 255, 0), -1)

        out.write(frame)

    out.release()
    return video_path


def test_split_screen_aspect_preservation_circle_undistorted(synthetic_split_gameplay_video, tmp_path):
    """Assert that circles in the bottom webcam pane remain undistorted (1:1 aspect ratio, w == h)."""
    out_clip = str(tmp_path / "out_undistorted_split.mp4")

    crop_filter = compute_crop_filter(
        synthetic_split_gameplay_video,
        0.0,
        1.0,
        mode="split",
    )
    assert crop_filter.layout_used == "split"

    render_clip_single_pass(
        video_path=synthetic_split_gameplay_video,
        start_time=0.0,
        end_time=1.0,
        crop_filter=crop_filter,
        ass_path="",
        output_path=out_clip,
    )

    cap = cv2.VideoCapture(out_clip)
    ret, frame = cap.read()
    cap.release()
    assert ret, "Failed to read rendered split screen frame"
    assert frame.shape == (1920, 1080, 3), f"Expected 1080x1920 output, got {frame.shape}"

    # Bottom pane is y in [960, 1920]
    bottom_pane = frame[960:1920, :]

    # Threshold for the black circle (R < 15, G < 15, B < 15)
    black_mask = cv2.inRange(bottom_pane, (0, 0, 0), (15, 15, 15))
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour corresponding to our test circle (area > 300)
    circle_contours = [c for c in contours if cv2.contourArea(c) > 300]
    assert len(circle_contours) >= 1, "Black test circle not found in bottom pane"

    c = max(circle_contours, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(c)

    # In an undistorted 9:8 crop scaled to 1080:960, the circle aspect ratio must be 1.0 (w == h within 2px)
    aspect_diff = abs(bw - bh)
    aspect_ratio = float(bw) / float(bh)
    assert aspect_diff <= 2, f"Circle distorted: w={bw}, h={bh}, diff={aspect_diff}px"
    assert 0.96 <= aspect_ratio <= 1.04, f"Circle aspect ratio distorted: {aspect_ratio:.3f}"


def test_split_screen_no_duplication_in_top_pane(synthetic_split_gameplay_video, tmp_path):
    """Assert unique webcam marker is strictly present in bottom pane and absent in top pane."""
    out_clip = str(tmp_path / "out_no_dupe_split.mp4")

    crop_filter = compute_crop_filter(
        synthetic_split_gameplay_video,
        0.0,
        1.0,
        mode="split",
    )

    render_clip_single_pass(
        video_path=synthetic_split_gameplay_video,
        start_time=0.0,
        end_time=1.0,
        crop_filter=crop_filter,
        ass_path="",
        output_path=out_clip,
    )

    cap = cv2.VideoCapture(out_clip)
    ret, frame = cap.read()
    cap.release()
    assert ret

    top_pane = frame[0:960, :]
    bottom_pane = frame[960:1920, :]

    # Green marker mask (pure green: G > 200, R < 50, B < 50)
    top_green = cv2.inRange(top_pane, (0, 200, 0), (50, 255, 50))
    bottom_green = cv2.inRange(bottom_pane, (0, 200, 0), (50, 255, 50))

    # Top pane must have ZERO green marker pixels (completely excluded!)
    assert np.count_nonzero(top_green) == 0, "Webcam marker duplicated in top pane!"
    # Bottom pane must contain the green marker
    assert np.count_nonzero(bottom_green) > 100, "Webcam marker missing from bottom pane!"


def test_split_screen_no_webcam_found_falls_back_to_fit_blur(tmp_path):
    """When video has no webcam, split mode must fall back to fit_blur with NO_WEBCAM_FOUND warning."""
    video_path = str(tmp_path / "no_face_gameplay.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 10, (1920, 1080))
    for _ in range(5):
        frame = np.full((1080, 1920, 3), 120, dtype=np.uint8)
        out.write(frame)
    out.release()

    res = compute_crop_filter(video_path, 0.0, 0.5, mode="split")
    assert res.layout_used == "fit_blur"
    assert any("NO_WEBCAM_FOUND" in w for w in res.warnings)


def test_split_screen_user_provided_webcam_region(tmp_path):
    """Assert user-provided webcam region overrides auto-detection and defines the bottom pane."""
    video_path = str(tmp_path / "blank_16_9.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 10, (1920, 1080))
    for _ in range(5):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    # User region in bottom-left: x=0.05, y=0.60, w=0.25, h=0.30
    user_region = WebcamRegion(x=0.05, y=0.60, w=0.25, h=0.30)
    res = compute_crop_filter(
        video_path,
        0.0,
        0.5,
        mode="split",
        webcam_region=user_region,
    )
    assert res.layout_used == "split"
    # Bottom crop should be positioned on the left side (x near 0.05 * 1920 ~ 96)
    assert "scale=1080:960" in res
    assert "[top][bottom]vstack=inputs=2[v]" in res


def test_split_screen_narrow_sources_render_without_crash(tmp_path):
    """Assert narrow sources (1:1 and 9:16) render cleanly without FFmpeg crop errors."""
    for size, name in [((1080, 1080), "square"), ((1080, 1920), "vertical")]:
        video_path = str(tmp_path / f"narrow_{name}.mp4")
        w, h = size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, 10, (w, h))
        for _ in range(5):
            frame = np.full((h, w, 3), 90, dtype=np.uint8)
            out.write(frame)
        out.release()

        # Provide a valid user region so split mode doesn't need face detection
        user_region = {"x": 0.1, "y": 0.5, "w": 0.8, "h": 0.4}
        res = compute_crop_filter(video_path, 0.0, 0.5, mode="split", webcam_region=user_region)
        assert res.layout_used == "split"

        out_clip = str(tmp_path / f"rendered_{name}.mp4")
        render_clip_single_pass(
            video_path=video_path,
            start_time=0.0,
            end_time=0.5,
            crop_filter=res,
            ass_path="",
            output_path=out_clip,
        )
        assert os.path.exists(out_clip)
        cap = cv2.VideoCapture(out_clip)
        ret, frame = cap.read()
        cap.release()
        assert ret
        assert frame.shape == (1920, 1080, 3)


def test_compute_split_pane_boxes_aspect_ratios():
    """Verify that compute_split_pane_boxes outputs 9:8 aspect ratio for both panes across diverse aspect ratios."""
    test_geometries = [
        (1920, 1080),  # 16:9
        (2560, 1080),  # 21:9
        (1440, 1080),  # 4:3
        (1080, 1080),  # 1:1
        (1080, 1920),  # 9:16
    ]
    for w, h in test_geometries:
        cam_box = get_default_webcam_region(w, h)
        top_pane, bottom_pane = compute_split_pane_boxes(w, h, cam_box)

        top_w, top_h = top_pane[2], top_pane[3]
        bot_w, bot_h = bottom_pane[2], bottom_pane[3]

        top_ar = top_w / top_h
        bot_ar = bot_w / bot_h

        # 9:8 is 1.125
        assert abs(top_ar - 1.125) < 0.03, f"Geometry {w}x{h} top pane aspect ratio {top_ar} is not ~1.125"
        assert abs(bot_ar - 1.125) < 0.03, f"Geometry {w}x{h} bottom pane aspect ratio {bot_ar} is not ~1.125"
        assert top_w % 2 == 0 and top_h % 2 == 0
        assert bot_w % 2 == 0 and bot_h % 2 == 0
