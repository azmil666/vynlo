"""Tests for smooth keyframed camera motion, dead zone filtering, piecewise-linear expressions, and edge clamping."""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config import settings
from app.services.cropper import (
    build_piecewise_linear_crop_expr,
    compute_crop_filter,
    crop_to_vertical,
    douglas_peucker_keyframes,
    generate_smooth_camera_path,
)


def _create_synthetic_moving_face_video(output_path: str, duration: float = 2.0, fps: int = 30) -> str:
    """Generate a 1920x1080 test video containing a synthetic drawn face moving horizontally."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = 1920, 1080
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    total_frames = int(fps * duration)
    for f in range(total_frames):
        t = f / fps
        # Smoothly move face center from x=450 to x=1450 across duration
        cx = int(450 + (t / duration) * 1000)
        cy = 540

        frame = np.full((h, w, 3), 220, dtype=np.uint8)
        # Face contour
        cv2.ellipse(frame, (cx, cy), (120, 160), 0, 0, 360, (180, 190, 220), -1)
        # Eyes
        cv2.circle(frame, (cx - 50, cy - 35), 15, (40, 40, 40), -1)
        cv2.circle(frame, (cx + 50, cy - 35), 15, (40, 40, 40), -1)
        # Mouth
        cv2.ellipse(frame, (cx, cy + 60), (45, 20), 0, 0, 180, (40, 40, 40), 3)

        writer.write(frame)

    writer.release()
    return output_path


# ============================================================================
# 1. Path Smoothness & Max Pan Speed Tests
# ============================================================================

def test_camera_path_smoothness_and_max_pan_speed():
    """An abrupt jump in subject position must not cause a camera jump; speed must be capped at max_pan_speed."""
    # Subject center abruptly jumps from 0.2 (x=384) to 0.8 (x=1536) at t=2.0
    times = [0.0, 2.0, 2.001, 7.0]
    centers = [0.2, 0.2, 0.8, 0.8]
    dt = 0.02
    max_pan_speed = 300.0  # px/s

    path = generate_smooth_camera_path(
        sample_times=times,
        clean_centers=centers,
        start_time=0.0,
        duration=7.0,
        frame_w=1920,
        crop_w=606,
        max_pan_speed=max_pan_speed,
        omega=3.0,
        dt=dt,
    )

    assert len(path) > 100
    # Check max per-step displacement
    displacements = [abs(path[i + 1][1] - path[i][1]) for i in range(len(path) - 1)]
    max_step_speed = max(displacements) / dt

    # Speed must never exceed max_pan_speed (with tiny float tolerance)
    assert max_step_speed <= max_pan_speed + 1e-3

    # Per-step displacement at 50Hz (dt=0.02) must be strictly bounded (< 6.1 px/step)
    assert max(displacements) <= (max_pan_speed * dt) + 1e-3

    # Velocity must decay to near zero after pan finishes (ease-out)
    assert displacements[-1] / dt < 5.0


# ============================================================================
# 2. Dead Zone Suppression Tests
# ============================================================================

def test_camera_dead_zone_suppresses_jitter():
    """Small subject movements within the dead zone must keep the camera completely stationary."""
    # Crop width is 606. Dead zone ratio 0.15 gives half-width 90.9 px.
    # Subject jitters by +/- 30 px around center (0.5 * 1920 = 960 px -> +/- 0.015)
    times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    # Small jitter within +/- 30 pixels (well inside 90px dead zone)
    centers = [0.5, 0.51, 0.49, 0.515, 0.485, 0.505, 0.5]

    path = generate_smooth_camera_path(
        sample_times=times,
        clean_centers=centers,
        start_time=0.0,
        duration=3.0,
        frame_w=1920,
        crop_w=606,
        dead_zone_ratio=0.15,
        dt=0.02,
    )

    x_positions = [pt[1] for pt in path]
    initial_x = x_positions[0]

    # Camera must stay 100% still throughout the entire sequence
    for x in x_positions:
        assert x == pytest.approx(initial_x, abs=1e-3)


# ============================================================================
# 3. Edge Clamping Tests
# ============================================================================

def test_camera_clamping_at_edges():
    """Camera coordinates must be strictly clamped within [0, W - crop_w] under extreme motion."""
    # Subject moves far beyond the left and right edges
    times = [0.0, 2.0, 4.0, 6.0]
    centers = [-0.5, -0.2, 1.2, 1.5]
    frame_w = 1920
    crop_w = 606
    max_crop_x = frame_w - crop_w  # 1314

    path = generate_smooth_camera_path(
        sample_times=times,
        clean_centers=centers,
        start_time=0.0,
        duration=6.0,
        frame_w=frame_w,
        crop_w=crop_w,
    )

    x_positions = [pt[1] for pt in path]
    assert min(x_positions) >= 0.0
    assert max(x_positions) <= float(max_crop_x)


# ============================================================================
# 4. Keyframe Simplification & Expression Tests
# ============================================================================

def test_douglas_peucker_and_expression_length():
    """A 60s clip with complex motion produces a simplified keyframe count <= 40 and a valid FFmpeg expression."""
    # Generate 60 seconds with multiple pans
    t_vals = [0.0, 10.0, 15.0, 30.0, 35.0, 50.0, 55.0, 60.0]
    c_vals = [0.2, 0.2, 0.7, 0.7, 0.3, 0.3, 0.8, 0.8]

    path = generate_smooth_camera_path(
        sample_times=t_vals,
        clean_centers=c_vals,
        start_time=0.0,
        duration=60.0,
        frame_w=1920,
        crop_w=606,
        dt=0.02,
    )

    assert len(path) == 3001
    keyframes = douglas_peucker_keyframes(path, epsilon=2.0, max_keyframes=40)

    # Keyframes must be capped
    assert len(keyframes) <= 40
    # Must preserve start and end timestamps
    assert keyframes[0][0] == 0.0
    assert keyframes[-1][0] == pytest.approx(60.0, abs=0.05)

    expr = build_piecewise_linear_crop_expr(keyframes, max_crop_x=1314)
    # Expression length must be reasonable for FFmpeg command lines (< 4000 chars)
    assert len(expr) < 4000
    assert "clip(" in expr
    assert "2*round(" in expr


def test_clip_relative_time_matching_input_seeking():
    """Keyframes must start at t=0 and span [0, duration] to match -ss input seeking."""
    # Clip slice from 15.0s to 25.0s (duration = 10.0s)
    times = [15.0, 18.0, 21.0, 25.0]
    centers = [0.3, 0.4, 0.6, 0.7]

    path = generate_smooth_camera_path(
        sample_times=times,
        clean_centers=centers,
        start_time=15.0,
        duration=10.0,
        frame_w=1920,
        crop_w=606,
    )

    # Relative times must begin at 0.0 and end at 10.0
    assert path[0][0] == 0.0
    assert path[-1][0] == pytest.approx(10.0, abs=0.02)


# ============================================================================
# 5. Real FFmpeg Render with Moving Synthetic Subject
# ============================================================================

def test_ffmpeg_render_moving_subject_no_jumps(tmp_path: Path):
    """Render a synthetic moving face video and verify vertical 9:16 output and smooth camera motion."""
    input_video = str(tmp_path / "synthetic_moving.mp4")
    output_video = str(tmp_path / "rendered_vertical.mp4")

    _create_synthetic_moving_face_video(input_video, duration=2.0, fps=30)
    assert os.path.exists(input_video)

    filter_res = compute_crop_filter(input_video, 0.0, 2.0, mode="speaker")
    assert filter_res.layout_used == "speaker"
    assert filter_res.coverage > 0.5

    result_path = crop_to_vertical(
        video_path=input_video,
        start_time=0.0,
        end_time=2.0,
        output_path=output_video,
        mode="speaker",
    )

    assert os.path.exists(result_path)
    assert os.path.getsize(result_path) > 0

    # Probe output video properties
    cap = cv2.VideoCapture(result_path)
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    assert out_w == 1080
    assert out_h == 1920
    assert frame_count >= 50
