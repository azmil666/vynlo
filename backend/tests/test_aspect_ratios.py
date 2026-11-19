"""Tests for correct, good-looking output for every source aspect ratio.

Verifies:
1. Geometry normalization: rotation and non-square SAR applied before crop decisions.
2. Decision logic: crops when 9:16 crop fits (Wcrop < W); uses fit_blur only when narrower than 9:16 or low coverage.
3. fit_blur: foreground centered, background scaled to cover 1080x1920, blurred and darkened, no black columns.
4. Exact 9:16 source passes through scaled with no distortion.
5. Render fixtures: 16:9, 4:3, 1:1, 4:5, 9:16, 21:9, rotated, and narrower (1:2). Output is 1080x1920, SAR=1.
"""

import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np
import pytest

from app.schemas import SourceInfo
from app.services.cropper import (
    FIT_BLUR_FILTER,
    compute_crop_filter,
    crop_to_vertical,
    normalize_geometry,
    probe_geometry,
)


def _probe_video(path: str) -> dict:
    """Extract width, height, and SAR using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,sample_aspect_ratio,display_aspect_ratio",
        "-of",
        "json",
        path,
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    data = json.loads(res.stdout)
    return data["streams"][0]


def _generate_fixture(
    path: str,
    width: int,
    height: int,
    color: str = "blue",
    rotation: int = 0,
    sar: str = "1:1",
    duration: float = 0.5,
) -> str:
    """Generate a lightweight synthetic MP4 fixture using FFmpeg."""
    if os.path.exists(path):
        return path

    os.makedirs(os.path.dirname(path), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:size={width}x{height}:rate=25:d={duration}",
    ]
    if sar not in ("1:1", "0:1", ""):
        cmd.extend(["-vf", f"setsar={sar}"])
    if rotation != 0:
        cmd.extend(["-metadata:s:v:0", f"rotate={rotation}"])

    cmd.extend([
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        path,
    ])
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return path


# ============================================================================
# 1. Geometry Normalization Tests
# ============================================================================

def test_normalize_geometry_all_ratios():
    """Verify geometry normalization for square, landscape, portrait, and rotated videos."""
    # 16:9
    w, h, sar, rot = normalize_geometry(1920, 1080, SourceInfo(width=1920, height=1080, duration=1.0))
    assert (w, h) == (1920, 1080)
    assert sar == 1.0

    # 4:3
    w, h, sar, rot = normalize_geometry(1440, 1080, SourceInfo(width=1440, height=1080, duration=1.0))
    assert (w, h) == (1440, 1080)

    # 1:1
    w, h, sar, rot = normalize_geometry(1080, 1080, SourceInfo(width=1080, height=1080, duration=1.0))
    assert (w, h) == (1080, 1080)

    # 4:5
    w, h, sar, rot = normalize_geometry(864, 1080, SourceInfo(width=864, height=1080, duration=1.0))
    assert (w, h) == (864, 1080)

    # 9:16
    w, h, sar, rot = normalize_geometry(1080, 1920, SourceInfo(width=1080, height=1920, duration=1.0))
    assert (w, h) == (1080, 1920)

    # 21:9
    w, h, sar, rot = normalize_geometry(2560, 1080, SourceInfo(width=2560, height=1080, duration=1.0))
    assert (w, h) == (2560, 1080)

    # Rotated 90° (1920x1080 rotated 90° becomes vertical 1080x1920)
    w, h, sar, rot = normalize_geometry(1920, 1080, SourceInfo(width=1920, height=1080, duration=1.0, rotation=90))
    assert (w, h) == (1080, 1920)
    assert rot == 90

    # Non-square SAR: 720x480 with SAR 32:27 -> 720 * 32/27 = 853.3 -> 854x480 (16:9 DAR)
    w, h, sar, rot = normalize_geometry(720, 480, SourceInfo(width=720, height=480, duration=1.0, sar="32:27"))
    assert w in (852, 854)
    assert h == 480
    assert sar == pytest.approx(32.0 / 27.0)


# ============================================================================
# 2. Aspect Ratio Crop Decision Tests
# ============================================================================

def test_crop_decision_wider_than_9_16_crops(tmp_path: Path):
    """When source is wider than 9:16 (Wcrop < W) and force_crop=True, crop is performed."""
    # 4:5 video (864x1080): Wcrop = 1080 * 9/16 = 606 < 864 -> crop fits!
    vid_4_5 = _generate_fixture(str(tmp_path / "vid_4_5.mp4"), 864, 1080)
    res = compute_crop_filter(vid_4_5, 0.0, 0.5, mode="speaker", force_crop=True)
    assert res.layout_used == "speaker"
    assert res.startswith("crop=606:1080:")
    assert "scale=1080:1920:flags=lanczos,setsar=1" in res

    # 4:3 video (1440x1080): Wcrop = 606 < 1440 -> crop fits!
    vid_4_3 = _generate_fixture(str(tmp_path / "vid_4_3.mp4"), 1440, 1080)
    res_4_3 = compute_crop_filter(vid_4_3, 0.0, 0.5, mode="speaker", force_crop=True)
    assert res_4_3.layout_used == "speaker"
    assert res_4_3.startswith("crop=606:1080:")

    # 21:9 video (2560x1080): Wcrop = 606 < 2560 -> crop fits!
    vid_21_9 = _generate_fixture(str(tmp_path / "vid_21_9.mp4"), 2560, 1080)
    res_21_9 = compute_crop_filter(vid_21_9, 0.0, 0.5, mode="speaker", force_crop=True)
    assert res_21_9.layout_used == "speaker"
    assert res_21_9.startswith("crop=606:1080:")


def test_crop_decision_exact_9_16_scaled_no_distortion(tmp_path: Path):
    """Exact 9:16 video passes through scaled directly with no distortion or black padding."""
    vid_9_16 = _generate_fixture(str(tmp_path / "vid_9_16.mp4"), 1080, 1920)
    res = compute_crop_filter(vid_9_16, 0.0, 0.5, mode="speaker")
    assert res.layout_used == "speaker"
    assert res == "scale=1080:1920:flags=lanczos,setsar=1"
    assert "pad=" not in res


def test_crop_decision_narrower_than_9_16_uses_fit_blur(tmp_path: Path):
    """Video narrower than 9:16 (e.g. 540x1080, 1:2) cannot fit 9:16 crop horizontally -> returns fit_blur."""
    vid_narrow = _generate_fixture(str(tmp_path / "vid_narrow.mp4"), 540, 1080)
    res = compute_crop_filter(vid_narrow, 0.0, 0.5, mode="speaker")
    assert res.layout_used == "fit_blur"
    assert res == FIT_BLUR_FILTER
    assert "boxblur=20:20" in res
    assert "drawbox=c=black@0.25:t=fill" in res
    assert "overlay=" in res


# ============================================================================
# 3. Full Rendering of All Fixtures & SAR=1 Assertions
# ============================================================================

@pytest.mark.parametrize(
    "name,width,height,color,rotation,sar",
    [
        ("16_9", 1920, 1080, "red", 0, "1:1"),
        ("4_3", 1440, 1080, "green", 0, "1:1"),
        ("1_1", 1080, 1080, "blue", 0, "1:1"),
        ("4_5", 864, 1080, "yellow", 0, "1:1"),
        ("9_16", 1080, 1920, "magenta", 0, "1:1"),
        ("21_9", 2560, 1080, "cyan", 0, "1:1"),
        ("rotated_90", 1920, 1080, "orange", 90, "1:1"),
        ("narrow_1_2", 540, 1080, "purple", 0, "1:1"),
    ],
)
def test_render_fixture_output_aspect_ratio_and_sar(
    tmp_path: Path,
    name: str,
    width: int,
    height: int,
    color: str,
    rotation: int,
    sar: str,
):
    """Render each fixture aspect ratio and verify:
    - Output dimensions are exactly 1080x1920
    - Sample Aspect Ratio is 1:1
    - Display Aspect Ratio is 9:16
    """
    input_file = _generate_fixture(
        str(tmp_path / f"input_{name}.mp4"),
        width=width,
        height=height,
        color=color,
        rotation=rotation,
        sar=sar,
    )
    output_file = str(tmp_path / f"output_{name}.mp4")

    # Render clip through the full vertical pipeline
    crop_to_vertical(input_file, 0.0, 0.5, output_file, mode="speaker")

    assert os.path.exists(output_file), f"Rendered output missing for {name}"

    probe = _probe_video(output_file)
    out_w = probe["width"]
    out_h = probe["height"]
    out_sar = probe.get("sample_aspect_ratio", "1:1")

    assert out_w == 1080, f"Expected width 1080 for {name}, got {out_w}"
    assert out_h == 1920, f"Expected height 1920 for {name}, got {out_h}"
    assert out_sar in ("1:1", "0:1", None), f"Expected SAR 1:1 for {name}, got {out_sar}"


# ============================================================================
# 4. fit_blur Edge Pixel Verification (No Black Columns)
# ============================================================================

def test_fit_blur_no_black_columns_edge_pixels(tmp_path: Path):
    """Verify that fit_blur completely covers the 1080x1920 canvas with blurred background.

    Samples left edge (x=10) and right edge (x=1070) pixels on a bright purple 540x1080 video
    and verifies that edge columns are filled with blurred content, NOT black padding.
    """
    input_file = _generate_fixture(
        str(tmp_path / "input_bright_narrow.mp4"),
        width=540,
        height=1080,
        color="purple",
        duration=0.5,
    )
    output_file = str(tmp_path / "output_bright_narrow.mp4")

    crop_to_vertical(input_file, 0.0, 0.5, output_file, mode="speaker")
    assert os.path.exists(output_file)

    # Read rendered frame with OpenCV
    cap = cv2.VideoCapture(output_file)
    ret, frame = cap.read()
    cap.release()

    assert ret is True and frame is not None
    h, w, c = frame.shape
    assert (w, h) == (1080, 1920)

    # Check horizontal midpoint y = 960
    left_edge_pixel = frame[960, 10]    # Left background column (x=10)
    right_edge_pixel = frame[960, 1070] # Right background column (x=1070)
    center_pixel = frame[960, 540]      # Center foreground (x=540)

    # Verify that edge pixels have significant color intensity (not black padding [0, 0, 0])
    left_intensity = int(np.sum(left_edge_pixel))
    right_intensity = int(np.sum(right_edge_pixel))
    center_intensity = int(np.sum(center_pixel))

    assert left_intensity > 30, f"Left edge column at x=10 appears black! Pixel: {left_edge_pixel}"
    assert right_intensity > 30, f"Right edge column at x=1070 appears black! Pixel: {right_edge_pixel}"
    assert center_intensity > 50, f"Center pixel appears black! Pixel: {center_pixel}"
