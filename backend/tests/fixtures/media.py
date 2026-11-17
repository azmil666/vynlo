import json
import os
from pathlib import Path
import subprocess


def probe(path: str | Path) -> dict:
    """Wrap ffprobe to inspect media streams and container metadata in JSON."""
    file_path = str(path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Media file not found for probing: {file_path}")

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        file_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def generate_16_9_1080p_30s(output_path: str | Path) -> str:
    """Generate 16:9 1080p (1920x1080) 30s video with sine wave audio using lavfi."""
    out = str(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=1920x1080:rate=30",
        "-f",
        "lavfi",
        "-i",
        "sine=f=1000:r=44100",
        "-t",
        "30",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        out,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return out


def generate_vertical_1080x1920(output_path: str | Path, duration: float = 2.0) -> str:
    """Generate vertical (1080x1920) video with sine audio using lavfi."""
    out = str(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=1080x1920:rate=30",
        "-f",
        "lavfi",
        "-i",
        "sine=f=1000:r=44100",
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        out,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return out


def generate_square(output_path: str | Path, duration: float = 2.0) -> str:
    """Generate square (1080x1080) video with sine audio using lavfi."""
    out = str(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=1080x1080:rate=30",
        "-f",
        "lavfi",
        "-i",
        "sine=f=1000:r=44100",
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        out,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return out


def generate_vfr(output_path: str | Path, duration: float = 2.0) -> str:
    """Generate variable frame rate (VFR) video using mpdecimate."""
    out = str(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    # Prefer -fps_mode vfr (FFmpeg 7+), fallback to -vsync vfr for older FFmpeg
    base_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=1920x1080:rate=30",
        "-vf",
        "mpdecimate",
    ]
    try:
        cmd = base_cmd + [
            "-fps_mode",
            "vfr",
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-an",
            out,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError:
        cmd = base_cmd + [
            "-vsync",
            "vfr",
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-an",
            out,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
    return out


def generate_no_audio(output_path: str | Path, duration: float = 2.0) -> str:
    """Generate 16:9 1080p video without an audio stream (-an)."""
    out = str(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=1920x1080:rate=30",
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-an",
        out,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return out


def generate_silent_audio(output_path: str | Path, duration: float = 2.0) -> str:
    """Generate 16:9 1080p video with a silent audio stream using anullsrc."""
    out = str(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=1920x1080:rate=30",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=44100:cl=stereo",
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        out,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return out
