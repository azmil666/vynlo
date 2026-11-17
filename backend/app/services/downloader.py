import os
import subprocess
from typing import Any, Optional

import yt_dlp

from app.config import settings


def get_video_metadata(url: str) -> dict:
    """Extract video metadata without downloading, checking safety duration limits."""
    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    if settings.YTDLP_COOKIES_PATH and os.path.exists(settings.YTDLP_COOKIES_PATH):
        ydl_opts["cookiefile"] = settings.YTDLP_COOKIES_PATH

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if not info:
        raise ValueError(f"Failed to extract metadata for URL: {url}")

    duration = float(info.get("duration") or 0)
    if duration > 14400:
        raise ValueError("Video exceeds maximum supported duration of 4 hours")

    extractor_key = str(info.get("extractor_key") or "video").lower()
    raw_id = str(info.get("id") or "")
    video_id = f"{extractor_key}:{raw_id}"

    return {
        "video_id": video_id,
        "duration": duration,
        "title": info.get("title") or "",
    }


def download_media(
    url: str, output_dir: str, extract_audio: bool = True
) -> tuple[str, Optional[str]]:
    """Download video from URL using yt-dlp and optionally extract 16kHz mono WAV audio."""
    os.makedirs(output_dir, exist_ok=True)

    video_path = os.path.join(output_dir, "source.mp4")
    audio_path = os.path.join(output_dir, "audio.wav") if extract_audio else None

    ydl_opts: dict[str, Any] = {
        "format": "bestvideo[vcodec^=avc1][height<=1080]+bestaudio[ext=m4a]/bestvideo[ext=mp4][vcodec!=av01][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][vcodec!=av01]/best",
        "outtmpl": os.path.join(output_dir, "source.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    if settings.YTDLP_COOKIES_PATH and os.path.exists(settings.YTDLP_COOKIES_PATH):
        ydl_opts["cookiefile"] = settings.YTDLP_COOKIES_PATH

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not os.path.exists(video_path):
        candidates = [
            f
            for f in os.listdir(output_dir)
            if f.startswith("source.") and not f.endswith(".wav")
        ]
        if candidates:
            source_candidate = os.path.join(output_dir, candidates[0])
            if source_candidate != video_path:
                os.replace(source_candidate, video_path)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Downloaded video file not found at {video_path}")

    if extract_audio and audio_path:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            audio_path,
        ]
        subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    return video_path, audio_path
