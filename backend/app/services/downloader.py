import os
import subprocess
from typing import Any

import yt_dlp

from app.config import settings


def download_media(url: str, output_dir: str) -> tuple[str, str]:
    """Download video from URL using yt-dlp and extract 16kHz mono WAV audio."""
    os.makedirs(output_dir, exist_ok=True)

    video_path = os.path.join(output_dir, "source.mp4")
    audio_path = os.path.join(output_dir, "audio.wav")

    ydl_opts: dict[str, Any] = {
        "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[ext=mp4]/best",
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

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return video_path, audio_path
