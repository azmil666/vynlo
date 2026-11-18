import json
import logging
import os
import subprocess
import time
from typing import Any, Optional

import yt_dlp

from app.config import settings
from app.schemas import PipelineError, PipelineErrorCode, SourceInfo

logger = logging.getLogger(__name__)


def classify_ytdlp_error(exc: Exception, url: str) -> PipelineError:
    """Classify yt-dlp exceptions into typed permanent vs transient PipelineError."""
    msg = str(exc)
    msg_lower = msg.lower()

    # 1. Live stream or upcoming premiere
    if any(k in msg_lower for k in ["live stream", "is a live", "premieres in", "live event", "is live"]):
        return PipelineError(
            code=PipelineErrorCode.LIVE_UNSUPPORTED,
            stage="downloading",
            user_message="Live and upcoming streams are not supported until broadcast has concluded.",
            retryable=False,
            detail=msg,
        )

    # 2. Format unavailable
    if any(k in msg_lower for k in ["requested format is not available", "no format found"]):
        return PipelineError(
            code=PipelineErrorCode.UNSUPPORTED_FORMAT,
            stage="downloading",
            user_message="No supported H.264/MP4 format found under the required resolution limit.",
            retryable=False,
            detail=msg,
        )

    # 3. Permanent: private, removed, copyright, geo-restricted, age-gated
    if any(k in msg_lower for k in [
        "private video",
        "this video is private",
        "video unavailable",
        "has been removed",
        "account has been terminated",
        "copyright",
        "not available in your country",
        "geo-restricted",
        "blocked in your country",
        "geographic restriction",
        "sign in to confirm your age",
        "age-restricted",
        "confirm your age",
        "requires authentication",
        "login required",
    ]):
        return PipelineError(
            code=PipelineErrorCode.VIDEO_UNAVAILABLE,
            stage="downloading",
            user_message="The video is unavailable, private, geo-blocked, or age-restricted.",
            retryable=False,
            detail=msg,
        )

    # 4. Permanent: invalid or unsupported URL
    if any(k in msg_lower for k in ["is not a valid url", "unsupported url", "url scheme"]):
        return PipelineError(
            code=PipelineErrorCode.INVALID_URL,
            stage="downloading",
            user_message="The provided URL is unsupported or invalid.",
            retryable=False,
            detail=msg,
        )

    # 5. Transient: HTTP 429, 5xx, or network dropouts
    is_transient = (
        isinstance(exc, (ConnectionError, TimeoutError))
        or any(k in msg_lower for k in [
            "429",
            "too many requests",
            "http error 5",
            "500",
            "502",
            "503",
            "504",
            "connection reset",
            "connection refused",
            "timed out",
            "timeout",
            "nodename nor servname",
            "temporary failure in name resolution",
            "name or service not known",
            "network is unreachable",
            "remote end closed",
            "broken pipe",
        ])
    )
    if is_transient:
        return PipelineError(
            code=PipelineErrorCode.DOWNLOAD_FAILED,
            stage="downloading",
            user_message="Temporary network or rate-limit error connecting to video host.",
            retryable=True,
            detail=msg,
        )

    # Default fallback
    return PipelineError(
        code=PipelineErrorCode.DOWNLOAD_FAILED,
        stage="downloading",
        user_message=f"Failed to access or download video: {msg[:120]}",
        retryable=False,
        detail=msg,
    )


def get_format_selector(meta_duration: Optional[float] = None) -> str:
    """Build format selection string preferring AVC1/H.264, capping height, and avoiding AV1/VP9.

    If meta_duration exceeds LONG_VIDEO_THRESHOLD_SECONDS, height is capped to LONG_VIDEO_MAX_HEIGHT
    to bound disk usage and rendering time. Unconditional trailing '/best' is omitted to fail
    with UNSUPPORTED_FORMAT if no acceptable format exists.
    """
    if meta_duration and meta_duration > settings.LONG_VIDEO_THRESHOLD_SECONDS:
        effective_height = min(settings.MAX_VIDEO_HEIGHT, settings.LONG_VIDEO_MAX_HEIGHT)
    else:
        effective_height = settings.MAX_VIDEO_HEIGHT

    return (
        f"bestvideo[vcodec^=avc1][height<={effective_height}]+bestaudio[ext=m4a]/"
        f"bestvideo[ext=mp4][vcodec!=av01][vcodec!=vp9][height<={effective_height}]+bestaudio[ext=m4a]/"
        f"bestvideo[vcodec^=avc1][height<={effective_height}]+bestaudio/"
        f"bestvideo[ext=mp4][vcodec!=av01][vcodec!=vp9][height<={effective_height}]+bestaudio/"
        f"best[ext=mp4][vcodec^=avc1][height<={effective_height}]/"
        f"best[ext=mp4][vcodec!=av01][vcodec!=vp9][height<={effective_height}]"
    )


def get_video_metadata(url: str, max_retries: int = 2) -> dict:
    """Extract video metadata without downloading, checking safety limits and stream types.

    Retries transient errors with capped backoff. Rejects live/upcoming streams.
    Handles missing duration gracefully for post-download validation.
    """
    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    if settings.YTDLP_COOKIES_PATH and os.path.exists(settings.YTDLP_COOKIES_PATH):
        ydl_opts["cookiefile"] = settings.YTDLP_COOKIES_PATH

    info = None
    for attempt in range(max_retries + 1):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            break
        except Exception as e:
            err = classify_ytdlp_error(e, url)
            if err.retryable and attempt < max_retries:
                backoff = min(5.0, 1.0 * (2 ** attempt))
                logger.warning(
                    "Transient error extracting metadata for %s (attempt %d/%d). Retrying in %.1fs: %s",
                    url,
                    attempt + 1,
                    max_retries,
                    backoff,
                    err.user_message,
                )
                time.sleep(backoff)
                continue
            raise err

    if not info:
        raise PipelineError(
            code=PipelineErrorCode.VIDEO_UNAVAILABLE,
            stage="downloading",
            user_message=f"Failed to extract metadata for URL: {url}",
            retryable=False,
        )

    # Reject live and upcoming streams
    is_live = bool(info.get("is_live"))
    is_upcoming = bool(info.get("is_upcoming"))
    live_status = str(info.get("live_status") or "").lower()
    if is_live or is_upcoming or live_status in ("is_live", "is_upcoming", "post_live"):
        raise PipelineError(
            code=PipelineErrorCode.LIVE_UNSUPPORTED,
            stage="downloading",
            user_message="Live and upcoming streams are not supported until broadcast has concluded.",
            retryable=False,
        )

    # Handle missing/None duration without crashing
    raw_duration = info.get("duration")
    duration: Optional[float] = None
    if raw_duration is not None:
        try:
            duration = float(raw_duration)
        except (ValueError, TypeError):
            duration = None

    if duration is not None and duration > settings.MAX_VIDEO_SECONDS:
        raise PipelineError(
            code=PipelineErrorCode.VIDEO_TOO_LONG,
            stage="downloading",
            user_message=f"Video exceeds maximum supported duration of {settings.MAX_VIDEO_SECONDS}s",
            retryable=False,
        )

    extractor_key = str(info.get("extractor_key") or "video").lower()
    raw_id = str(info.get("id") or "")
    video_id = f"{extractor_key}:{raw_id}"

    return {
        "video_id": video_id,
        "duration": duration,
        "title": info.get("title") or "",
    }


def validate_downloaded_media(
    video_path: str,
    meta_duration: Optional[float] = None,
) -> tuple[SourceInfo, list[dict]]:
    """Validate downloaded video file with ffprobe, ensuring frame decodability and container integrity.

    Returns SourceInfo and any non-fatal warnings (e.g. rotation, non-square SAR, VFR).
    """
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        raise PipelineError(
            code=PipelineErrorCode.DOWNLOAD_FAILED,
            stage="downloading",
            user_message="Downloaded video file is missing or 0 bytes.",
            retryable=True,
        )

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        probe_data = json.loads(res.stdout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        raise PipelineError(
            code=PipelineErrorCode.UNSUPPORTED_FORMAT,
            stage="downloading",
            user_message="Downloaded file is not a valid or decodable media container.",
            retryable=False,
            detail=str(e),
        )

    streams = probe_data.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if not video_stream:
        raise PipelineError(
            code=PipelineErrorCode.UNSUPPORTED_FORMAT,
            stage="downloading",
            user_message="No video stream found in downloaded media.",
            retryable=False,
        )

    if not audio_stream:
        raise PipelineError(
            code=PipelineErrorCode.NO_SPEECH,
            stage="downloading",
            user_message="No audio stream found in source video.",
            retryable=False,
        )

    # Determine real duration from container format or video stream
    format_duration = float(probe_data.get("format", {}).get("duration", 0) or 0)
    stream_duration = float(video_stream.get("duration", 0) or 0)
    actual_duration = format_duration or stream_duration

    if actual_duration <= 0:
        raise PipelineError(
            code=PipelineErrorCode.UNSUPPORTED_FORMAT,
            stage="downloading",
            user_message="Downloaded video has zero or invalid duration.",
            retryable=False,
        )

    # Enforce max duration guard on the real ffprobe duration
    if actual_duration > settings.MAX_VIDEO_SECONDS:
        raise PipelineError(
            code=PipelineErrorCode.VIDEO_TOO_LONG,
            stage="downloading",
            user_message=(
                f"Video exceeds maximum supported duration of {settings.MAX_VIDEO_SECONDS}s "
                f"(actual: {round(actual_duration, 1)}s)."
            ),
            retryable=False,
        )

    # Tolerance check if metadata duration was available
    if meta_duration is not None and meta_duration > 0:
        diff = abs(actual_duration - meta_duration)
        max_diff = max(settings.DURATION_TOLERANCE_SECONDS, meta_duration * 0.15)
        if diff > max_diff:
            raise PipelineError(
                code=PipelineErrorCode.DOWNLOAD_FAILED,
                stage="downloading",
                user_message=(
                    f"Downloaded duration ({round(actual_duration, 1)}s) diverges from "
                    f"metadata duration ({round(meta_duration, 1)}s) beyond tolerance ({round(max_diff, 1)}s)."
                ),
                retryable=True,
            )

    # Verify decodable first frame
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", "0", "-i", video_path, "-vframes", "1", "-f", "null", "-"],
            capture_output=True,
            check=True,
            timeout=15,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise PipelineError(
            code=PipelineErrorCode.DOWNLOAD_FAILED,
            stage="downloading",
            user_message="Failed to decode initial frame of downloaded video.",
            retryable=True,
            detail=str(e),
        )

    # Verify decodable last frame
    seek_last = max(0.0, actual_duration - 0.5)
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", str(seek_last), "-i", video_path, "-vframes", "1", "-f", "null", "-"],
            capture_output=True,
            check=True,
            timeout=15,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise PipelineError(
            code=PipelineErrorCode.DOWNLOAD_FAILED,
            stage="downloading",
            user_message="Failed to decode trailing frame of downloaded video (file may be truncated).",
            retryable=True,
            detail=str(e),
        )

    # Extract geometry, rotation, SAR, FPS, and VFR
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))

    rotation = 0
    tags = video_stream.get("tags", {})
    if "rotate" in tags:
        try:
            rotation = int(tags["rotate"]) % 360
        except (ValueError, TypeError):
            pass
    if rotation == 0:
        for sd in video_stream.get("side_data_list", []):
            if "rotation" in sd:
                try:
                    rotation = int(sd["rotation"]) % 360
                except (ValueError, TypeError):
                    pass

    sar = str(video_stream.get("sample_aspect_ratio", "1:1") or "1:1")

    raw_fps = str(video_stream.get("avg_frame_rate", "30/1") or "30/1")
    avg_fps = 30.0
    if "/" in raw_fps:
        try:
            num, den = raw_fps.split("/", 1)
            if float(den) > 0:
                avg_fps = round(float(num) / float(den), 2)
        except (ValueError, ZeroDivisionError):
            avg_fps = 30.0
    else:
        try:
            avg_fps = round(float(raw_fps), 2)
        except ValueError:
            avg_fps = 30.0

    r_fps_str = str(video_stream.get("r_frame_rate", "") or "")
    is_vfr = bool(r_fps_str and raw_fps and r_fps_str != raw_fps)

    # Timeline offset check (audio vs video presentation start time)
    format_start = float(probe_data.get("format", {}).get("start_time", 0) or 0)
    video_start = float(video_stream.get("start_time", format_start) or format_start or 0.0)
    audio_start = float(audio_stream.get("start_time", format_start) or format_start or 0.0)
    timeline_offset = round(audio_start - video_start, 4)

    warnings: list[dict] = []
    if abs(timeline_offset) > 0.02:
        warnings.append({
            "code": "AUDIO_TIMELINE_OFFSET",
            "message": f"Audio track has {timeline_offset:+.3f}s start offset relative to video timeline.",
        })
    if rotation != 0:
        warnings.append({
            "code": "ROTATED_SOURCE_VIDEO",
            "message": f"Source video contains {rotation}° rotation metadata.",
        })
    if sar not in ("1:1", "0:1", ""):
        warnings.append({
            "code": "NON_SQUARE_PIXEL_ASPECT_RATIO",
            "message": f"Source video has non-square SAR {sar}; normalized to 1:1 during rendering.",
        })
    if is_vfr:
        warnings.append({
            "code": "VARIABLE_FRAME_RATE",
            "message": "Source video has variable frame rate (VFR).",
        })

    source_info = SourceInfo(
        width=width,
        height=height,
        duration=round(actual_duration, 2),
        has_audio=True,
        rotation=rotation,
        sar=sar,
        avg_fps=avg_fps,
        is_vfr=is_vfr,
        timeline_offset=timeline_offset,
    )
    return source_info, warnings


def download_media(
    url: str,
    output_dir: str,
    extract_audio: bool = True,
    meta_duration: Optional[float] = None,
    max_retries: int = 2,
) -> tuple[str, Optional[str], SourceInfo, list[dict]]:
    """Download video from URL using yt-dlp, validate with ffprobe, and optionally extract WAV audio.

    Enforces format selection (avc1 preference, height cap, no unconditional fallback), retries
    transient network errors, and runs post-download validation via ffprobe.
    """
    os.makedirs(output_dir, exist_ok=True)

    video_path = os.path.join(output_dir, "source.mp4")
    audio_path = os.path.join(output_dir, "audio.wav") if extract_audio else None

    format_str = get_format_selector(meta_duration=meta_duration)

    ydl_opts: dict[str, Any] = {
        "format": format_str,
        "outtmpl": os.path.join(output_dir, "source.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    if settings.YTDLP_COOKIES_PATH and os.path.exists(settings.YTDLP_COOKIES_PATH):
        ydl_opts["cookiefile"] = settings.YTDLP_COOKIES_PATH

    for attempt in range(max_retries + 1):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            break
        except Exception as e:
            err = classify_ytdlp_error(e, url)
            if err.retryable and attempt < max_retries:
                backoff = min(5.0, 1.0 * (2 ** attempt))
                logger.warning(
                    "Transient error downloading %s (attempt %d/%d). Retrying in %.1fs: %s",
                    url,
                    attempt + 1,
                    max_retries,
                    backoff,
                    err.user_message,
                )
                time.sleep(backoff)
                continue
            raise err

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
        raise PipelineError(
            code=PipelineErrorCode.DOWNLOAD_FAILED,
            stage="downloading",
            user_message=f"Downloaded video file not found at {video_path}",
            retryable=True,
        )

    # Post-download validation using ffprobe
    source_info, warnings = validate_downloaded_media(video_path, meta_duration=meta_duration)

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

    return video_path, audio_path, source_info, warnings
