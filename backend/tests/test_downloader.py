import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yt_dlp

from app.config import settings
from app.schemas import JobStatusResponse, PipelineError, PipelineErrorCode, SourceInfo
from app.services.downloader import (
    classify_ytdlp_error,
    download_media,
    get_format_selector,
    get_video_metadata,
    validate_downloaded_media,
)
from app.worker import process_video_task
from tests.fixtures.media import (
    generate_16_9_1080p_30s,
    generate_no_audio,
    generate_square,
    generate_vertical_1080x1920,
    generate_vfr,
)


def test_classify_ytdlp_error_permanent():
    """Verify permanent errors are classified with retryable=False."""
    # Private video
    err = classify_ytdlp_error(Exception("ERROR: This video is private"), "https://youtu.be/test")
    assert err.code == PipelineErrorCode.VIDEO_UNAVAILABLE
    assert err.retryable is False

    # Removed video
    err = classify_ytdlp_error(Exception("ERROR: Video unavailable. This video has been removed"), "https://youtu.be/test")
    assert err.code == PipelineErrorCode.VIDEO_UNAVAILABLE
    assert err.retryable is False

    # Geo-restricted
    err = classify_ytdlp_error(Exception("ERROR: Video not available in your country due to geo-restricted rights"), "https://youtu.be/test")
    assert err.code == PipelineErrorCode.VIDEO_UNAVAILABLE
    assert err.retryable is False

    # Age-gated
    err = classify_ytdlp_error(Exception("ERROR: Sign in to confirm your age. This video may be inappropriate"), "https://youtu.be/test")
    assert err.code == PipelineErrorCode.VIDEO_UNAVAILABLE
    assert err.retryable is False

    # Invalid URL
    err = classify_ytdlp_error(Exception("ERROR: 'not_a_url' is not a valid URL."), "not_a_url")
    assert err.code == PipelineErrorCode.INVALID_URL
    assert err.retryable is False

    # Live stream
    err = classify_ytdlp_error(Exception("ERROR: This live stream has not started or is a live broadcast"), "https://youtu.be/test")
    assert err.code == PipelineErrorCode.LIVE_UNSUPPORTED
    assert err.retryable is False

    # Requested format not available
    err = classify_ytdlp_error(Exception("ERROR: Requested format is not available"), "https://youtu.be/test")
    assert err.code == PipelineErrorCode.UNSUPPORTED_FORMAT
    assert err.retryable is False


def test_classify_ytdlp_error_transient():
    """Verify transient network and rate-limit errors are classified with retryable=True."""
    # Rate limit (HTTP 429)
    err = classify_ytdlp_error(Exception("HTTP Error 429: Too Many Requests"), "https://youtu.be/test")
    assert err.code == PipelineErrorCode.DOWNLOAD_FAILED
    assert err.retryable is True

    # Server error (HTTP 503)
    err = classify_ytdlp_error(Exception("HTTP Error 503: Service Unavailable"), "https://youtu.be/test")
    assert err.code == PipelineErrorCode.DOWNLOAD_FAILED
    assert err.retryable is True

    # Socket connection timeout
    err = classify_ytdlp_error(TimeoutError("Connection timed out"), "https://youtu.be/test")
    assert err.code == PipelineErrorCode.DOWNLOAD_FAILED
    assert err.retryable is True

    # DNS lookup failure
    err = classify_ytdlp_error(Exception("nodename nor servname provided, or not known"), "https://youtu.be/test")
    assert err.code == PipelineErrorCode.DOWNLOAD_FAILED
    assert err.retryable is True


def test_get_video_metadata_transient_retry_success():
    """Verify transient errors retry with backoff and succeed if subsequent attempt passes."""
    with patch("yt_dlp.YoutubeDL") as mock_ydl_cls, patch("time.sleep") as mock_sleep:
        mock_ydl = MagicMock()
        mock_ydl.extract_info.side_effect = [
            Exception("HTTP Error 429: Too Many Requests"),
            {"id": "abc123", "duration": 120, "title": "Test Title"},
        ]
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

        meta = get_video_metadata("https://youtube.com/watch?v=abc123")
        assert meta["video_id"] == "video:abc123"
        assert meta["duration"] == 120.0
        assert mock_sleep.called


def test_get_video_metadata_transient_retry_exhausted():
    """Verify transient errors raise PipelineError(retryable=True) when retries are exhausted."""
    with patch("yt_dlp.YoutubeDL") as mock_ydl_cls, patch("time.sleep"):
        mock_ydl = MagicMock()
        mock_ydl.extract_info.side_effect = Exception("HTTP Error 503: Service Unavailable")
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

        with pytest.raises(PipelineError) as exc_info:
            get_video_metadata("https://youtube.com/watch?v=abc123", max_retries=1)

        assert exc_info.value.code == PipelineErrorCode.DOWNLOAD_FAILED
        assert exc_info.value.retryable is True


def test_get_video_metadata_permanent_no_retry():
    """Verify permanent errors raise immediately without retrying."""
    with patch("yt_dlp.YoutubeDL") as mock_ydl_cls, patch("time.sleep") as mock_sleep:
        mock_ydl = MagicMock()
        mock_ydl.extract_info.side_effect = Exception("ERROR: This video is private")
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

        with pytest.raises(PipelineError) as exc_info:
            get_video_metadata("https://youtube.com/watch?v=abc123", max_retries=2)

        assert exc_info.value.code == PipelineErrorCode.VIDEO_UNAVAILABLE
        assert exc_info.value.retryable is False
        mock_sleep.assert_not_called()


def test_get_video_metadata_live_and_upcoming_rejected():
    """Verify live and upcoming stream metadata is rejected with LIVE_UNSUPPORTED."""
    with patch("yt_dlp.YoutubeDL") as mock_ydl_cls:
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = {
            "id": "live123",
            "is_live": True,
            "title": "Live Broadcast",
        }
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

        with pytest.raises(PipelineError) as exc_info:
            get_video_metadata("https://youtube.com/watch?v=live123")

        assert exc_info.value.code == PipelineErrorCode.LIVE_UNSUPPORTED


def test_get_video_metadata_missing_duration_handled():
    """Verify missing/None duration in metadata does not crash and returns duration=None."""
    with patch("yt_dlp.YoutubeDL") as mock_ydl_cls:
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = {
            "id": "nodur123",
            "duration": None,
            "title": "No Duration Video",
        }
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

        meta = get_video_metadata("https://youtube.com/watch?v=nodur123")
        assert meta["video_id"] == "video:nodur123"
        assert meta["duration"] is None


def test_format_selector_capping():
    """Verify format selector enforces AVC1 preference, height limits, and omits trailing /best."""
    # Under long video threshold
    normal_fmt = get_format_selector(meta_duration=300)
    assert f"height<={settings.MAX_VIDEO_HEIGHT}" in normal_fmt
    assert "vcodec^=avc1" in normal_fmt
    assert not normal_fmt.endswith("/best")

    # Over long video threshold: capped to LONG_VIDEO_MAX_HEIGHT (720)
    long_fmt = get_format_selector(meta_duration=900)
    assert f"height<={settings.LONG_VIDEO_MAX_HEIGHT}" in long_fmt
    assert "vcodec!=av01" in long_fmt
    assert not long_fmt.endswith("/best")


def test_validate_downloaded_media_vertical_fixture(tmp_path):
    """Verify vertical 1080x1920 video passes ffprobe validation."""
    out_file = str(tmp_path / "vertical.mp4")
    generate_vertical_1080x1920(out_file, duration=2.0)

    source_info, warnings = validate_downloaded_media(out_file, meta_duration=2.0)
    assert source_info.width == 1080
    assert source_info.height == 1920
    assert source_info.has_audio is True
    assert source_info.duration >= 1.9


def test_validate_downloaded_media_square_fixture(tmp_path):
    """Verify square 1080x1080 video passes ffprobe validation."""
    out_file = str(tmp_path / "square.mp4")
    generate_square(out_file, duration=2.0)

    source_info, warnings = validate_downloaded_media(out_file, meta_duration=2.0)
    assert source_info.width == 1080
    assert source_info.height == 1080
    assert source_info.has_audio is True


def test_validate_downloaded_media_vfr_fixture(tmp_path):
    """Verify variable frame rate video flags is_vfr and emits warning."""
    out_file = str(tmp_path / "vfr.mp4")
    generate_vfr(out_file, duration=2.0)

    # Note: generate_vfr generates no audio by default, so we test no-audio error first
    with pytest.raises(PipelineError) as exc_info:
        validate_downloaded_media(out_file, meta_duration=2.0)
    assert exc_info.value.code == PipelineErrorCode.NO_SPEECH


def test_validate_downloaded_media_no_audio_fixture(tmp_path):
    """Verify video without audio stream fails validation with NO_SPEECH."""
    out_file = str(tmp_path / "no_audio.mp4")
    generate_no_audio(out_file, duration=2.0)

    with pytest.raises(PipelineError) as exc_info:
        validate_downloaded_media(out_file)
    assert exc_info.value.code == PipelineErrorCode.NO_SPEECH


def test_validate_downloaded_media_corrupt_files(tmp_path):
    """Verify 0-byte and truncated files fail validation."""
    # 0-byte file
    empty_file = str(tmp_path / "empty.mp4")
    Path(empty_file).touch()

    with pytest.raises(PipelineError) as exc_info:
        validate_downloaded_media(empty_file)
    assert exc_info.value.code == PipelineErrorCode.DOWNLOAD_FAILED

    # Corrupt / random binary file
    corrupt_file = str(tmp_path / "corrupt.mp4")
    with open(corrupt_file, "wb") as f:
        f.write(b"not a valid video stream header data 1234567890")

    with pytest.raises(PipelineError) as exc_info:
        validate_downloaded_media(corrupt_file)
    assert exc_info.value.code == PipelineErrorCode.UNSUPPORTED_FORMAT


def test_validate_downloaded_media_duration_tolerance(tmp_path):
    """Verify download whose duration severely diverges from metadata fails."""
    out_file = str(tmp_path / "short.mp4")
    generate_square(out_file, duration=2.0)

    # Claimed metadata was 60.0s, actual was 2.0s
    with pytest.raises(PipelineError) as exc_info:
        validate_downloaded_media(out_file, meta_duration=60.0)

    assert exc_info.value.code == PipelineErrorCode.DOWNLOAD_FAILED
    assert "diverges from metadata" in exc_info.value.user_message


def test_validate_downloaded_media_enforces_max_duration(tmp_path):
    """Verify ffprobe duration exceeding MAX_VIDEO_SECONDS is rejected with VIDEO_TOO_LONG."""
    out_file = str(tmp_path / "valid.mp4")
    generate_square(out_file, duration=2.0)

    with patch("app.config.settings.MAX_VIDEO_SECONDS", 1):
        with pytest.raises(PipelineError) as exc_info:
            validate_downloaded_media(out_file)
        assert exc_info.value.code == PipelineErrorCode.VIDEO_TOO_LONG


def test_worker_no_silent_hashed_id_on_pipeline_error(fake_redis):
    """Verify worker does not silently swallow permanent metadata PipelineError."""
    job_id = "test-perm-err-job"
    fake_redis.set(
        f"job:{job_id}",
        JobStatusResponse(job_id=job_id, stage="queued", status="queued", progress=0).model_dump_json(),
    )

    with patch("app.worker.get_video_metadata", side_effect=PipelineError(
        PipelineErrorCode.VIDEO_UNAVAILABLE, stage="downloading", user_message="This video is private"
    )):
        with pytest.raises(PipelineError) as exc_info:
            process_video_task(job_id, {"url": "https://youtu.be/private123"})

        assert exc_info.value.code == PipelineErrorCode.VIDEO_UNAVAILABLE

    # Verify status in Redis is marked failed immediately
    data = JobStatusResponse.model_validate_json(fake_redis.get(f"job:{job_id}"))
    assert data.status == "failed"
    assert data.error.code == PipelineErrorCode.VIDEO_UNAVAILABLE.value


def test_worker_missing_duration_recorded_warning(fake_redis):
    """Verify worker records warning when metadata lacks duration and passes meta_duration=None."""
    job_id = "test-no-dur-job"
    fake_redis.set(
        f"job:{job_id}",
        JobStatusResponse(job_id=job_id, stage="queued", status="queued", progress=0).model_dump_json(),
    )

    fake_source_info = SourceInfo(
        width=1080, height=1920, duration=10.0, has_audio=True
    )

    with patch("app.worker.get_video_metadata", return_value={"video_id": "test:nodur", "duration": None, "title": "T"}), \
         patch("app.worker.download_media", return_value=("/tmp/v.mp4", "/tmp/a.wav", fake_source_info, [])) as mock_dl, \
         patch("app.worker.transcribe_audio", return_value=[]), \
         patch("app.worker.find_viral_clips", return_value=[]):
        process_video_task(job_id, {"url": "https://youtu.be/nodur"})

        mock_dl.assert_called_once()
        _, kwargs = mock_dl.call_args
        assert kwargs.get("meta_duration") is None

    data = JobStatusResponse.model_validate_json(fake_redis.get(f"job:{job_id}"))
    assert any(w.code == "MISSING_METADATA_DURATION" for w in data.warnings)
