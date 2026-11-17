from datetime import datetime
import json
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import (
    ClipResult,
    JobErrorDetail,
    JobStatusResponse,
    JobWarning,
    PipelineError,
    PipelineErrorCode,
    map_exception_to_pipeline_error,
)


class TestPipelineErrorAndMapping:
    """Tests for PipelineError, error codes, and exception mapping."""

    def test_pipeline_error_instantiation_and_conversion(self):
        err = PipelineError(
            code=PipelineErrorCode.VIDEO_TOO_LONG,
            stage="downloading",
            user_message="Video exceeds 4 hours limit.",
            retryable=False,
            detail="Duration was 15000s.",
        )
        assert err.code == PipelineErrorCode.VIDEO_TOO_LONG
        assert err.stage == "downloading"
        assert err.retryable is False
        assert err.user_message == "Video exceeds 4 hours limit."
        assert str(err) == "Video exceeds 4 hours limit."

        detail = err.to_error_detail()
        assert isinstance(detail, JobErrorDetail)
        assert detail.code == PipelineErrorCode.VIDEO_TOO_LONG
        assert detail.message == "Video exceeds 4 hours limit."
        assert detail.retryable is False
        assert detail.detail == "Duration was 15000s."

    def test_pipeline_error_string_code_conversion(self):
        err = PipelineError(
            code="DOWNLOAD_FAILED",
            stage="downloading",
            user_message="Download error",
            retryable=True,
        )
        assert err.code == PipelineErrorCode.DOWNLOAD_FAILED

        # Unknown code maps to INTERNAL
        err_unknown = PipelineError(
            code="UNKNOWN_CUSTOM_CODE",
            stage="queued",
            user_message="Mystery",
        )
        assert err_unknown.code == PipelineErrorCode.INTERNAL

    def test_mapping_preserves_existing_pipeline_error(self):
        original = PipelineError(
            code=PipelineErrorCode.NO_QUALIFYING_CLIPS,
            stage="analyzing",
            user_message="No clips met score criteria.",
            retryable=False,
        )
        mapped = map_exception_to_pipeline_error(original, current_stage="analyzing")
        assert mapped is original

    @pytest.mark.parametrize(
        ("exc", "stage", "expected_code", "expected_retryable"),
        [
            (ValueError("Video exceeds maximum supported duration of 4 hours"), "downloading", PipelineErrorCode.VIDEO_TOO_LONG, False),
            (ValueError("Unsupported video domain"), "queued", PipelineErrorCode.INVALID_URL, False),
            (ValueError("No speech detected in audio"), "transcribing", PipelineErrorCode.NO_SPEECH, False),
            (Exception("ERROR: This live stream is currently ongoing"), "downloading", PipelineErrorCode.LIVE_UNSUPPORTED, False),
            (Exception("ERROR: [youtube] 123: Private video"), "downloading", PipelineErrorCode.VIDEO_UNAVAILABLE, False),
            (Exception("ERROR: [youtube] Video unavailable"), "downloading", PipelineErrorCode.VIDEO_UNAVAILABLE, False),
            (Exception("Download failed: connection reset by peer"), "downloading", PipelineErrorCode.DOWNLOAD_FAILED, True),
            (FileNotFoundError("Downloaded video file not found at /tmp/source.mp4"), "downloading", PipelineErrorCode.DOWNLOAD_FAILED, True),
            (Exception("Whisper model execution failed"), "transcribing", PipelineErrorCode.TRANSCRIPTION_FAILED, True),
            (Exception("Missing Gemini LLM_API_KEY"), "analyzing", PipelineErrorCode.LLM_CONFIG, False),
            (Exception("Gemini generation blocked by safety moderation filters"), "analyzing", PipelineErrorCode.LLM_BLOCKED, False),
            (Exception("LLM 503 service unavailable rate limit 429"), "analyzing", PipelineErrorCode.LLM_UNAVAILABLE, True),
            (Exception("Failed parsing LLM segment JSON analysis"), "analyzing", PipelineErrorCode.ANALYSIS_INVALID, True),
            (Exception("Failed to read valid video dimensions from /tmp/corrupted.mp4"), "rendering", PipelineErrorCode.UNSUPPORTED_FORMAT, False),
            (Exception("FFmpeg command failed with exit status 1"), "rendering", PipelineErrorCode.RENDER_FAILED, True),
            (Exception("boto3 upload error: Access Denied to s3"), "rendering", PipelineErrorCode.UPLOAD_FAILED, True),
            (Exception("celery.exceptions.TimeLimitExceeded: task timeout"), "rendering", PipelineErrorCode.TIMEOUT, False),
        ],
    )
    def test_map_exception_to_typed_pipeline_errors(
        self, exc: Exception, stage: str, expected_code: PipelineErrorCode, expected_retryable: bool
    ):
        mapped = map_exception_to_pipeline_error(exc, current_stage=stage)
        assert mapped.code == expected_code
        assert mapped.retryable == expected_retryable
        assert mapped.stage == stage
        assert len(mapped.user_message) > 0

    def test_unexpected_exception_maps_to_internal_with_traceback(self):
        exc = ZeroDivisionError("division by zero in custom code")
        mapped = map_exception_to_pipeline_error(exc, current_stage="analyzing")
        assert mapped.code == PipelineErrorCode.INTERNAL
        assert mapped.retryable is False
        assert mapped.stage == "analyzing"
        assert "internal error" in mapped.user_message.lower()
        assert mapped.detail is not None
        assert "ZeroDivisionError" in mapped.detail


class TestJobStatusSerializationAndRoundTrip:
    """Tests serialization round-trip, JSON shape, and backwards compatibility."""

    def test_full_job_status_response_round_trip(self):
        clip = ClipResult(
            clip_id="clip_1",
            title="Epic Win",
            virality_score=94,
            duration=34.2,
            video_url="http://localhost:8000/static/job_1/clip_1.mp4",
            is_fallback=False,
            description="Punchy hook description.",
            hashtags=["gaming", "viral"],
            selection_method="llm",
            layout_used="speaker",
            detector="yunet",
            caption_font="Liberation Sans",
            render_ok=True,
        )
        warning = JobWarning(code="AUDIO_LOW_VOLUME", message="Audio peaked below -24 LUFS", clip_index=1)
        resp = JobStatusResponse(
            job_id="job_full_abc",
            stage="completed",
            status="completed",
            progress=100,
            updated_at="2026-11-17T18:00:00Z",
            error=None,
            warnings=[warning],
            clips=[clip],
        )

        json_str = resp.model_dump_json()
        data = json.loads(json_str)

        assert data["job_id"] == "job_full_abc"
        assert data["stage"] == "completed"
        assert data["status"] == "completed"
        assert data["progress"] == 100
        assert data["updated_at"] == "2026-11-17T18:00:00Z"
        assert data["error"] is None
        assert len(data["warnings"]) == 1
        assert data["warnings"][0]["code"] == "AUDIO_LOW_VOLUME"
        assert data["warnings"][0]["clip_index"] == 1
        assert len(data["clips"]) == 1
        assert data["clips"][0]["selection_method"] == "llm"
        assert data["clips"][0]["layout_used"] == "speaker"
        assert data["clips"][0]["detector"] == "yunet"
        assert data["clips"][0]["caption_font"] == "Liberation Sans"
        assert data["clips"][0]["render_ok"] is True

        # Round-trip deserialization
        parsed = JobStatusResponse.model_validate_json(json_str)
        assert parsed.job_id == resp.job_id
        assert parsed.stage == resp.stage
        assert parsed.status == resp.status
        assert parsed.progress == resp.progress
        assert parsed.clips[0].selection_method == "llm"

    def test_failed_job_status_response_round_trip(self):
        err_detail = JobErrorDetail(
            code=PipelineErrorCode.RENDER_FAILED,
            message="FFmpeg rendering crashed.",
            retryable=True,
            detail="Error in filtergraph: vstack",
        )
        resp = JobStatusResponse(
            job_id="job_failed_123",
            stage="failed",
            status="failed",
            progress=65,
            error=err_detail,
        )

        json_str = resp.model_dump_json()
        data = json.loads(json_str)

        assert data["stage"] == "failed"
        assert data["status"] == "failed"
        assert data["error"]["code"] == "RENDER_FAILED"
        assert data["error"]["message"] == "FFmpeg rendering crashed."
        assert data["error"]["retryable"] is True
        assert data["error"]["detail"] == "Error in filtergraph: vstack"

        parsed = JobStatusResponse.model_validate_json(json_str)
        assert parsed.error is not None
        assert parsed.error.code == PipelineErrorCode.RENDER_FAILED
        assert parsed.error.retryable is True

    def test_backwards_compatible_legacy_string_error(self):
        # Old records stored in Redis had "error": "Some failure string"
        legacy_data = {
            "job_id": "legacy_job_1",
            "status": "failed",
            "progress": 0,
            "error": "Failed to extract metadata for URL",
            "clips": [],
        }
        json_str = json.dumps(legacy_data)
        parsed = JobStatusResponse.model_validate_json(json_str)

        assert parsed.status == "failed"
        assert parsed.stage == "failed"
        assert parsed.error is not None
        assert parsed.error.code == PipelineErrorCode.INTERNAL
        assert parsed.error.message == "Failed to extract metadata for URL"
        assert str(parsed.error) == "Failed to extract metadata for URL"
        assert parsed.warnings == []
        assert parsed.updated_at is not None

    def test_backwards_compatible_legacy_clips_and_fields(self):
        # Old records lacked stage, warnings, updated_at, and new clip fields
        legacy_data = {
            "job_id": "legacy_job_2",
            "status": "rendering",
            "progress": 80,
            "clips": [
                {
                    "clip_id": "clip_old",
                    "title": "Legacy Highlight",
                    "virality_score": 82,
                    "duration": 28.5,
                    "video_url": "http://localhost:8000/static/legacy/clip_1.mp4",
                    "is_fallback": True,
                }
            ],
        }
        json_str = json.dumps(legacy_data)
        parsed = JobStatusResponse.model_validate_json(json_str)

        assert parsed.stage == "rendering"
        assert parsed.status == "rendering"
        assert parsed.progress == 80
        assert parsed.warnings == []
        assert len(parsed.clips) == 1
        clip = parsed.clips[0]
        assert clip.clip_id == "clip_old"
        assert clip.selection_method == "llm"
        assert clip.layout_used == "speaker"
        assert clip.detector == "yunet"
        assert clip.caption_font == "Liberation Sans"
        assert clip.render_ok is True


class TestApiResponseShape:
    """Verifies FastAPI API endpoint response shapes match the structured model."""

    @pytest.fixture
    def client(self, fake_redis):
        return TestClient(app)

    def test_create_job_response_shape(self, client):
        payload = {
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "layout_mode": "speaker",
            "max_clips": 3,
        }
        with patch("app.main.process_video_task.delay") as mock_delay:
            res = client.post("/api/jobs", json=payload)
            assert res.status_code == 202
            mock_delay.assert_called_once()
            data = res.json()

        # Check all required top-level fields
        expected_keys = {"job_id", "status", "stage", "progress", "updated_at", "error", "warnings", "clips"}
        assert expected_keys.issubset(data.keys())
        assert data["status"] == "queued"
        assert data["stage"] == "queued"
        assert data["progress"] == 0
        assert data["error"] is None
        assert data["warnings"] == []
        assert data["clips"] == []

    def test_get_job_status_completed_response_shape(self, client, fake_redis):
        job_id = "test_completed_shape"
        job_data = JobStatusResponse(
            job_id=job_id,
            stage="completed",
            status="completed",
            progress=100,
            clips=[
                ClipResult(
                    clip_id="clip_1",
                    title="Amazing Moment",
                    virality_score=88,
                    duration=30.0,
                    video_url="http://localhost:8000/static/test/clip_1.mp4",
                    is_fallback=False,
                    selection_method="llm",
                    layout_used="speaker",
                    detector="yunet",
                    caption_font="Liberation Sans",
                    render_ok=True,
                )
            ],
            warnings=[JobWarning(code="INFO", message="Test warning", clip_index=1)],
        )
        fake_redis.set(f"job:{job_id}", job_data.model_dump_json())

        res = client.get(f"/api/jobs/{job_id}")
        assert res.status_code == 200
        data = res.json()

        assert data["job_id"] == job_id
        assert data["stage"] == "completed"
        assert data["status"] == "completed"
        assert data["progress"] == 100
        assert len(data["warnings"]) == 1
        assert data["warnings"][0]["code"] == "INFO"
        assert len(data["clips"]) == 1
        clip = data["clips"][0]
        assert clip["selection_method"] == "llm"
        assert clip["layout_used"] == "speaker"
        assert clip["detector"] == "yunet"
        assert clip["caption_font"] == "Liberation Sans"
        assert clip["render_ok"] is True

    def test_get_job_status_failed_response_shape(self, client, fake_redis):
        job_id = "test_failed_shape"
        job_data = JobStatusResponse(
            job_id=job_id,
            stage="failed",
            status="failed",
            progress=40,
            error=JobErrorDetail(
                code=PipelineErrorCode.NO_SPEECH,
                message="No clear speech was detected in the video audio.",
                retryable=False,
                detail="Whisper returned 0 segments",
            ),
        )
        fake_redis.set(f"job:{job_id}", job_data.model_dump_json())

        res = client.get(f"/api/jobs/{job_id}")
        assert res.status_code == 200
        data = res.json()

        assert data["stage"] == "failed"
        assert data["status"] == "failed"
        assert isinstance(data["error"], dict)
        assert data["error"]["code"] == "NO_SPEECH"
        assert data["error"]["message"] == "No clear speech was detected in the video audio."
        assert data["error"]["retryable"] is False
        assert data["error"]["detail"] == "Whisper returned 0 segments"
