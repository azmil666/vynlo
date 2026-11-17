from datetime import datetime, timezone
from enum import Enum
import traceback
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

ALLOWED_VIDEO_DOMAINS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "vimeo.com",
    "www.vimeo.com",
    "twitch.tv",
    "www.twitch.tv",
    "clips.twitch.tv",
}


class PipelineErrorCode(str, Enum):
    INVALID_URL = "INVALID_URL"
    VIDEO_UNAVAILABLE = "VIDEO_UNAVAILABLE"
    VIDEO_TOO_LONG = "VIDEO_TOO_LONG"
    LIVE_UNSUPPORTED = "LIVE_UNSUPPORTED"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    NO_SPEECH = "NO_SPEECH"
    TRANSCRIPTION_FAILED = "TRANSCRIPTION_FAILED"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    LLM_BLOCKED = "LLM_BLOCKED"
    LLM_CONFIG = "LLM_CONFIG"
    ANALYSIS_INVALID = "ANALYSIS_INVALID"
    NO_QUALIFYING_CLIPS = "NO_QUALIFYING_CLIPS"
    RENDER_FAILED = "RENDER_FAILED"
    UPLOAD_FAILED = "UPLOAD_FAILED"
    TIMEOUT = "TIMEOUT"
    WORKER_LOST = "WORKER_LOST"
    INTERNAL = "INTERNAL"


class JobErrorDetail(BaseModel):
    code: PipelineErrorCode
    message: str
    retryable: bool = False
    detail: Optional[str] = None

    def __str__(self) -> str:
        return self.message


class PipelineError(Exception):
    """Structured pipeline failure exception holding code, stage, retryable flag, user message, and detail."""

    def __init__(
        self,
        code: PipelineErrorCode | str,
        stage: str,
        user_message: str,
        retryable: bool = False,
        detail: Optional[str] = None,
    ):
        super().__init__(user_message)
        if isinstance(code, str):
            try:
                code = PipelineErrorCode(code)
            except ValueError:
                code = PipelineErrorCode.INTERNAL
        self.code = code
        self.stage = stage
        self.retryable = retryable
        self.user_message = user_message
        self.detail = detail

    def to_error_detail(self) -> JobErrorDetail:
        return JobErrorDetail(
            code=self.code,
            message=self.user_message,
            retryable=self.retryable,
            detail=self.detail,
        )


class JobWarning(BaseModel):
    code: str
    message: str
    clip_index: Optional[int] = None


class JobCreateRequest(BaseModel):
    url: HttpUrl
    layout_mode: Literal["speaker", "split"] = "speaker"
    max_clips: int = Field(default=3, ge=1, le=10)
    caption_preset: Literal["clean_white", "bold_yellow", "boxed", "minimal"] = "clean_white"
    platform_safe_zone: Literal["shorts", "reels", "tiktok"] = "shorts"
    highlight_color: str = Field(
        default="&H00FFFFFF",
        pattern=r"^&H[0-9A-Fa-f]{8}$",
    )
    show_hook_banner: bool = True

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: HttpUrl) -> HttpUrl:
        host = (v.host or "").lower()
        if host not in ALLOWED_VIDEO_DOMAINS:
            raise ValueError("Unsupported video domain")
        return v


class ClipResult(BaseModel):
    clip_id: str
    title: str
    virality_score: int
    duration: float
    video_url: str
    is_fallback: bool = False
    description: str = ""
    hashtags: list[str] = Field(default_factory=list)
    selection_method: Literal["llm", "heuristic"] = "llm"
    layout_used: Literal["speaker", "split"] = "speaker"
    detector: str = "yunet"
    caption_font: str = "Liberation Sans"
    render_ok: bool = True


PipelineStage = Literal[
    "queued", "downloading", "transcribing", "analyzing", "rendering", "completed", "failed"
]


class JobStatusResponse(BaseModel):
    job_id: str
    status: PipelineStage = "queued"
    stage: PipelineStage = "queued"
    progress: int = Field(default=0, ge=0, le=100)
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    error: Optional[JobErrorDetail] = None
    warnings: list[JobWarning] = Field(default_factory=list)
    clips: list[ClipResult] = Field(default_factory=list)

    @field_validator("error", mode="before")
    @classmethod
    def validate_error_field(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str):
            return JobErrorDetail(
                code=PipelineErrorCode.INTERNAL,
                message=v,
                retryable=False,
                detail=None,
            )
        if isinstance(v, PipelineError):
            return v.to_error_detail()
        return v

    @model_validator(mode="before")
    @classmethod
    def sync_stage_and_status(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Sync stage and status for backwards compatibility
            if "stage" in data and "status" not in data:
                data["status"] = data["stage"]
            elif "status" in data and "stage" not in data:
                data["stage"] = data["status"]
            # Map legacy raw string error if present
            raw_err = data.get("error")
            if isinstance(raw_err, str) and raw_err:
                data["error"] = {
                    "code": PipelineErrorCode.INTERNAL.value,
                    "message": raw_err,
                    "retryable": False,
                    "detail": None,
                }
        return data


def map_exception_to_pipeline_error(
    exc: Exception, current_stage: str = "failed"
) -> PipelineError:
    """Map any exception to a typed PipelineError, ensuring tracebacks are logged and never swallowed."""
    if isinstance(exc, PipelineError):
        return exc

    msg = str(exc)
    msg_lower = msg.lower()
    tb = traceback.format_exc()

    # 1. Timeouts (highest priority since they can occur during any stage)
    if "timelimitexceeded" in exc.__class__.__name__.lower() or "timeout" in msg_lower:
        return PipelineError(
            code=PipelineErrorCode.TIMEOUT,
            stage=current_stage,
            user_message="The video processing task timed out before completion.",
            retryable=False,
            detail=msg,
        )

    # 2. URL and metadata duration limits
    if "maximum supported duration" in msg_lower or "too long" in msg_lower:
        return PipelineError(
            code=PipelineErrorCode.VIDEO_TOO_LONG,
            stage=current_stage,
            user_message="Video exceeds maximum supported duration of 4 hours.",
            retryable=False,
            detail=msg,
        )

    if "unsupported video domain" in msg_lower or "invalid url" in msg_lower:
        return PipelineError(
            code=PipelineErrorCode.INVALID_URL,
            stage=current_stage,
            user_message="The video domain or URL format is unsupported.",
            retryable=False,
            detail=msg,
        )

    # 3. Live streams
    if "live stream" in msg_lower or "is a live" in msg_lower:
        return PipelineError(
            code=PipelineErrorCode.LIVE_UNSUPPORTED,
            stage=current_stage,
            user_message="Live streams are not supported until the broadcast has ended.",
            retryable=False,
            detail=msg,
        )

    # 4. Downloads
    if (
        "download" in msg_lower
        or "failed to download" in msg_lower
        or (isinstance(exc, FileNotFoundError) and "source.mp4" in msg)
    ):
        return PipelineError(
            code=PipelineErrorCode.DOWNLOAD_FAILED,
            stage=current_stage,
            user_message="Failed to download media from the provided URL.",
            retryable=True,
            detail=msg,
        )

    # 5. Video unavailable
    if (
        "video unavailable" in msg_lower
        or "private video" in msg_lower
        or "has been removed" in msg_lower
        or ("not found" in msg_lower and "video" in msg_lower)
    ):
        return PipelineError(
            code=PipelineErrorCode.VIDEO_UNAVAILABLE,
            stage=current_stage,
            user_message="The requested video is unavailable, private, or has been removed.",
            retryable=False,
            detail=msg,
        )

    # 6. Speech & transcription errors
    if "no speech detected" in msg_lower:
        return PipelineError(
            code=PipelineErrorCode.NO_SPEECH,
            stage=current_stage,
            user_message="No clear speech was detected in the video audio.",
            retryable=False,
            detail=msg,
        )

    if "whisper" in msg_lower or current_stage == "transcribing":
        return PipelineError(
            code=PipelineErrorCode.TRANSCRIPTION_FAILED,
            stage=current_stage,
            user_message="Audio transcription failed.",
            retryable=True,
            detail=msg,
        )

    # 7. LLM & analysis errors
    if "api key" in msg_lower or "llm_api_key" in msg_lower:
        return PipelineError(
            code=PipelineErrorCode.LLM_CONFIG,
            stage=current_stage,
            user_message="AI analysis is not configured properly.",
            retryable=False,
            detail=msg,
        )

    if "safety" in msg_lower or "blocked" in msg_lower or "moderation" in msg_lower:
        return PipelineError(
            code=PipelineErrorCode.LLM_BLOCKED,
            stage=current_stage,
            user_message="Content analysis was blocked by safety filters.",
            retryable=False,
            detail=msg,
        )

    if "llm" in msg_lower and any(k in msg_lower for k in ["unavailable", "503", "rate", "429"]):
        return PipelineError(
            code=PipelineErrorCode.LLM_UNAVAILABLE,
            stage=current_stage,
            user_message="AI analysis service is temporarily unavailable.",
            retryable=True,
            detail=msg,
        )

    if "no qualifying clips" in msg_lower:
        return PipelineError(
            code=PipelineErrorCode.NO_QUALIFYING_CLIPS,
            stage=current_stage,
            user_message="No qualifying viral clips could be identified for this video.",
            retryable=False,
            detail=msg,
        )

    if "analysis" in msg_lower and "json" in msg_lower:
        return PipelineError(
            code=PipelineErrorCode.ANALYSIS_INVALID,
            stage=current_stage,
            user_message="Failed to parse viral clip analysis response.",
            retryable=True,
            detail=msg,
        )

    # 8. Upload errors (check before general rendering errors)
    if "upload" in msg_lower or "s3" in msg_lower or "boto" in msg_lower:
        return PipelineError(
            code=PipelineErrorCode.UPLOAD_FAILED,
            stage=current_stage,
            user_message="Failed to upload rendered clips to storage.",
            retryable=True,
            detail=msg,
        )

    # 9. Format and rendering errors
    if (
        "unsupported format" in msg_lower
        or "failed to open video file" in msg_lower
        or "failed to read valid video dimensions" in msg_lower
    ):
        return PipelineError(
            code=PipelineErrorCode.UNSUPPORTED_FORMAT,
            stage=current_stage,
            user_message="The video file format is unsupported or corrupt.",
            retryable=False,
            detail=msg,
        )

    if current_stage == "rendering" or "ffmpeg" in msg_lower:
        return PipelineError(
            code=PipelineErrorCode.RENDER_FAILED,
            stage=current_stage,
            user_message="Failed to crop and render video clips.",
            retryable=True,
            detail=msg,
        )

    # 10. Unhandled unexpected exception -> INTERNAL
    return PipelineError(
        code=PipelineErrorCode.INTERNAL,
        stage=current_stage,
        user_message="An unexpected internal error occurred during video processing.",
        retryable=False,
        detail=f"{type(exc).__name__}: {msg}\n{tb}",
    )

