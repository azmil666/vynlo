from datetime import datetime, timezone
from enum import Enum
import traceback
from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from app.config import settings

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
    EXPIRED = "EXPIRED"
    INTERNAL = "INTERNAL"


class JobErrorDetail(BaseModel):
    """Structured error details for pipeline failures."""
    code: PipelineErrorCode = Field(description="Machine-readable pipeline error code.")
    message: str = Field(description="Human-readable explanation of the error.")
    retryable: bool = Field(default=False, description="Whether the job can be safely retried.")
    stage: Optional[str] = Field(default=None, description="Pipeline stage where the failure occurred.")
    detail: Optional[str] = Field(default=None, description="Technical diagnostic detail or stack trace summary.")

    def __str__(self) -> str:
        return self.message


class SourceInfo(BaseModel):
    """Source video media metadata extracted from ffprobe."""
    width: int
    height: int
    duration: float
    has_audio: bool = True
    rotation: int = 0
    sar: str = "1:1"
    avg_fps: float = 30.0
    is_vfr: bool = False
    timeline_offset: float = 0.0


class TranscriptValidation(BaseModel):
    """Quality and validation metrics computed for a generated transcript."""
    word_count: int
    speech_duration: float
    speech_coverage_ratio: float
    words_per_minute: float
    language: str = "en"
    language_probability: float = 1.0
    model: str = "base"


class PipelineError(ValueError):
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
            stage=self.stage,
            detail=self.detail,
        )


class JobWarning(BaseModel):
    """Non-fatal warning or degradation notice recorded during processing."""
    code: str = Field(description="Warning code identifier.")
    message: str = Field(description="Descriptive explanation of the non-fatal issue.")
    clip_index: Optional[int] = Field(default=None, description="Zero-based index of affected clip, or None if job-level.")


class WebcamRegion(BaseModel):
    """Normalized bounding box coordinates (0.0 to 1.0) defining a webcam region."""
    x: float = Field(..., ge=0.0, le=1.0, description="Left coordinate (0.0 to 1.0).")
    y: float = Field(..., ge=0.0, le=1.0, description="Top coordinate (0.0 to 1.0).")
    w: float = Field(..., gt=0.0, le=1.0, description="Width (0.0 to 1.0).")
    h: float = Field(..., gt=0.0, le=1.0, description="Height (0.0 to 1.0).")


class JobCreateRequest(BaseModel):
    url: HttpUrl = Field(description="Public media URL (e.g. YouTube, Vimeo, Twitch).")
    layout_mode: Literal["speaker", "split"] = Field(default="speaker", description="Target video framing: speaker or split layout.")
    webcam_region: Optional[WebcamRegion] = Field(default=None, description="Optional bounding box for webcam region in split layout.")
    max_clips: int = Field(default=3, ge=1, le=10, description="Maximum number of clips to extract (1 to 10).")
    caption_preset: Literal["clean_white", "bold_yellow", "boxed", "minimal"] = Field(default="clean_white", description="Caption styling preset.")
    platform_safe_zone: Literal["shorts", "reels", "tiktok"] = Field(default="shorts", description="Platform-specific safe margin layout.")
    highlight_color: str = Field(
        default="&H00FFFFFF",
        pattern=r"^&H[0-9A-Fa-f]{8}$",
        description="ASS hex color for active karaoke word highlighting (&HAABBGGRR).",
    )
    show_hook_banner: bool = Field(default=True, description="Whether to show the top hook title banner.")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: HttpUrl) -> HttpUrl:
        host = (v.host or "").lower()
        if host not in ALLOWED_VIDEO_DOMAINS:
            raise ValueError("Unsupported video domain")
        return v


class ClipResult(BaseModel):
    """Result and metadata for an individual generated clip."""
    model_config = ConfigDict(
        json_schema_extra=lambda schema: schema.get("required", []).extend(
            [f for f in ["video_url"] if f not in schema.get("required", [])]
        )
    )

    clip_id: str = Field(description="Unique identifier for the clip.")
    title: str = Field(description="Catchy title or hook generated for the clip.")
    virality_score: int = Field(description="Predicted virality score from 0 to 100.")
    duration: float = Field(description="Duration of the clip in seconds.")
    video_url: str = Field(default="", description="Public or presigned URL to access the rendered video.")
    selection_method: Literal["llm", "heuristic"] = Field(default="llm", description="Method used to select candidate segment: 'llm' or fallback 'heuristic'.")
    selection_reason: Optional[str] = Field(default=None, description="Explanation or reasoning for segment selection.")
    is_fallback: bool = Field(default=False, description="Deprecated alias for selection_method == 'heuristic'. Kept for backwards compatibility.")
    description: str = Field(default="", description="Social media post description / caption.")
    hashtags: list[str] = Field(default_factory=list, description="Relevant hashtags for discovery.")
    layout_used: Literal["speaker", "split", "fit_blur"] = Field(default="speaker", description="Layout style applied during vertical rendering.")
    detection_coverage: Optional[float] = Field(default=None, description="Fraction of frames where target subject/face was successfully tracked.")
    coverage: Optional[float] = Field(default=None, description="Alias for detection_coverage.")
    detector: str = Field(default="yunet", description="Face / subject detector used during visual tracking.")
    caption_font: str = Field(default="Liberation Sans", description="Font used for burned-in ASS captions.")
    render_ok: bool = Field(default=True, description="Whether rendering completed successfully.")
    ok: bool = Field(default=True, description="Whether clip rendering and upload succeeded without error.")
    error_code: Optional[str] = Field(default=None, description="Machine-readable error code if clip failed.")
    error_message: Optional[str] = Field(default=None, description="Human-readable error message if clip failed.")
    warnings: list[str | JobWarning] = Field(default_factory=list, description="Non-fatal warnings encountered during clip processing.")

    @model_validator(mode="before")
    @classmethod
    def sync_clip_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Sync ok and render_ok
            if "ok" in data and "render_ok" not in data:
                data["render_ok"] = bool(data["ok"])
            elif "render_ok" in data and "ok" not in data:
                data["ok"] = bool(data["render_ok"])
            elif "ok" in data and "render_ok" in data:
                data["ok"] = bool(data["ok"])
                data["render_ok"] = bool(data["render_ok"])

            if "selection_method" in data and "is_fallback" not in data:
                data["is_fallback"] = (data["selection_method"] == "heuristic")
            elif "is_fallback" in data and "selection_method" not in data:
                data["selection_method"] = "heuristic" if data["is_fallback"] else "llm"
            elif "selection_method" in data:
                data["is_fallback"] = (data["selection_method"] == "heuristic")
            # Sync coverage and detection_coverage
            if "detection_coverage" in data and "coverage" not in data:
                data["coverage"] = data["detection_coverage"]
            elif "coverage" in data and "detection_coverage" not in data:
                data["detection_coverage"] = data["coverage"]
        return data


PipelineStage = Literal[
    "queued",
    "downloading",
    "transcribing",
    "analyzing",
    "rendering",
    "completed",
    "completed_with_warnings",
    "partial",
    "failed",
    "expired",
]


class JobStatusResponse(BaseModel):
    """Job status response representation.

    Status String Mapping & Semantics:
    - "completed": All candidate clips rendered and uploaded successfully (all ok).
    - "completed_with_warnings": Partial success; at least one clip rendered/uploaded successfully,
      but one or more candidate clips failed (some ok). Failed clips have ok=False, error_code,
      and error_message, while successful clips are delivered.
    - "partial": Recognized alias for "completed_with_warnings".
    - "failed": All candidate clips failed (none ok), or an unrecoverable failure occurred before rendering.
    - "expired": The job or its storage retention window has elapsed.
    """
    model_config = ConfigDict(
        json_schema_extra=lambda schema: schema.get("required", []).extend(
            [f for f in ["status", "progress", "clips"] if f not in schema.get("required", [])]
        )
    )

    job_id: str = Field(description="Unique identifier for the job.")
    status: PipelineStage = Field(default="queued", description="High-level lifecycle status of the job.")
    stage: PipelineStage = Field(default="queued", description="Current execution stage of the pipeline.")
    progress: int = Field(default=0, ge=0, le=100, description="Overall pipeline completion percentage (0-100).")
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of the last status update."
    )
    error: Optional[JobErrorDetail] = Field(default=None, description="Detailed error information if the job failed.")
    warnings: list[JobWarning] = Field(default_factory=list, description="Non-fatal warnings recorded during pipeline execution.")
    clips: list[ClipResult] = Field(default_factory=list, description="List of generated clip results and their delivery metadata.")
    is_partial: bool = Field(default=False, description="Flag indicating if the job partially succeeded with warnings.")

    @model_validator(mode="before")
    @classmethod
    def sync_partial_flag(cls, data: Any) -> Any:
        if isinstance(data, dict):
            status = data.get("status")
            if status in ("completed_with_warnings", "partial"):
                data["is_partial"] = True
        return data

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
            user_message=f"Video exceeds maximum supported duration of {settings.MAX_VIDEO_SECONDS}s.",
            retryable=False,
            detail=msg,
        )

    if (
        "unsupported video domain" in msg_lower
        or "invalid url" in msg_lower
        or ("invalid" in msg_lower and "url" in msg_lower)
    ):
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

    # 4. Downloads & network errors
    if (
        isinstance(exc, (ConnectionError, TimeoutError))
        or "download" in msg_lower
        or "failed to download" in msg_lower
        or any(w in msg_lower for w in ["connection", "network error", "network disconnect", "http error 5", "failed to resolve", "nodename nor servname"])
        or (isinstance(exc, FileNotFoundError) and "source.mp4" in msg)
    ):
        return PipelineError(
            code=PipelineErrorCode.DOWNLOAD_FAILED,
            stage=current_stage,
            user_message="Failed to download media due to a network or connection error.",
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

