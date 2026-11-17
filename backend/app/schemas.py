from typing import Literal, Optional
from pydantic import BaseModel, Field, HttpUrl, field_validator

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


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "downloading", "transcribing", "analyzing", "rendering", "completed", "failed"]
    progress: int = Field(ge=0, le=100)
    error: Optional[str] = None
    clips: list[ClipResult] = Field(default_factory=list)
