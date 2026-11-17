from typing import Literal, Optional
from pydantic import BaseModel, Field, HttpUrl


class JobCreateRequest(BaseModel):
    url: HttpUrl
    layout_mode: Literal["speaker", "split"] = "speaker"
    max_clips: int = Field(default=3, ge=1, le=10)
    highlight_color: str = Field(default="&H0000FFFF")


class ClipResult(BaseModel):
    clip_id: str
    title: str
    virality_score: int
    duration: float
    video_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "downloading", "transcribing", "analyzing", "rendering", "completed", "failed"]
    progress: int = Field(ge=0, le=100)
    error: Optional[str] = None
    clips: list[ClipResult] = Field(default_factory=list)
