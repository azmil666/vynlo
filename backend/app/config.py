from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    REDIS_URL: str = "redis://redis:6379/0"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gemini-3.8-flash"
    GEMINI_MODEL: str = "gemini-3.8-flash"
    TEMP_DIR: str = "/tmp/clipper"
    STATIC_DIR: str = "/app/static"
    PUBLIC_BASE_URL: str = "http://localhost:8000"
    S3_ENDPOINT_URL: Optional[str] = None
    S3_ACCESS_KEY: Optional[str] = None
    S3_SECRET_KEY: Optional[str] = None
    S3_BUCKET_NAME: Optional[str] = None
    S3_PUBLIC_BASE_URL: Optional[str] = None
    YTDLP_COOKIES_PATH: Optional[str] = None

    # Video length limits and stage time budgets
    MAX_VIDEO_SECONDS: int = 1200  # 20 minutes default
    # Conservative throughput ratio: 0.5s processing per 1.0s audio on CPU.
    # TODO: calibrate in P8.1
    ASR_PROCESSING_RATIO: float = 0.5
    BUDGET_DOWNLOAD_SECONDS: int = 180
    BUDGET_ANALYSIS_SECONDS: int = 90
    BUDGET_RENDER_PER_CLIP_SECONDS: int = 60
    MAX_SUPPORTED_CLIPS: int = 10
    BUDGET_BUFFER_SECONDS: int = 120
    WORKER_HEARTBEAT_TIMEOUT_SECONDS: int = 300
    TEMP_DIR_CLEANUP_HOURS: float = 4.0

    @property
    def task_soft_time_limit(self) -> int:
        """Derived task soft time limit based on stage budgets and video length.
        Math: download + (video_duration * ASR_ratio) + analysis + (max_clips * render_budget) + buffer.
        """
        asr_budget = int(self.MAX_VIDEO_SECONDS * self.ASR_PROCESSING_RATIO)
        render_budget = self.MAX_SUPPORTED_CLIPS * self.BUDGET_RENDER_PER_CLIP_SECONDS
        return (
            self.BUDGET_DOWNLOAD_SECONDS
            + asr_budget
            + self.BUDGET_ANALYSIS_SECONDS
            + render_budget
            + self.BUDGET_BUFFER_SECONDS
        )

    @property
    def task_time_limit(self) -> int:
        """Derived task hard limit: soft limit + 60s grace period for cleanup."""
        return self.task_soft_time_limit + 60

    @property
    def broker_visibility_timeout(self) -> int:
        """Redis broker visibility timeout. Must be strictly greater than task_time_limit."""
        return max(3600, self.task_time_limit + 300)


settings = Settings()
