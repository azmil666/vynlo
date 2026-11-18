from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    REDIS_URL: str = "redis://redis:6379/0"
    # LLM Provider configuration
    LLM_PROVIDER: str = "gemini"  # "gemini" or "openai"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gemini-2.5-flash"
    GEMINI_MODEL: str = "gemini-2.5-flash"
    MIN_CLIP_DURATION_SECONDS: float = 5.0
    MAX_CLIP_DURATION_SECONDS: float = 60.0
    MIN_HASHTAGS_COUNT: int = 2
    MAX_HASHTAGS_COUNT: int = 8

    # LLM Analysis Windowing, Ranking & Superset Caching
    ANALYSIS_USE_WINDOWING: bool = False
    WINDOWING_THRESHOLD_SECONDS: float = 900.0
    WINDOW_SIZE_SECONDS: float = 600.0
    WINDOW_OVERLAP_MARGIN_SECONDS: float = 15.0
    WINDOW_OVERLAP_SECONDS: float = 75.0
    ANALYSIS_SUPERSET_SIZE: int = 10
    MIN_VIRALITY_SCORE: int = 60
    DEDUPE_IOU_THRESHOLD: float = 0.3
    DEDUPE_MIN_OVERLAP_RATIO: float = 0.5
    DEDUPE_TEXT_SIMILARITY_THRESHOLD: float = 0.7
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

    # Video format selection and validation thresholds
    MAX_VIDEO_HEIGHT: int = 1080
    LONG_VIDEO_THRESHOLD_SECONDS: int = 600
    LONG_VIDEO_MAX_HEIGHT: int = 720
    DURATION_TOLERANCE_SECONDS: float = 15.0

    # Faster-Whisper ASR configuration
    WHISPER_MODEL_SIZE: str = "base"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    WHISPER_LANGUAGE: Optional[str] = None  # None = auto-detect
    WHISPER_BEAM_SIZE: int = 5
    WHISPER_VAD_FILTER: bool = True
    WHISPER_VAD_MIN_SILENCE_DURATION_MS: int = 500
    WHISPER_VAD_SPEECH_PAD_MS: int = 400
    WHISPER_NO_SPEECH_THRESHOLD: float = 0.6
    WHISPER_COMPRESSION_RATIO_THRESHOLD: float = 2.4
    WHISPER_MIN_SPEECH_COVERAGE: float = 0.04  # 4% minimum speech coverage
    WHISPER_MIN_WORD_COUNT: int = 5

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
