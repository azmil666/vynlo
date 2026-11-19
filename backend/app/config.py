from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    REDIS_URL: str = "redis://redis:6379/0"
    CELERY_BROKER_URL: Optional[str] = None
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
    HEURISTIC_QUALITY_FLOOR: int = 65

    # Boundary Snapping & Lead-in/Tail Refinement
    SNAP_LEAD_IN_SECONDS: float = 0.20
    SNAP_TAIL_SECONDS: float = 0.35
    SNAP_PAUSE_THRESHOLD: float = 0.40
    SNAP_INITIAL_WINDOW_SECONDS: float = 1.5
    SNAP_WIDENED_WINDOW_SECONDS: float = 3.0

    # YuNet Neural Face Detection Configuration
    YUNET_MODEL_PATH: str = "/app/models/face_detection_yunet_2023mar.onnx"
    FACE_DETECTION_LONG_EDGE: int = 640
    FACE_DETECTION_SAMPLE_HZ: float = 4.0
    FACE_DETECTION_SCORE_THRESHOLD: float = 0.55
    FACE_DETECTION_NMS_THRESHOLD: float = 0.3
    FACE_DETECTION_TOP_K: int = 10

    # Face Reframing, Coverage & Layout Fallback Configuration
    FACE_COVERAGE_THRESHOLD: float = 0.40
    FACE_MAX_GAP_FILL_SECONDS: float = 1.5
    FACE_EASE_DURATION_SECONDS: float = 1.0
    FACE_NEUTRAL_POSITION: float = 0.5

    # Dynamic Camera Motion, Dead Zone & Keyframe Simplification
    CROP_DEAD_ZONE_RATIO: float = 0.15
    CROP_MAX_PAN_SPEED: float = 300.0
    CROP_EASE_OMEGA: float = 3.0
    CROP_SIMPLIFY_EPSILON: float = 2.0
    CROP_MAX_KEYFRAMES: int = 40

    # Active Speaker Tracking & Selection
    SPEAKER_TRACKING_ENABLED: bool = True
    SPEAKER_MIN_DWELL_SECONDS: float = 2.0
    SPEAKER_SWITCH_MARGIN: float = 1.3
    SPEAKER_MOUTH_WINDOW_SECONDS: float = 1.0
    SPEAKER_MOUTH_NOISE_FLOOR: float = 1e-4
    SPEAKER_DUAL_FRAME_MAX_RATIO: float = 0.85
    SPEAKER_IOU_THRESHOLD: float = 0.25
    SPEAKER_MAX_TRACK_GAP_SECONDS: float = 1.5
    SPEAKER_MIN_FACE_SIZE_PX: int = 35

    # Cache Configuration, Schema Versions & Guard Thresholds
    GLOBAL_CACHE_VERSION: str = "v1"
    TRANSCRIPT_SCHEMA_VERSION: str = "v1"
    CANDIDATE_SCHEMA_VERSION: str = "v1"
    PROMPT_VERSION: str = "v1"
    # Retention & Expiry Windows (Coherent Lifetime Management)
    # 1. Job Record TTL: Redis TTL for job status state. Default 24 hours (86,400s).
    #    User return window: users can refresh or check their job status for at least 24h.
    JOB_RECORD_TTL_SECONDS: int = 86400  # 24 hours

    # 2. Local Static Clip Retention: Time before local static clips and orphaned dirs are swept.
    #    Relationship: Clip files MUST live at least as long as the job record (>= JOB_RECORD_TTL_SECONDS).
    #    Default 48 hours (172,800s).
    STATIC_RETENTION_HOURS: float = 48.0  # 48 hours

    # 3. Temp Directory Retention: Scratch files in /tmp/clipper for active jobs.
    #    Relationship: Must live at least as long as the task time limit / heartbeat timeout
    #    so active jobs are not swept mid-render, but shorter than job/clip retention.
    #    Default 4 hours.
    TEMP_DIR_CLEANUP_HOURS: float = 4.0

    # 4. Independent Caches: Raw transcripts and candidate LLM extractions can be reused
    #    across different job runs for the same video URL.
    #    Independent of job record TTL and clip file retention. Default 7 days.
    CACHE_TRANSCRIPT_TTL_SECONDS: int = 604800  # 7 days
    CACHE_CANDIDATES_TTL_SECONDS: int = 604800  # 7 days

    # 5. S3 Retention & Presigned URL Expiry:
    #    S3 presigned URLs and lifecycle must cover the job record TTL.
    #    Default 7 days (604,800s).
    S3_PRESIGNED_EXPIRY_SECONDS: int = 604800  # 7 days
    S3_RETENTION_DAYS: int = 7  # 7 days
    CACHE_DURATION_TOLERANCE_SECONDS: float = 2.0  # Acceptable tolerance for duration mismatch
    CACHE_COMPRESSION_ENABLED: bool = True
    SINGLE_FLIGHT_LOCK_TTL_SECONDS: int = 300  # 5 minutes lock expiry
    SINGLE_FLIGHT_WAIT_TIMEOUT_SECONDS: float = 120.0  # Max wait time for concurrent worker
    SINGLE_FLIGHT_POLL_INTERVAL_SECONDS: float = 0.5  # Poll interval for waiting on single-flight lock
    TEMP_DIR: str = "/tmp/clipper"
    STATIC_DIR: str = "/app/static"
    PUBLIC_BASE_URL: str = "http://localhost:8000"
    STORAGE_BACKEND: str = "local"  # "local" or "s3"
    S3_URL_STYLE: str = "public"  # "public" or "presigned"
    ENVIRONMENT: str = "development"  # "development", "staging", "production"
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


def validate_retention_config(cfg: Settings) -> None:
    """Validate consistency of retention windows across jobs, static files, temp scratch, and caches.

    Documented relationships:
    1. Clip files must live at least as long as the job record:
       STATIC_RETENTION_HOURS * 3600 >= JOB_RECORD_TTL_SECONDS
    2. Job record must live at least as long as a user can reasonably return:
       JOB_RECORD_TTL_SECONDS >= 3600 (at least 1 hour, default 24h)
    3. Temp directory retention must be sufficient for active task budgets:
       TEMP_DIR_CLEANUP_HOURS * 3600 >= task_time_limit
    4. S3 presigned URL expiry and retention days must cover the job record TTL:
       If STORAGE_BACKEND == "s3":
         S3_PRESIGNED_EXPIRY_SECONDS >= JOB_RECORD_TTL_SECONDS
         S3_RETENTION_DAYS * 86400 >= JOB_RECORD_TTL_SECONDS
    """
    job_ttl = cfg.JOB_RECORD_TTL_SECONDS
    if job_ttl < 3600:
        raise ValueError(
            f"Invalid retention configuration: JOB_RECORD_TTL_SECONDS ({job_ttl}s) "
            f"must be at least 3600s (1 hour) to allow user return."
        )

    static_retention_seconds = cfg.STATIC_RETENTION_HOURS * 3600.0
    if static_retention_seconds < job_ttl:
        raise ValueError(
            f"Invalid retention configuration: clip files (STATIC_RETENTION_HOURS={cfg.STATIC_RETENTION_HOURS}h = "
            f"{static_retention_seconds}s) must live at least as long as the job record "
            f"(JOB_RECORD_TTL_SECONDS={job_ttl}s)."
        )

    temp_retention_seconds = cfg.TEMP_DIR_CLEANUP_HOURS * 3600.0
    if temp_retention_seconds < cfg.task_time_limit:
        raise ValueError(
            f"Invalid retention configuration: TEMP_DIR_CLEANUP_HOURS ({cfg.TEMP_DIR_CLEANUP_HOURS}h = "
            f"{temp_retention_seconds}s) is shorter than task_time_limit ({cfg.task_time_limit}s), "
            f"risking deletion of active scratch files."
        )

    if cfg.STORAGE_BACKEND == "s3":
        if cfg.S3_URL_STYLE == "presigned" and cfg.S3_PRESIGNED_EXPIRY_SECONDS < job_ttl:
            raise ValueError(
                f"Invalid retention configuration: S3_PRESIGNED_EXPIRY_SECONDS ({cfg.S3_PRESIGNED_EXPIRY_SECONDS}s) "
                f"must be >= JOB_RECORD_TTL_SECONDS ({job_ttl}s) to avoid dangling presigned URLs."
            )
        if cfg.S3_RETENTION_DAYS * 86400 < job_ttl:
            raise ValueError(
                f"Invalid retention configuration: S3_RETENTION_DAYS ({cfg.S3_RETENTION_DAYS}d = "
                f"{cfg.S3_RETENTION_DAYS * 86400}s) must be >= JOB_RECORD_TTL_SECONDS ({job_ttl}s)."
            )

