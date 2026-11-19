"""Unified production startup validator for API server and Celery workers.

Performs strict pre-flight validation across:
1. Required environment variables and configuration constraints
2. Pre-baked model files (YuNet ONNX model existence & SHA-256 checksum)
3. Core system executables (ffmpeg, ffprobe) and localized fonts
4. Redis state and broker connectivity
5. LLM provider and credentials readiness
6. Storage backend writability and credentials validation
7. PUBLIC_BASE_URL validity (enforcing non-loopback URLs in production)

Aggregates all failures and fails fast with a single actionable remediation list.
"""

import hashlib
import logging
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, List, Optional
from urllib.parse import urlparse

import redis

from app.config import settings
from app.services.cropper import YUNET_EXPECTED_SHA256, get_yunet_model_path

logger = logging.getLogger("startup_validator")


class StartupValidationError(RuntimeError):
    """Raised when one or more pre-flight startup checks fail."""

    def __init__(self, errors: List[str], service_name: str = "service"):
        self.errors = errors
        self.service_name = service_name
        formatted_list = "\n".join(f"  {i+1}. {err}" for i, err in enumerate(errors))
        message = (
            f"\n{'='*78}\n"
            f"CRITICAL: Startup validation failed for '{service_name}' ({len(errors)} error(s)):\n"
            f"{formatted_list}\n"
            f"{'='*78}\n"
            f"Action Required: Resolve the above configuration/environment issues before starting '{service_name}'."
        )
        super().__init__(message)


def validate_environment_variables() -> List[str]:
    """Verify core environment variables and configuration consistency."""
    errors = []
    cfg = settings

    # Storage backend
    backend = getattr(cfg, "STORAGE_BACKEND", "local").lower().strip()
    if backend not in ("local", "s3"):
        errors.append(
            f"[Env] STORAGE_BACKEND '{backend}' is invalid. Action: set STORAGE_BACKEND to 'local' or 's3'."
        )
    elif backend == "s3":
        missing_s3 = []
        if not getattr(cfg, "S3_BUCKET_NAME", None):
            missing_s3.append("S3_BUCKET_NAME")
        if not getattr(cfg, "S3_ACCESS_KEY", None):
            missing_s3.append("S3_ACCESS_KEY")
        if not getattr(cfg, "S3_SECRET_KEY", None):
            missing_s3.append("S3_SECRET_KEY")
        if not getattr(cfg, "S3_ENDPOINT_URL", None):
            missing_s3.append("S3_ENDPOINT_URL")
        if missing_s3:
            errors.append(
                f"[Env] S3 storage mode selected but required variables missing: {', '.join(missing_s3)}. "
                f"Action: configure S3 credentials and endpoint in environment."
            )

    # Redis URL
    redis_url = getattr(cfg, "REDIS_URL", "").strip()
    if not redis_url:
        errors.append("[Env] REDIS_URL is not set. Action: define REDIS_URL (e.g. redis://redis:6379/0).")

    # Public base URL
    env_name = getattr(cfg, "ENVIRONMENT", "development").lower().strip()
    public_url = getattr(cfg, "PUBLIC_BASE_URL", "").strip()
    if not public_url:
        errors.append(
            "[Env] PUBLIC_BASE_URL is empty. Action: set PUBLIC_BASE_URL to the public API root."
        )
    else:
        parsed = urlparse(public_url)
        if not parsed.scheme or not parsed.netloc:
            errors.append(
                f"[Env] PUBLIC_BASE_URL '{public_url}' is not a valid absolute URL. Action: provide a valid http(s) URL."
            )
        hostname = (parsed.hostname or "").lower()
        if env_name in ("production", "staging", "prod") and hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
            errors.append(
                f"[Env] PUBLIC_BASE_URL cannot use loopback/localhost ('{public_url}') in environment '{env_name}'. "
                f"Action: set PUBLIC_BASE_URL to the external production domain/hostname."
            )

    return errors


def validate_model_files() -> List[str]:
    """Verify presence and SHA-256 integrity of neural model files."""
    errors = []

    # YuNet face detector model
    try:
        model_path = get_yunet_model_path()
        if not os.path.exists(model_path):
            errors.append(
                f"[Model] YuNet face detection model not found at '{model_path}'. "
                f"Action: ensure face_detection_yunet_2023mar.onnx is baked into models directory."
            )
        else:
            with open(model_path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            if digest != YUNET_EXPECTED_SHA256:
                errors.append(
                    f"[Model] YuNet model checksum mismatch at '{model_path}'. "
                    f"Expected SHA-256 '{YUNET_EXPECTED_SHA256}', got '{digest}'. "
                    f"Action: replace with authentic face_detection_yunet_2023mar.onnx model file."
                )
    except Exception as e:
        errors.append(
            f"[Model] YuNet model verification failed: {e}. "
            f"Action: provide valid YUNET_MODEL_PATH in environment."
        )

    # Faster-whisper module / model availability check
    try:
        import faster_whisper
    except ImportError as e:
        errors.append(
            f"[Model] faster_whisper package is not importable: {e}. "
            f"Action: install faster-whisper dependency in the runtime environment."
        )

    return errors


def validate_executables_and_fonts() -> List[str]:
    """Verify ffmpeg, ffprobe, and required typography fonts are present and functional."""
    errors = []

    # FFmpeg executable
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        errors.append(
            "[System] ffmpeg executable not found on system PATH. Action: install FFmpeg (>= 4.4)."
        )
    else:
        try:
            res = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True, timeout=5)
            if res.returncode != 0:
                errors.append(
                    f"[System] ffmpeg executable failed with returncode {res.returncode}. Action: verify FFmpeg installation."
                )
        except Exception as e:
            errors.append(
                f"[System] Unable to execute ffmpeg at '{ffmpeg_path}': {e}. Action: check binary permissions."
            )

    # FFprobe executable
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        errors.append(
            "[System] ffprobe executable not found on system PATH. Action: install FFprobe."
        )
    else:
        try:
            res = subprocess.run([ffprobe_path, "-version"], capture_output=True, text=True, timeout=5)
            if res.returncode != 0:
                errors.append(
                    f"[System] ffprobe executable failed with returncode {res.returncode}. Action: verify FFprobe installation."
                )
        except Exception as e:
            errors.append(
                f"[System] Unable to execute ffprobe at '{ffprobe_path}': {e}. Action: check binary permissions."
            )

    # Fonts availability for ASS subtitle rendering
    app_dir = Path(__file__).resolve().parent.parent
    candidate_font_dirs = [
        app_dir / "fonts",
        Path("/app/fonts"),
        Path("/usr/share/fonts"),
        Path("/System/Library/Fonts"),
    ]
    fonts_found = []
    for fdir in candidate_font_dirs:
        if fdir.exists() and fdir.is_dir():
            fonts_found.extend(list(fdir.glob("*.ttf")) + list(fdir.glob("*.otf")))

    if not fonts_found:
        errors.append(
            "[System] No TrueType (.ttf) or OpenType (.otf) fonts discovered in fonts directories. "
            "Action: ensure NotoSans fonts are placed in app/fonts/."
        )

    return errors


def validate_redis_connectivity(client: Optional[redis.Redis] = None) -> List[str]:
    """Verify that state Redis (and Celery broker Redis if distinct) is reachable."""
    errors = []
    r_client = client

    if r_client is None:
        try:
            r_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True, socket_timeout=3.0)
        except Exception as e:
            return [
                f"[Redis] Failed to initialize Redis client from REDIS_URL '{settings.REDIS_URL}': {e}. "
                f"Action: check REDIS_URL format."
            ]

    try:
        pong = r_client.ping()
        if not pong:
            errors.append(
                f"[Redis] Redis ping returned unexpected response '{pong}' at '{settings.REDIS_URL}'. "
                f"Action: check Redis server health."
            )
    except Exception as e:
        errors.append(
            f"[Redis] Cannot connect to Redis at '{settings.REDIS_URL}': {e}. "
            f"Action: start or unblock Redis container/host."
        )

    # Check distinct broker URL if configured
    broker_url = getattr(settings, "CELERY_BROKER_URL", None)
    if broker_url and broker_url != settings.REDIS_URL and broker_url.startswith("redis"):
        try:
            broker_client = redis.Redis.from_url(broker_url, decode_responses=True, socket_timeout=3.0)
            if not broker_client.ping():
                errors.append(f"[Redis] Celery broker Redis at '{broker_url}' did not respond to ping.")
        except Exception as e:
            errors.append(
                f"[Redis] Cannot connect to Celery broker Redis at '{broker_url}': {e}. "
                f"Action: ensure broker Redis instance is running."
            )

    return errors


def validate_llm_configuration(dry_run: bool = True) -> List[str]:
    """Validate LLM provider credentials, model selection, and connectivity."""
    errors = []
    from app.services.analyzer import _llm_client, validate_llm_config

    # If mock/fake LLM client is already injected for tests, validation passes
    if _llm_client is not None:
        return []

    env_name = getattr(settings, "ENVIRONMENT", "development").lower().strip()
    api_key = str(getattr(settings, "LLM_API_KEY", "") or "").strip()

    try:
        validate_llm_config()
    except Exception as e:
        msg = str(e)
        if env_name in ("production", "staging", "prod"):
            errors.append(
                f"[LLM] LLM configuration error in '{env_name}': {msg}. "
                f"Action: configure valid LLM_PROVIDER ('gemini' or 'openai'), LLM_MODEL, and LLM_API_KEY."
            )
        else:
            logger.warning("Development LLM configuration note: %s", msg)
        return errors

    # In production/staging with valid API key, execute a cheap dry call
    if dry_run and api_key and api_key != "fake_key_for_testing" and env_name in ("production", "staging", "prod"):
        try:
            from app.services.analyzer import _get_llm_client
            client = _get_llm_client()
            if hasattr(client, "models") and hasattr(client.models, "list"):
                # Lightweight metadata call to verify credentials
                client.models.list()
        except Exception as e:
            errors.append(
                f"[LLM] LLM dry-run credential verification failed: {e}. "
                f"Action: verify that LLM_API_KEY has permission to access {settings.LLM_MODEL}."
            )

    return errors


def validate_storage_writable() -> List[str]:
    """Verify storage backend writability."""
    errors = []
    backend = getattr(settings, "STORAGE_BACKEND", "local").lower()

    if backend == "local":
        static_dir = getattr(settings, "STATIC_DIR", "/app/static")
        try:
            os.makedirs(static_dir, exist_ok=True)
            canary_path = os.path.join(static_dir, f".startup_canary_{os.getpid()}")
            with open(canary_path, "w") as f:
                f.write("canary_ok")
            if os.path.exists(canary_path):
                os.remove(canary_path)
        except Exception as e:
            errors.append(
                f"[Storage] STATIC_DIR '{static_dir}' is not writable: {e}. "
                f"Action: check permissions or docker volume mapping for static directory."
            )
    elif backend == "s3":
        # S3 credential presence checked in validate_environment_variables
        pass

    return errors


def validate_startup(
    service_name: str = "api",
    fail_fast: bool = True,
    dry_run_llm: bool = True,
    redis_client: Optional[redis.Redis] = None,
) -> List[str]:
    """Run all pre-flight startup validations and collect actionable errors.

    If fail_fast is True and errors exist, raises StartupValidationError.
    """
    all_errors: List[str] = []

    all_errors.extend(validate_environment_variables())
    all_errors.extend(validate_model_files())
    all_errors.extend(validate_executables_and_fonts())
    all_errors.extend(validate_redis_connectivity(redis_client))
    all_errors.extend(validate_llm_configuration(dry_run=dry_run_llm))
    all_errors.extend(validate_storage_writable())

    if all_errors:
        logger.error(
            "Startup pre-flight validation failed with %d error(s) for '%s'",
            len(all_errors),
            service_name,
        )
        if fail_fast:
            raise StartupValidationError(all_errors, service_name=service_name)
    else:
        logger.info("Startup pre-flight validation succeeded for '%s'", service_name)

    return all_errors
