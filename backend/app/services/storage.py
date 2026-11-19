import logging
import os
import shutil
import time
from typing import Any
from urllib.parse import urlparse

import boto3

from app.config import settings
from app.schemas import PipelineError, PipelineErrorCode

logger = logging.getLogger(__name__)

_s3_client: Any | None = None
_storage_handler: Any | None = None


def get_s3_client() -> Any | None:
    return _s3_client


def set_s3_client(client: Any | None) -> None:
    global _s3_client
    _s3_client = client


def get_storage_handler() -> Any | None:
    return _storage_handler


def set_storage_handler(handler: Any | None) -> None:
    global _storage_handler
    _storage_handler = handler


def validate_storage_config(current_settings: Any = None) -> None:
    """Validate storage configuration, writability, and public URLs at application startup."""
    cfg = current_settings if current_settings is not None else settings

    # 1. URL validation for non-development environments
    env_name = getattr(cfg, "ENVIRONMENT", "development").lower()
    public_url = getattr(cfg, "PUBLIC_BASE_URL", "").strip()

    if env_name in ("production", "prod", "staging"):
        if not public_url:
            raise ValueError(
                f"PUBLIC_BASE_URL must be set in non-development environment '{env_name}'."
            )
        parsed = urlparse(public_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(
                f"PUBLIC_BASE_URL '{public_url}' is not a valid URL in environment '{env_name}'."
            )
        hostname = (parsed.hostname or "").lower()
        if hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
            raise ValueError(
                f"PUBLIC_BASE_URL cannot use localhost/loopback ('{public_url}') in non-development environment '{env_name}'."
            )

    # 2. Storage backend validation
    backend = getattr(cfg, "STORAGE_BACKEND", "local").lower()
    if backend not in ("local", "s3"):
        raise ValueError(f"Invalid STORAGE_BACKEND '{backend}'. Must be 'local' or 's3'.")

    if backend == "s3":
        missing = []
        if not getattr(cfg, "S3_BUCKET_NAME", None):
            missing.append("S3_BUCKET_NAME")
        if not getattr(cfg, "S3_ACCESS_KEY", None):
            missing.append("S3_ACCESS_KEY")
        if not getattr(cfg, "S3_SECRET_KEY", None):
            missing.append("S3_SECRET_KEY")
        if not getattr(cfg, "S3_ENDPOINT_URL", None):
            missing.append("S3_ENDPOINT_URL")
        if missing:
            raise ValueError(
                f"STORAGE_BACKEND is 's3' but required S3 settings are missing: {', '.join(missing)}"
            )
    elif backend == "local":
        static_dir = getattr(cfg, "STATIC_DIR", "/app/static")
        try:
            os.makedirs(static_dir, exist_ok=True)
            probe_path = os.path.join(static_dir, f".write_test_{os.getpid()}")
            with open(probe_path, "w") as f:
                f.write("probe")
            os.remove(probe_path)
        except Exception as e:
            raise RuntimeError(
                f"STATIC_DIR '{static_dir}' is not writable. When running in local storage mode, "
                f"STATIC_DIR must be a shared writable volume between api and worker services: {e}"
            ) from e


def upload_clip(
    file_path: str,
    object_key: str,
    s3_client: Any | None = None,
    storage_handler: Any | None = None,
    max_retries: int = 3,
    initial_backoff: float = 0.5,
) -> str:
    """Upload rendered clip to S3/Cloudflare R2 or persist to local static directory.

    Never switches silently between backends.
    In S3 mode: retries with exponential backoff, verifies ContentLength via head_object,
    sets ContentType video/mp4 and Cache-Control, and raises typed UPLOAD_FAILED on failure.
    """
    active_handler = storage_handler if storage_handler is not None else get_storage_handler()
    if active_handler is not None:
        return active_handler(file_path, object_key)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found: {file_path}")

    clean_key = object_key.lstrip("/")
    backend = getattr(settings, "STORAGE_BACKEND", "local").lower()

    if backend == "s3":
        is_s3_configured = bool(
            settings.S3_ENDPOINT_URL
            and settings.S3_ACCESS_KEY
            and settings.S3_SECRET_KEY
            and settings.S3_BUCKET_NAME
        )
        if not is_s3_configured:
            raise PipelineError(
                code=PipelineErrorCode.UPLOAD_FAILED,
                stage="uploading",
                user_message="S3 storage is not properly configured.",
                retryable=False,
                detail="STORAGE_BACKEND is set to 's3' but required S3 configuration is missing. Refusing silent fallback to local storage.",
            )

        client = s3_client if s3_client is not None else get_s3_client()
        if client is None:
            client = boto3.client(
                "s3",
                endpoint_url=settings.S3_ENDPOINT_URL,
                aws_access_key_id=settings.S3_ACCESS_KEY,
                aws_secret_access_key=settings.S3_SECRET_KEY,
            )

        file_size = os.path.getsize(file_path)
        extra_args = {
            "ContentType": "video/mp4",
            "CacheControl": "public, max-age=31536000, immutable",
        }

        last_error: Exception | None = None
        backoff = initial_backoff

        for attempt in range(1, max_retries + 1):
            try:
                client.upload_file(
                    file_path,
                    settings.S3_BUCKET_NAME,
                    clean_key,
                    ExtraArgs=extra_args,
                )

                # Verify uploaded object with head_object
                head_res = client.head_object(
                    Bucket=settings.S3_BUCKET_NAME,
                    Key=clean_key,
                )
                remote_size = head_res.get("ContentLength")
                if remote_size is not None and int(remote_size) != file_size:
                    raise ValueError(
                        f"S3 size mismatch: local file is {file_size} bytes, remote head_object reported {remote_size} bytes."
                    )

                logger.info(
                    "Successfully uploaded and verified %s (%d bytes) to S3 bucket %s",
                    clean_key,
                    file_size,
                    settings.S3_BUCKET_NAME,
                )
                break
            except Exception as e:
                last_error = e
                logger.warning(
                    "S3 upload attempt %d/%d for '%s' failed: %s",
                    attempt,
                    max_retries,
                    clean_key,
                    e,
                )
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
                else:
                    raise PipelineError(
                        code=PipelineErrorCode.UPLOAD_FAILED,
                        stage="uploading",
                        user_message="Failed to upload clip to S3 storage.",
                        retryable=True,
                        detail=f"S3 upload failed after {max_retries} attempts: {last_error}",
                    ) from last_error

        # Generate output URL based on configuration
        url_style = getattr(settings, "S3_URL_STYLE", "public").lower()
        if url_style == "presigned":
            expiry = getattr(settings, "S3_PRESIGNED_EXPIRY_SECONDS", 604800)
            return client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.S3_BUCKET_NAME, "Key": clean_key},
                ExpiresIn=expiry,
            )
        elif settings.S3_PUBLIC_BASE_URL:
            base_url = settings.S3_PUBLIC_BASE_URL.rstrip("/")
            return f"{base_url}/{clean_key}"
        else:
            endpoint = str(settings.S3_ENDPOINT_URL).rstrip("/")
            return f"{endpoint}/{settings.S3_BUCKET_NAME}/{clean_key}"

    elif backend == "local":
        dest_path = os.path.join(settings.STATIC_DIR, clean_key)
        dest_dir = os.path.dirname(dest_path)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)

        shutil.copy2(file_path, dest_path)
        public_base = settings.PUBLIC_BASE_URL.rstrip("/")
        return f"{public_base}/static/{clean_key}"

    else:
        raise PipelineError(
            code=PipelineErrorCode.UPLOAD_FAILED,
            stage="uploading",
            user_message=f"Unsupported STORAGE_BACKEND '{backend}'.",
            retryable=False,
            detail=f"Configured STORAGE_BACKEND '{backend}' is not supported. Must be 'local' or 's3'.",
        )
