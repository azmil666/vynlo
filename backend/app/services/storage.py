import os
import shutil
from typing import Any

import boto3

from app.config import settings

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


def upload_clip(
    file_path: str,
    object_key: str,
    s3_client: Any | None = None,
    storage_handler: Any | None = None,
) -> str:
    """Upload rendered clip to S3/Cloudflare R2 or persist to local static directory."""
    active_handler = storage_handler if storage_handler is not None else get_storage_handler()
    if active_handler is not None:
        return active_handler(file_path, object_key)

    clean_key = object_key.lstrip("/")

    is_s3_configured = bool(
        settings.S3_ENDPOINT_URL
        and settings.S3_ACCESS_KEY
        and settings.S3_SECRET_KEY
        and settings.S3_BUCKET_NAME
    )

    if is_s3_configured:
        client = s3_client if s3_client is not None else get_s3_client()
        if client is None:
            client = boto3.client(
                "s3",
                endpoint_url=settings.S3_ENDPOINT_URL,
                aws_access_key_id=settings.S3_ACCESS_KEY,
                aws_secret_access_key=settings.S3_SECRET_KEY,
            )

        client.upload_file(
            file_path,
            settings.S3_BUCKET_NAME,
            clean_key,
            ExtraArgs={"ContentType": "video/mp4"},
        )

        if settings.S3_PUBLIC_BASE_URL:
            base_url = settings.S3_PUBLIC_BASE_URL.rstrip("/")
            return f"{base_url}/{clean_key}"
        else:
            endpoint = str(settings.S3_ENDPOINT_URL).rstrip("/")
            return f"{endpoint}/{settings.S3_BUCKET_NAME}/{clean_key}"

    # Local static fallback mode
    dest_path = os.path.join(settings.STATIC_DIR, clean_key)
    dest_dir = os.path.dirname(dest_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    shutil.copy2(file_path, dest_path)
    public_base = settings.PUBLIC_BASE_URL.rstrip("/")
    return f"{public_base}/static/{clean_key}"
