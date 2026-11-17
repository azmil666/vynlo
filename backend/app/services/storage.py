import os
import shutil

import boto3

from app.config import settings


def upload_clip(file_path: str, object_key: str) -> str:
    """Upload rendered clip to S3/Cloudflare R2 or persist to local static directory."""
    clean_key = object_key.lstrip("/")

    is_s3_configured = bool(
        settings.S3_ENDPOINT_URL
        and settings.S3_ACCESS_KEY
        and settings.S3_SECRET_KEY
        and settings.S3_BUCKET_NAME
    )

    if is_s3_configured:
        s3_client = boto3.client(
            "s3",
            endpoint_url=settings.S3_ENDPOINT_URL,
            aws_access_key_id=settings.S3_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SECRET_KEY,
        )

        s3_client.upload_file(
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
