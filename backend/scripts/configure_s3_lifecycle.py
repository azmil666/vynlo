"""S3 / R2 Bucket Lifecycle Policy Configuration Script.

Configures automated object expiration on the target S3/R2 bucket so that rendered
clips and media assets are permanently purged once their retention window elapses.

Usage:
    python -m scripts.configure_s3_lifecycle
    or
    python backend/scripts/configure_s3_lifecycle.py [--days 7] [--prefix ""]

Relationship to central retention config:
    S3_RETENTION_DAYS (default: 7 days) defines the object lifetime in S3.
    This must be >= JOB_RECORD_TTL_SECONDS (default 24h) and match S3_PRESIGNED_EXPIRY_SECONDS
    so presigned URLs remain valid while the clips are stored.
"""

import argparse
import logging
import sys
from typing import Any

import boto3
from botocore.client import Config

from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("s3_lifecycle")


def apply_s3_lifecycle_policy(
    s3_client: Any,
    bucket_name: str,
    retention_days: int = 7,
    rule_id: str = "clipper-clip-lifecycle",
    prefix: str = "",
) -> dict[str, Any]:
    """Apply an automated object expiration lifecycle rule to an S3 bucket."""
    lifecycle_configuration = {
        "Rules": [
            {
                "ID": rule_id,
                "Status": "Enabled",
                "Filter": {"Prefix": prefix},
                "Expiration": {"Days": retention_days},
            }
        ]
    }
    logger.info(
        "Applying lifecycle rule '%s' to bucket '%s': expire objects under prefix '%s' after %d days",
        rule_id,
        bucket_name,
        prefix,
        retention_days,
    )
    return s3_client.put_bucket_lifecycle_configuration(
        Bucket=bucket_name,
        LifecycleConfiguration=lifecycle_configuration,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure S3 lifecycle expiration policy for Clipper")
    parser.add_argument("--days", type=int, default=settings.S3_RETENTION_DAYS, help="Retention period in days")
    parser.add_argument("--prefix", type=str, default="", help="Object key prefix filter (default: all objects)")
    args = parser.parse_args()

    if not settings.S3_BUCKET_NAME:
        logger.error("S3_BUCKET_NAME is not configured in settings. Exiting.")
        sys.exit(1)

    s3_client = boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT_URL,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )

    try:
        apply_s3_lifecycle_policy(
            s3_client=s3_client,
            bucket_name=settings.S3_BUCKET_NAME,
            retention_days=args.days,
            prefix=args.prefix,
        )
        logger.info("Successfully applied lifecycle configuration to bucket '%s'.", settings.S3_BUCKET_NAME)
    except Exception as e:
        logger.error("Failed to apply lifecycle policy: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
