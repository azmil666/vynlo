import logging
import os
import shutil
import time
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


def cleanup_stale_temp_dirs(
    base_dir: str | None = None,
    max_age_hours: float | None = None,
    redis_client: Any | None = None,
    min_orphan_age_seconds: float = 3600.0,
) -> list[str]:
    """Scan base_dir for job directories older than max_age_hours and delete them.
    Also removes scratch directories for completed, failed, or missing jobs older than min_orphan_age_seconds.

    Returns a list of directory paths that were successfully removed.
    """
    target_dir = base_dir or settings.TEMP_DIR
    age_hours = (
        max_age_hours
        if max_age_hours is not None
        else settings.TEMP_DIR_CLEANUP_HOURS
    )
    max_age_seconds = age_hours * 3600.0

    if not os.path.exists(target_dir):
        return []

    removed: list[str] = []
    now = time.time()

    try:
        entries = os.listdir(target_dir)
    except OSError as e:
        logger.warning(
            "Could not list directory %s for stale cleanup: %s", target_dir, e
        )
        return []

    for entry in entries:
        if entry.startswith("."):
            continue
        entry_path = os.path.join(target_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        try:
            mtime = os.path.getmtime(entry_path)
            age = now - mtime

            # 1. Removal by age exceeding max retention
            if age > max_age_seconds:
                logger.info(
                    "Sweeper removing stale temp directory %s (age %.1fh > %.1fh)",
                    entry_path,
                    age / 3600.0,
                    age_hours,
                )
                shutil.rmtree(entry_path, ignore_errors=True)
                removed.append(entry_path)
                continue

            # 2. Removal of scratch dirs for finished or orphaned jobs older than grace period
            if redis_client is not None and age > min_orphan_age_seconds:
                job_id = entry
                try:
                    record = redis_client.get(f"job:{job_id}")
                    # If job record is missing or already terminal, temp scratch is safe to prune
                    should_prune = False
                    if not record:
                        should_prune = True
                    else:
                        import json
                        job_data = json.loads(record)
                        status = job_data.get("status")
                        if status in ("completed", "failed", "expired"):
                            should_prune = True

                    if should_prune:
                        logger.info(
                            "Sweeper pruning finished/orphaned temp directory %s (job %s, age %.1fh)",
                            entry_path,
                            job_id,
                            age / 3600.0,
                        )
                        shutil.rmtree(entry_path, ignore_errors=True)
                        removed.append(entry_path)
                except Exception as err:
                    logger.warning("Error inspecting redis status for temp dir %s: %s", entry_path, err)

        except Exception as e:
            logger.warning(
                "Failed inspecting or removing stale directory %s: %s",
                entry_path,
                e,
            )

    return removed


def cleanup_stale_static_files(
    static_dir: str | None = None,
    max_age_hours: float | None = None,
    redis_client: Any | None = None,
    min_orphan_age_seconds: float = 3600.0,
) -> list[str]:
    """Scan static_dir for:
    1. Static files and job directories older than max_age_hours (default settings.STATIC_RETENTION_HOURS).
    2. Orphaned job directories older than min_orphan_age_seconds whose job record no longer exists in Redis.

    Returns a list of paths (directories or files) that were successfully removed.
    """
    target_dir = static_dir or settings.STATIC_DIR
    age_hours = (
        max_age_hours
        if max_age_hours is not None
        else settings.STATIC_RETENTION_HOURS
    )
    max_age_seconds = age_hours * 3600.0

    if not os.path.exists(target_dir):
        return []

    removed: list[str] = []
    now = time.time()

    try:
        entries = os.listdir(target_dir)
    except OSError as e:
        logger.warning(
            "Could not list directory %s for static cleanup: %s", target_dir, e
        )
        return []

    for entry in entries:
        if entry.startswith("."):
            continue
        entry_path = os.path.join(target_dir, entry)

        try:
            mtime = os.path.getmtime(entry_path)
            age = now - mtime

            # 1. Stale by retention window
            if age > max_age_seconds:
                logger.info(
                    "Sweeper removing stale static entry %s (age %.1fh > %.1fh)",
                    entry_path,
                    age / 3600.0,
                    age_hours,
                )
                if os.path.isdir(entry_path):
                    shutil.rmtree(entry_path, ignore_errors=True)
                else:
                    try:
                        os.remove(entry_path)
                    except OSError:
                        pass
                removed.append(entry_path)
                continue

            # 2. Orphaned job directory check: older than min_orphan_age_seconds and job record missing from Redis
            if os.path.isdir(entry_path) and redis_client is not None and age > min_orphan_age_seconds:
                job_id = entry
                try:
                    record = redis_client.get(f"job:{job_id}")
                    if not record:
                        logger.info(
                            "Sweeper removing orphaned static job directory %s (job %s not in Redis, age %.1fh)",
                            entry_path,
                            job_id,
                            age / 3600.0,
                        )
                        shutil.rmtree(entry_path, ignore_errors=True)
                        removed.append(entry_path)
                except Exception as err:
                    logger.warning("Error checking redis for orphaned static job %s: %s", job_id, err)

        except Exception as e:
            logger.warning(
                "Failed inspecting or removing static entry %s: %s",
                entry_path,
                e,
            )

    return removed
