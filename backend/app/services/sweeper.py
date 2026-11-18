import logging
import os
import shutil
import time

from app.config import settings

logger = logging.getLogger(__name__)


def cleanup_stale_temp_dirs(
    base_dir: str | None = None,
    max_age_hours: float | None = None,
) -> list[str]:
    """Scan base_dir for job directories older than max_age_hours and delete them.

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
        entry_path = os.path.join(target_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        try:
            mtime = os.path.getmtime(entry_path)
            age = now - mtime
            if age > max_age_seconds:
                logger.info(
                    "Sweeper removing stale directory %s (age %.1fh > %.1fh)",
                    entry_path,
                    age / 3600.0,
                    age_hours,
                )
                shutil.rmtree(entry_path, ignore_errors=True)
                removed.append(entry_path)
        except Exception as e:
            logger.warning(
                "Failed inspecting or removing stale directory %s: %s",
                entry_path,
                e,
            )

    return removed
