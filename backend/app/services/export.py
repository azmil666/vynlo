import logging
import os
import re
import tempfile
from typing import Any
import zipfile

import requests

from app.config import settings
from app.schemas import JobStatusResponse

logger = logging.getLogger(__name__)

# Strict limits to preserve low-memory bounds and avoid unbounded transfers
MAX_CLIP_EXPORT_BYTES: int = 250 * 1024 * 1024  # 250 MB max per clip
SPOOLED_ZIP_MAX_MEMORY_BYTES: int = 10 * 1024 * 1024  # 10 MB in RAM before spilling to disk
REMOTE_CONNECT_TIMEOUT_SECONDS: float = 5.0
REMOTE_READ_TIMEOUT_SECONDS: float = 30.0


def slugify_title(text: str) -> str:
    """Convert text to safe lowercase filename slug."""
    slug = re.sub(r"[^\w\s-]", "", text or "").strip().lower()
    slug = re.sub(r"[-\s]+", "_", slug)
    slug = slug.strip("_")
    return slug[:40] or "clip"


def fetch_remote_clip_stream(
    url: str,
    dest_stream: Any,
    max_bytes: int = MAX_CLIP_EXPORT_BYTES,
    timeout: tuple[float, float] = (
        REMOTE_CONNECT_TIMEOUT_SECONDS,
        REMOTE_READ_TIMEOUT_SECONDS,
    ),
) -> int:
    """Stream a remote clip over HTTP directly into dest_stream in 64KB chunks.
    Enforces connect/read timeouts and strict size limits.
    """
    total_bytes = 0
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            if chunk:
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise ValueError(
                        f"Remote clip exceeds maximum allowed size ({max_bytes} bytes)"
                    )
                dest_stream.write(chunk)
    return total_bytes


def build_batch_export_archive(
    job_id: str,
    job_status: JobStatusResponse,
    static_dir: str | None = None,
    max_spooled_memory: int = SPOOLED_ZIP_MAX_MEMORY_BYTES,
    max_clip_bytes: int = MAX_CLIP_EXPORT_BYTES,
) -> tempfile.SpooledTemporaryFile:
    """Build a zip archive containing generated clips and a comprehensive text manifest.

    Memory & Performance design:
    - Uses SpooledTemporaryFile with max_spooled_memory to strictly cap RAM usage.
    - Uses ZIP_STORED for MP4 video entries to avoid redundant CPU/RAM compression overhead.
    - Uses ZIP_DEFLATED for text manifest.
    - Streams local files in 64KB chunks directly into the archive without loading into RAM.
    - Streams remote clips in chunks with timeouts and size limits.
    - Records missing/failed clips in manifest notes rather than aborting export.
    """
    base_static = static_dir or settings.STATIC_DIR
    spooled_zip = tempfile.SpooledTemporaryFile(max_size=max_spooled_memory)

    manifest_lines: list[str] = [
        f"=== VYNLO STUDIO BATCH EXPORT: JOB {job_id} ===",
        f"Status: {job_status.status}",
        f"Stage: {job_status.stage}",
        f"Total Clips: {len(job_status.clips)}",
        "",
    ]

    with zipfile.ZipFile(spooled_zip, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for i, clip in enumerate(job_status.clips, start=1):
            slug = slugify_title(clip.title)
            mp4_filename = f"{i:02d}_{slug}.mp4"
            status_note = "AVAILABLE"
            clip_included = False

            # Check if clip is marked failed in the response model
            if clip.ok is False or clip.render_ok is False:
                status_note = (
                    f"UNAVAILABLE: Clip rendering failed [{clip.error_code or 'UNKNOWN'}] "
                    f"({clip.error_message or 'Render error'})"
                )
            else:
                # 1. Try reading from local static storage
                local_path = os.path.join(base_static, job_id, f"clip_{i}.mp4")
                if os.path.isfile(local_path):
                    try:
                        # Write local file with ZIP_STORED (streamed in 64KB chunks by ZipFile.write)
                        zf.write(local_path, arcname=mp4_filename, compress_type=zipfile.ZIP_STORED)
                        clip_included = True
                        status_note = "AVAILABLE (Local Storage)"
                    except Exception as e:
                        logger.warning("Error reading local clip file %s: %s", local_path, e)
                        status_note = f"UNAVAILABLE: Local read error ({e})"

                # 2. Try fetching from remote URL if not local
                if not clip_included and clip.video_url:
                    try:
                        # Stream remote clip chunks directly into the zip entry
                        with zf.open(mp4_filename, mode="w", force_zip64=True) as dest_entry:
                            fetch_remote_clip_stream(
                                url=clip.video_url,
                                dest_stream=dest_entry,
                                max_bytes=max_clip_bytes,
                            )
                        clip_included = True
                        status_note = "AVAILABLE (Remote Storage)"
                    except Exception as e:
                        logger.warning("Error streaming remote clip %s (%s): %s", i, clip.video_url, e)
                        status_note = f"UNAVAILABLE: Remote fetch failed ({e})"

                if not clip_included and status_note == "AVAILABLE":
                    status_note = "UNAVAILABLE: File missing or expired from storage"

            # Build detailed metadata block for manifest
            manifest_lines.append(f"--- CLIP {i:02d}: {clip.title} ---")
            manifest_lines.append(f"File: {mp4_filename if clip_included else 'NOT_INCLUDED'}")
            manifest_lines.append(f"Title: {clip.title}")
            manifest_lines.append(f"Status: {status_note}")
            manifest_lines.append(f"Duration: {clip.duration}s | Score: {clip.virality_score}/100")
            selection_info = clip.selection_method
            if clip.selection_reason:
                selection_info += f" ({clip.selection_reason})"
            manifest_lines.append(f"Selection Method: {selection_info}")
            manifest_lines.append(f"Layout Used: {clip.layout_used}")
            manifest_lines.append(f"Video URL: {clip.video_url or 'N/A'}")

            if clip.description:
                manifest_lines.append(f"Description: {clip.description}")

            if clip.hashtags:
                formatted_tags = " ".join(f"#{t.lstrip('#')}" for t in clip.hashtags if str(t).strip())
                if formatted_tags:
                    manifest_lines.append(f"Hashtags: {formatted_tags}")

            # Collect all warnings: per-clip warnings and matching job-level warnings
            clip_warns: list[str] = []
            for w in getattr(clip, "warnings", []):
                clip_warns.append(str(w))
            for jw in job_status.warnings:
                if jw.clip_index == i:
                    clip_warns.append(f"[{jw.code}] {jw.message}")

            if clip_warns:
                manifest_lines.append("Warnings:")
                for cw in clip_warns:
                    manifest_lines.append(f"  - {cw}")

            manifest_lines.append("")

        # Add manifest compressed with deflate
        manifest_text = "\n".join(manifest_lines)
        zf.writestr(
            "captions_and_tags.txt",
            manifest_text.encode("utf-8"),
            compress_type=zipfile.ZIP_DEFLATED,
        )

    spooled_zip.seek(0)
    return spooled_zip
