from contextlib import asynccontextmanager
from datetime import datetime, timezone
import io
import logging
import os
import re
import urllib.request
from uuid import uuid4
import zipfile

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import redis

from app.config import settings, validate_retention_config
from app.schemas import (
    JobCreateRequest,
    JobErrorDetail,
    JobStatusResponse,
    PipelineErrorCode,
)
from app.services.storage import validate_storage_config
from app.services.sweeper import cleanup_stale_static_files, cleanup_stale_temp_dirs
from app.worker import process_video_task

logger = logging.getLogger(__name__)

# Ensure configured static directory exists for StaticFiles mount
os.makedirs(settings.STATIC_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validate retention windows and documented relationships
    validate_retention_config(settings)

    # Validate storage configuration, writability, and URL settings at startup
    validate_storage_config()

    # Sweep stale temp media directories on application startup
    try:
        cleaned_temp = cleanup_stale_temp_dirs()
        if cleaned_temp:
            logger.info("Cleaned %d stale temp directories on startup", len(cleaned_temp))
    except Exception as e:
        logger.warning("Startup stale temp cleanup failed: %s", e)

    # Sweep stale local static files and orphaned job directories on application startup
    try:
        cleaned_static = cleanup_stale_static_files(redis_client=get_redis_client())
        if cleaned_static:
            logger.info("Cleaned %d stale static entries on startup", len(cleaned_static))
    except Exception as e:
        logger.warning("Startup stale static cleanup failed: %s", e)

    # Validate LLM configuration on startup
    try:
        from app.services.analyzer import validate_llm_config
        validate_llm_config()
    except Exception as e:
        logger.warning("LLM configuration validation warning on startup: %s", e)

    yield


app = FastAPI(title="Clipper MVP API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

_redis_client = None


def get_redis_client() -> redis.Redis:
    global _redis_client, redis_client
    if _redis_client is not None:
        return _redis_client
    if redis_client is not None:
        return redis_client
    _redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis_client


def set_redis_client(client: redis.Redis | None) -> None:
    global _redis_client, redis_client
    _redis_client = client
    redis_client = client


redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)


def _slugify(text: str) -> str:
    """Convert text to safe lowercase filename slug."""
    slug = re.sub(r"[^\w\s-]", "", text).strip().lower()
    slug = re.sub(r"[-\s]+", "_", slug)
    return slug[:40] or "clip"


@app.post("/api/jobs", response_model=JobStatusResponse, status_code=status.HTTP_202_ACCEPTED)
def create_job(req: JobCreateRequest):
    job_id = str(uuid4())
    initial_status = JobStatusResponse(
        job_id=job_id,
        stage="queued",
        status="queued",
        progress=0,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    get_redis_client().set(
        f"job:{job_id}",
        initial_status.model_dump_json(),
        ex=settings.JOB_RECORD_TTL_SECONDS,
    )
    process_video_task.delay(job_id, req.model_dump(mode="json"))
    return initial_status


ACTIVE_PROCESSING_STAGES = {
    "queued",
    "downloading",
    "transcribing",
    "analyzing",
    "rendering",
}


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    r = get_redis_client()
    data = r.get(f"job:{job_id}")
    if not data:
        # Check if static files exist for this job (job record expired but files remain)
        static_job_dir = os.path.join(settings.STATIC_DIR, job_id)
        if os.path.isdir(static_job_dir):
            return JobStatusResponse(
                job_id=job_id,
                status="expired",
                stage="expired",
                progress=100,
                error=JobErrorDetail(
                    code=PipelineErrorCode.EXPIRED,
                    message="Job record has expired and is no longer available.",
                    retryable=False,
                    stage="expired",
                    detail="The job status record has expired from cache, though static files are pending cleanup.",
                ),
                clips=[],
            )
        raise HTTPException(status_code=404, detail="Job not found")

    job_status = JobStatusResponse.model_validate_json(data)

    # Check if job is completed but clip files have expired/been deleted (record remains but files expired)
    if job_status.status == "completed" and job_status.clips:
        is_expired = False
        if job_status.updated_at:
            try:
                raw_updated = job_status.updated_at
                updated_dt = datetime.fromisoformat(raw_updated.replace("Z", "+00:00"))
                if updated_dt.tzinfo is None:
                    updated_dt = updated_dt.replace(tzinfo=timezone.utc)
                age_seconds = (datetime.now(timezone.utc) - updated_dt).total_seconds()

                # Stale by exceeding static retention window
                if age_seconds > (settings.STATIC_RETENTION_HOURS * 3600.0):
                    is_expired = True
                # Stale if older than grace period (1 hour) and all files are missing from storage
                elif age_seconds > 3600.0 and settings.STORAGE_BACKEND == "local":
                    all_missing = True
                    for i, clip in enumerate(job_status.clips, start=1):
                        clip_path = os.path.join(settings.STATIC_DIR, job_id, f"clip_{i}.mp4")
                        if os.path.exists(clip_path):
                            all_missing = False
                            break
                    if all_missing:
                        is_expired = True
            except Exception:
                pass

        if is_expired:
            logger.warning(
                "Job %s record exists but static clip files have expired/been removed.", job_id
            )
            job_status.status = "expired"
            job_status.stage = "expired"
            job_status.clips = []  # Do not return dangling URLs
            job_status.error = JobErrorDetail(
                code=PipelineErrorCode.EXPIRED,
                message="Clip files have expired and are no longer available.",
                retryable=False,
                stage="expired",
                detail="The storage retention window for this job has elapsed.",
            )

    # Detect worker crash or killed worker process via stale heartbeat
    is_active = (
        job_status.status in ACTIVE_PROCESSING_STAGES
        or job_status.stage in ACTIVE_PROCESSING_STAGES
    )
    if is_active and job_status.updated_at:
        try:
            raw_updated = job_status.updated_at
            updated_dt = datetime.fromisoformat(raw_updated.replace("Z", "+00:00"))
            if updated_dt.tzinfo is None:
                updated_dt = updated_dt.replace(tzinfo=timezone.utc)

            age_seconds = (datetime.now(timezone.utc) - updated_dt).total_seconds()
            if age_seconds > settings.WORKER_HEARTBEAT_TIMEOUT_SECONDS:
                previous_stage = job_status.stage or job_status.status or "queued"
                logger.warning(
                    "Job %s stale heartbeat detected (%.1fs > %ds). Marking as WORKER_LOST.",
                    job_id,
                    age_seconds,
                    settings.WORKER_HEARTBEAT_TIMEOUT_SECONDS,
                )
                job_status.status = "failed"
                job_status.stage = "failed"
                job_status.error = JobErrorDetail(
                    code=PipelineErrorCode.WORKER_LOST.value,
                    message=f"Worker lost or unresponsive for more than {settings.WORKER_HEARTBEAT_TIMEOUT_SECONDS}s.",
                    retryable=True,
                    stage=previous_stage,
                    detail=f"Last updated at {raw_updated} ({int(age_seconds)}s ago)",
                )
                job_status.updated_at = datetime.now(timezone.utc).isoformat()

                # Persist recovered state back to Redis
                job_key = f"job:{job_id}"
                ttl = r.ttl(job_key)
                if ttl is None or ttl <= 0:
                    ttl = settings.JOB_RECORD_TTL_SECONDS
                r.set(job_key, job_status.model_dump_json(), ex=ttl)
        except Exception as e:
            logger.warning("Error evaluating worker heartbeat for job %s: %s", job_id, e)

    return job_status


@app.get("/api/jobs/{job_id}/download-all")
def download_all_clips(job_id: str):
    """Bundle all generated clips and captions into a single zip archive for batch download."""
    data = get_redis_client().get(f"job:{job_id}")
    if not data:
        static_job_dir = os.path.join(settings.STATIC_DIR, job_id)
        if os.path.isdir(static_job_dir):
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail="Job record has expired and download is no longer available",
            )
        raise HTTPException(status_code=404, detail="Job not found")

    job_status = JobStatusResponse.model_validate_json(data)
    if job_status.status == "expired":
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Job clips have expired and are no longer available",
        )
    if job_status.status != "completed" or not job_status.clips:
        raise HTTPException(
            status_code=400, detail="Job is not completed or has no clips available"
        )

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        captions_txt_lines: list[str] = [
            f"=== CLIPPER STUDIO BATCH EXPORT: JOB {job_id} ===",
            f"Total Clips: {len(job_status.clips)}",
            "",
        ]

        for i, clip in enumerate(job_status.clips, start=1):
            slug_title = _slugify(clip.title)
            mp4_filename = f"{i:02d}_{slug_title}.mp4"

            # 1. Read clip video bytes from local static store or remote S3/R2 URL
            video_bytes = None
            local_path = os.path.join(settings.STATIC_DIR, job_id, f"clip_{i}.mp4")
            if os.path.exists(local_path):
                try:
                    with open(local_path, "rb") as f:
                        video_bytes = f.read()
                except Exception:
                    pass

            if not video_bytes and clip.video_url:
                try:
                    # Attempt fetch from URL
                    with urllib.request.urlopen(clip.video_url, timeout=15) as resp:
                        video_bytes = resp.read()
                except Exception:
                    pass

            if video_bytes:
                zip_file.writestr(mp4_filename, video_bytes)

            # 2. Append social caption & hashtags to text summary
            formatted_tags = " ".join(
                f"#{t.lstrip('#')}" for t in clip.hashtags if str(t).strip()
            )
            captions_txt_lines.append(f"--- CLIP {i:02d}: {clip.title} ---")
            captions_txt_lines.append(f"File: {mp4_filename}")
            captions_txt_lines.append(f"Title: {clip.title}")
            if clip.description:
                captions_txt_lines.append(f"Description: {clip.description}")
            if formatted_tags:
                captions_txt_lines.append(f"Hashtags: {formatted_tags}")
            captions_txt_lines.append(
                f"Duration: {clip.duration}s | Score: {clip.virality_score}/100"
            )
            captions_txt_lines.append(
                f"Selection Method: {clip.selection_method}"
                + (f" ({clip.selection_reason})" if clip.selection_reason else "")
            )
            captions_txt_lines.append(f"Video URL: {clip.video_url}")
            captions_txt_lines.append("")

        # If no video files could be read, the clips have expired from storage
        video_files_in_zip = [f for f in zip_file.filelist if f.filename.endswith(".mp4")]
        if not video_files_in_zip:
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail="Job clips have expired or are missing from storage",
            )

        zip_file.writestr("captions_and_tags.txt", "\n".join(captions_txt_lines))

    zip_buffer.seek(0)
    filename = f"shorts_{job_id[:8]}.zip"
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/health")
def health_check():
    return {"status": "ok"}
