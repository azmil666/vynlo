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

from app.config import settings
from app.schemas import (
    JobCreateRequest,
    JobErrorDetail,
    JobStatusResponse,
    PipelineErrorCode,
)
from app.services.sweeper import cleanup_stale_temp_dirs
from app.worker import process_video_task

logger = logging.getLogger(__name__)

try:
    os.makedirs(settings.STATIC_DIR, exist_ok=True)
except OSError:
    fallback_static = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))
    os.makedirs(fallback_static, exist_ok=True)
    settings.STATIC_DIR = fallback_static


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Sweep stale temp media directories on application startup
    try:
        cleaned = cleanup_stale_temp_dirs()
        if cleaned:
            logger.info("Cleaned %d stale temp directories on startup", len(cleaned))
    except Exception as e:
        logger.warning("Startup stale temp cleanup failed: %s", e)

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
    get_redis_client().set(f"job:{job_id}", initial_status.model_dump_json(), ex=86400)
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
        raise HTTPException(status_code=404, detail="Job not found")

    job_status = JobStatusResponse.model_validate_json(data)

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
                    ttl = 86400
                r.set(job_key, job_status.model_dump_json(), ex=ttl)
        except Exception as e:
            logger.warning("Error evaluating worker heartbeat for job %s: %s", job_id, e)

    return job_status


@app.get("/api/jobs/{job_id}/download-all")
def download_all_clips(job_id: str):
    """Bundle all generated clips and captions into a single zip archive for batch download."""
    data = get_redis_client().get(f"job:{job_id}")
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")

    job_status = JobStatusResponse.model_validate_json(data)
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
