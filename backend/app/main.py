import io
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
from app.schemas import JobCreateRequest, JobStatusResponse
from app.worker import process_video_task

os.makedirs(settings.STATIC_DIR, exist_ok=True)

app = FastAPI(title="Clipper MVP API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)


def _slugify(text: str) -> str:
    """Convert text to safe lowercase filename slug."""
    slug = re.sub(r"[^\w\s-]", "", text).strip().lower()
    slug = re.sub(r"[-\s]+", "_", slug)
    return slug[:40] or "clip"


@app.post("/api/jobs", response_model=JobStatusResponse, status_code=status.HTTP_202_ACCEPTED)
def create_job(req: JobCreateRequest):
    job_id = str(uuid4())
    initial_status = JobStatusResponse(job_id=job_id, status="queued", progress=0)
    redis_client.set(f"job:{job_id}", initial_status.model_dump_json(), ex=86400)
    process_video_task.delay(job_id, req.model_dump(mode="json"))
    return initial_status


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    data = redis_client.get(f"job:{job_id}")
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse.model_validate_json(data)


@app.get("/api/jobs/{job_id}/download-all")
def download_all_clips(job_id: str):
    """Bundle all generated clips and captions into a single zip archive for batch download."""
    data = redis_client.get(f"job:{job_id}")
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
