import os
from uuid import uuid4

import redis
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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


@app.get("/health")
def health_check():
    return {"status": "ok"}
