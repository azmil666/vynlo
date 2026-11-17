import json
import logging
import os
import shutil

from celery import Celery
import redis

from app.config import settings
from app.schemas import JobStatusResponse
from app.services.downloader import download_media
from app.services.transcriber import transcribe_audio
from app.services.analyzer import find_viral_clips
from app.services.cropper import compute_crop_filter
from app.services.captions import generate_ass_subtitles, render_clip_single_pass
from app.services.storage import upload_clip

logger = logging.getLogger(__name__)

celery_app = Celery("clipper_worker", broker=settings.REDIS_URL, backend=settings.REDIS_URL)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)


@celery_app.task(
    name="process_video_task",
    bind=True,
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    max_retries=2,
    soft_time_limit=600,
    time_limit=660,
    acks_late=True,
)
def process_video_task(self, job_id: str, options: dict) -> None:
    def update_job_state(
        status: str,
        progress: int,
        clips: list[dict] | None = None,
        error: str | None = None,
    ) -> None:
        job_key = f"job:{job_id}"
        ttl = redis_client.ttl(job_key)
        if ttl is None or ttl <= 0:
            ttl = 86400

        current_clips: list[dict] = []
        current_error: str | None = None
        existing_raw = redis_client.get(job_key)
        if existing_raw:
            try:
                existing_data = json.loads(existing_raw)
                current_clips = existing_data.get("clips", [])
                current_error = existing_data.get("error")
            except Exception:
                pass

        state = JobStatusResponse(
            job_id=job_id,
            status=status,
            progress=progress,
            error=error if error is not None else current_error,
            clips=clips if clips is not None else current_clips,
        )
        redis_client.set(job_key, state.model_dump_json(), ex=ttl)

    work_dir = os.path.join(settings.TEMP_DIR, job_id)
    os.makedirs(work_dir, exist_ok=True)

    try:
        update_job_state("downloading", 10)
        video_path, audio_path = download_media(options["url"], work_dir)

        update_job_state("transcribing", 30)
        words = transcribe_audio(audio_path)

        update_job_state("analyzing", 50)
        candidates = find_viral_clips(words, max_clips=options.get("max_clips", 3))

        completed_clips: list[dict] = []
        total_candidates = len(candidates)

        for i, clip in enumerate(candidates, start=1):
            progress_step = 30 / total_candidates if total_candidates > 0 else 30
            current_progress = int(60 + (i - 1) * progress_step)
            update_job_state("rendering", current_progress)

            ass_path = os.path.join(work_dir, f"clip_{i}.ass")
            final_clip_path = os.path.join(work_dir, f"final_clip_{i}.mp4")

            crop_filter = compute_crop_filter(
                video_path,
                clip["start_time"],
                clip["end_time"],
                mode=options.get("layout_mode", "speaker"),
            )
            generate_ass_subtitles(
                words,
                clip["start_time"],
                clip["end_time"],
                ass_path,
                preset=options.get("caption_preset", "clean_white"),
                safe_zone=options.get("platform_safe_zone", "shorts"),
                highlight_color=options.get("highlight_color", "&H00FFFFFF"),
            )
            render_clip_single_pass(
                video_path=video_path,
                start_time=clip["start_time"],
                end_time=clip["end_time"],
                crop_filter=crop_filter,
                ass_path=ass_path,
                output_path=final_clip_path,
            )
            public_url = upload_clip(final_clip_path, f"{job_id}/clip_{i}.mp4")

            completed_clips.append(
                {
                    "clip_id": f"clip_{i}",
                    "title": clip["title"],
                    "virality_score": clip["virality_score"],
                    "duration": round(clip["end_time"] - clip["start_time"], 2),
                    "video_url": public_url,
                }
            )

        update_job_state("completed", 100, clips=completed_clips)

    except Exception as e:
        logger.exception("Task failed for job %s: %s", job_id, e)
        update_job_state("failed", 0, error=str(e))
        raise
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
