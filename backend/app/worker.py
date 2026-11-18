from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import shutil
import subprocess
from typing import Any
from uuid import uuid4

from celery import Celery
from celery.exceptions import Retry, SoftTimeLimitExceeded
from celery.signals import worker_process_init, worker_ready
import cv2
import redis

from app.config import settings
from app.schemas import (
    ClipResult,
    JobErrorDetail,
    JobStatusResponse,
    JobWarning,
    PipelineError,
    PipelineErrorCode,
    map_exception_to_pipeline_error,
)
from app.services.downloader import download_media, get_video_metadata
from app.services.transcriber import transcribe_audio
from app.services.analyzer import find_viral_clips
from app.services.cropper import (
    _get_face_detector,
    compute_crop_filter,
    verify_yunet_model_present,
)
from app.services.captions import generate_ass_subtitles, render_clip_single_pass
from app.services.storage import upload_clip
from app.services.sweeper import cleanup_stale_temp_dirs
from app.services.cache import (
    acquire_single_flight_lock,
    get_cached_candidates,
    get_cached_transcript,
    release_single_flight_lock,
    set_cached_candidates,
    set_cached_transcript,
    wait_for_single_flight_completion,
)

logger = logging.getLogger(__name__)

celery_app = Celery("clipper_worker", broker=settings.REDIS_URL, backend=settings.REDIS_URL)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    broker_transport_options={"visibility_timeout": settings.broker_visibility_timeout},
    task_reject_on_worker_lost=False,
)


@worker_process_init.connect
def on_worker_process_init(**kwargs: Any) -> None:
    """Verify bundled AI/CV models at worker process initialization and fail fast if missing."""
    try:
        model_path = verify_yunet_model_present()
        logger.info("YuNet face detection model verified at worker startup: %s", model_path)
    except Exception as e:
        logger.critical("Fatal: YuNet model missing or invalid at worker startup: %s", e)
        raise

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


@celery_app.task(name="sweep_stale_temp_dirs_task")
def sweep_stale_temp_dirs_task() -> list[str]:
    """Celery task to clean up old temp job directories."""
    return cleanup_stale_temp_dirs()


@celery_app.task(
    name="process_video_task",
    bind=True,
    max_retries=2,
    soft_time_limit=settings.task_soft_time_limit,
    time_limit=settings.task_time_limit,
    acks_late=True,
)
def process_video_task(self, job_id: str, options: dict) -> None:
    current_stage = "queued"
    warnings_list: list[dict] = []
    r = get_redis_client()
    job_key = f"job:{job_id}"
    lock_key = f"lock:job:{job_id}"
    worker_task_id = (
        self.request.id
        if getattr(self, "request", None) and getattr(self.request, "id", None)
        else str(uuid4())
    )

    # 1. Idempotency guard: if job is already completed, do not duplicate work
    existing_raw = r.get(job_key)
    if existing_raw:
        try:
            existing_data = json.loads(existing_raw)
            if (
                existing_data.get("status") == "completed"
                or existing_data.get("stage") == "completed"
            ):
                logger.info(
                    "Job %s is already completed. Skipping redelivered task.",
                    job_id,
                )
                return
        except Exception:
            pass

    # 2. Redis distributed lock: prevent duplicate concurrent processing on redelivery
    acquired_lock = bool(
        r.set(lock_key, worker_task_id, nx=True, ex=settings.task_time_limit)
    )
    if not acquired_lock:
        lock_owner = r.get(lock_key)
        if lock_owner != worker_task_id:
            logger.warning(
                "Job %s is currently locked by another worker (%s). Skipping duplicate task.",
                job_id,
                lock_owner,
            )
            return

    def update_job_state(
        stage: str,
        progress: int,
        clips: list[dict] | None = None,
        error: Any | None = None,
        warnings: list[dict] | None = None,
    ) -> None:
        nonlocal current_stage
        current_stage = stage
        r_conn = get_redis_client()
        j_key = f"job:{job_id}"
        ttl = r_conn.ttl(j_key)
        if ttl is None or ttl <= 0:
            ttl = 86400

        current_clips: list[dict] = []
        current_error: Any = None
        current_warnings: list[dict] = []
        existing_status_raw = r_conn.get(j_key)
        if existing_status_raw:
            try:
                existing_data = json.loads(existing_status_raw)
                current_clips = existing_data.get("clips", [])
                current_error = existing_data.get("error")
                current_warnings = existing_data.get("warnings", [])
            except Exception:
                pass

        active_warnings = (
            warnings if warnings is not None else (warnings_list or current_warnings)
        )

        state = JobStatusResponse(
            job_id=job_id,
            stage=stage,
            status=stage,
            progress=progress,
            updated_at=datetime.now(timezone.utc).isoformat(),
            error=error if error is not None else current_error,
            warnings=active_warnings,
            clips=clips if clips is not None else current_clips,
        )
        r_conn.set(j_key, state.model_dump_json(), ex=ttl)

    work_dir = os.path.join(settings.TEMP_DIR, job_id)
    os.makedirs(work_dir, exist_ok=True)

    try:
        url = options["url"]
        max_clips = int(options.get("max_clips", 3))

        # 1. Resolve video_id via metadata with restricted fallback
        meta_duration: Optional[float] = None
        try:
            meta = get_video_metadata(url)
            video_id = meta["video_id"]
            meta_duration = meta.get("duration")
            if meta_duration is None:
                warnings_list.append({
                    "code": "MISSING_METADATA_DURATION",
                    "message": "Video metadata did not specify duration; duration limits will be enforced via ffprobe.",
                })
        except PipelineError:
            # Re-raise classified pipeline errors immediately (no silent continuation)
            raise
        except Exception as e:
            # Only for non-classified generic errors on direct media URLs: record warning,
            # use URL-hash ID, and post-download validation will strictly enforce ffprobe duration.
            logger.warning(
                "Metadata extraction failed for %s: %s. Using URL hash fallback and enforcing duration post-download.",
                url,
                e,
            )
            warnings_list.append({
                "code": "METADATA_FALLBACK_HASHED_ID",
                "message": "Metadata extraction unavailable; using hashed URL id and enforcing duration via ffprobe.",
            })
            url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
            video_id = f"url:{url_hash}"

        r = get_redis_client()

        # 2. Check Redis for cached transcript before download (preliminary check to skip audio extraction)
        cached_words, cached_val, _ = get_cached_transcript(r, video_id)
        has_cached_transcript = cached_words is not None

        # 3. Download media: skip audio extraction if transcript is already cached
        update_job_state("downloading", 10)
        video_path, audio_path, source_info, dl_warnings = download_media(
            url,
            work_dir,
            extract_audio=(not has_cached_transcript),
            meta_duration=meta_duration,
        )
        warnings_list.extend(dl_warnings)

        # 4. Transcribe or load cached transcript (with freshly probed duration validation)
        update_job_state("transcribing", 30)
        probed_dur = source_info.duration if source_info else None
        words, validation_dict, reject_reason = get_cached_transcript(
            r, video_id, probed_duration=probed_dur
        )

        if reject_reason and reject_reason != "cache_miss":
            warnings_list.append(
                {
                    "code": "TRANSCRIPT_CACHE_REJECTED",
                    "message": f"Cached transcript rejected ({reject_reason}); re-transcribing.",
                }
            )

        if words is None:
            # Single-flight lock for transcription to prevent duplicate computation
            lock_acquired = acquire_single_flight_lock(r, f"transcription:{video_id}", worker_task_id)
            if not lock_acquired:
                logger.info("Concurrent transcription in progress for %s. Waiting for completion...", video_id)
                def check_t():
                    w, _, _ = get_cached_transcript(r, video_id, probed_duration=probed_dur)
                    return w
                waited_words = wait_for_single_flight_completion(r, f"transcription:{video_id}", check_t)
                if waited_words is not None:
                    words = waited_words
                else:
                    lock_acquired = acquire_single_flight_lock(r, f"transcription:{video_id}", worker_task_id)

            if words is None:
                try:
                    if not audio_path:
                        audio_path = os.path.join(work_dir, "audio.wav")
                        cmd = [
                            "ffmpeg",
                            "-y",
                            "-i",
                            video_path,
                            "-vn",
                            "-acodec",
                            "pcm_s16le",
                            "-ar",
                            "16000",
                            "-ac",
                            "1",
                            audio_path,
                        ]
                        subprocess.run(
                            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                        )
                    words = transcribe_audio(
                        audio_path,
                        timeline_offset=source_info.timeline_offset if source_info else 0.0,
                        media_duration=source_info.duration if source_info else None,
                    )
                    validation_dict = (
                        words.validation.model_dump()
                        if hasattr(words, "validation") and hasattr(words.validation, "model_dump")
                        else {}
                    )
                    set_cached_transcript(
                        r,
                        video_id,
                        words,
                        media_duration=probed_dur,
                        validation_report=validation_dict,
                    )
                finally:
                    if lock_acquired:
                        release_single_flight_lock(r, f"transcription:{video_id}", worker_task_id)

        # 5. Viral analysis or load cached candidates
        update_job_state("analyzing", 50)
        candidates, cache_hit = get_cached_candidates(r, video_id, words, max_clips=max_clips)
        if not cache_hit:
            # Single-flight lock for viral analysis
            lock_acquired = acquire_single_flight_lock(r, f"analysis:{video_id}", worker_task_id)
            if not lock_acquired:
                logger.info("Concurrent analysis in progress for %s. Waiting for completion...", video_id)
                def check_c():
                    c_list, hit = get_cached_candidates(r, video_id, words, max_clips=max_clips)
                    return c_list if hit else None
                waited_candidates = wait_for_single_flight_completion(r, f"analysis:{video_id}", check_c)
                if waited_candidates is not None:
                    candidates = waited_candidates
                    cache_hit = True
                else:
                    lock_acquired = acquire_single_flight_lock(r, f"analysis:{video_id}", worker_task_id)

            if not cache_hit:
                try:
                    superset_target = max(max_clips, settings.ANALYSIS_SUPERSET_SIZE)
                    all_candidates = find_viral_clips(words, max_clips=superset_target)
                    set_cached_candidates(r, video_id, words, all_candidates)
                    candidates = all_candidates[:max_clips]
                finally:
                    if lock_acquired:
                        release_single_flight_lock(r, f"analysis:{video_id}", worker_task_id)

        if len(candidates) < max_clips:
            warnings_list.append(
                {
                    "code": "FEWER_CLIPS_THAN_REQUESTED",
                    "message": f"Only {len(candidates)} qualifying clips found (requested {max_clips}).",
                }
            )

        for clip in candidates:
            for w in clip.get("warnings", []):
                warnings_list.append(JobWarning(code="BOUNDARY_SNAP", message=str(w)))

        completed_clips: list[dict] = []
        total_candidates = len(candidates)

        # Determine face detector name and layout
        layout_mode = options.get("layout_mode", "speaker")
        detector_obj = _get_face_detector()
        if layout_mode == "split":
            detector_name = "split"
        elif isinstance(detector_obj, getattr(cv2, "FaceDetectorYN", tuple())):
            detector_name = "yunet"
        elif isinstance(detector_obj, getattr(cv2, "CascadeClassifier", tuple())):
            detector_name = "haar"
        else:
            detector_name = "none"

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
                mode=layout_mode,
                source_info=source_info,
            )
            generate_ass_subtitles(
                words,
                clip["start_time"],
                clip["end_time"],
                ass_path,
                preset=options.get("caption_preset", "clean_white"),
                safe_zone=options.get("platform_safe_zone", "shorts"),
                highlight_color=options.get("highlight_color", "&H00FFFFFF"),
                hook_title=clip.get("title"),
                show_hook_banner=options.get("show_hook_banner", True),
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

            is_fb = bool(
                clip.get("is_fallback", False)
                or clip.get("selection_method") == "heuristic"
            )
            sel_method = clip.get("selection_method", "heuristic" if is_fb else "llm")
            sel_reason = clip.get("selection_reason")
            if is_fb:
                warnings_list.append(
                    {
                        "code": "HEURISTIC_FALLBACK_CLIP",
                        "message": sel_reason or f"Clip {i} generated using heuristic transcript selection.",
                        "clip_index": i,
                    }
                )

            completed_clips.append(
                {
                    "clip_id": f"clip_{i}",
                    "title": clip["title"],
                    "virality_score": clip["virality_score"],
                    "duration": round(clip["end_time"] - clip["start_time"], 2),
                    "video_url": public_url,
                    "is_fallback": is_fb,
                    "selection_method": sel_method,
                    "selection_reason": sel_reason,
                    "description": clip.get("description", ""),
                    "hashtags": clip.get("hashtags", []),
                    "layout_used": layout_mode,
                    "detector": detector_name,
                    "caption_font": "Liberation Sans",
                    "render_ok": True,
                }
            )

        update_job_state("completed", 100, clips=completed_clips, warnings=warnings_list)

    except Retry:
        raise
    except SoftTimeLimitExceeded as e:
        logger.error(
            "Job %s exceeded soft time limit of %ds",
            job_id,
            settings.task_soft_time_limit,
        )
        timeout_err = JobErrorDetail(
            code=PipelineErrorCode.TIMEOUT.value,
            message=f"Job exceeded execution time limit ({settings.task_soft_time_limit}s).",
            retryable=True,
            stage=current_stage,
            detail=str(e),
        )
        update_job_state(
            stage="failed",
            progress=0,
            error=timeout_err,
            warnings=warnings_list,
        )
        raise
    except Exception as e:
        pipeline_err = map_exception_to_pipeline_error(e, current_stage=current_stage)
        logger.exception(
            "Task failed for job %s at stage '%s' [%s]: %s",
            job_id,
            pipeline_err.stage,
            pipeline_err.code.value,
            pipeline_err.user_message,
        )

        max_retries = getattr(self, "max_retries", 2)
        current_retries = (
            getattr(self.request, "retries", 0)
            if getattr(self, "request", None)
            else 0
        )

        # Retryable error policy with exponential backoff: only retry retryable errors
        if pipeline_err.retryable and current_retries < max_retries:
            countdown = int(5 * (2 ** current_retries))
            logger.info(
                "Retrying job %s (attempt %d/%d) in %ds due to retryable error: %s",
                job_id,
                current_retries + 1,
                max_retries,
                countdown,
                pipeline_err.user_message,
            )
            retry_warning = {
                "code": "TASK_RETRYING",
                "message": (
                    f"Transient error at stage '{current_stage}': {pipeline_err.user_message}. "
                    f"Retrying in {countdown}s (attempt {current_retries + 1}/{max_retries})."
                ),
            }
            update_job_state(
                stage=current_stage,
                progress=0,
                warnings=warnings_list + [retry_warning],
            )
            # Release lock before retrying so subsequent attempt can re-acquire
            if acquired_lock:
                try:
                    if r.get(lock_key) == worker_task_id:
                        r.delete(lock_key)
                        acquired_lock = False
                except Exception:
                    pass
            raise self.retry(exc=e, countdown=countdown)

        update_job_state(
            stage="failed",
            progress=0,
            error=pipeline_err.to_error_detail(),
            warnings=warnings_list,
        )
        raise
    finally:
        try:
            if acquired_lock:
                current_owner = r.get(lock_key)
                if current_owner == worker_task_id:
                    r.delete(lock_key)
        except Exception as e:
            logger.warning("Failed releasing lock for job %s: %s", job_id, e)
        shutil.rmtree(work_dir, ignore_errors=True)
