from datetime import datetime, timedelta, timezone
import json
import os
import shutil
import time
from unittest.mock import MagicMock, patch

from celery.exceptions import Retry, SoftTimeLimitExceeded
import pytest

from app.config import settings
from app.main import app
from app.schemas import JobStatusResponse, PipelineErrorCode, map_exception_to_pipeline_error
from app.services.downloader import get_video_metadata
from app.services.sweeper import cleanup_stale_temp_dirs
from app.worker import celery_app, process_video_task


def test_settings_time_budgets_and_limits():
    """Verify stage budgets, soft limit, hard limit, and broker visibility timeout math."""
    assert settings.MAX_VIDEO_SECONDS == 1200  # 20 minutes default
    assert settings.ASR_PROCESSING_RATIO == 0.5

    # Download(180) + ASR(1200*0.5=600) + Analysis(90) + Render(10*60=600) + Buffer(120) = 1590
    expected_soft = 180 + 600 + 90 + 600 + 120
    assert settings.task_soft_time_limit == expected_soft
    assert settings.task_time_limit == expected_soft + 60  # 1650
    assert settings.broker_visibility_timeout > settings.task_time_limit
    assert settings.broker_visibility_timeout >= 3600

    # Verify Celery conf
    conf = celery_app.conf
    assert conf.task_reject_on_worker_lost is False
    assert conf.broker_transport_options.get("visibility_timeout") == settings.broker_visibility_timeout


def test_downloader_enforces_max_video_seconds():
    """Verify downloader duration guard rejects videos exceeding MAX_VIDEO_SECONDS."""
    with patch("yt_dlp.YoutubeDL") as mock_ydl_cls:
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = {
            "id": "abc1234",
            "duration": settings.MAX_VIDEO_SECONDS + 5,
            "title": "Too Long Video",
        }
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

        with pytest.raises(ValueError) as exc_info:
            get_video_metadata("https://youtube.com/watch?v=abc1234")

        err_msg = str(exc_info.value)
        assert f"exceeds maximum supported duration of {settings.MAX_VIDEO_SECONDS}s" in err_msg

        mapped = map_exception_to_pipeline_error(exc_info.value, current_stage="downloading")
        assert mapped.code == PipelineErrorCode.VIDEO_TOO_LONG
        assert mapped.retryable is False


def test_simulated_soft_time_limit(fake_redis):
    """Verify SoftTimeLimitExceeded marks job as TIMEOUT (failed, retryable) and cleans up work_dir."""
    job_id = "test-soft-limit-job"
    fake_redis.set(
        f"job:{job_id}",
        JobStatusResponse(job_id=job_id, stage="transcribing", status="transcribing", progress=30).model_dump_json(),
    )

    work_dir = os.path.join(settings.TEMP_DIR, job_id)
    os.makedirs(work_dir, exist_ok=True)
    probe_file = os.path.join(work_dir, "temp.txt")
    with open(probe_file, "w") as f:
        f.write("data")

    # Simulate task running and encountering SoftTimeLimitExceeded
    with patch("app.worker.get_video_metadata", return_value={"video_id": "test:dummy", "duration": 60, "title": "Dummy"}), patch(
        "app.worker.download_media", side_effect=SoftTimeLimitExceeded()
    ):
        with pytest.raises(SoftTimeLimitExceeded):
            process_video_task(job_id, {"url": "https://youtu.be/dummy"})

    # Check job status in redis
    data = fake_redis.get(f"job:{job_id}")
    assert data is not None
    job = JobStatusResponse.model_validate_json(data)
    assert job.status == "failed"
    assert job.stage == "failed"
    assert job.error is not None
    assert job.error.code == PipelineErrorCode.TIMEOUT.value
    assert job.error.retryable is True
    assert not os.path.exists(probe_file)


def test_stale_heartbeat_worker_lost_detection(test_client, fake_redis):
    """Verify GET /api/jobs/{id} detects stale processing jobs and marks them WORKER_LOST."""
    job_id = "stale-worker-job"
    stale_time = (datetime.now(timezone.utc) - timedelta(seconds=settings.WORKER_HEARTBEAT_TIMEOUT_SECONDS + 30)).isoformat()

    fake_redis.set(
        f"job:{job_id}",
        JobStatusResponse(
            job_id=job_id,
            stage="transcribing",
            status="transcribing",
            progress=30,
            updated_at=stale_time,
        ).model_dump_json(),
    )

    resp = test_client.get(f"/api/jobs/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "failed"
    assert data["stage"] == "failed"
    assert data["error"]["code"] == PipelineErrorCode.WORKER_LOST.value
    assert data["error"]["retryable"] is True
    assert data["error"]["stage"] == "transcribing"

    # Verify persisted in Redis
    persisted = JobStatusResponse.model_validate_json(fake_redis.get(f"job:{job_id}"))
    assert persisted.status == "failed"
    assert persisted.error.code == PipelineErrorCode.WORKER_LOST.value


def test_active_heartbeat_not_marked_worker_lost(test_client, fake_redis):
    """Verify recent heartbeat preserves normal active status."""
    job_id = "fresh-worker-job"
    fresh_time = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()

    fake_redis.set(
        f"job:{job_id}",
        JobStatusResponse(
            job_id=job_id,
            stage="analyzing",
            status="analyzing",
            progress=50,
            updated_at=fresh_time,
        ).model_dump_json(),
    )

    resp = test_client.get(f"/api/jobs/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "analyzing"
    assert data["error"] is None


def test_completed_job_with_old_heartbeat_unchanged(test_client, fake_redis):
    """Verify completed jobs with old updated_at are not marked as WORKER_LOST."""
    job_id = "completed-old-job"
    old_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()

    fake_redis.set(
        f"job:{job_id}",
        JobStatusResponse(
            job_id=job_id,
            stage="completed",
            status="completed",
            progress=100,
            updated_at=old_time,
        ).model_dump_json(),
    )

    resp = test_client.get(f"/api/jobs/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"


def test_redelivery_guard_already_completed(fake_redis):
    """Verify redelivered task exits early if job is already completed."""
    job_id = "already-done-job"
    fake_redis.set(
        f"job:{job_id}",
        JobStatusResponse(
            job_id=job_id,
            stage="completed",
            status="completed",
            progress=100,
        ).model_dump_json(),
    )

    with patch("app.worker.get_video_metadata") as mock_meta, patch("app.worker.download_media") as mock_dl:
        process_video_task(job_id, {"url": "https://youtu.be/dummy"})
        mock_meta.assert_not_called()
        mock_dl.assert_not_called()


def test_redelivery_guard_concurrent_lock(fake_redis):
    """Verify redelivered task skips execution if lock is held by another worker."""
    job_id = "concurrent-locked-job"
    fake_redis.set(
        f"job:{job_id}",
        JobStatusResponse(
            job_id=job_id,
            stage="queued",
            status="queued",
            progress=0,
        ).model_dump_json(),
    )
    # Manually hold lock with another worker's task id
    fake_redis.set(f"lock:job:{job_id}", "other-worker-task-id", ex=300)

    with patch("app.worker.get_video_metadata") as mock_meta:
        process_video_task(job_id, {"url": "https://youtu.be/dummy"})
        mock_meta.assert_not_called()


def test_retry_policy_transient_error(fake_redis):
    """Verify transient error calls self.retry with exponential backoff and sets TASK_RETRYING warning."""
    job_id = "retryable-error-job"
    fake_redis.set(
        f"job:{job_id}",
        JobStatusResponse(
            job_id=job_id,
            stage="downloading",
            status="downloading",
            progress=10,
        ).model_dump_json(),
    )

    with patch.object(process_video_task, "retry", side_effect=Retry("Retrying...")) as mock_retry:
        with patch.object(process_video_task.request, "retries", 0), patch.object(
            process_video_task.request, "id", "task-id-123"
        ):
            with patch("app.worker.get_video_metadata", return_value={"video_id": "test:video123"}), patch(
                "app.worker.download_media", side_effect=ConnectionError("Temporary network disconnect")
            ):
                with pytest.raises(Retry):
                    process_video_task(job_id, {"url": "https://youtu.be/dummy"})

                mock_retry.assert_called_once()
                _, kwargs = mock_retry.call_args
                assert kwargs["countdown"] == 5  # 5 * (2 ** 0)

    # Verify warning added
    data = JobStatusResponse.model_validate_json(fake_redis.get(f"job:{job_id}"))
    assert any(w.code == "TASK_RETRYING" for w in data.warnings)
    # Verify lock was released prior to retry
    assert fake_redis.get(f"lock:job:{job_id}") is None


def test_retry_policy_non_retryable_error(fake_redis):
    """Verify permanent/non-retryable errors are not retried."""
    job_id = "non-retryable-job"
    fake_redis.set(
        f"job:{job_id}",
        JobStatusResponse(
            job_id=job_id,
            stage="queued",
            status="queued",
            progress=0,
        ).model_dump_json(),
    )

    with patch.object(process_video_task, "retry") as mock_retry:
        with patch.object(process_video_task.request, "retries", 0), patch.object(
            process_video_task.request, "id", "task-id-456"
        ):
            # A non-retryable error (e.g. invalid URL)
            with patch("app.worker.get_video_metadata", side_effect=ValueError("Invalid YouTube URL")):
                with pytest.raises(ValueError):
                    process_video_task(job_id, {"url": "invalid://url"})

                mock_retry.assert_not_called()

    data = JobStatusResponse.model_validate_json(fake_redis.get(f"job:{job_id}"))
    assert data.status == "failed"
    assert data.error.code == PipelineErrorCode.INVALID_URL.value
    assert data.error.retryable is False


def test_sweeper_stale_directories(tmp_path):
    """Verify cleanup_stale_temp_dirs removes directories older than threshold."""
    base_dir = tmp_path / "temp_clipper"
    base_dir.mkdir()

    # Create stale directory
    stale_dir = base_dir / "job_stale"
    stale_dir.mkdir()
    (stale_dir / "test.mp4").write_text("dummy")

    # Create fresh directory
    fresh_dir = base_dir / "job_fresh"
    fresh_dir.mkdir()
    (fresh_dir / "test.mp4").write_text("dummy")

    # Create a non-directory file
    regular_file = base_dir / "keep_file.txt"
    regular_file.write_text("not a dir")

    # Set stale_dir mtime to 5 hours ago
    past_mtime = time.time() - (5 * 3600)
    os.utime(stale_dir, (past_mtime, past_mtime))

    cleaned = cleanup_stale_temp_dirs(base_dir=str(base_dir), max_age_hours=4.0)

    assert str(stale_dir) in cleaned
    assert not os.path.exists(stale_dir)
    assert os.path.exists(fresh_dir)
    assert os.path.exists(regular_file)
