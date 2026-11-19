import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import time
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, settings, validate_retention_config
from app.schemas import ClipResult, JobStatusResponse, PipelineErrorCode
from app.services.sweeper import cleanup_stale_static_files, cleanup_stale_temp_dirs
from scripts.configure_s3_lifecycle import apply_s3_lifecycle_policy


class TestConfigConsistencyValidator:
    """Validate consistency of retention windows across jobs, static files, temp scratch, and caches."""

    def test_default_settings_pass_validation(self):
        """Default application settings satisfy all documented lifetime relationships."""
        validate_retention_config(settings)

    def test_job_ttl_too_short_raises_error(self):
        """Job record TTL must be at least 1 hour (3600s) to allow reasonable user return."""
        cfg = Settings(JOB_RECORD_TTL_SECONDS=1800)
        with pytest.raises(ValueError, match="must be at least 3600s"):
            validate_retention_config(cfg)

    def test_static_retention_shorter_than_job_ttl_raises_error(self):
        """Clip files must live at least as long as the job record to prevent dangling clip references."""
        # Job record 24h (86,400s), but static retention only 12h (43,200s)
        cfg = Settings(JOB_RECORD_TTL_SECONDS=86400, STATIC_RETENTION_HOURS=12.0)
        with pytest.raises(ValueError, match="must live at least as long as the job record"):
            validate_retention_config(cfg)

    def test_temp_retention_shorter_than_task_budget_raises_error(self):
        """Temp scratch directories must not be swept before the task time limit expires."""
        # Cleanup hours 0.05h = 180s < task_time_limit (~1000s+)
        cfg = Settings(TEMP_DIR_CLEANUP_HOURS=0.05)
        with pytest.raises(ValueError, match="risking deletion of active scratch files"):
            validate_retention_config(cfg)

    def test_s3_presigned_expiry_shorter_than_job_ttl_raises_error(self):
        """In S3 mode with presigned URLs, presigned URLs must not expire before the job record."""
        cfg = Settings(
            STORAGE_BACKEND="s3",
            S3_URL_STYLE="presigned",
            JOB_RECORD_TTL_SECONDS=86400,
            S3_PRESIGNED_EXPIRY_SECONDS=3600,  # 1 hour < 24 hours
            STATIC_RETENTION_HOURS=48.0,
        )
        with pytest.raises(ValueError, match="S3_PRESIGNED_EXPIRY_SECONDS .* must be >= JOB_RECORD_TTL_SECONDS"):
            validate_retention_config(cfg)

    def test_s3_retention_days_shorter_than_job_ttl_raises_error(self):
        """In S3 mode, S3 bucket retention days must cover the job record TTL."""
        cfg = Settings(
            STORAGE_BACKEND="s3",
            JOB_RECORD_TTL_SECONDS=86400 * 2,  # 2 days
            S3_RETENTION_DAYS=1,  # 1 day < 2 days
            STATIC_RETENTION_HOURS=72.0,
            S3_PRESIGNED_EXPIRY_SECONDS=86400 * 3,
        )
        with pytest.raises(ValueError, match="S3_RETENTION_DAYS .* must be >= JOB_RECORD_TTL_SECONDS"):
            validate_retention_config(cfg)

    def test_caches_independent_of_job_ttl(self):
        """Transcript and candidate caches can have independent TTLs without violating retention consistency."""
        cfg = Settings(
            JOB_RECORD_TTL_SECONDS=86400,
            STATIC_RETENTION_HOURS=48.0,
            CACHE_TRANSCRIPT_TTL_SECONDS=3600,  # 1 hour cache
            CACHE_CANDIDATES_TTL_SECONDS=86400 * 30,  # 30 day cache
        )
        # Should validate without error
        validate_retention_config(cfg)


class TestSweeperAgeAndOrphanLogic:
    """Tests sweeping logic for stale temp directories, local static files, and orphaned folders."""

    def test_temp_sweeper_removes_old_scratch_dirs_by_age(self, tmp_path):
        """Dirs older than max_age_hours are swept from temp media."""
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()

        stale_dir = temp_dir / "job_stale"
        stale_dir.mkdir()
        recent_dir = temp_dir / "job_recent"
        recent_dir.mkdir()

        now = time.time()
        # Set stale directory mtime to 6 hours ago (retention: 4 hours)
        os.utime(str(stale_dir), (now - 21600, now - 21600))
        # Set recent directory mtime to 30 mins ago
        os.utime(str(recent_dir), (now - 1800, now - 1800))

        removed = cleanup_stale_temp_dirs(base_dir=str(temp_dir), max_age_hours=4.0)

        assert str(stale_dir) in removed
        assert not stale_dir.exists()
        assert recent_dir.exists()

    def test_temp_sweeper_removes_finished_job_scratch_older_than_grace_period(self, tmp_path, fake_redis):
        """Scratch directory for a completed or failed job older than 1h grace period is pruned."""
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()

        job_dir = temp_dir / "job_finished"
        job_dir.mkdir()

        now = time.time()
        # 1.5 hours old (> 1.0h grace period, but < 4h max retention)
        os.utime(str(job_dir), (now - 5400, now - 5400))

        # Mark job completed in Redis
        status = JobStatusResponse(job_id="job_finished", status="completed", progress=100)
        fake_redis.set("job:job_finished", status.model_dump_json())

        removed = cleanup_stale_temp_dirs(
            base_dir=str(temp_dir),
            max_age_hours=4.0,
            redis_client=fake_redis,
            min_orphan_age_seconds=3600.0,
        )

        assert str(job_dir) in removed
        assert not job_dir.exists()

    def test_temp_sweeper_preserves_active_job_scratch(self, tmp_path, fake_redis):
        """Scratch directory for an active (rendering/transcribing) job is preserved even past grace period."""
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()

        job_dir = temp_dir / "job_active"
        job_dir.mkdir()

        now = time.time()
        # 2 hours old
        os.utime(str(job_dir), (now - 7200, now - 7200))

        # Mark job rendering in Redis
        status = JobStatusResponse(job_id="job_active", status="rendering", progress=50)
        fake_redis.set("job:job_active", status.model_dump_json())

        removed = cleanup_stale_temp_dirs(
            base_dir=str(temp_dir),
            max_age_hours=4.0,
            redis_client=fake_redis,
            min_orphan_age_seconds=3600.0,
        )

        assert str(job_dir) not in removed
        assert job_dir.exists()

    def test_static_sweeper_removes_old_files_and_dirs_by_age(self, tmp_path):
        """Files and directories exceeding STATIC_RETENTION_HOURS are swept."""
        static_dir = tmp_path / "static"
        static_dir.mkdir()

        stale_file = static_dir / "old_clip.mp4"
        stale_file.write_bytes(b"dummy mp4")

        stale_job_dir = static_dir / "job_old"
        stale_job_dir.mkdir()
        (stale_job_dir / "clip_1.mp4").write_bytes(b"clip content")

        recent_job_dir = static_dir / "job_recent"
        recent_job_dir.mkdir()
        (recent_job_dir / "clip_1.mp4").write_bytes(b"recent content")

        now = time.time()
        # Stale items: 50 hours old (> 48.0h retention)
        os.utime(str(stale_file), (now - 180000, now - 180000))
        os.utime(str(stale_job_dir), (now - 180000, now - 180000))
        # Recent items: 10 hours old (< 48.0h retention)
        os.utime(str(recent_job_dir), (now - 36000, now - 36000))

        removed = cleanup_stale_static_files(static_dir=str(static_dir), max_age_hours=48.0)

        assert str(stale_file) in removed
        assert str(stale_job_dir) in removed
        assert not stale_file.exists()
        assert not stale_job_dir.exists()
        assert recent_job_dir.exists()

    def test_static_sweeper_removes_orphaned_job_dir_when_record_missing_from_redis(self, tmp_path, fake_redis):
        """Job directory whose job record has expired/disappeared from Redis is swept if older than grace period."""
        static_dir = tmp_path / "static"
        static_dir.mkdir()

        orphaned_dir = static_dir / "job_orphaned"
        orphaned_dir.mkdir()
        (orphaned_dir / "clip_1.mp4").write_bytes(b"orphaned clip")

        now = time.time()
        # 3 hours old (> 1h grace period)
        os.utime(str(orphaned_dir), (now - 10800, now - 10800))

        # Do NOT set job_orphaned in fake_redis (it has expired / gone)
        removed = cleanup_stale_static_files(
            static_dir=str(static_dir),
            max_age_hours=48.0,
            redis_client=fake_redis,
            min_orphan_age_seconds=3600.0,
        )

        assert str(orphaned_dir) in removed
        assert not orphaned_dir.exists()

    def test_static_sweeper_preserves_job_dir_when_record_present_in_redis(self, tmp_path, fake_redis):
        """Job directory whose record is still present in Redis is preserved within retention window."""
        static_dir = tmp_path / "static"
        static_dir.mkdir()

        active_dir = static_dir / "job_active"
        active_dir.mkdir()
        (active_dir / "clip_1.mp4").write_bytes(b"active clip")

        now = time.time()
        # 3 hours old
        os.utime(str(active_dir), (now - 10800, now - 10800))

        # Set job record in Redis
        status = JobStatusResponse(job_id="job_active", status="completed", progress=100)
        fake_redis.set("job:job_active", status.model_dump_json())

        removed = cleanup_stale_static_files(
            static_dir=str(static_dir),
            max_age_hours=48.0,
            redis_client=fake_redis,
            min_orphan_age_seconds=3600.0,
        )

        assert str(active_dir) not in removed
        assert active_dir.exists()

    def test_static_sweeper_preserves_recent_dir_during_grace_period_even_without_redis_record(self, tmp_path, fake_redis):
        """Directory created very recently (< 1h grace period) is preserved even if Redis record check runs early."""
        static_dir = tmp_path / "static"
        static_dir.mkdir()

        brand_new_dir = static_dir / "job_new"
        brand_new_dir.mkdir()

        now = time.time()
        # 10 minutes old (< 1h grace period)
        os.utime(str(brand_new_dir), (now - 600, now - 600))

        removed = cleanup_stale_static_files(
            static_dir=str(static_dir),
            max_age_hours=48.0,
            redis_client=fake_redis,
            min_orphan_age_seconds=3600.0,
        )

        assert str(brand_new_dir) not in removed
        assert brand_new_dir.exists()


class TestExpiredJobResponse:
    """Validate API behavior when job records expire but files remain or vice versa."""

    def test_job_record_expired_but_files_remain_returns_expired_state(self, test_client, fake_redis, tmp_path):
        """When job record expired from Redis but static directory still exists, API returns expired status."""
        job_id = "job_rec_expired_files_exist"
        job_static_dir = Path(settings.STATIC_DIR) / job_id
        job_static_dir.mkdir(parents=True, exist_ok=True)
        (job_static_dir / "clip_1.mp4").write_bytes(b"clip content")

        # fake_redis has NO key job:job_rec_expired_files_exist
        resp = test_client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id
        assert data["status"] == "expired"
        assert data["stage"] == "expired"
        assert data["clips"] == []  # No dangling URLs
        assert data["error"]["code"] == "EXPIRED"

    def test_job_record_exists_but_files_expired_returns_expired_state_and_no_dangling_urls(self, test_client, fake_redis):
        """When job record in Redis is completed but storage files were removed, API returns expired state."""
        job_id = "job_rec_exists_files_gone"

        # Record in Redis with completed status and clips pointing to static path
        clip = ClipResult(
            clip_id="c1",
            title="Clip 1",
            virality_score=85,
            duration=30.0,
            video_url=f"http://localhost:8000/static/{job_id}/clip_1.mp4",
        )
        two_hours_ago = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        status = JobStatusResponse(
            job_id=job_id,
            status="completed",
            stage="completed",
            progress=100,
            updated_at=two_hours_ago,
            clips=[clip],
        )
        fake_redis.set(f"job:{job_id}", status.model_dump_json())

        # Ensure no static files exist on disk for this job
        job_static_dir = Path(settings.STATIC_DIR) / job_id
        if job_static_dir.exists():
            shutil.rmtree(job_static_dir)

        resp = test_client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id
        assert data["status"] == "expired"
        assert data["stage"] == "expired"
        assert data["clips"] == []  # Prevent dangling URLs
        assert data["error"]["code"] == "EXPIRED"

    def test_download_all_returns_410_when_job_expired_in_redis_with_files_remaining(self, test_client, fake_redis):
        """download-all returns HTTP 410 Gone when job record has expired."""
        job_id = "job_download_expired_rec"
        job_static_dir = Path(settings.STATIC_DIR) / job_id
        job_static_dir.mkdir(parents=True, exist_ok=True)
        (job_static_dir / "clip_1.mp4").write_bytes(b"content")

        resp = test_client.get(f"/api/jobs/{job_id}/download-all")
        assert resp.status_code == 410
        assert "expired" in resp.json()["detail"].lower()

    def test_download_all_returns_410_when_job_record_exists_but_files_missing(self, test_client, fake_redis):
        """download-all returns HTTP 410 Gone when clip files have expired from storage."""
        job_id = "job_download_files_missing"
        clip = ClipResult(
            clip_id="c1",
            title="Clip 1",
            virality_score=85,
            duration=30.0,
            video_url=f"http://localhost:8000/static/{job_id}/clip_1.mp4",
        )
        two_hours_ago = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        status = JobStatusResponse(
            job_id=job_id,
            status="completed",
            stage="completed",
            progress=100,
            updated_at=two_hours_ago,
            clips=[clip],
        )
        fake_redis.set(f"job:{job_id}", status.model_dump_json())

        job_static_dir = Path(settings.STATIC_DIR) / job_id
        if job_static_dir.exists():
            shutil.rmtree(job_static_dir)

        resp = test_client.get(f"/api/jobs/{job_id}/download-all")
        assert resp.status_code == 410
        assert "expired" in resp.json()["detail"].lower()

    def test_non_existent_job_returns_404(self, test_client, fake_redis):
        """Job with no Redis record and no storage files returns standard 404 Not Found."""
        resp = test_client.get("/api/jobs/completely_unknown_job_id")
        assert resp.status_code == 404


class TestS3LifecycleScript:
    """Validate S3 lifecycle configuration policy builder."""

    def test_apply_s3_lifecycle_policy(self):
        mock_client = MagicMock()
        apply_s3_lifecycle_policy(
            s3_client=mock_client,
            bucket_name="test-bucket",
            retention_days=7,
            prefix="clips/",
        )

        mock_client.put_bucket_lifecycle_configuration.assert_called_once()
        call_kwargs = mock_client.put_bucket_lifecycle_configuration.call_args[1]
        assert call_kwargs["Bucket"] == "test-bucket"
        rules = call_kwargs["LifecycleConfiguration"]["Rules"]
        assert len(rules) == 1
        rule = rules[0]
        assert rule["Status"] == "Enabled"
        assert rule["Expiration"]["Days"] == 7
        assert rule["Filter"]["Prefix"] == "clips/"
