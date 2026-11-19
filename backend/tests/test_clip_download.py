"""Unit and integration tests for single-clip download endpoint and byte-range requests."""

import os
from pathlib import Path
import pytest
from app.config import settings
from app.schemas import ClipResult, JobStatusResponse


class TestClipDownloadEndpoint:
    """Test GET /api/jobs/{job_id}/clips/{clip_id}/download."""

    def test_download_local_clip_returns_file_response_with_attachment_header(
        self, test_client, fake_redis, tmp_path
    ):
        """When clip file exists in local storage, endpoint returns attachment with filename."""
        job_id = "job_download_test"
        job_dir = Path(settings.STATIC_DIR) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        clip_file = job_dir / "clip_1.mp4"
        test_video_bytes = b"fake video bytes for testing download" * 10
        clip_file.write_bytes(test_video_bytes)

        status = JobStatusResponse(
            job_id=job_id,
            status="completed",
            stage="completed",
            progress=100,
            clips=[
                ClipResult(
                    clip_id="c1",
                    title="Viral Clip One",
                    virality_score=88,
                    duration=15.0,
                    video_url=f"/static/{job_id}/clip_1.mp4",
                )
            ],
        )
        fake_redis.set(f"job:{job_id}", status.model_dump_json())

        resp = test_client.get(f"/api/jobs/{job_id}/clips/c1/download")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "video/mp4"
        assert "attachment;" in resp.headers["content-disposition"]
        assert "c1.mp4" in resp.headers["content-disposition"]
        assert resp.content == test_video_bytes

    def test_download_supports_byte_range_requests(
        self, test_client, fake_redis
    ):
        """Byte-range requests return HTTP 206 Partial Content (required by Safari and iOS)."""
        job_id = "job_range_test"
        job_dir = Path(settings.STATIC_DIR) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        clip_file = job_dir / "clip_1.mp4"
        # 100 bytes
        payload = bytes([i % 256 for i in range(100)])
        clip_file.write_bytes(payload)

        status = JobStatusResponse(
            job_id=job_id,
            status="completed",
            stage="completed",
            progress=100,
            clips=[
                ClipResult(
                    clip_id="c_range",
                    title="Range Test Clip",
                    virality_score=75,
                    duration=10.0,
                    video_url=f"/static/{job_id}/clip_1.mp4",
                )
            ],
        )
        fake_redis.set(f"job:{job_id}", status.model_dump_json())

        # Request bytes 0-49
        headers = {"Range": "bytes=0-49"}
        resp = test_client.get(f"/api/jobs/{job_id}/clips/c_range/download", headers=headers)
        assert resp.status_code == 206
        assert resp.headers["content-range"] == "bytes 0-49/100"
        assert len(resp.content) == 50
        assert resp.content == payload[0:50]

    def test_download_nonexistent_clip_returns_404(
        self, test_client, fake_redis
    ):
        """Requesting a clip that is not part of the job returns 404."""
        job_id = "job_missing_clip"
        status = JobStatusResponse(
            job_id=job_id,
            status="completed",
            progress=100,
            clips=[
                ClipResult(
                    clip_id="c_exist",
                    title="Clip Exist",
                    virality_score=80,
                    duration=10.0,
                    video_url="",
                )
            ],
        )
        fake_redis.set(f"job:{job_id}", status.model_dump_json())

        resp = test_client.get(f"/api/jobs/{job_id}/clips/c_nonexistent/download")
        assert resp.status_code == 404
        assert "Clip not found" in resp.json()["detail"]

    def test_download_expired_job_returns_410(
        self, test_client, fake_redis
    ):
        """When job has expired, download endpoint returns HTTP 410 Gone."""
        job_id = "job_expired_test"
        status = JobStatusResponse(
            job_id=job_id,
            status="expired",
            stage="expired",
            progress=100,
            clips=[],
        )
        fake_redis.set(f"job:{job_id}", status.model_dump_json())

        resp = test_client.get(f"/api/jobs/{job_id}/clips/c1/download")
        assert resp.status_code == 410
        assert "expired" in resp.json()["detail"].lower()
