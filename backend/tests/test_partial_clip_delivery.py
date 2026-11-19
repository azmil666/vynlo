import io
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import zipfile
import pytest

from app.schemas import (
    ClipResult,
    JobStatusResponse,
    PipelineError,
    PipelineErrorCode,
    SourceInfo,
)
from app.worker import process_video_task
from app.config import settings


@pytest.fixture
def mock_pipeline_prerequisites():
    """Sets up standard mocks for download, audio extraction, and transcription."""
    mock_meta = {
        "title": "Test Multi Clip",
        "duration": 120.0,
        "width": 1920,
        "height": 1080,
    }
    mock_source = SourceInfo(
        width=1920,
        height=1080,
        duration=120.0,
        has_audio=True,
    )
    mock_words = [
        {"word": f"word{i}", "start": float(i), "end": float(i) + 0.5, "probability": 0.99}
        for i in range(100)
    ]
    mock_candidates = [
        {
            "title": f"Clip Title {i}",
            "start_time": float(i * 20),
            "end_time": float(i * 20 + 15),
            "virality_score": 80 + i,
            "description": f"Description {i}",
            "hashtags": ["#viral"],
            "selection_method": "llm",
        }
        for i in range(1, 4)
    ]

    return mock_meta, mock_source, mock_words, mock_candidates


class TestPartialClipDeliveryAndFailureIsolation:
    """Verify that failures during render/upload for one clip do not abort the job."""

    def test_render_failure_at_clip_2_of_3_delivers_partial_status_and_successful_clips(
        self, fake_redis, mock_pipeline_prerequisites
    ):
        job_id = "job_render_fail_clip_2"
        mock_meta, mock_source, mock_words, mock_candidates = mock_pipeline_prerequisites

        def mock_render(video_path, start_time, end_time, crop_filter, ass_path, output_path):
            if "final_clip_2.mp4" in output_path:
                raise PipelineError(
                    code=PipelineErrorCode.RENDER_FAILED,
                    stage="rendering",
                    user_message="FFmpeg process failed encoding clip 2",
                    retryable=False,
                )
            # Create a mock output file
            Path(output_path).touch()
            return output_path

        def mock_upload(file_path, key):
            return f"http://storage/{key}"

        with patch("app.worker.get_video_metadata", return_value=mock_meta), \
             patch("app.worker.download_media", return_value=("/tmp/in.mp4", "/tmp/in.wav", mock_source, [])), \
             patch("app.worker.transcribe_audio", return_value=mock_words), \
             patch("app.worker.find_viral_clips", return_value=mock_candidates), \
             patch("app.services.cropper.sample_faces_sequentially", return_value=[]), \
             patch("app.worker.generate_ass_subtitles", return_value="/tmp/clip.ass"), \
             patch("app.worker.render_clip_single_pass", side_effect=mock_render), \
             patch("app.worker.upload_clip", side_effect=mock_upload), \
             patch("cv2.VideoCapture") as mock_vc:

            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: 1920 if prop == 3 else 1080
            mock_vc.return_value = mock_cap

            # Run process_video_task: should NOT raise an exception
            process_video_task(job_id, {"url": "https://youtu.be/dummy", "max_clips": 3})

        raw = fake_redis.get(f"job:{job_id}")
        assert raw is not None
        job_data = json.loads(raw)

        # Status must be completed_with_warnings (or partial)
        assert job_data["status"] == "completed_with_warnings"
        assert job_data["stage"] == "completed_with_warnings"
        assert job_data["progress"] == 100
        assert len(job_data["clips"]) == 3

        c1, c2, c3 = job_data["clips"]

        # Clip 1 succeeded
        assert c1["clip_id"] == "clip_1"
        assert c1["ok"] is True
        assert c1["render_ok"] is True
        assert c1["video_url"] == f"http://storage/{job_id}/clip_1.mp4"
        assert c1["error_code"] is None

        # Clip 2 failed
        assert c2["clip_id"] == "clip_2"
        assert c2["ok"] is False
        assert c2["render_ok"] is False
        assert c2["video_url"] == ""
        assert c2["error_code"] == "RENDER_FAILED"
        assert "FFmpeg process failed" in c2["error_message"]

        # Clip 3 succeeded
        assert c3["clip_id"] == "clip_3"
        assert c3["ok"] is True
        assert c3["render_ok"] is True
        assert c3["video_url"] == f"http://storage/{job_id}/clip_3.mp4"
        assert c3["error_code"] is None

        # Warning recorded in job warnings list
        clip_2_warnings = [w for w in job_data["warnings"] if w.get("clip_index") == 2]
        assert len(clip_2_warnings) >= 1
        assert any(w["code"] == "RENDER_FAILED" for w in clip_2_warnings)

    def test_upload_failure_at_clip_2_of_3_delivers_partial_status_and_successful_clips(
        self, fake_redis, mock_pipeline_prerequisites
    ):
        job_id = "job_upload_fail_clip_2"
        mock_meta, mock_source, mock_words, mock_candidates = mock_pipeline_prerequisites

        def mock_upload(file_path, key):
            if "clip_2.mp4" in key:
                raise PipelineError(
                    code=PipelineErrorCode.UPLOAD_FAILED,
                    stage="uploading",
                    user_message="S3 head_object verification failed for clip 2",
                    retryable=False,
                )
            return f"http://storage/{key}"

        with patch("app.worker.get_video_metadata", return_value=mock_meta), \
             patch("app.worker.download_media", return_value=("/tmp/in.mp4", "/tmp/in.wav", mock_source, [])), \
             patch("app.worker.transcribe_audio", return_value=mock_words), \
             patch("app.worker.find_viral_clips", return_value=mock_candidates), \
             patch("app.services.cropper.sample_faces_sequentially", return_value=[]), \
             patch("app.worker.generate_ass_subtitles", return_value="/tmp/clip.ass"), \
             patch("app.worker.render_clip_single_pass", return_value="/tmp/final.mp4"), \
             patch("app.worker.upload_clip", side_effect=mock_upload), \
             patch("cv2.VideoCapture") as mock_vc:

            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: 1920 if prop == 3 else 1080
            mock_vc.return_value = mock_cap

            process_video_task(job_id, {"url": "https://youtu.be/dummy", "max_clips": 3})

        raw = fake_redis.get(f"job:{job_id}")
        assert raw is not None
        job_data = json.loads(raw)

        assert job_data["status"] == "completed_with_warnings"
        assert len(job_data["clips"]) == 3

        c1, c2, c3 = job_data["clips"]
        assert c1["ok"] is True
        assert c1["video_url"] == f"http://storage/{job_id}/clip_1.mp4"

        assert c2["ok"] is False
        assert c2["error_code"] == "UPLOAD_FAILED"
        assert c2["video_url"] == ""

        assert c3["ok"] is True
        assert c3["video_url"] == f"http://storage/{job_id}/clip_3.mp4"

    def test_all_clips_failing_sets_job_to_failed(
        self, fake_redis, mock_pipeline_prerequisites
    ):
        job_id = "job_all_clips_failed"
        mock_meta, mock_source, mock_words, mock_candidates = mock_pipeline_prerequisites

        def mock_render_fail(*args, **kwargs):
            raise PipelineError(
                code=PipelineErrorCode.RENDER_FAILED,
                stage="rendering",
                user_message="Encoder initialization error",
            )

        with patch("app.worker.get_video_metadata", return_value=mock_meta), \
             patch("app.worker.download_media", return_value=("/tmp/in.mp4", "/tmp/in.wav", mock_source, [])), \
             patch("app.worker.transcribe_audio", return_value=mock_words), \
             patch("app.worker.find_viral_clips", return_value=mock_candidates), \
             patch("app.services.cropper.sample_faces_sequentially", return_value=[]), \
             patch("app.worker.generate_ass_subtitles", return_value="/tmp/clip.ass"), \
             patch("app.worker.render_clip_single_pass", side_effect=mock_render_fail), \
             patch("cv2.VideoCapture") as mock_vc:

            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: 1920 if prop == 3 else 1080
            mock_vc.return_value = mock_cap

            process_video_task(job_id, {"url": "https://youtu.be/dummy", "max_clips": 3})

        raw = fake_redis.get(f"job:{job_id}")
        assert raw is not None
        job_data = json.loads(raw)

        # None succeeded -> failed
        assert job_data["status"] == "failed"
        assert job_data["stage"] == "failed"
        assert job_data["error"]["code"] == "RENDER_FAILED"
        assert all(c["ok"] is False for c in job_data["clips"])

    def test_all_clips_succeeding_sets_job_to_completed(
        self, fake_redis, mock_pipeline_prerequisites
    ):
        job_id = "job_all_clips_ok"
        mock_meta, mock_source, mock_words, mock_candidates = mock_pipeline_prerequisites

        with patch("app.worker.get_video_metadata", return_value=mock_meta), \
             patch("app.worker.download_media", return_value=("/tmp/in.mp4", "/tmp/in.wav", mock_source, [])), \
             patch("app.worker.transcribe_audio", return_value=mock_words), \
             patch("app.worker.find_viral_clips", return_value=mock_candidates), \
             patch("app.services.cropper.sample_faces_sequentially", return_value=[]), \
             patch("app.worker.generate_ass_subtitles", return_value="/tmp/clip.ass"), \
             patch("app.worker.render_clip_single_pass", return_value="/tmp/final.mp4"), \
             patch("app.worker.upload_clip", side_effect=lambda p, k: f"http://storage/{k}"), \
             patch("cv2.VideoCapture") as mock_vc:

            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: 1920 if prop == 3 else 1080
            mock_vc.return_value = mock_cap

            process_video_task(job_id, {"url": "https://youtu.be/dummy", "max_clips": 3})

        raw = fake_redis.get(f"job:{job_id}")
        assert raw is not None
        job_data = json.loads(raw)

        assert job_data["status"] == "completed"
        assert job_data["stage"] == "completed"
        assert len(job_data["clips"]) == 3
        assert all(c["ok"] is True for c in job_data["clips"])

    def test_idempotent_clip_retry_preserves_previously_completed_clips(
        self, fake_redis, mock_pipeline_prerequisites, tmp_path
    ):
        job_id = "job_idempotent_retry"
        mock_meta, mock_source, mock_words, mock_candidates = mock_pipeline_prerequisites

        # Simulate prior attempt where clip 1 succeeded
        prior_clip_1 = {
            "clip_id": "clip_1",
            "title": "Clip 1",
            "virality_score": 88,
            "duration": 15.0,
            "video_url": f"http://localhost:8000/static/{job_id}/clip_1.mp4",
            "ok": True,
            "render_ok": True,
            "error_code": None,
            "warnings": [],
        }
        prior_status = JobStatusResponse(
            job_id=job_id,
            status="rendering",
            stage="rendering",
            progress=70,
            clips=[ClipResult(**prior_clip_1)],
        )
        fake_redis.set(f"job:{job_id}", prior_status.model_dump_json())

        # Ensure local static file exists for clip 1
        local_dir = Path(settings.STATIC_DIR) / job_id
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "clip_1.mp4").write_bytes(b"existing video content")

        render_mock = MagicMock(return_value="/tmp/final.mp4")
        upload_mock = MagicMock(side_effect=lambda p, k: f"http://storage/{k}")

        with patch("app.worker.get_video_metadata", return_value=mock_meta), \
             patch("app.worker.download_media", return_value=("/tmp/in.mp4", "/tmp/in.wav", mock_source, [])), \
             patch("app.worker.transcribe_audio", return_value=mock_words), \
             patch("app.worker.find_viral_clips", return_value=mock_candidates), \
             patch("app.services.cropper.sample_faces_sequentially", return_value=[]), \
             patch("app.worker.generate_ass_subtitles", return_value="/tmp/clip.ass"), \
             patch("app.worker.render_clip_single_pass", render_mock), \
             patch("app.worker.upload_clip", upload_mock), \
             patch("cv2.VideoCapture") as mock_vc:

            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: 1920 if prop == 3 else 1080
            mock_vc.return_value = mock_cap

            process_video_task(job_id, {"url": "https://youtu.be/dummy", "max_clips": 3})

        # Clip 1 was reused: render_mock and upload_mock should only have been called for clips 2 and 3 (2 times)
        assert render_mock.call_count == 2
        assert upload_mock.call_count == 2

    def test_download_all_zip_includes_successful_clips_when_partial(
        self, test_client, fake_redis, tmp_path
    ):
        job_id = "job_download_partial"
        job_dir = Path(settings.STATIC_DIR) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Clip 1 exists
        (job_dir / "clip_1.mp4").write_bytes(b"clip 1 mp4 bytes")
        # Clip 2 failed (no file on disk)
        # Clip 3 exists
        (job_dir / "clip_3.mp4").write_bytes(b"clip 3 mp4 bytes")

        clips = [
            ClipResult(
                clip_id="clip_1",
                title="First Clip",
                virality_score=85,
                duration=15.0,
                video_url=f"http://localhost:8000/static/{job_id}/clip_1.mp4",
                ok=True,
            ),
            ClipResult(
                clip_id="clip_2",
                title="Failed Clip",
                virality_score=75,
                duration=15.0,
                video_url="",
                ok=False,
                error_code="RENDER_FAILED",
                error_message="Encoding failed",
            ),
            ClipResult(
                clip_id="clip_3",
                title="Third Clip",
                virality_score=90,
                duration=15.0,
                video_url=f"http://localhost:8000/static/{job_id}/clip_3.mp4",
                ok=True,
            ),
        ]
        status = JobStatusResponse(
            job_id=job_id,
            status="completed_with_warnings",
            stage="completed_with_warnings",
            progress=100,
            clips=clips,
        )
        fake_redis.set(f"job:{job_id}", status.model_dump_json())

        resp = test_client.get(f"/api/jobs/{job_id}/download-all")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"

        with zipfile.ZipFile(io.BytesIO(resp.content), "r") as zf:
            filenames = zf.namelist()
            # Must include clip 1 and clip 3, captions text, but NOT clip 2
            mp4_files = [f for f in filenames if f.endswith(".mp4")]
            assert len(mp4_files) == 2
            assert any("01_" in f for f in mp4_files)
            assert any("03_" in f for f in mp4_files)
            assert not any("02_" in f for f in mp4_files)
            assert "captions_and_tags.txt" in filenames
