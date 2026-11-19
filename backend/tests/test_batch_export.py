import io
import json
import os
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock, patch
import zipfile
import pytest

from app.config import settings
from app.schemas import ClipResult, JobStatusResponse, JobWarning
from app.services.export import (
    build_batch_export_archive,
    fetch_remote_clip_stream,
    slugify_title,
)


class DummyVideoServer(BaseHTTPRequestHandler):
    """Simple in-process HTTP server for testing remote clip streaming."""

    payload = b"FAKE_REMOTE_VIDEO_STREAM_BYTES_12345" * 100

    def do_GET(self):
        if self.path == "/slow_clip.mp4":
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)
        elif self.path == "/404.mp4":
            self.send_response(404)
            self.end_headers()
        else:
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)

    def log_message(self, format, *args):
        pass  # Quiet logging during tests


@pytest.fixture
def fake_http_server():
    server = HTTPServer(("127.0.0.1", 0), DummyVideoServer)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()


class TestBatchExportArchive:
    """Test suite for low-memory, non-blocking batch export in local and remote modes."""

    def test_batch_export_local_mode(self, test_client, fake_redis, tmp_path):
        """Export in local mode packs files with ZIP_STORED, manifest with ZIP_DEFLATED, and metadata."""
        job_id = "job_export_local_ok"
        job_dir = Path(settings.STATIC_DIR) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        clip1_data = b"MOCK_LOCAL_CLIP_ONE_MP4_CONTENT"
        clip2_data = b"MOCK_LOCAL_CLIP_TWO_MP4_CONTENT"
        (job_dir / "clip_1.mp4").write_bytes(clip1_data)
        (job_dir / "clip_2.mp4").write_bytes(clip2_data)

        clips = [
            ClipResult(
                clip_id="clip_1",
                title="First Great Moment!",
                virality_score=92,
                duration=25.5,
                video_url=f"http://localhost:8000/static/{job_id}/clip_1.mp4",
                selection_method="llm",
                description="Amazing highlight reel",
                hashtags=["#epic", "#viral"],
                layout_used="speaker",
                warnings=["Boundary snapped by 0.2s"],
            ),
            ClipResult(
                clip_id="clip_2",
                title="Second Clip Scene",
                virality_score=78,
                duration=18.0,
                video_url=f"http://localhost:8000/static/{job_id}/clip_2.mp4",
                selection_method="heuristic",
                selection_reason="Transcript energy peak fallback",
                is_fallback=True,
                layout_used="fit_blur",
            ),
        ]
        status = JobStatusResponse(
            job_id=job_id,
            status="completed",
            stage="completed",
            progress=100,
            clips=clips,
            warnings=[JobWarning(code="AUDIO_NORM", message="Loudnorm applied", clip_index=1)],
        )
        fake_redis.set(f"job:{job_id}", status.model_dump_json())

        resp = test_client.get(f"/api/jobs/{job_id}/download-all")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"
        assert f"shorts_{job_id[:8]}.zip" in resp.headers["content-disposition"]

        with zipfile.ZipFile(io.BytesIO(resp.content), "r") as zf:
            infolist = zf.infolist()
            info_by_name = {info.filename: info for info in infolist}

            # Check that MP4 files use ZIP_STORED (compression=0)
            mp4_entries = [info for info in infolist if info.filename.endswith(".mp4")]
            assert len(mp4_entries) == 2
            for mp4_info in mp4_entries:
                assert mp4_info.compress_type == zipfile.ZIP_STORED, "MP4 files must use ZIP_STORED"

            # Check that text manifest uses ZIP_DEFLATED
            assert "captions_and_tags.txt" in info_by_name
            manifest_info = info_by_name["captions_and_tags.txt"]
            assert manifest_info.compress_type == zipfile.ZIP_DEFLATED

            # Verify contents
            manifest_text = zf.read("captions_and_tags.txt").decode("utf-8")
            assert "=== CLIPPER STUDIO BATCH EXPORT: JOB job_export_local_ok ===" in manifest_text
            assert "Selection Method: llm" in manifest_text
            assert "Selection Method: heuristic (Transcript energy peak fallback)" in manifest_text
            assert "Status: AVAILABLE (Local Storage)" in manifest_text
            assert "Warnings:" in manifest_text
            assert "Boundary snapped by 0.2s" in manifest_text
            assert "[AUDIO_NORM] Loudnorm applied" in manifest_text
            assert "#epic #viral" in manifest_text

    def test_batch_export_remote_mode_with_fake_server(self, test_client, fake_redis):
        """Export with remote S3/HTTP URLs streams chunks directly into the archive with ZIP_STORED."""
        job_id = "job_export_remote_s3"
        remote_url_1 = "https://s3.example.com/bucket/clip_1.mp4"
        remote_url_2 = "https://s3.example.com/bucket/clip_2.mp4"
        payload = b"FAKE_REMOTE_VIDEO_STREAM_BYTES_12345" * 100

        clips = [
            ClipResult(
                clip_id="clip_1",
                title="Remote Cloud Clip 1",
                virality_score=89,
                duration=30.0,
                video_url=remote_url_1,
                selection_method="llm",
            ),
            ClipResult(
                clip_id="clip_2",
                title="Remote Cloud Clip 2",
                virality_score=81,
                duration=20.0,
                video_url=remote_url_2,
                selection_method="llm",
            ),
        ]
        status = JobStatusResponse(
            job_id=job_id,
            status="completed",
            stage="completed",
            progress=100,
            clips=clips,
        )
        fake_redis.set(f"job:{job_id}", status.model_dump_json())

        # Ensure no local static files exist for this job
        local_dir = Path(settings.STATIC_DIR) / job_id
        if local_dir.exists():
            import shutil
            shutil.rmtree(local_dir)

        class MockStreamResponse:
            def __init__(self, data):
                self.data = data

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def raise_for_status(self):
                pass

            def iter_content(self, chunk_size=64 * 1024):
                # Yield in realistic small chunks
                for idx in range(0, len(self.data), chunk_size):
                    yield self.data[idx : idx + chunk_size]

        with patch("app.services.export.requests.get", side_effect=lambda url, stream, timeout: MockStreamResponse(payload)):
            resp = test_client.get(f"/api/jobs/{job_id}/download-all")
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "application/zip"

            with zipfile.ZipFile(io.BytesIO(resp.content), "r") as zf:
                mp4_files = [f for f in zf.namelist() if f.endswith(".mp4")]
                assert len(mp4_files) == 2

                for filename in mp4_files:
                    file_info = zf.getinfo(filename)
                    assert file_info.compress_type == zipfile.ZIP_STORED
                    content = zf.read(filename)
                    assert content == payload

                manifest_text = zf.read("captions_and_tags.txt").decode("utf-8")
                assert "Status: AVAILABLE (Remote Storage)" in manifest_text

    def test_batch_export_missing_file_records_manifest_note_without_aborting(
        self, test_client, fake_redis, tmp_path
    ):
        """When a clip is missing or expired, export does not abort; it notes UNAVAILABLE in manifest."""
        job_id = "job_export_missing_clip"
        job_dir = Path(settings.STATIC_DIR) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Clip 1 exists
        (job_dir / "clip_1.mp4").write_bytes(b"CLIP_1_PRESENT_CONTENT")
        # Clip 2 does NOT exist locally and has no valid remote URL
        # Clip 3 exists
        (job_dir / "clip_3.mp4").write_bytes(b"CLIP_3_PRESENT_CONTENT")

        clips = [
            ClipResult(
                clip_id="clip_1",
                title="Good Clip 1",
                virality_score=85,
                duration=15.0,
                video_url=f"http://localhost:8000/static/{job_id}/clip_1.mp4",
                ok=True,
            ),
            ClipResult(
                clip_id="clip_2",
                title="Missing Expired Clip 2",
                virality_score=70,
                duration=15.0,
                video_url="",
                ok=False,
                error_code="RENDER_FAILED",
                error_message="Single-pass render crashed",
            ),
            ClipResult(
                clip_id="clip_3",
                title="Good Clip 3",
                virality_score=88,
                duration=20.0,
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

        with zipfile.ZipFile(io.BytesIO(resp.content), "r") as zf:
            filenames = zf.namelist()
            # Only clips 1 and 3 should be present as MP4s
            mp4_files = [f for f in filenames if f.endswith(".mp4")]
            assert len(mp4_files) == 2
            assert any("01_" in f for f in mp4_files)
            assert any("03_" in f for f in mp4_files)
            assert not any("02_" in f for f in mp4_files)

            manifest_text = zf.read("captions_and_tags.txt").decode("utf-8")
            assert "--- CLIP 02: Missing Expired Clip 2 ---" in manifest_text
            assert "Status: UNAVAILABLE: Clip rendering failed [RENDER_FAILED]" in manifest_text
            assert "File: NOT_INCLUDED" in manifest_text

    def test_batch_export_large_file_memory_bound(self, tmp_path):
        """Verify SpooledTemporaryFile spills to disk and bounds RAM when archive exceeds threshold."""
        job_id = "job_mem_bound_test"
        job_dir = tmp_path / "static" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Create a 2MB video payload
        large_payload = b"X" * (2 * 1024 * 1024)
        (job_dir / "clip_1.mp4").write_bytes(large_payload)

        clip = ClipResult(
            clip_id="clip_1",
            title="Large Clip",
            virality_score=90,
            duration=60.0,
            video_url=f"http://localhost:8000/static/{job_id}/clip_1.mp4",
        )
        status = JobStatusResponse(
            job_id=job_id,
            status="completed",
            stage="completed",
            progress=100,
            clips=[clip],
        )

        # Build archive with a strict 256KB in-memory spool threshold
        spooled = build_batch_export_archive(
            job_id=job_id,
            job_status=status,
            static_dir=str(tmp_path / "static"),
            max_spooled_memory=256 * 1024,  # 256 KB memory ceiling
        )

        try:
            # When payload (2MB) exceeds 256KB, SpooledTemporaryFile must have spilled to disk
            internal_file = getattr(spooled, "_file", None)
            assert internal_file is not None
            # On rollover, _file switches from io.BytesIO to a file wrapper on disk
            assert not isinstance(internal_file, io.BytesIO), "Spooled file must have rolled over to disk"

            # Check that the archive is complete and readable
            spooled.seek(0)
            with zipfile.ZipFile(spooled, mode="r") as zf:
                assert len(zf.namelist()) == 2
                assert any(f.endswith(".mp4") for f in zf.namelist())
                assert "captions_and_tags.txt" in zf.namelist()
        finally:
            spooled.close()

    def test_remote_fetch_timeout_and_error_handling(self):
        """Verify remote fetch failure writes note to manifest instead of raising unhandled error."""
        job_id = "job_remote_fail_handling"
        clip = ClipResult(
            clip_id="clip_1",
            title="Unreachable Remote Clip",
            virality_score=80,
            duration=30.0,
            # Point to an unreachable IP/port that will immediately refuse connection
            video_url="http://127.0.0.1:59999/nonexistent.mp4",
        )
        status = JobStatusResponse(
            job_id=job_id,
            status="completed",
            stage="completed",
            progress=100,
            clips=[clip],
        )

        spooled = build_batch_export_archive(
            job_id=job_id,
            job_status=status,
            static_dir="/tmp/nonexistent_static_dir",
        )

        try:
            with zipfile.ZipFile(spooled, mode="r") as zf:
                manifest_text = zf.read("captions_and_tags.txt").decode("utf-8")
                assert "UNAVAILABLE: Remote fetch failed" in manifest_text
        finally:
            spooled.close()

    def test_slugify_title_edge_cases(self):
        assert slugify_title("Hello World 123!") == "hello_world_123"
        assert slugify_title("   ---Spaces & Symbols---   ") == "spaces_symbols"
        assert slugify_title("") == "clip"
