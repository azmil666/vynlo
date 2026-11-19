import os
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, settings
from app.main import app
from app.schemas import PipelineError, PipelineErrorCode
from app.services.storage import (
    set_s3_client,
    set_storage_handler,
    upload_clip,
    validate_storage_config,
)


class TestStorageUploadLocalMode:
    """Characterization tests for storage upload in local and injected modes."""

    def test_local_upload_copies_file_and_returns_url(self, local_storage_dir: Path, tmp_path: Path):
        source_file = tmp_path / "dummy_clip.mp4"
        source_file.write_bytes(b"dummy video data 12345")

        old_backend = settings.STORAGE_BACKEND
        settings.STORAGE_BACKEND = "local"
        try:
            url = upload_clip(str(source_file), "job_abc/clip_1.mp4")

            expected_dest = local_storage_dir / "job_abc" / "clip_1.mp4"
            assert expected_dest.exists()
            assert expected_dest.read_bytes() == b"dummy video data 12345"
            assert url == "http://localhost:8000/static/job_abc/clip_1.mp4"
        finally:
            settings.STORAGE_BACKEND = old_backend

    def test_local_upload_strips_leading_slashes(self, local_storage_dir: Path, tmp_path: Path):
        source_file = tmp_path / "clip.mp4"
        source_file.write_bytes(b"video bytes")

        old_backend = settings.STORAGE_BACKEND
        settings.STORAGE_BACKEND = "local"
        try:
            url = upload_clip(str(source_file), "///subfolder/clip.mp4")

            expected_dest = local_storage_dir / "subfolder" / "clip.mp4"
            assert expected_dest.exists()
            assert url == "http://localhost:8000/static/subfolder/clip.mp4"
        finally:
            settings.STORAGE_BACKEND = old_backend

    def test_local_upload_missing_source_raises_error(self, local_storage_dir: Path):
        with pytest.raises(FileNotFoundError):
            upload_clip("/non/existent/file.mp4", "job_1/clip.mp4")

    def test_injected_storage_handler_seam(self, tmp_path: Path):
        mock_handler = MagicMock(return_value="https://custom-cdn.example.com/clip.mp4")
        set_storage_handler(mock_handler)
        try:
            url = upload_clip("/any/file.mp4", "job_custom/clip.mp4")
            mock_handler.assert_called_once_with("/any/file.mp4", "job_custom/clip.mp4")
            assert url == "https://custom-cdn.example.com/clip.mp4"
        finally:
            set_storage_handler(None)


class TestStorageUploadS3Mode:
    """Tests for S3/R2 upload, headers, retries with backoff, verification, and errors."""

    @pytest.fixture(autouse=True)
    def setup_s3_settings(self):
        old_backend = settings.STORAGE_BACKEND
        old_endpoint = settings.S3_ENDPOINT_URL
        old_key = settings.S3_ACCESS_KEY
        old_secret = settings.S3_SECRET_KEY
        old_bucket = settings.S3_BUCKET_NAME
        old_public = settings.S3_PUBLIC_BASE_URL
        old_url_style = settings.S3_URL_STYLE

        settings.STORAGE_BACKEND = "s3"
        settings.S3_ENDPOINT_URL = "https://s3.us-east-1.amazonaws.com"
        settings.S3_ACCESS_KEY = "mock_key"
        settings.S3_SECRET_KEY = "mock_secret"
        settings.S3_BUCKET_NAME = "my-bucket"
        settings.S3_PUBLIC_BASE_URL = "https://cdn.example.com"
        settings.S3_URL_STYLE = "public"

        yield

        settings.STORAGE_BACKEND = old_backend
        settings.S3_ENDPOINT_URL = old_endpoint
        settings.S3_ACCESS_KEY = old_key
        settings.S3_SECRET_KEY = old_secret
        settings.S3_BUCKET_NAME = old_bucket
        settings.S3_PUBLIC_BASE_URL = old_public
        settings.S3_URL_STYLE = old_url_style
        set_s3_client(None)

    def test_s3_upload_successful_with_headers_and_head_object_verification(self, tmp_path: Path):
        source_file = tmp_path / "clip.mp4"
        source_file.write_bytes(b"data-123456")

        mock_s3 = MagicMock()
        mock_s3.head_object.return_value = {"ContentLength": len(b"data-123456")}
        set_s3_client(mock_s3)

        url = upload_clip(str(source_file), "job_s3/clip_1.mp4")

        mock_s3.upload_file.assert_called_once_with(
            str(source_file),
            "my-bucket",
            "job_s3/clip_1.mp4",
            ExtraArgs={
                "ContentType": "video/mp4",
                "CacheControl": "public, max-age=31536000, immutable",
            },
        )
        mock_s3.head_object.assert_called_once_with(
            Bucket="my-bucket",
            Key="job_s3/clip_1.mp4",
        )
        assert url == "https://cdn.example.com/job_s3/clip_1.mp4"

    def test_s3_upload_retry_with_backoff_on_transient_failure(self, tmp_path: Path):
        source_file = tmp_path / "clip.mp4"
        source_file.write_bytes(b"test data")

        mock_s3 = MagicMock()
        # Fail attempt 1, succeed attempt 2
        mock_s3.upload_file.side_effect = [Exception("Network error on attempt 1"), None]
        mock_s3.head_object.return_value = {"ContentLength": len(b"test data")}
        set_s3_client(mock_s3)

        url = upload_clip(
            str(source_file),
            "job_s3/clip_retry.mp4",
            max_retries=3,
            initial_backoff=0.01,
        )

        assert mock_s3.upload_file.call_count == 2
        assert url == "https://cdn.example.com/job_s3/clip_retry.mp4"

    def test_s3_upload_retry_exhaustion_raises_upload_failed(self, tmp_path: Path):
        source_file = tmp_path / "clip.mp4"
        source_file.write_bytes(b"test data")

        mock_s3 = MagicMock()
        mock_s3.upload_file.side_effect = Exception("Permanent S3 500 error")
        set_s3_client(mock_s3)

        with pytest.raises(PipelineError) as exc_info:
            upload_clip(
                str(source_file),
                "job_s3/clip_fail.mp4",
                max_retries=3,
                initial_backoff=0.01,
            )

        err = exc_info.value
        assert err.code == PipelineErrorCode.UPLOAD_FAILED
        assert err.stage == "uploading"
        assert "Permanent S3 500 error" in err.detail
        assert mock_s3.upload_file.call_count == 3

    def test_s3_upload_size_mismatch_raises_upload_failed(self, tmp_path: Path):
        source_file = tmp_path / "clip.mp4"
        source_file.write_bytes(b"1234567890")  # 10 bytes

        mock_s3 = MagicMock()
        # Remote head_object reports 5 bytes (truncated upload)
        mock_s3.head_object.return_value = {"ContentLength": 5}
        set_s3_client(mock_s3)

        with pytest.raises(PipelineError) as exc_info:
            upload_clip(
                str(source_file),
                "job_s3/mismatch.mp4",
                max_retries=2,
                initial_backoff=0.01,
            )

        err = exc_info.value
        assert err.code == PipelineErrorCode.UPLOAD_FAILED
        assert "size mismatch" in err.detail

    def test_s3_mode_unconfigured_never_falls_back_to_local_silently(self, tmp_path: Path):
        source_file = tmp_path / "clip.mp4"
        source_file.write_bytes(b"clip content")

        # In S3 mode but credentials missing:
        settings.S3_BUCKET_NAME = None

        with pytest.raises(PipelineError) as exc_info:
            upload_clip(str(source_file), "job_s3/no_fallback.mp4")

        err = exc_info.value
        assert err.code == PipelineErrorCode.UPLOAD_FAILED
        assert "Refusing silent fallback to local storage" in err.detail

    def test_s3_presigned_url_generation(self, tmp_path: Path):
        source_file = tmp_path / "clip.mp4"
        source_file.write_bytes(b"clip content")

        settings.S3_URL_STYLE = "presigned"
        settings.S3_PRESIGNED_EXPIRY_SECONDS = 604800

        mock_s3 = MagicMock()
        mock_s3.head_object.return_value = {"ContentLength": len(b"clip content")}
        mock_s3.generate_presigned_url.return_value = "https://s3.amazonaws.com/my-bucket/clip.mp4?signature=xyz"
        set_s3_client(mock_s3)

        url = upload_clip(str(source_file), "job_s3/presigned.mp4")

        mock_s3.generate_presigned_url.assert_called_once_with(
            "get_object",
            Params={"Bucket": "my-bucket", "Key": "job_s3/presigned.mp4"},
            ExpiresIn=604800,
        )
        assert url == "https://s3.amazonaws.com/my-bucket/clip.mp4?signature=xyz"


class TestStorageConfigAndValidation:
    """Tests for startup storage and URL validation."""

    def test_startup_validation_rejects_localhost_in_production(self):
        cfg = Settings(
            ENVIRONMENT="production",
            PUBLIC_BASE_URL="http://localhost:8000",
            STORAGE_BACKEND="local",
        )
        with pytest.raises(ValueError) as exc_info:
            validate_storage_config(cfg)
        assert "PUBLIC_BASE_URL cannot use localhost/loopback" in str(exc_info.value)

    def test_startup_validation_accepts_valid_domain_in_production(self, tmp_path: Path):
        cfg = Settings(
            ENVIRONMENT="production",
            PUBLIC_BASE_URL="https://api.clipmvp.com",
            STORAGE_BACKEND="local",
            STATIC_DIR=str(tmp_path / "static"),
        )
        validate_storage_config(cfg)

    def test_startup_validation_accepts_localhost_in_development(self, tmp_path: Path):
        cfg = Settings(
            ENVIRONMENT="development",
            PUBLIC_BASE_URL="http://localhost:8000",
            STORAGE_BACKEND="local",
            STATIC_DIR=str(tmp_path / "static"),
        )
        validate_storage_config(cfg)

    def test_startup_validation_fails_when_static_dir_not_writable(self, tmp_path: Path):
        non_writable = tmp_path / "readonly_dir"
        non_writable.mkdir(mode=0o444)

        cfg = Settings(
            ENVIRONMENT="development",
            STORAGE_BACKEND="local",
            STATIC_DIR=str(non_writable / "sub_static"),
        )
        # Attempting to write into read-only directory fails
        try:
            os.chmod(str(non_writable), 0o444)
            with pytest.raises(RuntimeError) as exc_info:
                validate_storage_config(cfg)
            assert "not writable" in str(exc_info.value)
        finally:
            os.chmod(str(non_writable), 0o777)

    def test_startup_validation_fails_when_s3_missing_credentials(self):
        cfg = Settings(
            STORAGE_BACKEND="s3",
            S3_BUCKET_NAME="bucket",
            S3_ACCESS_KEY="",
            S3_SECRET_KEY="",
            S3_ENDPOINT_URL="",
        )
        with pytest.raises(ValueError) as exc_info:
            validate_storage_config(cfg)
        assert "required S3 settings are missing" in str(exc_info.value)


class TestLocalServingRangeRequests:
    """Verify HTTP Range requests for local static serving (needed for Safari/iOS playback)."""

    def test_range_requests_safari_probe_bytes_0_1(self, tmp_path: Path):
        # Create a test video in STATIC_DIR
        video_name = "safari_probe_test.mp4"
        file_path = Path(settings.STATIC_DIR) / video_name
        data = b"0123456789" * 20  # 200 bytes
        file_path.write_bytes(data)

        try:
            client = TestClient(app)
            # Safari initially probes the first 2 bytes: bytes=0-1
            response = client.get(f"/static/{video_name}", headers={"Range": "bytes=0-1"})

            assert response.status_code == 206
            assert response.headers.get("accept-ranges") == "bytes"
            assert response.headers.get("content-range") == "bytes 0-1/200"
            assert response.headers.get("content-length") == "2"
            assert response.content == b"01"
        finally:
            if file_path.exists():
                file_path.unlink()

    def test_range_requests_suffix_bytes(self, tmp_path: Path):
        video_name = "suffix_range_test.mp4"
        file_path = Path(settings.STATIC_DIR) / video_name
        data = b"0123456789" * 10  # 100 bytes
        file_path.write_bytes(data)

        try:
            client = TestClient(app)
            # Request from byte 80 to end
            response = client.get(f"/static/{video_name}", headers={"Range": "bytes=80-"})

            assert response.status_code == 206
            assert response.headers.get("content-range") == "bytes 80-99/100"
            assert response.headers.get("content-length") == "20"
            assert response.content == data[80:]
        finally:
            if file_path.exists():
                file_path.unlink()

    def test_range_requests_invalid_range_returns_416(self, tmp_path: Path):
        video_name = "invalid_range_test.mp4"
        file_path = Path(settings.STATIC_DIR) / video_name
        file_path.write_bytes(b"short file")

        try:
            client = TestClient(app)
            response = client.get(f"/static/{video_name}", headers={"Range": "bytes=500-600"})
            assert response.status_code == 416
        finally:
            if file_path.exists():
                file_path.unlink()
