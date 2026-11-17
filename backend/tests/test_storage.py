import os
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from app.config import settings
from app.services.storage import set_s3_client, set_storage_handler, upload_clip


class TestStorageUploadLocalMode:
    """Characterization tests for storage upload in local and injected modes."""

    def test_local_upload_copies_file_and_returns_url(self, local_storage_dir: Path, tmp_path: Path):
        source_file = tmp_path / "dummy_clip.mp4"
        source_file.write_bytes(b"dummy video data 12345")

        url = upload_clip(str(source_file), "job_abc/clip_1.mp4")

        expected_dest = local_storage_dir / "job_abc" / "clip_1.mp4"
        assert expected_dest.exists()
        assert expected_dest.read_bytes() == b"dummy video data 12345"
        assert url == "http://localhost:8000/static/job_abc/clip_1.mp4"

    def test_local_upload_strips_leading_slashes(self, local_storage_dir: Path, tmp_path: Path):
        source_file = tmp_path / "clip.mp4"
        source_file.write_bytes(b"video bytes")

        url = upload_clip(str(source_file), "///subfolder/clip.mp4")

        expected_dest = local_storage_dir / "subfolder" / "clip.mp4"
        assert expected_dest.exists()
        assert url == "http://localhost:8000/static/subfolder/clip.mp4"

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

    def test_injected_s3_client_seam(self, tmp_path: Path):
        source_file = tmp_path / "clip.mp4"
        source_file.write_bytes(b"data")

        # Configure S3 settings
        old_endpoint = settings.S3_ENDPOINT_URL
        old_key = settings.S3_ACCESS_KEY
        old_secret = settings.S3_SECRET_KEY
        old_bucket = settings.S3_BUCKET_NAME
        old_public = settings.S3_PUBLIC_BASE_URL

        settings.S3_ENDPOINT_URL = "https://s3.us-east-1.amazonaws.com"
        settings.S3_ACCESS_KEY = "mock_key"
        settings.S3_SECRET_KEY = "mock_secret"
        settings.S3_BUCKET_NAME = "my-bucket"
        settings.S3_PUBLIC_BASE_URL = "https://cdn.example.com"

        mock_s3 = MagicMock()
        set_s3_client(mock_s3)
        try:
            url = upload_clip(str(source_file), "job_s3/clip_1.mp4")
            mock_s3.upload_file.assert_called_once_with(
                str(source_file),
                "my-bucket",
                "job_s3/clip_1.mp4",
                ExtraArgs={"ContentType": "video/mp4"},
            )
            assert url == "https://cdn.example.com/job_s3/clip_1.mp4"
        finally:
            set_s3_client(None)
            settings.S3_ENDPOINT_URL = old_endpoint
            settings.S3_ACCESS_KEY = old_key
            settings.S3_SECRET_KEY = old_secret
            settings.S3_BUCKET_NAME = old_bucket
            settings.S3_PUBLIC_BASE_URL = old_public
