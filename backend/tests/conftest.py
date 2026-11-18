import json
import os
from pathlib import Path

# Ensure host-compatible directories before app config loads
os.environ["STATIC_DIR"] = str((Path(__file__).parent.parent / "static").resolve())
os.environ.setdefault("TEMP_DIR", "/tmp/clipper")

import pytest
import fakeredis
from fastapi.testclient import TestClient

from app.config import settings
import app.main as app_main
import app.worker as app_worker
import app.services.analyzer as app_analyzer
import app.services.storage as app_storage

from tests.fixtures.fake_llm import FakeLLM
from tests.fixtures.media import (
    generate_16_9_1080p_30s,
    generate_no_audio,
    generate_silent_audio,
    generate_square,
    generate_vertical_1080x1920,
    generate_vfr,
    probe,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TRANSCRIPTS_DIR = FIXTURES_DIR / "transcripts"


@pytest.fixture
def fake_redis():
    """Isolated FakeRedis instance injected into main and worker modules."""
    client = fakeredis.FakeRedis(decode_responses=True)
    orig_main_client = app_main.get_redis_client()
    orig_worker_client = app_worker.get_redis_client()

    app_main.set_redis_client(client)
    app_worker.set_redis_client(client)
    try:
        yield client
    finally:
        app_main.set_redis_client(orig_main_client)
        app_worker.set_redis_client(orig_worker_client)


@pytest.fixture
def test_client(fake_redis):
    """TestClient configured with fake_redis for isolated API route testing."""
    return TestClient(app_main.app)


@pytest.fixture
def fake_llm():
    """Configurable FakeLLM instance injected into the analyzer module."""
    llm = FakeLLM()
    orig_llm = app_analyzer.get_llm_client()
    app_analyzer.set_llm_client(llm)
    try:
        yield llm
    finally:
        app_analyzer.set_llm_client(orig_llm)


@pytest.fixture
def transcript_normal() -> list[dict]:
    with open(TRANSCRIPTS_DIR / "normal_speech.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def transcript_no_punctuation() -> list[dict]:
    with open(TRANSCRIPTS_DIR / "no_punctuation.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def transcript_long_pauses() -> list[dict]:
    with open(TRANSCRIPTS_DIR / "long_pauses.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def transcript_empty() -> list[dict]:
    with open(TRANSCRIPTS_DIR / "empty.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def probe_helper():
    return probe


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("media_fixtures")


@pytest.fixture(scope="session")
def video_16_9_30s(media_dir: Path) -> str:
    path = media_dir / "16_9_1080p_30s.mp4"
    if not path.exists():
        generate_16_9_1080p_30s(path)
    return str(path)


@pytest.fixture(scope="session")
def video_vertical(media_dir: Path) -> str:
    path = media_dir / "vertical_1080x1920.mp4"
    if not path.exists():
        generate_vertical_1080x1920(path, duration=2.0)
    return str(path)


@pytest.fixture(scope="session")
def video_square(media_dir: Path) -> str:
    path = media_dir / "square_1080x1080.mp4"
    if not path.exists():
        generate_square(path, duration=2.0)
    return str(path)


@pytest.fixture(scope="session")
def video_vfr(media_dir: Path) -> str:
    path = media_dir / "vfr.mp4"
    if not path.exists():
        generate_vfr(path, duration=2.0)
    return str(path)


@pytest.fixture(scope="session")
def video_no_audio(media_dir: Path) -> str:
    path = media_dir / "no_audio.mp4"
    if not path.exists():
        generate_no_audio(path, duration=2.0)
    return str(path)


@pytest.fixture(scope="session")
def video_silent_audio(media_dir: Path) -> str:
    path = media_dir / "silent_audio.mp4"
    if not path.exists():
        generate_silent_audio(path, duration=2.0)
    return str(path)


@pytest.fixture
def local_storage_dir(tmp_path: Path):
    """Temporary static directory for testing local storage uploads."""
    old_static = settings.STATIC_DIR
    old_base_url = settings.PUBLIC_BASE_URL
    old_s3_endpoint = settings.S3_ENDPOINT_URL
    old_s3_key = settings.S3_ACCESS_KEY
    old_s3_secret = settings.S3_SECRET_KEY
    old_s3_bucket = settings.S3_BUCKET_NAME

    settings.STATIC_DIR = str(tmp_path / "static")
    settings.PUBLIC_BASE_URL = "http://localhost:8000"
    settings.S3_ENDPOINT_URL = None
    settings.S3_ACCESS_KEY = None
    settings.S3_SECRET_KEY = None
    settings.S3_BUCKET_NAME = None

    try:
        yield tmp_path / "static"
    finally:
        settings.STATIC_DIR = old_static
        settings.PUBLIC_BASE_URL = old_base_url
        settings.S3_ENDPOINT_URL = old_s3_endpoint
        settings.S3_ACCESS_KEY = old_s3_key
        settings.S3_SECRET_KEY = old_s3_secret
        settings.S3_BUCKET_NAME = old_s3_bucket
