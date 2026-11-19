"""Unit tests for startup validation and enhanced health inspection endpoints."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import fakeredis
from fastapi.testclient import TestClient
import pytest

from app.config import settings
from app.main import app, set_redis_client
from app.schemas import HealthCheckResponse
from app.services.startup_validator import (
    StartupValidationError,
    validate_environment_variables,
    validate_executables_and_fonts,
    validate_llm_configuration,
    validate_model_files,
    validate_redis_connectivity,
    validate_startup,
    validate_storage_writable,
)


@pytest.fixture
def fake_redis():
    client = fakeredis.FakeRedis(decode_responses=True)
    set_redis_client(client)
    yield client
    set_redis_client(None)


@pytest.fixture
def test_client(fake_redis):
    with TestClient(app) as client:
        yield client


# ============================================================================
# Startup Validation Tests
# ============================================================================


def test_validate_environment_variables_valid():
    errors = validate_environment_variables()
    assert len(errors) == 0


def test_validate_environment_variables_invalid_storage(monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "invalid_backend")
    errors = validate_environment_variables()
    assert any("STORAGE_BACKEND 'invalid_backend' is invalid" in e for e in errors)


def test_validate_environment_variables_production_loopback(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "http://localhost:8000")
    errors = validate_environment_variables()
    assert any("cannot use loopback/localhost" in e for e in errors)


def test_validate_model_files_success():
    errors = validate_model_files()
    assert len(errors) == 0


def test_validate_model_files_corrupted(monkeypatch, tmp_path):
    bad_model = tmp_path / "corrupt_yunet.onnx"
    bad_model.write_text("corrupted content")
    monkeypatch.setattr("app.services.startup_validator.get_yunet_model_path", lambda: str(bad_model))

    errors = validate_model_files()
    assert any("checksum mismatch" in e for e in errors)


def test_validate_executables_and_fonts():
    errors = validate_executables_and_fonts()
    # On the test system, ffmpeg, ffprobe, and fonts exist
    assert len(errors) == 0


def test_validate_executables_missing_ffmpeg(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda cmd: None if cmd == "ffmpeg" else "/usr/bin/ffprobe")
    errors = validate_executables_and_fonts()
    assert any("ffmpeg executable not found" in e for e in errors)


def test_validate_redis_connectivity_fake_redis(fake_redis):
    errors = validate_redis_connectivity(fake_redis)
    assert len(errors) == 0


def test_validate_redis_connectivity_failure():
    dead_client = MagicMock()
    dead_client.ping.side_effect = ConnectionRefusedError("Redis down")
    errors = validate_redis_connectivity(dead_client)
    assert any("Cannot connect to Redis" in e for e in errors)


def test_validate_storage_writable_local():
    errors = validate_storage_writable()
    assert len(errors) == 0


def test_validate_startup_aggregates_all_errors(monkeypatch, fake_redis):
    # Invalidate both storage backend and loopback URL
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "invalid_store")
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "http://127.0.0.1:8000")

    with pytest.raises(StartupValidationError) as exc_info:
        validate_startup(service_name="test_service", fail_fast=True, redis_client=fake_redis)

    err = str(exc_info.value)
    assert "Startup validation failed for 'test_service'" in err
    assert "STORAGE_BACKEND 'invalid_store' is invalid" in err
    assert "cannot use loopback/localhost" in err
    assert "Action Required:" in err


# ============================================================================
# Enhanced Health Endpoint Tests
# ============================================================================


def test_health_check_healthy(test_client, fake_redis):
    # Populate queue and job in Redis
    fake_redis.rpush("celery", "task1")
    fake_redis.set("job:test1", '{"job_id": "test1", "status": "rendering", "stage": "rendering"}')

    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] in ("ok", "degraded")
    assert data["service"] == "vynlo-api"
    assert data["redis"]["status"] == "ok"
    assert data["redis"]["latency_ms"] is not None
    assert data["queue"]["depth"] == 1
    assert data["queue"]["active_jobs"] == 1
    assert data["storage"]["status"] == "ok"
    assert data["workers"]["status"] in ("ok", "degraded")


def test_health_check_redis_failure(test_client):
    dead_redis = MagicMock()
    dead_redis.ping.side_effect = ConnectionError("Connection refused")
    set_redis_client(dead_redis)

    response = test_client.get("/health")
    assert response.status_code == 503
    data = response.json()

    assert data["status"] == "unhealthy"
    assert data["redis"]["status"] == "unhealthy"
    assert "Connection refused" in data["redis"]["detail"]


def test_api_health_alias(test_client, fake_redis):
    response = test_client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "vynlo-api"
    assert data["redis"]["status"] == "ok"
