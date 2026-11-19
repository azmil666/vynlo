"""API contract tests validating OpenAPI schema and generated TypeScript types."""

import json
from pathlib import Path
import pytest
from app.main import app
from scripts.generate_types import (
    generate_typescript,
    get_openapi_spec,
    schema_to_ts_type,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIR = ROOT_DIR / "frontend"
OPENAPI_SPEC_PATH = BACKEND_DIR / "openapi.json"
TYPESCRIPT_TYPES_PATH = FRONTEND_DIR / "src" / "types" / "api.ts"


class TestAPIContractAndTypeSync:
    """Contract tests ensuring backend schemas and frontend TypeScript types remain in lockstep."""

    def test_openapi_json_is_in_sync_with_fastapi_schema(self):
        """The committed backend/openapi.json must exactly match app.openapi()."""
        spec = get_openapi_spec()
        expected_json = json.dumps(spec, indent=2, sort_keys=True) + "\n"

        assert OPENAPI_SPEC_PATH.exists(), f"Missing {OPENAPI_SPEC_PATH}. Run 'npm run generate-types' to create it."
        actual_json = OPENAPI_SPEC_PATH.read_text(encoding="utf-8")

        assert actual_json == expected_json, (
            "backend/openapi.json is out of sync with FastAPI schema. "
            "Run 'npm run generate-types' or 'backend/.venv/bin/python backend/scripts/generate_types.py' to update it."
        )

    def test_typescript_api_types_are_in_sync_with_fastapi_schema(self):
        """The committed frontend/src/types/api.ts must match freshly generated TypeScript code."""
        spec = get_openapi_spec()
        expected_ts = generate_typescript(spec)

        assert TYPESCRIPT_TYPES_PATH.exists(), (
            f"Missing {TYPESCRIPT_TYPES_PATH}. Run 'npm run generate-types' to create it."
        )
        actual_ts = TYPESCRIPT_TYPES_PATH.read_text(encoding="utf-8")

        assert actual_ts == expected_ts, (
            "frontend/src/types/api.ts is out of sync with backend FastAPI OpenAPI schema. "
            "Run 'npm run generate-types' or 'backend/.venv/bin/python backend/scripts/generate_types.py' to update it."
        )

    def test_contract_detects_schema_divergence(self):
        """Ensure the contract check fails when schema divergence is injected."""
        spec = get_openapi_spec()
        # Mutate spec to simulate backend schema change
        mutated_spec = json.loads(json.dumps(spec))
        mutated_spec["components"]["schemas"]["JobStatusResponse"]["properties"]["new_uncommitted_field"] = {
            "type": "string",
            "title": "New Uncommitted Field",
        }

        generated_from_mutated = generate_typescript(mutated_spec)
        current_committed_ts = TYPESCRIPT_TYPES_PATH.read_text(encoding="utf-8")

        assert "new_uncommitted_field?: string;" in generated_from_mutated
        assert current_committed_ts != generated_from_mutated

    def test_job_status_response_schema_contract(self):
        """JobStatusResponse must specify status, stage, progress, error, warnings, clips, and required fields."""
        spec = get_openapi_spec()
        schemas = spec["components"]["schemas"]
        assert "JobStatusResponse" in schemas

        job_status = schemas["JobStatusResponse"]
        props = job_status["properties"]
        required = job_status.get("required", [])

        # Required fields for response guarantee
        for req_field in ("job_id", "status", "progress", "clips"):
            assert req_field in required, f"{req_field} must be in JobStatusResponse.required"

        # Key fields must exist
        for key in ("job_id", "status", "stage", "progress", "updated_at", "error", "warnings", "clips", "is_partial"):
            assert key in props, f"Missing property '{key}' in JobStatusResponse schema"

        # Status and Stage enums
        expected_stages = [
            "queued",
            "downloading",
            "transcribing",
            "analyzing",
            "rendering",
            "completed",
            "completed_with_warnings",
            "partial",
            "failed",
            "expired",
        ]
        assert props["status"]["enum"] == expected_stages
        assert props["stage"]["enum"] == expected_stages

        # Warnings references JobWarning
        assert props["warnings"]["items"]["$ref"] == "#/components/schemas/JobWarning"

        # Clips references ClipResult
        assert props["clips"]["items"]["$ref"] == "#/components/schemas/ClipResult"

    def test_clip_result_schema_contract(self):
        """ClipResult must contain selection_method, layout_used, warnings, and preserve deprecated is_fallback."""
        spec = get_openapi_spec()
        schemas = spec["components"]["schemas"]
        assert "ClipResult" in schemas

        clip = schemas["ClipResult"]
        props = clip["properties"]
        required = clip.get("required", [])

        for req_field in ("clip_id", "title", "virality_score", "duration", "video_url"):
            assert req_field in required, f"{req_field} must be in ClipResult.required"

        # Selection method
        assert props["selection_method"]["enum"] == ["llm", "heuristic"]

        # Layout used
        assert props["layout_used"]["enum"] == ["speaker", "split", "fit_blur"]

        # Deprecated is_fallback preserved
        assert "is_fallback" in props
        assert props["is_fallback"]["type"] == "boolean"

        # Warnings supports strings and JobWarning objects
        warning_items = props["warnings"]["items"]
        assert "anyOf" in warning_items
        refs_or_types = [
            item.get("$ref") or item.get("type") for item in warning_items["anyOf"]
        ]
        assert "string" in refs_or_types
        assert "#/components/schemas/JobWarning" in refs_or_types

    def test_job_error_detail_schema_contract(self):
        """JobErrorDetail must provide code referencing PipelineErrorCode, message, retryable, stage, detail."""
        spec = get_openapi_spec()
        schemas = spec["components"]["schemas"]
        assert "JobErrorDetail" in schemas

        err = schemas["JobErrorDetail"]
        props = err["properties"]
        required = err.get("required", [])

        assert "code" in required
        assert "message" in required
        assert props["code"]["$ref"] == "#/components/schemas/PipelineErrorCode"
        assert "retryable" in props
        assert "stage" in props
        assert "detail" in props

    def test_typescript_file_content_validations(self):
        """Validate structure and backward compatibility exports in frontend/src/types/api.ts."""
        content = TYPESCRIPT_TYPES_PATH.read_text(encoding="utf-8")

        # Top-level types and interfaces must be exported
        assert "export type PipelineStage =" in content
        assert "export type PipelineErrorCode =" in content
        assert "export interface JobStatusResponse {" in content
        assert "export interface ClipResult {" in content
        assert "export interface JobErrorDetail {" in content
        assert "export interface JobWarning {" in content
        assert "export interface JobCreateRequest {" in content
        assert "export interface WebcamRegion {" in content

        # Deprecated fields and backwards compatibility aliases
        assert "is_fallback?: boolean;" in content
        assert "export type JobStatus = JobStatusResponse;" in content
