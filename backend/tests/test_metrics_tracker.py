import json
import logging
import time
from unittest.mock import MagicMock

import pytest

from app.schemas import JobStatusResponse
from app.services.metrics import (
    PipelineMetricsTracker,
    get_current_tracker,
    get_peak_rss_mb,
    normalize_stage_name,
)


class TestPipelineMetricsTracker:
    """Unit tests validating wall-clock timing accuracy, stage aggregation, nesting, and resource reporting."""

    def test_normalize_stage_name(self):
        assert normalize_stage_name("audio extract") == "audio_extract"
        assert normalize_stage_name("Audio-Extract") == "audio_extract"
        assert normalize_stage_name("face detection") == "face_detection"
        assert normalize_stage_name("metadata") == "metadata"

    def test_single_stage_timing(self):
        tracker = PipelineMetricsTracker(job_id="test-job-single")
        with tracker.stage("metadata"):
            time.sleep(0.02)

        duration = tracker.get_stage_duration("metadata")
        assert duration >= 0.019
        assert duration <= 0.15

        metrics = tracker.to_dict()
        assert "metadata" in metrics["stages"]
        assert metrics["stages"]["metadata"] == duration
        assert metrics["total_duration"] >= duration

    def test_stage_accumulation_across_multiple_calls(self):
        tracker = PipelineMetricsTracker(job_id="test-job-accum")

        for _ in range(3):
            with tracker.stage("render"):
                time.sleep(0.01)

        total_render = tracker.get_stage_duration("render")
        assert total_render >= 0.025
        assert total_render <= 0.20

    def test_per_clip_timing_breakdown(self):
        tracker = PipelineMetricsTracker(job_id="test-job-clips")

        # Clip 1
        with tracker.stage("face_detection", clip_index=1):
            time.sleep(0.01)
        with tracker.stage("render", clip_index=1):
            time.sleep(0.01)
        with tracker.stage("upload", clip_index=1):
            time.sleep(0.005)

        # Clip 2
        with tracker.stage("face_detection", clip_index=2):
            time.sleep(0.01)
        with tracker.stage("render", clip_index=2):
            time.sleep(0.01)
        with tracker.stage("upload", clip_index=2):
            time.sleep(0.005)

        metrics = tracker.to_dict()
        clips = metrics["clips"]
        assert len(clips) == 2

        assert clips[0]["clip_index"] == 1
        assert "face_detection" in clips[0]
        assert "render" in clips[0]
        assert "upload" in clips[0]
        assert clips[0]["total"] >= 0.02

        assert clips[1]["clip_index"] == 2
        assert clips[1]["total"] >= 0.02

        # Stages dict must hold sum
        assert metrics["stages"]["face_detection"] >= 0.018
        assert metrics["stages"]["render"] >= 0.018
        assert metrics["stages"]["upload"] >= 0.008

    def test_nested_stage_timing_net_deduction(self):
        """When audio_extract is nested inside download, download time excludes audio_extract time."""
        tracker = PipelineMetricsTracker(job_id="test-job-nested")

        with tracker.stage("download"):
            time.sleep(0.015)
            with tracker.stage("audio_extract"):
                time.sleep(0.02)
            time.sleep(0.01)

        metrics = tracker.to_dict()
        audio_time = metrics["stages"]["audio_extract"]
        download_net_time = metrics["stages"]["download"]

        # Inner stage took ~0.02s
        assert audio_time >= 0.018
        # Net download took ~0.025s (not the total 0.045s)
        assert download_net_time >= 0.02
        assert download_net_time <= 0.04
        # Total sum accounts for the full duration
        assert (download_net_time + audio_time) >= 0.04

    def test_peak_rss_reporting(self):
        rss_mb = get_peak_rss_mb()
        assert isinstance(rss_mb, float)
        assert rss_mb > 0.0

        tracker = PipelineMetricsTracker(job_id="test-job-rss")
        metrics = tracker.to_dict()
        assert "resource_usage" in metrics
        assert metrics["resource_usage"]["peak_rss_mb"] > 0.0
        assert metrics["resource_usage"]["start_rss_mb"] > 0.0

    def test_structured_json_logging(self):
        tracker = PipelineMetricsTracker(job_id="test-job-log")
        with tracker.stage("transcribe"):
            time.sleep(0.01)

        mock_logger = MagicMock(spec=logging.Logger)
        tracker.log_metrics(mock_logger, job_id="custom-job-id", extra={"custom_tag": "perf"})

        assert mock_logger.info.called
        log_call_arg = mock_logger.info.call_args[0][0]
        parsed = json.loads(log_call_arg)

        assert parsed["event"] == "pipeline_metrics"
        assert parsed["job_id"] == "custom-job-id"
        assert parsed["custom_tag"] == "perf"
        assert "transcribe" in parsed["stages"]
        assert "resource_usage" in parsed

    def test_context_var_lifecycle(self):
        assert get_current_tracker() is None

        tracker = PipelineMetricsTracker(job_id="test-job-ctx")
        with tracker:
            active = get_current_tracker()
            assert active is tracker

        assert get_current_tracker() is None

    def test_job_status_response_includes_metrics_in_serialization(self):
        tracker = PipelineMetricsTracker(job_id="test-job-schema")
        with tracker.stage("metadata"):
            time.sleep(0.005)

        metrics_payload = tracker.to_dict()
        resp = JobStatusResponse(
            job_id="test-job-schema",
            status="completed",
            stage="completed",
            progress=100,
            clips=[],
            metrics=metrics_payload,
        )

        serialized = resp.model_dump_json()
        deserialized = JobStatusResponse.model_validate_json(serialized)

        assert deserialized.metrics is not None
        assert deserialized.metrics["stages"]["metadata"] > 0.0
        assert deserialized.metrics["resource_usage"]["peak_rss_mb"] > 0.0
