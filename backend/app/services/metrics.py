import contextvars
import json
import logging
import resource
import sys
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Context variable to hold the active tracker across nested function calls
_current_tracker_var: contextvars.ContextVar[Optional["PipelineMetricsTracker"]] = (
    contextvars.ContextVar("current_pipeline_metrics_tracker", default=None)
)


def get_peak_rss_mb() -> float:
    """Return the peak resident set size (RSS) in megabytes.

    On macOS (Darwin), ru_maxrss is reported in bytes.
    On Linux, ru_maxrss is reported in kilobytes (KiB).
    """
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return round(ru / (1024.0 * 1024.0), 2)
        return round(ru / 1024.0, 2)
    except Exception:
        return 0.0


def normalize_stage_name(name: str) -> str:
    """Normalize stage names to snake_case.

    Maps 'audio extract' -> 'audio_extract', 'face detection' -> 'face_detection'.
    """
    cleaned = name.strip().lower()
    return cleaned.replace(" ", "_").replace("-", "_")


def get_current_tracker() -> Optional["PipelineMetricsTracker"]:
    """Retrieve the currently active PipelineMetricsTracker from context."""
    return _current_tracker_var.get()


def set_current_tracker(
    tracker: Optional["PipelineMetricsTracker"],
) -> contextvars.Token:
    """Set the currently active PipelineMetricsTracker in context."""
    return _current_tracker_var.set(tracker)


class StageContext:
    """Context manager measuring wall-clock time for an individual pipeline stage."""

    def __init__(
        self,
        tracker: "PipelineMetricsTracker",
        stage_name: str,
        clip_index: Optional[int] = None,
    ):
        self.tracker = tracker
        self.raw_name = stage_name
        self.normalized_name = normalize_stage_name(stage_name)
        self.clip_index = clip_index
        self.start_time: float = 0.0
        self.elapsed: float = 0.0
        self.nested_children_elapsed: float = 0.0
        self.parent_stage: Optional["StageContext"] = None

    def __enter__(self) -> "StageContext":
        self.parent_stage = self.tracker._active_stage_context
        self.tracker._active_stage_context = self
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.elapsed = max(0.0, time.perf_counter() - self.start_time)
        self.tracker._active_stage_context = self.parent_stage

        # Net time for this stage excludes child stage durations that were nested inside it
        net_elapsed = max(0.0, self.elapsed - self.nested_children_elapsed)

        if self.parent_stage is not None:
            self.parent_stage.nested_children_elapsed += self.elapsed

        self.tracker._record_stage_completion(
            stage_name=self.normalized_name,
            duration=net_elapsed,
            clip_index=self.clip_index,
        )


class PipelineMetricsTracker:
    """Tracks per-stage wall-clock timing and peak resource consumption for video clipping jobs."""

    STANDARD_STAGES = (
        "metadata",
        "download",
        "audio_extract",
        "transcribe",
        "analyze",
        "face_detection",
        "render",
        "upload",
    )

    def __init__(self, job_id: Optional[str] = None):
        self.job_id = job_id
        self.start_time: float = time.perf_counter()
        self.start_rss_mb: float = get_peak_rss_mb()
        self._active_stage_context: Optional[StageContext] = None
        self._context_token: Optional[contextvars.Token] = None
        self._lock = threading.Lock()

        # Cumulative elapsed seconds per stage
        self.stage_durations: dict[str, float] = {
            stage: 0.0 for stage in self.STANDARD_STAGES
        }

        # Per-clip breakdowns: clip_index -> { "face_detection": ..., "render": ..., "upload": ... }
        self.clip_metrics: dict[int, dict[str, float]] = {}

    def __enter__(self) -> "PipelineMetricsTracker":
        self._context_token = set_current_tracker(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._context_token is not None:
            _current_tracker_var.reset(self._context_token)
            self._context_token = None

    def stage(
        self,
        stage_name: str,
        clip_index: Optional[int] = None,
    ) -> StageContext:
        """Context manager to measure the execution time of a specific pipeline stage."""
        return StageContext(self, stage_name, clip_index=clip_index)

    def _record_stage_completion(
        self,
        stage_name: str,
        duration: float,
        clip_index: Optional[int] = None,
    ) -> None:
        """Internal callback when a StageContext finishes."""
        with self._lock:
            if stage_name not in self.stage_durations:
                self.stage_durations[stage_name] = 0.0
            self.stage_durations[stage_name] += duration

            if clip_index is not None:
                if clip_index not in self.clip_metrics:
                    self.clip_metrics[clip_index] = {}
                if stage_name not in self.clip_metrics[clip_index]:
                    self.clip_metrics[clip_index][stage_name] = 0.0
                self.clip_metrics[clip_index][stage_name] += duration

    def record_stage(
        self,
        stage_name: str,
        duration: float,
        clip_index: Optional[int] = None,
    ) -> None:
        """Explicitly record a stage's wall-clock duration in seconds."""
        norm_name = normalize_stage_name(stage_name)
        self._record_stage_completion(
            norm_name, max(0.0, float(duration)), clip_index=clip_index
        )

    def get_stage_duration(self, stage_name: str) -> float:
        """Retrieve total seconds spent in a named stage."""
        norm_name = normalize_stage_name(stage_name)
        return round(self.stage_durations.get(norm_name, 0.0), 3)

    @property
    def total_duration(self) -> float:
        """Total elapsed pipeline wall-clock time since tracker instantiation."""
        return max(0.0, time.perf_counter() - self.start_time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics into structured dictionary conforming to JobStatusResponse internal field."""
        peak_rss = get_peak_rss_mb()
        stages_dict: dict[str, float] = {}
        clips_list: list[dict[str, Any]] = []

        with self._lock:
            for stage in self.STANDARD_STAGES:
                stages_dict[stage] = round(self.stage_durations.get(stage, 0.0), 3)

            # Include any custom/extra stages recorded
            for stage, duration in self.stage_durations.items():
                if stage not in stages_dict:
                    stages_dict[stage] = round(duration, 3)

            for idx in sorted(self.clip_metrics.keys()):
                clip_data = self.clip_metrics[idx]
                clip_entry: dict[str, Any] = {"clip_index": idx}
                clip_total = 0.0
                for k, v in clip_data.items():
                    rounded_v = round(v, 3)
                    clip_entry[k] = rounded_v
                    clip_total += rounded_v
                clip_entry["total"] = round(clip_total, 3)
                clips_list.append(clip_entry)

        return {
            "stages": stages_dict,
            "total_duration": round(self.total_duration, 3),
            "resource_usage": {
                "peak_rss_mb": peak_rss,
                "start_rss_mb": self.start_rss_mb,
            },
            "clips": clips_list,
        }

    def log_metrics(
        self,
        target_logger: logging.Logger,
        job_id: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log structured JSON metrics to the specified logger."""
        effective_job_id = job_id or self.job_id or "unknown"
        metrics_data = self.to_dict()

        log_payload = {
            "event": "pipeline_metrics",
            "job_id": effective_job_id,
            "total_duration_seconds": metrics_data["total_duration"],
            "stages": metrics_data["stages"],
            "resource_usage": metrics_data["resource_usage"],
            "clips_count": len(metrics_data["clips"]),
        }
        if extra:
            log_payload.update(extra)

        target_logger.info(json.dumps(log_payload))
