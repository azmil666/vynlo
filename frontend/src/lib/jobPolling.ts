/**
 * Job polling, resilience, error mapping, and persistence utilities.
 */

import type {
  JobErrorDetail,
  JobStatusResponse,
  JobWarning,
  PipelineErrorCode,
  PipelineStage,
} from "@/types/api";

export interface StructuredErrorInfo {
  code: string;
  title: string;
  message: string;
  detail?: string | null;
  retryable: boolean;
}

export const ERROR_CODE_MAP: Record<
  string,
  { title: string; defaultMessage: string; retryable: boolean }
> = {
  INVALID_URL: {
    title: "Invalid Media URL",
    defaultMessage: "The provided video link is invalid or comes from an unsupported host.",
    retryable: false,
  },
  VIDEO_UNAVAILABLE: {
    title: "Video Unavailable",
    defaultMessage: "The source video is private, removed, age-restricted, or region-blocked.",
    retryable: false,
  },
  VIDEO_TOO_LONG: {
    title: "Video Exceeds Duration Limit",
    defaultMessage: "The source video duration exceeds the allowed processing limit.",
    retryable: false,
  },
  LIVE_UNSUPPORTED: {
    title: "Live Stream Unsupported",
    defaultMessage: "Live streams cannot be processed. Please provide an on-demand video URL.",
    retryable: false,
  },
  DOWNLOAD_FAILED: {
    title: "Download Failed",
    defaultMessage: "Failed to download the media stream from the hosting provider.",
    retryable: true,
  },
  UNSUPPORTED_FORMAT: {
    title: "Unsupported Format",
    defaultMessage: "The audio or video codec could not be decoded.",
    retryable: false,
  },
  NO_SPEECH: {
    title: "No Speech Detected",
    defaultMessage: "No spoken audio was found to generate viral clips or subtitles.",
    retryable: false,
  },
  TRANSCRIPTION_FAILED: {
    title: "Transcription Failed",
    defaultMessage: "Automatic speech recognition encountered an unexpected error.",
    retryable: true,
  },
  LLM_UNAVAILABLE: {
    title: "AI Service Unavailable",
    defaultMessage: "The viral hook analysis service is temporarily offline or rate-limited.",
    retryable: true,
  },
  LLM_BLOCKED: {
    title: "Content Safety Filtered",
    defaultMessage: "AI analysis was blocked due to safety or moderation filters.",
    retryable: false,
  },
  LLM_CONFIG: {
    title: "AI Configuration Error",
    defaultMessage: "The AI analysis model or prompt configuration is invalid.",
    retryable: false,
  },
  ANALYSIS_INVALID: {
    title: "Analysis Malformed",
    defaultMessage: "Failed to parse viral candidate timestamps from AI response.",
    retryable: true,
  },
  NO_QUALIFYING_CLIPS: {
    title: "No Viral Clips Found",
    defaultMessage: "No segments met the virality criteria for 9:16 extraction.",
    retryable: false,
  },
  RENDER_FAILED: {
    title: "Video Render Failed",
    defaultMessage: "FFmpeg encountered an error during framing, subtitle burning, or encoding.",
    retryable: true,
  },
  UPLOAD_FAILED: {
    title: "Upload Failed",
    defaultMessage: "Failed to persist the rendered clip to storage.",
    retryable: true,
  },
  TIMEOUT: {
    title: "Processing Timed Out",
    defaultMessage: "Job processing exceeded the time limit.",
    retryable: true,
  },
  WORKER_LOST: {
    title: "Worker Process Lost",
    defaultMessage: "The background worker process crashed or became unresponsive.",
    retryable: true,
  },
  EXPIRED: {
    title: "Job Retention Expired",
    defaultMessage: "This job and its generated video files have expired from storage.",
    retryable: false,
  },
  JOB_NOT_FOUND: {
    title: "Job Not Found",
    defaultMessage: "The requested job ID was not found on the server. It may have expired or never existed.",
    retryable: false,
  },
  INTERNAL: {
    title: "Internal Pipeline Error",
    defaultMessage: "An unexpected system error occurred during pipeline execution.",
    retryable: true,
  },
};

/**
 * Resolve the API base URL.
 * Never defaults to localhost in production builds.
 */
export function getApiBaseUrl(): string {
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL.replace(/\/$/, "");
  }
  // In production builds, do not hardcode localhost
  if (process.env.NODE_ENV === "production") {
    return typeof window !== "undefined" && window.location ? window.location.origin : "";
  }
  return "http://localhost:8000";
}

/**
 * Resolve a relative or absolute video URL against the active API base URL.
 */
export function resolveVideoUrl(rawUrl: string, apiBaseUrl = getApiBaseUrl()): string {
  if (!rawUrl) return "";
  if (rawUrl.startsWith("http://") || rawUrl.startsWith("https://")) {
    return rawUrl;
  }
  const base = apiBaseUrl.replace(/\/$/, "");
  return base ? `${base}/${rawUrl.replace(/^\//, "")}` : rawUrl;
}

/**
 * Translate a structured or string error into a user-friendly structured error.
 */
export function getStructuredError(
  error?: JobErrorDetail | string | null,
  fallbackStatus?: string
): StructuredErrorInfo {
  if (!error) {
    if (fallbackStatus === "expired") {
      return {
        code: "EXPIRED",
        title: ERROR_CODE_MAP.EXPIRED.title,
        message: ERROR_CODE_MAP.EXPIRED.defaultMessage,
        retryable: false,
      };
    }
    return {
      code: "INTERNAL",
      title: ERROR_CODE_MAP.INTERNAL.title,
      message: "An unexpected error occurred.",
      retryable: true,
    };
  }

  if (typeof error === "string") {
    const matched = ERROR_CODE_MAP[error];
    if (matched) {
      return {
        code: error,
        title: matched.title,
        message: matched.defaultMessage,
        retryable: matched.retryable,
      };
    }
    return {
      code: "ERROR",
      title: "Pipeline Error",
      message: error,
      retryable: true,
    };
  }

  const code = error.code || "INTERNAL";
  const matched = ERROR_CODE_MAP[code];

  return {
    code,
    title: matched ? matched.title : "Pipeline Error",
    message: error.message || (matched ? matched.defaultMessage : "An unexpected failure occurred."),
    detail: error.detail || null,
    retryable: typeof error.retryable === "boolean" ? error.retryable : (matched ? matched.retryable : true),
  };
}

/**
 * Retrieve the active job ID from URL query parameter (?job=...) with fallback to localStorage.
 */
export function getStoredJobId(): string | null {
  if (typeof window === "undefined") return null;

  try {
    const params = new URLSearchParams(window.location.search);
    const urlJob = params.get("job");
    if (urlJob && urlJob.trim()) {
      return urlJob.trim();
    }
  } catch {
    // Ignore URL search parsing errors
  }

  try {
    const localJob = localStorage.getItem("clipping_job_id");
    if (localJob && localJob.trim()) {
      return localJob.trim();
    }
  } catch {
    // Ignore localStorage access restrictions
  }

  return null;
}

/**
 * Persist or clear the active job ID in both URL query param (?job=...) and localStorage.
 */
export function setStoredJobId(jobId: string | null): void {
  if (typeof window === "undefined") return;

  try {
    const url = new URL(window.location.href);
    if (jobId && jobId.trim()) {
      url.searchParams.set("job", jobId.trim());
      localStorage.setItem("clipping_job_id", jobId.trim());
    } else {
      url.searchParams.delete("job");
      localStorage.removeItem("clipping_job_id");
    }
    window.history.replaceState(null, "", url.toString());
  } catch {
    // Fallback if URL or localStorage manipulation is blocked
  }
}

/**
 * Check whether a status represents a terminal state.
 */
export function isTerminalStatus(status?: string): boolean {
  return (
    status === "completed" ||
    status === "completed_with_warnings" ||
    status === "partial" ||
    status === "failed" ||
    status === "expired"
  );
}

/**
 * Check whether an active job is stalled based on updated_at timestamp.
 * Defaults to 180s timeout if stage hasn't updated.
 */
export function isJobStalled(
  jobState: JobStatusResponse | null,
  timeoutSeconds = 180
): boolean {
  if (!jobState || isTerminalStatus(jobState.status)) {
    return false;
  }

  if (!jobState.updated_at) {
    return false;
  }

  try {
    const updatedTime = new Date(jobState.updated_at).getTime();
    if (isNaN(updatedTime)) return false;

    const elapsedSeconds = (Date.now() - updatedTime) / 1000;
    return elapsedSeconds > timeoutSeconds;
  } catch {
    return false;
  }
}

/**
 * Calculate backoff delay in milliseconds for consecutive network/polling errors.
 */
export function calculatePollingBackoff(
  consecutiveFailures: number,
  baseIntervalMs = 2500,
  maxIntervalMs = 10000
): number {
  if (consecutiveFailures <= 0) return baseIntervalMs;
  const factor = Math.pow(1.5, Math.min(consecutiveFailures, 5));
  return Math.min(maxIntervalMs, Math.round(baseIntervalMs * factor));
}
