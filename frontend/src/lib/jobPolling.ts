/**
 * Job polling, resilience, error mapping, and persistence utilities.
 */

import type {
  ClipResult,
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

/**
 * Copy text to clipboard with modern Clipboard API and fallback to execCommand.
 * Returns true if copy succeeded, false on error.
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  if (typeof window === "undefined") return false;

  // 1. Try modern async Clipboard API if available and permitted
  if (
    navigator.clipboard &&
    typeof navigator.clipboard.writeText === "function"
  ) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // If permission denied or restricted, fall through to fallback
    }
  }

  // 2. Fallback using temporary off-screen textarea with execCommand
  try {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    // Position fixed and off-screen to avoid scroll jumping on iOS/Safari
    textarea.style.position = "fixed";
    textarea.style.top = "0";
    textarea.style.left = "-9999px";
    textarea.style.opacity = "0";
    textarea.setAttribute("readonly", "");
    document.body.appendChild(textarea);

    textarea.focus();
    textarea.select();
    textarea.setSelectionRange(0, text.length);

    const successful = document.execCommand("copy");
    document.body.removeChild(textarea);
    return Boolean(successful);
  } catch {
    return false;
  }
}

/**
 * Download a clip file handling cross-origin restrictions.
 * Fetches as a local Blob or falls back to backend Content-Disposition attachment endpoint.
 */
export async function downloadClip(
  videoUrl: string,
  filename: string,
  fallbackEndpoint?: string,
  onProgress?: (percent: number) => void
): Promise<void> {
  if (typeof window === "undefined") return;

  try {
    const res = await fetch(videoUrl, { mode: "cors" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const contentLength = Number(res.headers.get("content-length")) || 0;
    let blob: Blob;

    if (res.body && contentLength > 0 && onProgress) {
      const reader = res.body.getReader();
      let receivedBytes = 0;
      const chunks: BlobPart[] = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value) {
          chunks.push(value);
          receivedBytes += value.length;
          onProgress(
            Math.min(99, Math.round((receivedBytes / contentLength) * 100))
          );
        }
      }
      blob = new Blob(chunks, { type: "video/mp4" });
      onProgress(100);
    } else {
      blob = await res.blob();
      if (onProgress) onProgress(100);
    }

    const blobUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = blobUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(() => URL.revokeObjectURL(blobUrl), 2000);
  } catch (err) {
    // If CORS or blob fetch fails, use backend Content-Disposition endpoint
    if (fallbackEndpoint) {
      const link = document.createElement("a");
      link.href = fallbackEndpoint;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      return;
    }
    throw err;
  }
}

/**
 * Extract human-readable layout and framing notes from clip metadata and warnings.
 */
export function getLayoutNote(
  clip: ClipResult,
  jobWarnings?: JobWarning[]
): { label: string; tooltip?: string; isDegraded: boolean } | null {
  const warnings = clip.warnings || [];
  const warningTexts = warnings.map((w) =>
    typeof w === "string" ? w : w.message
  );

  // Check job warnings matching clip_index
  if (jobWarnings) {
    for (const jw of jobWarnings) {
      if (jw.clip_index !== null && jw.clip_index !== undefined) {
        warningTexts.push(jw.message);
      }
    }
  }

  // 1. Check for no-face / full-frame degradation
  const hasNoFace = warningTexts.some(
    (t) =>
      t.toLowerCase().includes("no face") ||
      t.toLowerCase().includes("full-frame") ||
      t.includes("NO_FACE_DETECTED")
  );
  if (hasNoFace) {
    return {
      label: "Full-frame (no face)",
      tooltip:
        "No human face was detected; framed as full-frame 9:16 instead of auto-tracking.",
      isDegraded: true,
    };
  }

  // 2. Check for fit_blur layout
  if (clip.layout_used === "fit_blur") {
    return {
      label: "Fit-blur",
      tooltip: "Rendered with padded blurred background.",
      isDegraded: false,
    };
  }

  // 3. Check for split layout
  if (clip.layout_used === "split") {
    return {
      label: "Split layout",
      tooltip: "Top webcam face pinned over bottom gameplay / screen pane.",
      isDegraded: false,
    };
  }

  // 4. Default speaker tracked layout
  if (clip.layout_used === "speaker") {
    return {
      label: "Speaker tracked",
      tooltip: "Smart 9:16 speaker tracking crop centered on detected face.",
      isDegraded: false,
    };
  }

  return null;
}
