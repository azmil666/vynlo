"use client";

import { useEffect, useState, useCallback, useMemo, useRef } from "react";
import type {
  ClipResult,
  JobErrorDetail,
  JobStatus,
  JobStatusResponse,
  JobWarning,
  PipelineErrorCode,
  PipelineStage,
} from "@/types/api";
import {
  calculatePollingBackoff,
  copyToClipboard,
  downloadClip,
  getApiBaseUrl,
  getLayoutNote,
  getStoredJobId,
  getStructuredError,
  isJobStalled,
  isTerminalStatus,
  resolveVideoUrl,
  setStoredJobId,
} from "@/lib/jobPolling";

const HIGHLIGHT_COLORS = [
  { label: "Yellow", value: "&H0000FFFF", hex: "#FFFF00" },
  { label: "Neon Green", value: "&H0000FF00", hex: "#00FF00" },
  { label: "Cyan", value: "&H00FFFF00", hex: "#00FFFF" },
  { label: "Hot Pink", value: "&H00B469FF", hex: "#FF69B4" },
];

const ORDERED_STAGES = [
  { key: "queued", label: "Job queued" },
  { key: "downloading", label: "Downloading media" },
  { key: "transcribing", label: "Transcribing audio" },
  { key: "analyzing", label: "Identifying viral hooks" },
  { key: "rendering", label: "Cropping & rendering 9:16" },
  { key: "completed", label: "Pipeline completed" },
] as const;

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="inline-flex items-center justify-center font-mono text-[10px] px-1.5 py-0.5 rounded bg-white/[0.08] text-[#888888] border border-white/[0.12] leading-none">
      {children}
    </kbd>
  );
}

function StatusDot({
  status,
  pulse = false,
}: {
  status: "online" | "offline" | "checking" | "active" | "error" | "done";
  pulse?: boolean;
}) {
  const colorClass = useMemo(() => {
    switch (status) {
      case "online":
      case "done":
        return "bg-[#50E3C2]";
      case "offline":
      case "error":
        return "bg-[#FF4D4D]";
      case "active":
        return "bg-[#EDEDED]";
      case "checking":
      default:
        return "bg-[#888888]";
    }
  }, [status]);

  return (
    <span className="relative flex h-2 w-2 items-center justify-center">
      {pulse && (
        <span
          className={`absolute inline-flex h-full w-full rounded-full opacity-60 animate-ping ${colorClass}`}
        />
      )}
      <span
        className={`relative inline-flex h-1.5 w-1.5 rounded-full ${colorClass}`}
      />
    </span>
  );
}

function SegmentedProgressBar({
  progress,
  isFailed = false,
  totalSegments = 24,
}: {
  progress: number;
  isFailed?: boolean;
  totalSegments?: number;
}) {
  const safeProgress = Math.min(100, Math.max(0, progress));
  const activeCount = Math.round((safeProgress / 100) * totalSegments);

  return (
    <div
      role="progressbar"
      aria-valuenow={safeProgress}
      aria-valuemin={0}
      aria-valuemax={100}
      className="flex w-full items-center gap-[3px]"
    >
      {Array.from({ length: totalSegments }).map((_, idx) => {
        const isFilled = idx < activeCount;
        const isCurrent = idx === activeCount - 1 && !isFailed && safeProgress < 100;

        let bgClass = "bg-white/[0.08]";
        if (isFailed) {
          bgClass = isFilled ? "bg-[#FF4D4D]" : "bg-white/[0.08]";
        } else if (isFilled) {
          bgClass = "bg-white";
        }

        return (
          <div
            key={idx}
            className={`h-1.5 flex-1 rounded-[1px] transition-colors duration-150 ${bgClass} ${
              isCurrent ? "animate-shimmer" : ""
            }`}
          />
        );
      })}
    </div>
  );
}

export default function HomePage() {
  const [url, setUrl] = useState("");
  const [layoutMode, setLayoutMode] = useState<"speaker" | "split">("speaker");
  const [maxClips, setMaxClips] = useState(3);
  const [captionPreset, setCaptionPreset] = useState<
    "clean_white" | "bold_yellow" | "boxed" | "minimal"
  >("clean_white");
  const [platformSafeZone, setPlatformSafeZone] = useState<
    "shorts" | "reels" | "tiktok"
  >("shorts");
  const [highlightColor, setHighlightColor] = useState("&H0000FFFF");
  const [showHookBanner, setShowHookBanner] = useState(true);

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [jobState, setJobState] = useState<JobStatus | null>(null);
  const [isReconnecting, setIsReconnecting] = useState(false);
  const [reconnectAttempt, setReconnectAttempt] = useState(0);
  const [lastSubmittedOptions, setLastSubmittedOptions] = useState<{
    url: string;
    layout_mode: "speaker" | "split";
    max_clips: number;
    caption_preset: "clean_white" | "bold_yellow" | "boxed" | "minimal";
    platform_safe_zone: "shorts" | "reels" | "tiktok";
    highlight_color: string;
    show_hook_banner: boolean;
  } | null>(null);

  const [isMac, setIsMac] = useState(false);
  const [copiedKeys, setCopiedKeys] = useState<Record<string, boolean>>({});
  const [copyStatus, setCopyStatus] = useState<Record<string, "idle" | "copied" | "error">>({});
  const [downloadProgress, setDownloadProgress] = useState<Record<string, number>>({});
  const [videoErrors, setVideoErrors] = useState<Record<string, boolean>>({});
  const [videoReloadKey, setVideoReloadKey] = useState<Record<string, number>>({});
  const [liveAnnouncement, setLiveAnnouncement] = useState("");

  // Detect platform in useEffect to avoid hydration mismatch
  useEffect(() => {
    if (typeof window !== "undefined") {
      setIsMac(/(Mac|iPhone|iPod|iPad)/i.test(navigator.userAgent));
    }
  }, []);

  // Restore job from URL query parameter (?job=...) or localStorage on mount
  useEffect(() => {
    const stored = getStoredJobId();
    if (stored) {
      setCurrentJobId(stored);
      setStoredJobId(stored);
    }
  }, []);

  // Check backend health on mount
  useEffect(() => {
    let isMounted = true;
    const checkHealth = async () => {
      try {
        const res = await fetch(`${getApiBaseUrl()}/health`);
        if (isMounted) setBackendOnline(res.ok);
      } catch {
        if (isMounted) setBackendOnline(false);
      }
    };
    checkHealth();
    return () => {
      isMounted = false;
    };
  }, []);

  // Poll job status with AbortController, exponential backoff, and terminal state detection
  useEffect(() => {
    if (!currentJobId) {
      setIsReconnecting(false);
      setReconnectAttempt(0);
      return;
    }

    let isMounted = true;
    let pollTimer: ReturnType<typeof setTimeout> | null = null;
    let failures = 0;
    const controller = new AbortController();

    const poll = async () => {
      try {
        const apiUrl = `${getApiBaseUrl()}/api/jobs/${currentJobId}`;
        const res = await fetch(apiUrl, { signal: controller.signal });

        if (!res.ok) {
          if (res.status === 404) {
            if (isMounted) {
              setJobState({
                job_id: currentJobId,
                status: "failed",
                stage: "failed",
                progress: 0,
                clips: [],
                error: {
                  code: "JOB_NOT_FOUND" as PipelineErrorCode,
                  message:
                    "Job not found. The job may have expired or never existed.",
                  retryable: false,
                },
              });
              setIsReconnecting(false);
              setReconnectAttempt(0);
            }
            return; // Terminal state: stop polling
          }

          if (res.status === 410) {
            const errJson = await res.json().catch(() => ({}));
            if (isMounted) {
              setJobState({
                job_id: currentJobId,
                status: "expired",
                stage: "expired",
                progress: 100,
                clips: [],
                error: {
                  code: "EXPIRED" as PipelineErrorCode,
                  message:
                    errJson.detail ||
                    "Job clips have expired and are no longer available.",
                  retryable: false,
                },
              });
              setIsReconnecting(false);
              setReconnectAttempt(0);
            }
            return; // Terminal state: stop polling
          }

          throw new Error(`HTTP ${res.status}`);
        }

        const data: JobStatus = await res.json();
        if (isMounted) {
          failures = 0;
          setIsReconnecting(false);
          setReconnectAttempt(0);
          setJobState(data);

          if (isTerminalStatus(data.status)) {
            return; // Stop polling on terminal status
          }

          pollTimer = setTimeout(poll, 2500);
        }
      } catch (err: unknown) {
        if (!isMounted || controller.signal.aborted) return;
        failures += 1;
        setIsReconnecting(true);
        setReconnectAttempt(failures);
        const delay = calculatePollingBackoff(failures);
        pollTimer = setTimeout(poll, delay);
      }
    };

    poll();

    return () => {
      isMounted = false;
      controller.abort();
      if (pollTimer) clearTimeout(pollTimer);
    };
  }, [currentJobId]);

  const handleSubmit = useCallback(
    async (
      e?: React.FormEvent,
      overrideOptions?: {
        url: string;
        layout_mode: "speaker" | "split";
        max_clips: number;
        caption_preset: "clean_white" | "bold_yellow" | "boxed" | "minimal";
        platform_safe_zone: "shorts" | "reels" | "tiktok";
        highlight_color: string;
        show_hook_banner: boolean;
      }
    ) => {
      if (e) e.preventDefault();
      const targetOptions = overrideOptions || {
        url: url.trim(),
        layout_mode: layoutMode,
        max_clips: Number(maxClips),
        caption_preset: captionPreset,
        platform_safe_zone: platformSafeZone,
        highlight_color:
          captionPreset === "bold_yellow" ? highlightColor : "&H00FFFFFF",
        show_hook_banner: showHookBanner,
      };

      if (!targetOptions.url || isSubmitting) return;
      if (jobState && !isTerminalStatus(jobState.status)) {
        return;
      }

      setIsSubmitting(true);
      setSubmitError(null);
      setJobState(null);
      setCurrentJobId(null);
      setIsReconnecting(false);
      setReconnectAttempt(0);
      setLastSubmittedOptions(targetOptions);

      // If overrideOptions provided (e.g. retry), sync form state
      if (overrideOptions) {
        setUrl(overrideOptions.url);
        setLayoutMode(overrideOptions.layout_mode);
        setMaxClips(overrideOptions.max_clips);
        setCaptionPreset(overrideOptions.caption_preset);
        setPlatformSafeZone(overrideOptions.platform_safe_zone);
        setHighlightColor(overrideOptions.highlight_color);
        setShowHookBanner(overrideOptions.show_hook_banner);
      }

      try {
        const res = await fetch(`${getApiBaseUrl()}/api/jobs`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(targetOptions),
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || "Failed to create clipping job");
        }

        const initialData: JobStatus = await res.json();
        setCurrentJobId(initialData.job_id);
        setStoredJobId(initialData.job_id);
        setJobState(initialData);
      } catch (err: unknown) {
        const msg =
          err instanceof Error
            ? err.message
            : "Failed to connect to API server";
        setSubmitError(msg);
      } finally {
        setIsSubmitting(false);
      }
    },
    [
      url,
      layoutMode,
      maxClips,
      captionPreset,
      platformSafeZone,
      highlightColor,
      showHookBanner,
      isSubmitting,
      jobState,
    ]
  );

  const handleRetry = useCallback(() => {
    if (lastSubmittedOptions) {
      handleSubmit(undefined, lastSubmittedOptions);
    } else if (url.trim()) {
      handleSubmit();
    }
  }, [handleSubmit, lastSubmittedOptions, url]);

  const handleReset = useCallback(() => {
    setCurrentJobId(null);
    setJobState(null);
    setUrl("");
    setSubmitError(null);
    setIsReconnecting(false);
    setReconnectAttempt(0);
    setStoredJobId(null);
  }, []);

  const handleCopy = useCallback(
    async (key: string, text: string, itemLabel = "Content") => {
      const success = await copyToClipboard(text);
      if (success) {
        setCopiedKeys((prev) => ({ ...prev, [key]: true }));
        setCopyStatus((prev) => ({ ...prev, [key]: "copied" }));
        setLiveAnnouncement(`${itemLabel} copied to clipboard.`);
        setTimeout(() => {
          setCopiedKeys((prev) => ({ ...prev, [key]: false }));
          setCopyStatus((prev) => ({ ...prev, [key]: "idle" }));
        }, 1800);
      } else {
        setCopiedKeys((prev) => ({ ...prev, [key]: false }));
        setCopyStatus((prev) => ({ ...prev, [key]: "error" }));
        setLiveAnnouncement(
          `Failed to copy ${itemLabel}. Please select and copy manually.`
        );
        setTimeout(() => {
          setCopyStatus((prev) => ({ ...prev, [key]: "idle" }));
        }, 2500);
      }
    },
    []
  );

  const handleDownloadClip = useCallback(
    async (clip: ClipResult) => {
      if (!currentJobId) return;
      const resolvedUrl = resolveVideoUrl(clip.video_url);
      const fallbackEndpoint = `${getApiBaseUrl()}/api/jobs/${currentJobId}/clips/${clip.clip_id}/download`;
      const filename = `${currentJobId.slice(0, 8)}_${clip.clip_id}.mp4`;

      setDownloadProgress((prev) => ({ ...prev, [clip.clip_id]: 0 }));
      setLiveAnnouncement(`Starting download for ${clip.title}`);

      try {
        await downloadClip(
          resolvedUrl,
          filename,
          fallbackEndpoint,
          (percent) => {
            setDownloadProgress((prev) => ({ ...prev, [clip.clip_id]: percent }));
          }
        );
        setLiveAnnouncement(`Download completed for ${clip.title}`);
      } catch (err) {
        console.error("Download failed:", err);
        setLiveAnnouncement(`Download failed for ${clip.title}`);
      } finally {
        setTimeout(() => {
          setDownloadProgress((prev) => {
            const next = { ...prev };
            delete next[clip.clip_id];
            return next;
          });
        }, 1200);
      }
    },
    [currentJobId]
  );

  // Keyboard shortcut listener (Cmd/Ctrl + Enter)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        handleSubmit();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [handleSubmit]);

  const selectedColorObj = useMemo(
    () =>
      HIGHLIGHT_COLORS.find((c) => c.value === highlightColor) ||
      HIGHLIGHT_COLORS[0],
    [highlightColor]
  );

  const isJobActive = Boolean(
    jobState &&
      jobState.status !== "completed" &&
      jobState.status !== "completed_with_warnings" &&
      jobState.status !== "partial" &&
      jobState.status !== "failed" &&
      jobState.status !== "expired"
  );

  const isStalled = useMemo(() => isJobStalled(jobState), [jobState]);

  const currentStageIndex = useMemo(() => {
    if (!jobState) return -1;
    if (
      jobState.status === "completed" ||
      jobState.status === "completed_with_warnings" ||
      jobState.status === "partial"
    ) {
      return ORDERED_STAGES.length;
    }
    return ORDERED_STAGES.findIndex((s) => s.key === jobState.status);
  }, [jobState]);

  return (
    <div className="relative min-h-screen bg-black text-[#EDEDED] antialiased">
      {/* Screen-reader accessible live announcement region */}
      <div
        role="status"
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
      >
        {liveAnnouncement}
      </div>

      {/* Background Architectural Dot Grid */}
      <div className="pointer-events-none fixed inset-0 dot-grid z-0 opacity-80" />

      {/* Main Content Container */}
      <div className="relative z-10 mx-auto max-w-6xl px-4 sm:px-6 py-10 sm:py-14 space-y-8">
        {/* Header Bar */}
        <header className="flex items-center justify-between border-b border-white/[0.08] pb-4">
          <div className="flex items-center gap-2.5">
            <svg
              className="w-4 h-4 text-white"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <line x1="4" y1="20" x2="20" y2="4" />
              <polyline points="4,4 4,20 20,20" />
            </svg>
            <span className="font-semibold text-sm tracking-tight text-[#EDEDED]">
              Vynlo Studio
            </span>
            <span className="text-[11px] font-mono text-[#888888] px-1.5 py-0.5 rounded border border-white/[0.08] bg-[#0A0A0A]">
              v0.1
            </span>
          </div>

          <div className="flex items-center gap-2 text-xs font-mono">
            <StatusDot
              status={
                backendOnline === true
                  ? "online"
                  : backendOnline === false
                  ? "offline"
                  : "checking"
              }
              pulse={backendOnline === true}
            />
            <span className="text-[#888888]">
              {backendOnline === true
                ? "api · online"
                : backendOnline === false
                ? "api · offline"
                : "api · checking"}
            </span>
          </div>
        </header>

        {/* Hero Section */}
        <section className="space-y-1.5 pt-2">
          <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight text-[#EDEDED]">
            YouTube to Shorts
          </h1>
          <p className="text-sm text-[#888888] max-w-xl">
            Extract high-retention vertical clips with MediaPipe face tracking,
            speech transcription, and animated karaoke captions.
          </p>
        </section>

        {/* Composer Panel */}
        <section className="bg-[#0A0A0A] border border-white/[0.08] rounded-lg p-5 sm:p-6 space-y-6">
          <form onSubmit={handleSubmit} className="space-y-5">
            {/* URL Input */}
            <div className="space-y-2">
              <label
                htmlFor="video-url"
                className="block text-xs uppercase tracking-wider font-mono text-[#888888]"
              >
                Source Stream / Video URL
              </label>
              <div className="relative flex items-center">
                <span className="absolute left-3.5 text-[#888888] pointer-events-none">
                  <svg
                    className="w-4 h-4"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.8"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden="true"
                  >
                    <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
                    <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
                  </svg>
                </span>
                <input
                  id="video-url"
                  type="url"
                  required
                  placeholder="https://youtube.com/watch?v=..."
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  disabled={isSubmitting || isJobActive}
                  className="w-full bg-black border border-white/[0.08] hover:border-white/[0.16] focus:border-white/[0.24] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white/30 rounded-md pl-10 pr-4 py-2.5 text-sm font-mono text-[#EDEDED] placeholder-[#555555] transition-colors disabled:opacity-50"
                />
              </div>
            </div>

            {/* Config Controls Grid */}
            <div className="space-y-4 pt-1">
              {/* Layout Mode Segmented Control */}
              <div className="space-y-2">
                <span className="block text-xs uppercase tracking-wider font-mono text-[#888888]">
                  Layout Framing
                </span>
                <div
                  role="radiogroup"
                  aria-label="Layout Framing"
                  className="grid grid-cols-1 sm:grid-cols-2 p-1 bg-black rounded-md border border-white/[0.08] gap-1"
                >
                  <button
                    type="button"
                    role="radio"
                    aria-checked={layoutMode === "speaker"}
                    onClick={() => setLayoutMode("speaker")}
                    disabled={isSubmitting || isJobActive}
                    className={`flex items-start gap-3 p-2.5 rounded text-left transition-colors ${
                      layoutMode === "speaker"
                        ? "bg-[#111111] text-[#EDEDED] border border-white/[0.24]"
                        : "text-[#888888] hover:text-[#EDEDED] border border-transparent"
                    }`}
                  >
                    <svg
                      className={`w-3.5 h-3.5 flex-shrink-0 mt-0.5 ${
                        layoutMode === "speaker" ? "text-white" : "text-[#888888]"
                      }`}
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.8"
                    >
                      <rect x="5" y="3" width="14" height="18" rx="2" />
                      <circle cx="12" cy="10" r="3" />
                      <path d="M7 19c0-2.5 2-4 5-4s5 1.5 5 4" />
                    </svg>
                    <div className="flex flex-col gap-0.5">
                      <span className="text-xs font-medium text-[#EDEDED]">
                        Smart Face Crop
                      </span>
                      <span className="text-[11px] text-[#888888] leading-tight">
                        Podcasts, interviews & talking heads
                      </span>
                    </div>
                  </button>

                  <button
                    type="button"
                    role="radio"
                    aria-checked={layoutMode === "split"}
                    onClick={() => setLayoutMode("split")}
                    disabled={isSubmitting || isJobActive}
                    className={`flex items-start gap-3 p-2.5 rounded text-left transition-colors ${
                      layoutMode === "split"
                        ? "bg-[#111111] text-[#EDEDED] border border-white/[0.24]"
                        : "text-[#888888] hover:text-[#EDEDED] border border-transparent"
                    }`}
                  >
                    <svg
                      className={`w-3.5 h-3.5 flex-shrink-0 mt-0.5 ${
                        layoutMode === "split" ? "text-white" : "text-[#888888]"
                      }`}
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.8"
                    >
                      <rect x="5" y="3" width="14" height="18" rx="2" />
                      <line x1="5" y1="12" x2="19" y2="12" />
                      <circle cx="15.5" cy="16.5" r="1.5" />
                    </svg>
                    <div className="flex flex-col gap-0.5">
                      <span className="text-xs font-medium text-[#EDEDED]">
                        Split Screen
                      </span>
                      <span className="text-[11px] text-[#888888] leading-tight">
                        Gaming & streams with corner facecam
                      </span>
                    </div>
                  </button>
                </div>
                <p className="text-xs text-[#666666] font-mono">
                  {layoutMode === "speaker"
                    ? "Tracks speaker face dynamically across 9:16 full-screen frame."
                    : "Stacks gameplay content (top) over detected facecam (bottom) without distortion."}
                </p>
              </div>

              {/* Safe Zone, Max Clips & Hook Banner Row */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                {/* Platform Safe Zone Toggle */}
                <div className="space-y-2">
                  <span className="block text-xs uppercase tracking-wider font-mono text-[#888888]">
                    Safe Zone
                  </span>
                  <div
                    role="radiogroup"
                    aria-label="Platform Safe Zone"
                    className="grid grid-cols-3 p-1 bg-black rounded-md border border-white/[0.08] gap-1"
                  >
                    {(
                      [
                        { id: "shorts", label: "Shorts" },
                        { id: "reels", label: "Reels" },
                        { id: "tiktok", label: "TikTok" },
                      ] as const
                    ).map(({ id, label }) => (
                      <button
                        key={id}
                        type="button"
                        role="radio"
                        aria-checked={platformSafeZone === id}
                        onClick={() => setPlatformSafeZone(id)}
                        disabled={isSubmitting || isJobActive}
                        className={`py-1.5 px-2 rounded text-xs font-medium transition-colors ${
                          platformSafeZone === id
                            ? "bg-[#111111] text-[#EDEDED] border border-white/[0.16]"
                            : "text-[#888888] hover:text-[#EDEDED] border border-transparent"
                        }`}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Max Clips Segmented Control */}
                <div className="space-y-2">
                  <span className="block text-xs uppercase tracking-wider font-mono text-[#888888]">
                    Max Clips
                  </span>
                  <div
                    role="radiogroup"
                    aria-label="Max Clips"
                    className="grid grid-cols-3 p-1 bg-black rounded-md border border-white/[0.08] gap-1"
                  >
                    {[1, 3, 5].map((num) => (
                      <button
                        key={num}
                        type="button"
                        role="radio"
                        aria-checked={maxClips === num}
                        onClick={() => setMaxClips(num)}
                        disabled={isSubmitting || isJobActive}
                        className={`py-1.5 px-2 rounded text-xs font-mono tabular-nums transition-colors ${
                          maxClips === num
                            ? "bg-[#111111] text-[#EDEDED] border border-white/[0.16]"
                            : "text-[#888888] hover:text-[#EDEDED] border border-transparent"
                        }`}
                      >
                        {num}
                      </button>
                    ))}
                  </div>
                </div>

                {/* 3s Hook Banner Toggle */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="block text-xs uppercase tracking-wider font-mono text-[#888888]">
                      Hook Banner
                    </span>
                    <span className="text-[10px] font-mono text-[#666666]">
                      3.2s
                    </span>
                  </div>
                  <button
                    type="button"
                    role="switch"
                    aria-checked={showHookBanner}
                    onClick={() => setShowHookBanner(!showHookBanner)}
                    disabled={isSubmitting || isJobActive}
                    className={`w-full h-[38px] px-3 rounded-md border flex items-center justify-between text-xs font-medium transition-colors ${
                      showHookBanner
                        ? "bg-[#111111] text-[#EDEDED] border-white/[0.16]"
                        : "bg-black text-[#888888] border-white/[0.08] hover:text-[#EDEDED]"
                    }`}
                  >
                    <span className="truncate">Top Hook Pill</span>
                    <span
                      className={`relative inline-flex h-4 w-7 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out ${
                        showHookBanner ? "bg-white" : "bg-white/20"
                      }`}
                    >
                      <span
                        className={`pointer-events-none inline-block h-3 w-3 transform rounded-full bg-black shadow-lg ring-0 transition duration-200 ease-in-out ${
                          showHookBanner ? "translate-x-3" : "translate-x-0"
                        }`}
                      />
                    </span>
                  </button>
                </div>
              </div>

              {/* Subtitles & Styling Controls */}
              <div
                className={`grid grid-cols-1 ${
                  captionPreset === "bold_yellow"
                    ? "md:grid-cols-3"
                    : "md:grid-cols-1"
                } gap-4 items-start`}
              >
                {/* Caption Style Segmented Selector */}
                <div
                  className={`space-y-2 ${
                    captionPreset === "bold_yellow"
                      ? "md:col-span-2"
                      : "w-full"
                  }`}
                >
                  <span className="block text-xs uppercase tracking-wider font-mono text-[#888888]">
                    Caption Style
                  </span>
                  <div
                    role="radiogroup"
                    aria-label="Caption Style Preset"
                    className="grid grid-cols-2 sm:grid-cols-4 p-1 bg-black rounded-md border border-white/[0.08] gap-1"
                  >
                    {(
                      [
                        { id: "clean_white", label: "Clean White" },
                        { id: "bold_yellow", label: "Bold Yellow" },
                        { id: "boxed", label: "Boxed" },
                        { id: "minimal", label: "Minimal" },
                      ] as const
                    ).map(({ id, label }) => (
                      <button
                        key={id}
                        type="button"
                        role="radio"
                        aria-checked={captionPreset === id}
                        onClick={() => setCaptionPreset(id)}
                        disabled={isSubmitting || isJobActive}
                        className={`py-1.5 px-2.5 rounded text-xs font-medium text-center transition-colors ${
                          captionPreset === id
                            ? "bg-[#111111] text-[#EDEDED] border border-white/[0.16]"
                            : "text-[#888888] hover:text-[#EDEDED] border border-transparent"
                        }`}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Subtitle Highlight Color Picker - ONLY shown when Bold Yellow is selected */}
                {captionPreset === "bold_yellow" && (
                  <div className="space-y-2">
                    <span className="block text-xs uppercase tracking-wider font-mono text-[#888888]">
                      Highlight Accent
                    </span>
                    <div className="flex items-center justify-between p-1 px-3 h-[38px] bg-black rounded-md border border-white/[0.08]">
                      <div
                        role="radiogroup"
                        aria-label="Highlight Color"
                        className="flex items-center gap-2.5"
                      >
                        {HIGHLIGHT_COLORS.map((c) => {
                          const isSelected = highlightColor === c.value;
                          return (
                            <button
                              key={c.value}
                              type="button"
                              role="radio"
                              aria-label={c.label}
                              aria-checked={isSelected}
                              onClick={() => setHighlightColor(c.value)}
                              disabled={isSubmitting || isJobActive}
                              className={`w-4 h-4 rounded-full transition-transform focus:outline-none ${
                                isSelected
                                  ? "ring-1 ring-white ring-offset-2 ring-offset-[#0A0A0A] scale-110"
                                  : "opacity-75 hover:opacity-100"
                              }`}
                              style={{ backgroundColor: c.hex }}
                            />
                          );
                        })}
                      </div>
                      <span className="text-[11px] font-mono text-[#888888]">
                        {selectedColorObj.label}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Error Banner */}
            {submitError && (
              <div className="p-3 rounded-md bg-[#FF4D4D]/[0.05] border border-[#FF4D4D]/20 text-xs font-mono text-[#FF4D4D] flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <svg
                    className="w-3.5 h-3.5 flex-shrink-0"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <circle cx="12" cy="12" r="10" />
                    <line x1="12" y1="8" x2="12" y2="12" />
                    <line x1="12" y1="16" x2="12.01" y2="16" />
                  </svg>
                  <span>{submitError}</span>
                </div>
                <button
                  type="button"
                  onClick={() => setSubmitError(null)}
                  className="text-[11px] underline opacity-80 hover:opacity-100"
                >
                  Dismiss
                </button>
              </div>
            )}

            {/* Action Bar */}
            <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-2">
              <div className="flex items-center gap-2 text-xs text-[#888888]">
                <span>Shortcut</span>
                <Kbd>{isMac ? "⌘ ↵" : "Ctrl ↵"}</Kbd>
              </div>

              <button
                type="submit"
                disabled={isSubmitting || !url.trim() || isJobActive}
                className="w-full sm:w-auto h-9 px-4 rounded-md bg-white text-black hover:bg-[#EDEDED] active:scale-[0.98] font-medium text-xs tracking-tight transition-all disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {isSubmitting ? (
                  <>
                    <svg
                      className="w-3.5 h-3.5 animate-spin"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2.5"
                    >
                      <circle cx="12" cy="12" r="10" strokeOpacity="0.25" />
                      <path d="M12 2a10 10 0 0 1 10 10" />
                    </svg>
                    <span>Submitting...</span>
                  </>
                ) : isJobActive ? (
                  <>
                    <svg
                      className="w-3.5 h-3.5 animate-spin"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2.5"
                    >
                      <circle cx="12" cy="12" r="10" strokeOpacity="0.25" />
                      <path d="M12 2a10 10 0 0 1 10 10" />
                    </svg>
                    <span>Running ({jobState?.progress ?? 0}%)</span>
                  </>
                ) : (
                  <span>Generate Shorts</span>
                )}
              </button>
            </div>
          </form>
        </section>

        {/* Pipeline Status Panel */}
        {jobState && (
          <section className="bg-[#0A0A0A] border border-white/[0.08] rounded-lg p-5 space-y-4">
            {/* Header bar */}
            <div className="flex items-center justify-between border-b border-white/[0.08] pb-3">
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase font-mono tracking-wider text-[#888888]">
                  Job
                </span>
                <button
                  type="button"
                  onClick={() => handleCopy("job-id", jobState.job_id)}
                  title="Click to copy Job ID"
                  className="font-mono text-xs text-[#EDEDED] hover:text-white px-1.5 py-0.5 rounded hover:bg-[#111111] transition-colors flex items-center gap-1.5"
                >
                  <span>{jobState.job_id.slice(0, 12)}…</span>
                  {copiedKeys["job-id"] ? (
                    <span className="text-[10px] text-[#50E3C2]">copied</span>
                  ) : (
                    <svg
                      className="w-3 h-3 text-[#888888]"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.8"
                    >
                      <rect x="9" y="9" width="13" height="13" rx="2" />
                      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                    </svg>
                  )}
                </button>
              </div>

              <div
                aria-live="polite"
                className="flex items-center gap-2 font-mono text-xs text-[#888888]"
              >
                <span className="text-[#EDEDED] capitalize">
                  {jobState.status}
                </span>
                <span>·</span>
                <span className="tabular-nums">{jobState.progress}%</span>
              </div>
            </div>

            {/* Reconnecting Banner */}
            {isReconnecting && isJobActive && (
              <div className="p-2.5 rounded bg-amber-500/10 border border-amber-500/30 text-amber-300 font-mono text-xs flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping" />
                  <span>
                    Network connection interrupted. Reconnecting (attempt{" "}
                    {reconnectAttempt})…
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    if (currentJobId) {
                      const id = currentJobId;
                      setCurrentJobId(null);
                      setTimeout(() => setCurrentJobId(id), 50);
                    }
                  }}
                  className="underline hover:text-amber-100 text-[11px]"
                >
                  Retry now
                </button>
              </div>
            )}

            {/* Stalled Worker Alert */}
            {isStalled && isJobActive && (
              <div className="p-3 rounded bg-amber-500/10 border border-amber-500/30 text-amber-300 font-mono text-xs flex items-center justify-between gap-3">
                <div>
                  <p className="font-semibold">Worker Stall Detected</p>
                  <p className="text-[11px] opacity-80">
                    No pipeline updates received for over 3 minutes. The background
                    worker process may have been lost or stalled.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={handleRetry}
                  className="h-7 px-3 rounded bg-amber-500/20 hover:bg-amber-500/30 text-amber-200 border border-amber-500/40 text-xs font-semibold shrink-0 transition-colors"
                >
                  Restart Job
                </button>
              </div>
            )}

            {/* Segmented linear progress bar */}
            <SegmentedProgressBar
              progress={jobState.progress}
              isFailed={jobState.status === "failed"}
            />

            {/* Live Pipeline Step List */}
            <div className="pt-2 space-y-2 font-mono text-xs">
              {ORDERED_STAGES.map((stage, idx) => {
                const isCompleted =
                  jobState.status === "completed" ||
                  jobState.status === "completed_with_warnings" ||
                  jobState.status === "partial" ||
                  (currentStageIndex > idx && jobState.status !== "failed");
                const isCurrent =
                  stage.key === jobState.status &&
                  stage.key !== "completed";
                const isFailed =
                  jobState.status === "failed" && currentStageIndex === idx;

                return (
                  <div key={stage.key} className="space-y-1">
                    <div className="flex items-center justify-between py-1 px-1.5 rounded transition-colors">
                      <div className="flex items-center gap-2.5">
                        {isCompleted ? (
                          <span className="w-3.5 h-3.5 rounded-full bg-[#50E3C2]/10 text-[#50E3C2] flex items-center justify-center text-[10px]">
                            ✓
                          </span>
                        ) : isCurrent ? (
                          <span className="w-3.5 h-3.5 flex items-center justify-center">
                            <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse-dot" />
                          </span>
                        ) : isFailed ? (
                          <span className="w-3.5 h-3.5 rounded-full bg-[#FF4D4D]/10 text-[#FF4D4D] flex items-center justify-center text-[10px]">
                            ✕
                          </span>
                        ) : (
                          <span className="w-3.5 h-3.5 flex items-center justify-center">
                            <span className="w-1.5 h-1.5 rounded-full border border-white/20" />
                          </span>
                        )}

                        <span
                          className={
                            isCurrent
                              ? "text-[#EDEDED] font-medium"
                              : isCompleted
                              ? "text-[#888888]"
                              : isFailed
                              ? "text-[#FF4D4D]"
                              : "text-[#888888]/50"
                          }
                        >
                          {stage.label}
                        </span>
                      </div>

                      <span className="text-[11px] text-[#555555]">
                        {isCompleted
                          ? "done"
                          : isCurrent
                          ? "running…"
                          : isFailed
                          ? "failed"
                          : "pending"}
                      </span>
                    </div>

                    {/* Per-clip rendering progress */}
                    {stage.key === "rendering" &&
                      isCurrent &&
                      jobState.clips &&
                      jobState.clips.length > 0 && (
                        <div className="pl-6 pb-1 text-[11px] text-[#888888] flex items-center gap-2">
                          <span className="text-[#50E3C2]">●</span>
                          <span>
                            Clips rendered:{" "}
                            {
                              jobState.clips.filter(
                                (c) => c.ok !== false && c.render_ok !== false
                              ).length
                            }{" "}
                            / {jobState.clips.length}
                          </span>
                        </div>
                      )}
                  </div>
                );
              })}
            </div>

            {/* Job-Level Warnings Panel */}
            {jobState.warnings && jobState.warnings.length > 0 && (
              <div className="p-3 rounded-lg bg-amber-500/[0.08] border border-amber-500/20 text-amber-300 font-mono text-xs space-y-1.5">
                <div className="flex items-center gap-2 font-semibold">
                  <span>⚠ Pipeline Notices ({jobState.warnings.length})</span>
                </div>
                <ul className="list-disc list-inside space-y-1 text-[11px] text-amber-200/90 pl-1">
                  {jobState.warnings.map((w, idx) => (
                    <li key={idx}>
                      <span className="font-mono text-amber-400 font-medium">
                        [{w.code}]
                      </span>{" "}
                      {w.message}
                      {w.clip_index !== null && w.clip_index !== undefined && (
                        <span className="opacity-75 text-[10px] ml-1">
                          (Clip #{w.clip_index + 1})
                        </span>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Failure Box with Structured Error & Try Again */}
            {jobState.status === "failed" &&
              (() => {
                const structuredErr = getStructuredError(
                  jobState.error,
                  jobState.status
                );
                return (
                  <div className="pt-2 space-y-3">
                    <div className="p-3.5 rounded bg-[#FF4D4D]/[0.08] border border-[#FF4D4D]/30 font-mono text-xs text-[#FF4D4D] space-y-1.5">
                      <div className="flex items-center justify-between">
                        <p className="font-semibold text-sm">
                          {structuredErr.title}
                        </p>
                        <span className="px-1.5 py-0.5 rounded text-[10px] bg-[#FF4D4D]/20 text-[#FF4D4D] border border-[#FF4D4D]/40 font-mono">
                          [{structuredErr.code}]
                        </span>
                      </div>
                      <p className="opacity-90 font-sans text-xs">
                        {structuredErr.message}
                      </p>
                      {structuredErr.detail && (
                        <p className="font-mono text-[11px] text-[#FF4D4D]/75 pt-1 border-t border-[#FF4D4D]/20">
                          Detail: {structuredErr.detail}
                        </p>
                      )}
                    </div>

                    <div className="flex items-center justify-end gap-2">
                      <button
                        type="button"
                        onClick={handleReset}
                        className="h-8 px-3 rounded text-xs font-mono text-[#888888] hover:text-[#EDEDED] border border-white/[0.08] hover:bg-[#111111] transition-colors"
                      >
                        Try another link
                      </button>

                      {structuredErr.retryable && (
                        <button
                          type="button"
                          onClick={handleRetry}
                          className="h-8 px-4 rounded text-xs font-mono font-medium text-black bg-white hover:bg-[#E0E0E0] transition-colors"
                        >
                          Try Again
                        </button>
                      )}
                    </div>
                  </div>
                );
              })()}

            {/* Expired Box with Structured Error & Re-render Option */}
            {jobState.status === "expired" &&
              (() => {
                const structuredErr = getStructuredError(
                  jobState.error,
                  "expired"
                );
                return (
                  <div className="pt-2 space-y-3">
                    <div className="p-3.5 rounded bg-amber-500/[0.08] border border-amber-500/20 font-mono text-xs text-amber-300 space-y-1.5">
                      <div className="flex items-center justify-between">
                        <p className="font-semibold text-sm">
                          {structuredErr.title}
                        </p>
                        <span className="px-1.5 py-0.5 rounded text-[10px] bg-amber-500/20 text-amber-300 border border-amber-500/40">
                          [{structuredErr.code}]
                        </span>
                      </div>
                      <p className="opacity-90 font-sans text-xs">
                        {structuredErr.message}
                      </p>
                    </div>

                    <div className="flex justify-end gap-2">
                      <button
                        type="button"
                        onClick={handleReset}
                        className="h-8 px-3 rounded text-xs font-mono text-[#888888] hover:text-[#EDEDED] border border-white/[0.08] hover:bg-[#111111] transition-colors"
                      >
                        Start new job
                      </button>
                      <button
                        type="button"
                        onClick={handleRetry}
                        className="h-8 px-4 rounded text-xs font-mono font-medium text-black bg-white hover:bg-[#E0E0E0] transition-colors"
                      >
                        Re-render Video
                      </button>
                    </div>
                  </div>
                );
              })()}
          </section>
        )}

        {/* Results Section */}
        <section className="space-y-4">
          <div className="flex items-center justify-between border-b border-white/[0.08] pb-3">
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-medium text-[#EDEDED]">Clips</h2>
              <span className="font-mono text-xs px-1.5 py-0.5 rounded border border-white/[0.08] text-[#888888] tabular-nums">
                {jobState?.clips?.length ?? 0}
              </span>
            </div>

            {(jobState?.status === "completed" ||
              jobState?.status === "completed_with_warnings" ||
              jobState?.status === "partial") &&
              jobState.clips &&
              jobState.clips.some((c) => c.ok !== false && c.render_ok !== false) && (
                <a
                  href={`${getApiBaseUrl()}/api/jobs/${jobState.job_id}/download-all`}
                  download={`vynlo-clips_${jobState.job_id.slice(0, 8)}.zip`}
                  className="h-7 px-2.5 rounded text-xs font-mono text-[#EDEDED] bg-white/[0.08] hover:bg-white/[0.12] border border-white/[0.12] transition-colors flex items-center gap-1.5"
                >
                  <svg
                    className="w-3.5 h-3.5"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.8"
                  >
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                  <span>Download All (.ZIP)</span>
                </a>
              )}
          </div>

          {(jobState?.status === "completed_with_warnings" ||
            jobState?.status === "partial") && (
            <div className="p-3 rounded-lg bg-amber-500/[0.08] border border-amber-500/20 font-mono text-xs text-amber-300">
              <span className="font-semibold">Partial Completion:</span> One or more clips failed to render or upload, but successful clips are delivered below.
            </div>
          )}

          {jobState && jobState.clips && jobState.clips.length > 0 ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {jobState.clips.map((clip, idx) => {
                const resolvedUrl = resolveVideoUrl(clip.video_url);
                const captionStatus = copyStatus[`${clip.clip_id}-caption`];
                const linkStatus = copyStatus[`${clip.clip_id}-link`];
                const isDownloading = downloadProgress[clip.clip_id] !== undefined;
                const dlPercent = downloadProgress[clip.clip_id];

                const captionText = [
                  clip.title,
                  clip.description,
                  clip.hashtags && clip.hashtags.length > 0
                    ? clip.hashtags
                        .map((t) => `#${t.replace(/^#/, "")}`)
                        .join(" ")
                    : "",
                ]
                  .filter(Boolean)
                  .join("\n\n");

                const scoreColor =
                  clip.virality_score >= 80
                    ? "text-[#EDEDED]"
                    : clip.virality_score >= 50
                    ? "text-[#888888]"
                    : "text-[#555555]";

                const isHeuristic =
                  clip.selection_method === "heuristic" || clip.is_fallback;
                const layoutNote = getLayoutNote(clip, jobState.warnings);

                return (
                  <div
                    key={clip.clip_id}
                    style={{ animationDelay: `${idx * 40}ms` }}
                    className="animate-fade-in bg-[#0A0A0A] border border-white/[0.08] hover:border-white/[0.16] rounded-lg overflow-hidden transition-colors flex flex-col"
                  >
                    {/* Header Metadata Bar */}
                    <div className="px-3 h-9 border-b border-white/[0.08] flex items-center justify-between gap-2 text-xs">
                      <span
                        title={clip.title}
                        className="truncate font-medium text-[#EDEDED] text-xs max-w-[50%]"
                      >
                        {clip.title}
                      </span>
                      <div className="flex items-center gap-1.5 font-mono tabular-nums text-[11px] shrink-0">
                        <span className="text-[#888888]">
                          {clip.duration}s
                        </span>

                        {/* AI score ONLY shown for LLM selected clips */}
                        {!isHeuristic ? (
                          <>
                            <span className="text-[#555555]">·</span>
                            <span
                              className={scoreColor}
                              title="Predicted Virality Score (0-100)"
                            >
                              {clip.virality_score}/100
                            </span>
                          </>
                        ) : (
                          /* Heuristic label driven by selection_method with tooltip */
                          <span
                            className="px-1.5 py-0.5 rounded text-[10px] font-mono tracking-wider bg-amber-500/10 text-amber-400 border border-amber-500/20"
                            title={
                              clip.selection_reason ||
                              "Auto-selected via speech heuristics (not AI-ranked)"
                            }
                          >
                            Auto-selected (not AI-ranked)
                          </span>
                        )}

                        {/* Layout notes from warnings and layout_used */}
                        {layoutNote && (
                          <span
                            className={`px-1.5 py-0.5 rounded text-[10px] font-mono ${
                              layoutNote.isDegraded
                                ? "bg-amber-500/10 text-amber-300 border border-amber-500/30"
                                : "bg-white/[0.06] text-[#888888] border border-white/[0.08]"
                            }`}
                            title={layoutNote.tooltip}
                          >
                            {layoutNote.label}
                          </span>
                        )}
                      </div>
                    </div>

                    {/* 9:16 Video Frame / Error State */}
                    <div className="aspect-[9/16] bg-black w-full relative flex items-center justify-center p-4">
                      {clip.ok === false || clip.render_ok === false ? (
                        <div className="text-center space-y-2 font-mono">
                          <span className="w-8 h-8 rounded-full bg-[#FF4D4D]/10 text-[#FF4D4D] flex items-center justify-center mx-auto text-sm font-bold">
                            ✕
                          </span>
                          <p className="text-xs text-[#FF4D4D] font-medium">
                            Render Failed
                          </p>
                          {clip.error_code && (
                            <span className="inline-block px-1.5 py-0.5 rounded text-[10px] bg-[#FF4D4D]/10 text-[#FF4D4D] border border-[#FF4D4D]/20">
                              [{clip.error_code}]
                            </span>
                          )}
                          {clip.error_message && (
                            <p className="text-[11px] text-[#888888] line-clamp-3 font-sans">
                              {clip.error_message}
                            </p>
                          )}
                        </div>
                      ) : videoErrors[clip.clip_id] ? (
                        <div className="text-center space-y-2 font-mono p-4">
                          <span className="w-8 h-8 rounded-full bg-[#FF4D4D]/10 text-[#FF4D4D] flex items-center justify-center mx-auto text-sm font-bold">
                            !
                          </span>
                          <p className="text-xs text-[#FF4D4D] font-medium">
                            Playback Error
                          </p>
                          <p className="text-[11px] text-[#888888] font-sans">
                            Failed to stream video or decode byte range.
                          </p>
                          <button
                            type="button"
                            onClick={() => {
                              setVideoErrors((prev) => ({
                                ...prev,
                                [clip.clip_id]: false,
                              }));
                              setVideoReloadKey((prev) => ({
                                ...prev,
                                [clip.clip_id]: (prev[clip.clip_id] || 0) + 1,
                              }));
                            }}
                            className="mt-2 h-7 px-3 rounded text-xs font-mono text-white bg-white/[0.1] hover:bg-white/[0.2] border border-white/[0.15] transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white/40"
                          >
                            Retry Playback
                          </button>
                        </div>
                      ) : (
                        <video
                          key={`${clip.clip_id}-${
                            videoReloadKey[clip.clip_id] || 0
                          }`}
                          src={resolvedUrl}
                          controls
                          playsInline
                          preload="metadata"
                          className="w-full h-full object-contain bg-black"
                          onError={() => {
                            setVideoErrors((prev) => ({
                              ...prev,
                              [clip.clip_id]: true,
                            }));
                          }}
                        />
                      )}
                    </div>

                    {/* Social Kit: Description & Hashtags */}
                    {(clip.description ||
                      (clip.hashtags && clip.hashtags.length > 0)) && (
                      <div className="p-3 bg-black/40 border-t border-white/[0.04] space-y-1.5 text-left">
                        {clip.description && (
                          <p className="line-clamp-2 text-xs text-[#888888] leading-relaxed">
                            {clip.description}
                          </p>
                        )}
                        {clip.hashtags && clip.hashtags.length > 0 && (
                          <p className="font-mono text-[11px] text-[#666666] truncate">
                            {clip.hashtags
                              .map((t) => `#${t.replace(/^#/, "")}`)
                              .join(" ")}
                          </p>
                        )}
                      </div>
                    )}

                    {/* Footer Action Bar */}
                    <div className="px-3 h-9 border-t border-white/[0.08] flex items-center justify-between text-xs">
                      {clip.ok === false || clip.render_ok === false ? (
                        <span className="text-[11px] font-mono text-[#FF4D4D]">
                          Clip unavailable
                        </span>
                      ) : (
                        <>
                          <div className="flex items-center gap-1">
                            <button
                              type="button"
                              onClick={() =>
                                handleCopy(
                                  `${clip.clip_id}-caption`,
                                  captionText,
                                  `Caption for ${clip.title}`
                                )
                              }
                              aria-label={`Copy caption and hashtags for ${clip.title}`}
                              title="Copy title, description, and hashtags"
                              className={`text-[11px] font-mono px-2 py-1 rounded transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white/40 ${
                                captionStatus === "copied"
                                  ? "text-[#50E3C2] bg-[#50E3C2]/10"
                                  : captionStatus === "error"
                                  ? "text-[#FF4D4D] bg-[#FF4D4D]/10"
                                  : "text-[#888888] hover:text-[#EDEDED] hover:bg-[#111111]"
                              }`}
                            >
                              {captionStatus === "copied"
                                ? "Copied"
                                : captionStatus === "error"
                                ? "Copy Failed"
                                : "Copy Caption"}
                            </button>

                            <button
                              type="button"
                              onClick={() =>
                                handleCopy(
                                  `${clip.clip_id}-link`,
                                  resolvedUrl,
                                  `Video URL for ${clip.title}`
                                )
                              }
                              aria-label={`Copy video URL for ${clip.title}`}
                              title="Copy direct video URL"
                              className={`text-[11px] font-mono px-2 py-1 rounded transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white/40 ${
                                linkStatus === "copied"
                                  ? "text-[#50E3C2] bg-[#50E3C2]/10"
                                  : linkStatus === "error"
                                  ? "text-[#FF4D4D] bg-[#FF4D4D]/10"
                                  : "text-[#888888] hover:text-[#EDEDED] hover:bg-[#111111]"
                              }`}
                            >
                              {linkStatus === "copied"
                                ? "Copied"
                                : linkStatus === "error"
                                ? "Copy Failed"
                                : "Copy Link"}
                            </button>
                          </div>

                          <button
                            type="button"
                            onClick={() => handleDownloadClip(clip)}
                            aria-label={`Download video clip for ${clip.title}`}
                            disabled={isDownloading}
                            className="text-[11px] font-mono text-[#888888] hover:text-[#EDEDED] hover:bg-[#111111] px-2 py-1 rounded transition-colors flex items-center gap-1 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white/40 disabled:opacity-60"
                          >
                            {isDownloading ? (
                              <>
                                <span className="w-2.5 h-2.5 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                                <span>
                                  {typeof dlPercent === "number" && dlPercent > 0
                                    ? `${dlPercent}%`
                                    : "Downloading…"}
                                </span>
                              </>
                            ) : (
                              <>
                                <span>Download</span>
                                <svg
                                  className="w-3 h-3"
                                  viewBox="0 0 24 24"
                                  fill="none"
                                  stroke="currentColor"
                                  strokeWidth="1.8"
                                  aria-hidden="true"
                                >
                                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                  <polyline points="7 10 12 15 17 10" />
                                  <line x1="12" y1="15" x2="12" y2="3" />
                                </svg>
                              </>
                            )}
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            /* Idle & Empty State */
            <div className="border border-dashed border-white/[0.08] rounded-lg p-10 text-center">
              <p className="text-xs font-mono text-[#888888]">
                No clips yet. Paste a URL and press{" "}
                <Kbd>{isMac ? "⌘ ↵" : "Ctrl ↵"}</Kbd>
              </p>
            </div>
          )}
        </section>

        {/* Minimal Footer */}
        <footer className="pt-8 border-t border-white/[0.08] flex items-center justify-between text-[11px] font-mono text-[#888888]">
          <span>Vynlo Studio</span>
          <span>API · {getApiBaseUrl() || "same-origin"}</span>
        </footer>
      </div>
    </div>
  );
}
