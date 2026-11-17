"use client";

import { useEffect, useState, useCallback, useMemo } from "react";

interface ClipResult {
  clip_id: string;
  title: string;
  virality_score: number;
  duration: number;
  video_url: string;
  is_fallback?: boolean;
  description?: string;
  hashtags?: string[];
}

interface JobStatus {
  job_id: string;
  status:
    | "queued"
    | "downloading"
    | "transcribing"
    | "analyzing"
    | "rendering"
    | "completed"
    | "failed";
  progress: number;
  error?: string | null;
  clips: ClipResult[];
}

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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

function resolveVideoUrl(rawUrl: string): string {
  if (!rawUrl) return "";
  if (rawUrl.startsWith("http://") || rawUrl.startsWith("https://")) {
    return rawUrl;
  }
  return `${API_BASE_URL.replace(/\/$/, "")}/${rawUrl.replace(/^\//, "")}`;
}

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

  const [isMac, setIsMac] = useState(false);
  const [copiedKeys, setCopiedKeys] = useState<Record<string, boolean>>({});

  // Detect platform in useEffect to avoid hydration mismatch
  useEffect(() => {
    if (typeof window !== "undefined") {
      setIsMac(/(Mac|iPhone|iPod|iPad)/i.test(navigator.userAgent));
    }
  }, []);

  // Check backend health on mount
  useEffect(() => {
    let isMounted = true;
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/health`);
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

  // Poll job status every 2500ms
  useEffect(() => {
    if (!currentJobId) return;

    let isMounted = true;
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/jobs/${currentJobId}`);
        if (!res.ok) {
          throw new Error("Failed to fetch job status");
        }
        const data: JobStatus = await res.json();
        if (isMounted) {
          setJobState(data);
          if (data.status === "completed" || data.status === "failed") {
            clearInterval(interval);
          }
        }
      } catch (err) {
        console.error("Polling error:", err);
      }
    }, 2500);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [currentJobId]);

  const handleSubmit = useCallback(
    async (e?: React.FormEvent) => {
      if (e) e.preventDefault();
      if (!url.trim() || isSubmitting) return;
      if (
        jobState &&
        jobState.status !== "completed" &&
        jobState.status !== "failed"
      ) {
        return;
      }

      setIsSubmitting(true);
      setSubmitError(null);
      setJobState(null);
      setCurrentJobId(null);

      try {
        const res = await fetch(`${API_BASE_URL}/api/jobs`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            url: url.trim(),
            layout_mode: layoutMode,
            max_clips: Number(maxClips),
            caption_preset: captionPreset,
            platform_safe_zone: platformSafeZone,
            highlight_color:
              captionPreset === "bold_yellow" ? highlightColor : "&H00FFFFFF",
            show_hook_banner: showHookBanner,
          }),
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || "Failed to create clipping job");
        }

        const initialData: JobStatus = await res.json();
        setCurrentJobId(initialData.job_id);
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

  const handleReset = useCallback(() => {
    setCurrentJobId(null);
    setJobState(null);
    setUrl("");
    setSubmitError(null);
  }, []);

  const handleCopy = useCallback(async (key: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedKeys((prev) => ({ ...prev, [key]: true }));
      setTimeout(() => {
        setCopiedKeys((prev) => ({ ...prev, [key]: false }));
      }, 1500);
    } catch {
      // Fallback silently if clipboard permissions are restricted
    }
  }, []);

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
      jobState.status !== "failed"
  );

  const currentStageIndex = useMemo(() => {
    if (!jobState) return -1;
    if (jobState.status === "completed") return ORDERED_STAGES.length;
    return ORDERED_STAGES.findIndex((s) => s.key === jobState.status);
  }, [jobState]);

  return (
    <div className="relative min-h-screen bg-black text-[#EDEDED] antialiased">
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
              Clipper Studio
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
                    : "Stacks gameplay center-crop (top) over bottom-right facecam (bottom)."}
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
                  (currentStageIndex > idx && jobState.status !== "failed");
                const isCurrent =
                  stage.key === jobState.status &&
                  stage.key !== "completed";
                const isFailed =
                  jobState.status === "failed" && currentStageIndex === idx;

                return (
                  <div
                    key={stage.key}
                    className="flex items-center justify-between py-1 px-1.5 rounded transition-colors"
                  >
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
                );
              })}
            </div>

            {/* Failure Box */}
            {jobState.status === "failed" && (
              <div className="pt-2 space-y-3">
                <div className="p-3 rounded bg-[#FF4D4D]/[0.05] border border-[#FF4D4D]/20 font-mono text-xs text-[#FF4D4D]">
                  <p className="font-semibold mb-1">Execution Failure:</p>
                  <p className="opacity-90">
                    {jobState.error || "An unexpected pipeline error occurred."}
                  </p>
                </div>
                <div className="flex justify-end">
                  <button
                    type="button"
                    onClick={handleReset}
                    className="h-8 px-3 rounded text-xs font-mono text-[#888888] hover:text-[#EDEDED] border border-white/[0.08] hover:bg-[#111111] transition-colors"
                  >
                    Try another link
                  </button>
                </div>
              </div>
            )}
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

            {jobState?.status === "completed" &&
              jobState.clips &&
              jobState.clips.length > 0 && (
                <a
                  href={`${API_BASE_URL}/api/jobs/${jobState.job_id}/download-all`}
                  download={`clips_${jobState.job_id}.zip`}
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

          {jobState && jobState.clips && jobState.clips.length > 0 ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {jobState.clips.map((clip, idx) => {
                const resolvedUrl = resolveVideoUrl(clip.video_url);
                const isCaptionCopied = copiedKeys[`${clip.clip_id}-caption`];
                const isLinkCopied = copiedKeys[`${clip.clip_id}-link`];

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
                        className="truncate font-medium text-[#EDEDED] text-xs max-w-[55%]"
                      >
                        {clip.title}
                      </span>
                      <div className="flex items-center gap-2 font-mono tabular-nums text-[11px]">
                        <span className="text-[#888888]">
                          {clip.duration}s
                        </span>
                        <span className="text-[#555555]">·</span>
                        <span className={scoreColor} title="Virality Score">
                          {clip.virality_score}/100
                        </span>
                        {clip.is_fallback && (
                          <span className="px-1.5 py-0.5 rounded text-[10px] font-mono uppercase tracking-wider bg-amber-500/10 text-amber-400 border border-amber-500/20">
                            Fallback
                          </span>
                        )}
                      </div>
                    </div>

                    {/* 9:16 Video Frame */}
                    <div className="aspect-[9/16] bg-black w-full relative">
                      <video
                        src={resolvedUrl}
                        controls
                        playsInline
                        preload="metadata"
                        className="w-full h-full object-contain bg-black"
                      />
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
                      <div className="flex items-center gap-1">
                        <button
                          type="button"
                          onClick={() =>
                            handleCopy(`${clip.clip_id}-caption`, captionText)
                          }
                          title="Copy title, description, and hashtags"
                          className="text-[11px] font-mono text-[#888888] hover:text-[#EDEDED] hover:bg-[#111111] px-2 py-1 rounded transition-colors"
                        >
                          {isCaptionCopied ? "Copied" : "Copy Caption"}
                        </button>

                        <button
                          type="button"
                          onClick={() =>
                            handleCopy(`${clip.clip_id}-link`, resolvedUrl)
                          }
                          className="text-[11px] font-mono text-[#888888] hover:text-[#EDEDED] hover:bg-[#111111] px-2 py-1 rounded transition-colors"
                        >
                          {isLinkCopied ? "Copied" : "Copy Link"}
                        </button>
                      </div>

                      <a
                        href={resolvedUrl}
                        download={`${clip.clip_id}.mp4`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-[11px] font-mono text-[#888888] hover:text-[#EDEDED] hover:bg-[#111111] px-2 py-1 rounded transition-colors flex items-center gap-1"
                      >
                        <span>Download</span>
                        <svg
                          className="w-3 h-3"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="1.8"
                        >
                          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                          <polyline points="7 10 12 15 17 10" />
                          <line x1="12" y1="15" x2="12" y2="3" />
                        </svg>
                      </a>
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
          <span>Clipper AI Studio</span>
          <span>API · {API_BASE_URL}</span>
        </footer>
      </div>
    </div>
  );
}
