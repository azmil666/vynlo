"use client";

import { useEffect, useState } from "react";
import {
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  Clock,
  Download,
  Flame,
  Gamepad2,
  Loader2,
  Scissors,
  Sparkles,
  User,
  Video,
} from "lucide-react";

interface ClipResult {
  clip_id: string;
  title: string;
  virality_score: number;
  duration: number;
  video_url: string;
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
  { label: "Classic Yellow", value: "&H0000FFFF", hex: "#FACC15" },
  { label: "Neon Green", value: "&H0000FF00", hex: "#22C55E" },
  { label: "Vibrant Cyan", value: "&H00FFFF00", hex: "#06B6D4" },
  { label: "Hot Pink", value: "&H00B469FF", hex: "#EC4899" },
];

export default function HomePage() {
  const [url, setUrl] = useState("");
  const [layoutMode, setLayoutMode] = useState<"speaker" | "split">("speaker");
  const [maxClips, setMaxClips] = useState(3);
  const [highlightColor, setHighlightColor] = useState("&H0000FFFF");

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [jobState, setJobState] = useState<JobStatus | null>(null);

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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url.trim()) return;

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
          highlight_color: highlightColor,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || "Failed to create clipping job");
      }

      const initialData: JobStatus = await res.json();
      setCurrentJobId(initialData.job_id);
      setJobState(initialData);
    } catch (err: any) {
      setSubmitError(err.message || "Failed to connect to API");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleReset = () => {
    setCurrentJobId(null);
    setJobState(null);
    setUrl("");
    setSubmitError(null);
  };

  const stages = [
    { key: "downloading", label: "Downloading Media", threshold: 10 },
    { key: "transcribing", label: "Transcribing Audio", threshold: 30 },
    { key: "analyzing", label: "Finding Viral Hooks", threshold: 50 },
    { key: "rendering", label: "Cropping & Rendering", threshold: 90 },
  ];

  const getStageStatus = (stageThreshold: number) => {
    if (!jobState) return "waiting";
    if (jobState.status === "failed") return "error";
    if (jobState.status === "completed") return "done";
    if (jobState.progress >= stageThreshold) return "done";
    if (
      jobState.progress >= stageThreshold - 25 &&
      jobState.progress < stageThreshold
    )
      return "current";
    return "waiting";
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 py-10 px-4 sm:px-6 lg:px-8">
      <div className="max-w-6xl mx-auto space-y-10">
        {/* Header */}
        <header className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-zinc-800 pb-6">
          <div className="flex items-center gap-3">
            <div className="w-11 h-11 rounded-xl bg-gradient-to-tr from-amber-500 to-orange-500 flex items-center justify-center shadow-lg shadow-orange-500/20 text-black">
              <Scissors className="w-6 h-6 stroke-[2.5]" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
                Clipper AI
                <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20">
                  MVP Studio
                </span>
              </h1>
              <p className="text-xs text-zinc-400">
                Stream to Viral Shorts & Reels Pipeline
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 text-xs">
            <span
              className={`w-2 h-2 rounded-full ${
                backendOnline === true
                  ? "bg-emerald-500 shadow-sm shadow-emerald-500/50"
                  : backendOnline === false
                  ? "bg-rose-500"
                  : "bg-zinc-600 animate-pulse"
              }`}
            />
            <span className="text-zinc-400">
              {backendOnline === true
                ? "API Connected (port 8000)"
                : backendOnline === false
                ? "API Offline (Start Docker)"
                : "Checking Backend..."}
            </span>
          </div>
        </header>

        {/* Form Card */}
        <div className="bg-zinc-900/70 border border-zinc-800 rounded-2xl p-6 sm:p-8 backdrop-blur shadow-xl">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label
                htmlFor="video-url"
                className="block text-sm font-medium text-zinc-200 mb-2"
              >
                YouTube / Stream URL
              </label>
              <div className="relative">
                <input
                  id="video-url"
                  type="url"
                  required
                  placeholder="https://www.youtube.com/watch?v=..."
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  disabled={isSubmitting || !!(jobState && jobState.status !== "completed" && jobState.status !== "failed")}
                  className="w-full bg-zinc-950 border border-zinc-700 focus:border-amber-500 rounded-xl px-4 py-3 text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-amber-500/20 transition disabled:opacity-50"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Layout Mode */}
              <div>
                <label className="block text-sm font-medium text-zinc-300 mb-2">
                  Layout Framing
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => setLayoutMode("speaker")}
                    className={`flex flex-col items-center justify-center gap-1.5 p-3 rounded-xl border text-xs font-medium transition ${
                      layoutMode === "speaker"
                        ? "bg-amber-500/10 border-amber-500/50 text-amber-300"
                        : "bg-zinc-950/60 border-zinc-800 text-zinc-400 hover:border-zinc-700"
                    }`}
                  >
                    <User className="w-4 h-4" />
                    <span>Smart Face Crop</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setLayoutMode("split")}
                    className={`flex flex-col items-center justify-center gap-1.5 p-3 rounded-xl border text-xs font-medium transition ${
                      layoutMode === "split"
                        ? "bg-amber-500/10 border-amber-500/50 text-amber-300"
                        : "bg-zinc-950/60 border-zinc-800 text-zinc-400 hover:border-zinc-700"
                    }`}
                  >
                    <Gamepad2 className="w-4 h-4" />
                    <span>Split Screen</span>
                  </button>
                </div>
              </div>

              {/* Max Clips */}
              <div>
                <label className="block text-sm font-medium text-zinc-300 mb-2">
                  Clips to Generate
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {[1, 3, 5].map((num) => (
                    <button
                      key={num}
                      type="button"
                      onClick={() => setMaxClips(num)}
                      className={`p-3 rounded-xl border text-xs font-medium transition ${
                        maxClips === num
                          ? "bg-amber-500/10 border-amber-500/50 text-amber-300"
                          : "bg-zinc-950/60 border-zinc-800 text-zinc-400 hover:border-zinc-700"
                      }`}
                    >
                      {num} {num === 1 ? "Clip" : "Clips"}
                    </button>
                  ))}
                </div>
              </div>

              {/* Highlight Color */}
              <div>
                <label className="block text-sm font-medium text-zinc-300 mb-2">
                  Subtitle Highlight
                </label>
                <div className="flex items-center gap-2 h-11">
                  {HIGHLIGHT_COLORS.map((c) => (
                    <button
                      key={c.value}
                      type="button"
                      onClick={() => setHighlightColor(c.value)}
                      title={c.label}
                      className={`w-9 h-9 rounded-xl border-2 transition flex items-center justify-center ${
                        highlightColor === c.value
                          ? "border-white scale-110 shadow-lg"
                          : "border-transparent opacity-70 hover:opacity-100"
                      }`}
                      style={{ backgroundColor: c.hex }}
                    >
                      {highlightColor === c.value && (
                        <span className="w-2 h-2 rounded-full bg-black/80" />
                      )}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {submitError && (
              <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-sm flex items-center gap-3">
                <AlertCircle className="w-5 h-5 flex-shrink-0" />
                <span>{submitError}</span>
              </div>
            )}

            <button
              type="submit"
              disabled={
                isSubmitting ||
                !url.trim() ||
                (!!jobState &&
                  jobState.status !== "completed" &&
                  jobState.status !== "failed")
              }
              className="w-full py-3.5 px-6 rounded-xl bg-gradient-to-r from-amber-500 to-orange-500 text-black font-semibold text-sm hover:from-amber-400 hover:to-orange-400 focus:outline-none focus:ring-2 focus:ring-amber-500/50 transition shadow-lg shadow-orange-500/20 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Submitting Video...</span>
                </>
              ) : jobState &&
                jobState.status !== "completed" &&
                jobState.status !== "failed" ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Pipeline Running ({jobState.progress}%)...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Generate Viral Shorts</span>
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </form>
        </div>

        {/* Live Pipeline Tracker */}
        {jobState && (
          <div className="bg-zinc-900/60 border border-zinc-800 rounded-2xl p-6 sm:p-8 space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-zinc-800/80 pb-4">
              <div>
                <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                  <span>Processing Pipeline</span>
                  <span className="text-xs text-zinc-500 font-mono">
                    ID: {jobState.job_id.slice(0, 8)}...
                  </span>
                </h2>
                <p className="text-xs text-zinc-400 capitalize">
                  Current Status: {jobState.status}
                </p>
              </div>
              <div className="text-2xl font-bold font-mono text-amber-400">
                {jobState.progress}%
              </div>
            </div>

            {/* Stepper */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {stages.map((st, idx) => {
                const status = getStageStatus(st.threshold);
                return (
                  <div
                    key={st.key}
                    className={`p-4 rounded-xl border transition ${
                      status === "done"
                        ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-300"
                        : status === "current"
                        ? "bg-amber-500/10 border-amber-500/40 text-amber-300 ring-1 ring-amber-500/20"
                        : status === "error"
                        ? "bg-rose-500/10 border-rose-500/20 text-rose-400"
                        : "bg-zinc-950/40 border-zinc-800/80 text-zinc-500"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-mono font-medium opacity-60">
                        Step 0{idx + 1}
                      </span>
                      {status === "done" ? (
                        <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      ) : status === "current" ? (
                        <Loader2 className="w-4 h-4 animate-spin text-amber-400" />
                      ) : (
                        <Clock className="w-4 h-4 opacity-40" />
                      )}
                    </div>
                    <div className="text-xs font-semibold">{st.label}</div>
                  </div>
                );
              })}
            </div>

            {/* Progress Bar */}
            <div className="w-full bg-zinc-950 rounded-full h-2 overflow-hidden border border-zinc-800">
              <div
                className={`h-full transition-all duration-500 rounded-full ${
                  jobState.status === "failed"
                    ? "bg-rose-500"
                    : jobState.status === "completed"
                    ? "bg-emerald-500"
                    : "bg-gradient-to-r from-amber-500 to-orange-500"
                }`}
                style={{ width: `${Math.max(5, jobState.progress)}%` }}
              />
            </div>

            {/* Error Message */}
            {jobState.status === "failed" && (
              <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <AlertCircle className="w-5 h-5 flex-shrink-0 text-rose-400" />
                  <span>
                    Pipeline failed: {jobState.error || "Unknown error occurred"}
                  </span>
                </div>
                <button
                  onClick={handleReset}
                  className="px-4 py-2 bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 border border-rose-500/40 text-xs font-medium rounded-lg transition"
                >
                  Try Another Link
                </button>
              </div>
            )}
          </div>
        )}

        {/* Generated Shorts Gallery */}
        {jobState && jobState.clips && jobState.clips.length > 0 && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-amber-400" />
                  Generated Viral Shorts
                </h2>
                <p className="text-xs text-zinc-400">
                  {jobState.clips.length} vertical 9:16 clips produced with dynamic karaoke subtitles
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {jobState.clips.map((clip) => (
                <div
                  key={clip.clip_id}
                  className="bg-zinc-900 border border-zinc-800 rounded-2xl overflow-hidden shadow-2xl flex flex-col group hover:border-zinc-700 transition"
                >
                  {/* Video Player */}
                  <div className="relative aspect-[9/16] bg-black">
                    <video
                      src={clip.video_url}
                      controls
                      playsInline
                      preload="metadata"
                      className="w-full h-full object-cover"
                    />
                  </div>

                  {/* Clip Info */}
                  <div className="p-4 flex-1 flex flex-col justify-between gap-4 bg-zinc-900/90">
                    <div>
                      <div className="flex items-center justify-between gap-2 mb-2">
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold bg-orange-500/10 text-orange-400 border border-orange-500/20">
                          <Flame className="w-3.5 h-3.5" />
                          {clip.virality_score}/100 Virality
                        </span>
                        <span className="text-xs text-zinc-400 font-mono">
                          {clip.duration}s
                        </span>
                      </div>
                      <h3 className="text-sm font-semibold text-zinc-100 line-clamp-2">
                        {clip.title}
                      </h3>
                    </div>

                    <a
                      href={clip.video_url}
                      download={`${clip.clip_id}.mp4`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="w-full py-2.5 px-4 rounded-xl bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-xs font-medium transition flex items-center justify-center gap-2 border border-zinc-700"
                    >
                      <Download className="w-4 h-4" />
                      <span>Download MP4</span>
                    </a>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
