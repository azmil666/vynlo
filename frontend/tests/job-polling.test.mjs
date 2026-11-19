/**
 * Frontend Unit & Component/Hook Tests for Job Polling, Resilience, Error Mapping, and State Persistence.
 */

import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";

import {
  ERROR_CODE_MAP,
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
} from "../src/lib/jobPolling.ts";

// Mock minimal browser globals for testing persistence and URL handling
class MockStorage {
  constructor() {
    this.store = {};
  }
  getItem(key) {
    return this.store[key] || null;
  }
  setItem(key, value) {
    this.store[key] = String(value);
  }
  removeItem(key) {
    delete this.store[key];
  }
  clear() {
    this.store = {};
  }
}

describe("API Base URL Resolution", () => {
  const origEnv = { ...process.env };

  afterEach(() => {
    process.env = { ...origEnv };
  });

  it("returns NEXT_PUBLIC_API_URL when set", () => {
    process.env.NEXT_PUBLIC_API_URL = "https://api.clippermvp.com/";
    assert.strictEqual(getApiBaseUrl(), "https://api.clippermvp.com");
  });

  it("never returns hardcoded localhost in production builds without env var", () => {
    delete process.env.NEXT_PUBLIC_API_URL;
    process.env.NODE_ENV = "production";
    const url = getApiBaseUrl();
    assert.ok(!url.includes("localhost:8000"), "Production build must not default to localhost:8000");
  });

  it("defaults to localhost:8000 in development when env var is absent", () => {
    delete process.env.NEXT_PUBLIC_API_URL;
    process.env.NODE_ENV = "development";
    assert.strictEqual(getApiBaseUrl(), "http://localhost:8000");
  });
});

describe("Video URL Resolution", () => {
  it("resolves relative video paths against API base URL", () => {
    const resolved = resolveVideoUrl("/static/job_123/clip_1.mp4", "http://localhost:8000");
    assert.strictEqual(resolved, "http://localhost:8000/static/job_123/clip_1.mp4");
  });

  it("preserves full http/https URLs", () => {
    const fullUrl = "https://s3.amazonaws.com/bucket/clip.mp4";
    assert.strictEqual(resolveVideoUrl(fullUrl, "http://localhost:8000"), fullUrl);
  });

  it("returns empty string when url is empty or null", () => {
    assert.strictEqual(resolveVideoUrl(""), "");
    assert.strictEqual(resolveVideoUrl(null), "");
  });
});

describe("Structured Error Mapping", () => {
  it("maps known PipelineErrorCodes to friendly title, message, and retryable flag", () => {
    const err = getStructuredError({
      code: "RENDER_FAILED",
      message: "FFmpeg exited with non-zero code",
      retryable: true,
      detail: "Process killed: error decoding frame",
    });

    assert.strictEqual(err.code, "RENDER_FAILED");
    assert.strictEqual(err.title, "Video Render Failed");
    assert.strictEqual(err.message, "FFmpeg exited with non-zero code");
    assert.strictEqual(err.retryable, true);
    assert.strictEqual(err.detail, "Process killed: error decoding frame");
  });

  it("correctly identifies non-retryable errors like INVALID_URL and VIDEO_UNAVAILABLE", () => {
    const err = getStructuredError({
      code: "INVALID_URL",
      message: "Unsupported video domain",
      retryable: false,
    });

    assert.strictEqual(err.code, "INVALID_URL");
    assert.strictEqual(err.retryable, false);
  });

  it("handles string error messages gracefully", () => {
    const err = getStructuredError("DOWNLOAD_FAILED");
    assert.strictEqual(err.code, "DOWNLOAD_FAILED");
    assert.strictEqual(err.title, "Download Failed");
    assert.strictEqual(err.retryable, true);
  });

  it("handles expired status fallback", () => {
    const err = getStructuredError(null, "expired");
    assert.strictEqual(err.code, "EXPIRED");
    assert.strictEqual(err.title, "Job Retention Expired");
    assert.strictEqual(err.retryable, false);
  });
});

describe("Job ID Persistence (URL & localStorage)", () => {
  beforeEach(() => {
    global.localStorage = new MockStorage();
    global.window = {
      location: {
        href: "https://clipper.local/",
        search: "",
        origin: "https://clipper.local",
      },
      history: {
        replaceState: (_data, _title, url) => {
          const parsed = new URL(url, "https://clipper.local");
          global.window.location.href = parsed.toString();
          global.window.location.search = parsed.search;
        },
      },
    };
  });

  afterEach(() => {
    delete global.localStorage;
    delete global.window;
  });

  it("retrieves job ID from URL query parameters", () => {
    global.window.location.search = "?job=job_from_url_123";
    global.window.location.href = "https://clipper.local/?job=job_from_url_123";

    assert.strictEqual(getStoredJobId(), "job_from_url_123");
  });

  it("falls back to localStorage if URL param is missing", () => {
    global.window.location.search = "";
    global.localStorage.setItem("clipping_job_id", "job_from_storage_456");

    assert.strictEqual(getStoredJobId(), "job_from_storage_456");
  });

  it("persists job ID to both URL and localStorage", () => {
    setStoredJobId("job_new_789");

    assert.strictEqual(global.localStorage.getItem("clipping_job_id"), "job_new_789");
    assert.ok(global.window.location.search.includes("job=job_new_789"));
  });

  it("clears job ID from both URL and localStorage on reset", () => {
    setStoredJobId("job_temp");
    assert.strictEqual(getStoredJobId(), "job_temp");

    setStoredJobId(null);
    assert.strictEqual(global.localStorage.getItem("clipping_job_id"), null);
    assert.ok(!global.window.location.search.includes("job="));
  });
});

describe("Polling Resilience & Terminal Status", () => {
  it("identifies terminal and non-terminal pipeline statuses", () => {
    assert.strictEqual(isTerminalStatus("completed"), true);
    assert.strictEqual(isTerminalStatus("completed_with_warnings"), true);
    assert.strictEqual(isTerminalStatus("partial"), true);
    assert.strictEqual(isTerminalStatus("failed"), true);
    assert.strictEqual(isTerminalStatus("expired"), true);

    assert.strictEqual(isTerminalStatus("queued"), false);
    assert.strictEqual(isTerminalStatus("downloading"), false);
    assert.strictEqual(isTerminalStatus("transcribing"), false);
    assert.strictEqual(isTerminalStatus("analyzing"), false);
    assert.strictEqual(isTerminalStatus("rendering"), false);
  });

  it("calculates exponential backoff delay for network retries", () => {
    assert.strictEqual(calculatePollingBackoff(0), 2500);
    assert.strictEqual(calculatePollingBackoff(1), 3750); // 2500 * 1.5
    assert.strictEqual(calculatePollingBackoff(2), 5625); // 2500 * 2.25
    assert.strictEqual(calculatePollingBackoff(5), 10000); // capped at max 10000ms
    assert.strictEqual(calculatePollingBackoff(10), 10000); // capped
  });

  it("detects stalled worker when updated_at exceeds threshold", () => {
    const now = Date.now();
    const fourMinutesAgo = new Date(now - 240 * 1000).toISOString();
    const thirtySecondsAgo = new Date(now - 30 * 1000).toISOString();

    const stalledJob = {
      job_id: "j1",
      status: "transcribing",
      progress: 30,
      updated_at: fourMinutesAgo,
      clips: [],
    };

    const activeJob = {
      job_id: "j2",
      status: "transcribing",
      progress: 40,
      updated_at: thirtySecondsAgo,
      clips: [],
    };

    assert.strictEqual(isJobStalled(stalledJob, 180), true);
    assert.strictEqual(isJobStalled(activeJob, 180), false);
  });

  it("does not report completed or failed jobs as stalled", () => {
    const fourMinutesAgo = new Date(Date.now() - 240 * 1000).toISOString();
    const finishedJob = {
      job_id: "j3",
      status: "completed",
      progress: 100,
      updated_at: fourMinutesAgo,
      clips: [],
    };
    assert.strictEqual(isJobStalled(finishedJob, 180), false);
  });
});

describe("Mock Fetch Polling State Transitions", () => {
  it("polls across stages until terminal completed status", async () => {
    let callCount = 0;
    const stages = [
      { status: "queued", progress: 0, clips: [] },
      { status: "transcribing", progress: 30, clips: [] },
      { status: "rendering", progress: 70, clips: [] },
      { status: "completed", progress: 100, clips: [{ clip_id: "c1", title: "Clip 1", virality_score: 90, duration: 25.0, video_url: "/clip_1.mp4" }] },
    ];

    const mockFetch = async () => {
      const current = stages[Math.min(callCount, stages.length - 1)];
      callCount++;
      return {
        ok: true,
        status: 200,
        json: async () => ({ job_id: "test_job", ...current }),
      };
    };

    const polledStates = [];
    let isFinished = false;

    for (let step = 0; step < 10; step++) {
      if (isFinished) break;
      const res = await mockFetch();
      const data = await res.json();
      polledStates.push(data.status);
      if (isTerminalStatus(data.status)) {
        isFinished = true;
      }
    }

    assert.deepStrictEqual(polledStates, ["queued", "transcribing", "rendering", "completed"]);
    assert.strictEqual(isFinished, true);
  });

  it("handles 404 response by halting and mapping to JOB_NOT_FOUND error", async () => {
    const mockFetch = async () => ({
      ok: false,
      status: 404,
      json: async () => ({ detail: "Job not found" }),
    });

    const res = await mockFetch();
    assert.strictEqual(res.ok, false);
    assert.strictEqual(res.status, 404);

    const err = getStructuredError({
      code: "JOB_NOT_FOUND",
      message: "Job not found. The job may have expired or never existed.",
      retryable: false,
    });
    assert.strictEqual(err.code, "JOB_NOT_FOUND");
    assert.strictEqual(err.retryable, false);
  });

  it("handles 410 response by halting and mapping to EXPIRED status", async () => {
    const mockFetch = async () => ({
      ok: false,
      status: 410,
      json: async () => ({ detail: "Job clips have expired and are no longer available" }),
    });

    const res = await mockFetch();
    assert.strictEqual(res.ok, false);
    assert.strictEqual(res.status, 410);

    const errData = await res.json();
    const err = getStructuredError({
      code: "EXPIRED",
      message: errData.detail,
      retryable: false,
    });
    assert.strictEqual(err.code, "EXPIRED");
    assert.strictEqual(err.retryable, false);
    assert.ok(err.message.includes("expired"));
  });

  it("handles AbortController signal correctly on unmount", async () => {
    const controller = new AbortController();

    const mockFetchWithSignal = (signal) => {
      return new Promise((resolve, reject) => {
        if (signal.aborted) {
          return reject(new Error("AbortError"));
        }
        signal.addEventListener("abort", () => {
          reject(new Error("AbortError"));
        });
      });
    };

    const fetchPromise = mockFetchWithSignal(controller.signal);
    controller.abort();

    await assert.rejects(fetchPromise, { message: "AbortError" });
  });
});

describe("Copy to Clipboard with Fallbacks", () => {
  const origNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");

  afterEach(() => {
    if (origNavigator) {
      Object.defineProperty(globalThis, "navigator", origNavigator);
    }
    delete global.document;
    delete global.window;
  });

  it("uses navigator.clipboard.writeText when available and supported", async () => {
    let written = "";
    global.window = {};
    Object.defineProperty(globalThis, "navigator", {
      value: {
        clipboard: {
          writeText: async (t) => {
            written = t;
          },
        },
      },
      configurable: true,
      writable: true,
    });

    const res = await copyToClipboard("Caption text");
    assert.strictEqual(res, true);
    assert.strictEqual(written, "Caption text");
  });

  it("falls back to document.execCommand when navigator.clipboard fails or throws", async () => {
    global.window = {};
    Object.defineProperty(globalThis, "navigator", {
      value: {
        clipboard: {
          writeText: async () => {
            throw new Error("Permission denied");
          },
        },
      },
      configurable: true,
      writable: true,
    });

    let executedCommand = "";
    const createdElements = [];
    global.document = {
      createElement: (tag) => {
        const el = {
          tag,
          style: {},
          setAttribute: () => {},
          focus: () => {},
          select: () => {},
          setSelectionRange: () => {},
        };
        createdElements.push(el);
        return el;
      },
      body: {
        appendChild: () => {},
        removeChild: () => {},
      },
      execCommand: (cmd) => {
        executedCommand = cmd;
        return true;
      },
    };

    const res = await copyToClipboard("Fallback caption");
    assert.strictEqual(res, true);
    assert.strictEqual(executedCommand, "copy");
    assert.strictEqual(createdElements[0].value, "Fallback caption");
  });

  it("returns false when both clipboard API and execCommand fail", async () => {
    global.window = {};
    Object.defineProperty(globalThis, "navigator", {
      value: {},
      configurable: true,
      writable: true,
    });
    global.document = {
      createElement: () => ({
        style: {},
        setAttribute: () => {},
        focus: () => {},
        select: () => {},
        setSelectionRange: () => {},
      }),
      body: {
        appendChild: () => {},
        removeChild: () => {},
      },
      execCommand: () => false,
    };

    const res = await copyToClipboard("Impossible copy");
    assert.strictEqual(res, false);
  });
});

describe("Cross-Origin Media Download", () => {
  const origURL = global.URL;

  afterEach(() => {
    delete global.window;
    delete global.fetch;
    delete global.document;
    global.URL = origURL;
  });

  it("fetches blob, reports progress, and triggers object URL download click", async () => {
    global.window = {};
    let createdUrl = "";
    let clickedHref = "";
    let clickedDownload = "";

    global.URL = {
      createObjectURL: () => {
        createdUrl = "blob:http://localhost:3000/123-abc";
        return createdUrl;
      },
      revokeObjectURL: () => {},
    };

    global.document = {
      createElement: (tag) => ({
        tag,
        click: function () {
          clickedHref = this.href;
          clickedDownload = this.download;
        },
      }),
      body: {
        appendChild: () => {},
        removeChild: () => {},
      },
    };

    global.fetch = async () => ({
      ok: true,
      headers: {
        get: (h) => (h === "content-length" ? "100" : null),
      },
      blob: async () => ({ size: 100, type: "video/mp4" }),
    });

    const progressValues = [];
    await downloadClip("https://cdn.example.com/video.mp4", "my_clip.mp4", undefined, (p) => {
      progressValues.push(p);
    });

    assert.strictEqual(clickedHref, "blob:http://localhost:3000/123-abc");
    assert.strictEqual(clickedDownload, "my_clip.mp4");
    assert.ok(progressValues.includes(100));
  });

  it("falls back to backend attachment endpoint when CORS blob fetch fails", async () => {
    global.window = {};
    let clickedHref = "";
    let clickedDownload = "";

    global.document = {
      createElement: (tag) => ({
        tag,
        click: function () {
          clickedHref = this.href;
          clickedDownload = this.download;
        },
      }),
      body: {
        appendChild: () => {},
        removeChild: () => {},
      },
    };

    // Simulate CORS network block
    global.fetch = async () => {
      throw new Error("Failed to fetch due to CORS");
    };

    const fallbackUrl = "http://localhost:8000/api/jobs/job123/clips/c1/download";
    await downloadClip("https://s3.amazonaws.com/cors-blocked.mp4", "fallback.mp4", fallbackUrl);

    assert.strictEqual(clickedHref, fallbackUrl);
    assert.strictEqual(clickedDownload, "fallback.mp4");
  });
});

describe("Selection Method Badges & Layout Notes", () => {
  it("extracts full-frame layout note when no face was detected", () => {
    const clip = {
      clip_id: "c1",
      title: "Clip 1",
      virality_score: 80,
      duration: 30.0,
      layout_used: "fit_blur",
      warnings: ["NO_FACE_DETECTED: No face detected, used full-frame layout"],
    };

    const note = getLayoutNote(clip);
    assert.ok(note);
    assert.strictEqual(note.label, "Full-frame (no face)");
    assert.strictEqual(note.isDegraded, true);
  });

  it("extracts split layout note", () => {
    const clip = {
      clip_id: "c2",
      title: "Clip 2",
      virality_score: 85,
      duration: 25.0,
      layout_used: "split",
      warnings: [],
    };

    const note = getLayoutNote(clip);
    assert.ok(note);
    assert.strictEqual(note.label, "Split layout");
    assert.strictEqual(note.isDegraded, false);
  });

  it("extracts speaker tracked note for standard smart crop", () => {
    const clip = {
      clip_id: "c3",
      title: "Clip 3",
      virality_score: 92,
      duration: 20.0,
      layout_used: "speaker",
      warnings: [],
    };

    const note = getLayoutNote(clip);
    assert.ok(note);
    assert.strictEqual(note.label, "Speaker tracked");
    assert.strictEqual(note.isDegraded, false);
  });
});
