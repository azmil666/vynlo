# Vynlo

Automated pipeline for ingesting long-form videos and producing vertical 9:16 short clips with word-level captions, face and active-speaker tracking, and loudness normalization.

---

## Architecture & Pipeline

```
+-----------------------------------------------------------------------+
|                           Next.js 14 Web UI                           |
|      (State polling, video scrubbing, clip cards, batch export)       |
+-----------------------------------+-----------------------------------+
                                    | HTTP REST
                                    v
+-----------------------------------------------------------------------+
|                           FastAPI Gateway                             |
|  (Lifespan startup validation, task dispatch, job polling, /health)   |
+------------------+--------------------------------+-------------------+
                   |                                |
                   v                                v
+----------------------------------+ +----------------------------------+
|          Redis (DB 0)            | |          Redis (DB 1)            |
|       Celery Task Broker         | |  Job Status Records (TTL: 24h)   |
|   (Queue backlog, visibility)    | |  Versioned Caches (TTL: 7d)      |
|                                  | |  Distributed Single-Flight Locks |
+------------------+---------------+ +----------------------------------+
                   |
                   v
+-----------------------------------------------------------------------+
|                    Celery Worker & Beat Scheduler                     |
|  (Worker pool, heartbeat updates, single-pass render, hourly sweep)   |
+------------------+--------------------------------+-------------------+
                   |                                |
                   v                                v
+----------------------------------+ +----------------------------------+
|           Local Mount            | |            S3 / R2               |
|      (/app/static, TTL: 48h)     | |      Object Storage Bucket       |
+----------------------------------+ +----------------------------------+
```

### Pipeline Stages

1. **Metadata & Duration Guard**: Probes the submitted media URL using `yt-dlp` to extract duration, title, and available formats. Validates domain against `ALLOWED_VIDEO_DOMAINS` and rejects videos exceeding `MAX_VIDEO_SECONDS` (default: 1200 seconds / 20 minutes) before allocating compute resources.
2. **Capped Media Download & FFprobe Validation**: Downloads the best video and audio stream clamped to 1080p (or 720p for long videos) into an isolated directory in `/tmp/clipper`. Runs `ffprobe` to validate stream integrity, container duration, frame rate, sample aspect ratio (SAR), and pixel format, detecting variable frame rate (VFR) sources.
3. **Faster-Whisper Transcription & Speech Validation**: Extracts a 16 kHz mono audio track and transcribes it using `faster-whisper` (CTranslate2 backend with Silero VAD) to produce word-level timestamps. Validates speech coverage ratio and minimum word count to fail fast on silent or purely musical media.
4. **Segment-ID LLM Analysis & Rubric Ranking**: Submits transcript segment IDs and text windows to the configured LLM (`gemini-2.5-flash` or `gpt-4o-mini`) using structured JSON schema output. Scores clips against engagement, hook strength, and flow rubrics. Deduplicates overlapping candidate spans. Automatically falls back to transcript-driven heuristic segmentation if the LLM is unreachable or blocked.
5. **YuNet Face/Active-Speaker Tracking & Layout Framing**: Decodes video frames at 4 Hz and executes neural face detection via OpenCV YuNet (`face_detection_yunet_2023mar.onnx`). Tracks bounding boxes, computes mouth motion energy to identify the active speaker, and derives smooth camera trajectories using an exponential dead-zone filter. Supports centered speaker crops or stacked split-screen layout (screen recording top 50%, webcam bottom 50%).
6. **Multi-Script ASS Subtitle Generation & CFR FFmpeg Render**: Generates Advanced SubStation Alpha (`.ass`) subtitle streams featuring word-by-word karaoke highlight timing, platform safe-zone margins, and fallback fonts across Latin, CJK, and Arabic scripts. Executes a single-pass FFmpeg filtergraph enforcing constant frame rate (30 fps CFR), 1080x1920 scaling, subtitle rendering, and two-pass EBU R128 audio loudness normalization (-14 LUFS integrated, -1 dBFS true peak).
7. **Verified Upload & Per-Clip Isolation**: Renders each clip within its own isolated execution boundary. Verifies file presence and byte size on disk or via S3 `head_object` check before setting status. Successful clips are marked available even if an adjacent clip fails, preventing total job failure.

---

## Key Engineering Decisions

- **Typed PipelineError Codes**: Exceptions raise structured `PipelineError` instances containing machine-readable `PipelineErrorCode` enums, failure stage labels, user-facing error messages, and retryable flags. Errors are never silently swallowed.
- **Heartbeat & WORKER_LOST Recovery**: Celery workers continuously write UTC heartbeat timestamps to the Redis job record during long-running tasks. The API evaluates heartbeat age against `WORKER_HEARTBEAT_TIMEOUT_SECONDS` (300 seconds); if a worker is killed or crashes mid-render, the API transitions the job from stuck active stages to `status="failed"`, `code="WORKER_LOST"`, and `retryable=true`.
- **Transcript-Driven Heuristic Fallback**: If the LLM provider encounters API rate limits (HTTP 429), connection timeouts, or content safety filter blocks, the pipeline automatically switches to heuristic candidate selection based on speech density, sentence boundaries, and duration targets. The resulting clips are marked with `selection_method="heuristic"` and delivered normally.
- **Versioned Cache Keys & Single-Flight Locks**: Transcripts and LLM candidates are cached in Redis using versioned schema keys (`cache:v1:transcript:...` and `cache:v1:candidates:...`). Concurrent jobs requesting the same media URL acquire a single-flight distributed lock, allowing the first worker to compute and cache results while subsequent workers wait, avoiding redundant transcription and LLM inference costs.
- **Two-Pass Loudness Normalization**: Audio tracks are normalized to broadcast standards (-14 LUFS, -1 dBFS true peak) using FFmpeg's `loudnorm` filter with measured dual-pass parameters, preventing audio clipping and volume jumps across clips.
- **HTTP Range Playback & Streaming Export**: Video endpoints support HTTP Byte-Range requests for low-latency scrubbing in browsers. Batch ZIP downloads (`GET /api/jobs/{job_id}/download-all`) stream through a memory-bounded `SpooledTemporaryFile` without holding full archive payloads in application memory.
- **Retention Sweepers & Coherent Lifetimes**: Celery Beat runs periodic sweepers enforcing a documented retention hierarchy: S3 Storage (7 days) >= Static Clip Retention (48 hours) >= Job Record TTL (24 hours) > Temp Scratch Cleanup (4 hours). When job records expire while static files remain (or vice versa), the API returns a structured `expired` status rather than dangling or broken URLs.

---

## Tech Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| Frontend Web UI | Next.js 14, React 18, TypeScript, Tailwind CSS, Lucide | User interface, job submission, status polling, video preview, and clip download |
| API Gateway | FastAPI, Uvicorn, Pydantic v2, Pydantic-Settings | REST API, request validation, startup checks, streaming ZIP exports, and health probes |
| Asynchronous Execution | Celery, Celery Beat | Asynchronous pipeline execution, task retry handling, and scheduled retention sweepers |
| State & Cache Store | Redis 7 (alpine) | Celery task broker (DB 0), job status records (DB 1), versioned caches, and distributed locks |
| Ingestion & Inspection | yt-dlp, FFprobe | Video and audio extraction, cookie authentication, stream validation, and metadata parsing |
| Speech-to-Text | faster-whisper (CTranslate2, Silero VAD) | Word-level timestamped transcription, speech validation, and language detection |
| Face Detection & Vision | OpenCV (cv2), YuNet ONNX | 4 Hz face detection, active speaker mouth-motion tracking, and smoothed reframing |
| Content Analysis | Google Gemini (`google-genai`), OpenAI (`openai`) | Structured JSON segment scoring, virality ranking, titles, and hashtags |
| Video Processing | FFmpeg (libx264, aac, loudnorm, ass) | Single-pass 1080x1920 cropping, 30 fps CFR conversion, subtitle burn-in, and loudness normalization |
| Storage Backends | Local Volume Mount, AWS S3 / Cloudflare R2 (`boto3`) | MP4 clip storage with faststart container flags and presigned or direct delivery |
| Testing & Verification | Pytest, FakeRedis, HTTPX, Node.js test runner | Unit and integration test suites, API contract validation, soak tests, and quality evaluation harness |

---

## Quick Start (Local & Docker)

### 1. Configure Environment Variables

Copy the example environment configuration into `.env` at the root of the repository:

```bash
cp .env.example .env
```

Set your LLM credentials and provider:

```env
LLM_PROVIDER=gemini
LLM_API_KEY=your_api_key_here
LLM_MODEL=gemini-2.5-flash
```

### 2. Boot Backend Infrastructure

Start Redis, the FastAPI gateway, and the Celery worker via Docker Compose:

```bash
docker compose up --build
```

Verify that all services started and pass startup validation:

```bash
curl -s http://localhost:8000/health | jq
```

Expected response status is `"ok"` with responsive Redis, active Celery workers, and writable storage.

### 3. Start Frontend

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` in your browser.

---

## Configuration

Essential environment variables loaded by `backend/app/config.py`:

| Variable | Default | Allowed Values | Description |
| :--- | :--- | :--- | :--- |
| `LLM_PROVIDER` | `gemini` | `gemini`, `openai` | Active AI provider for candidate segment analysis. |
| `LLM_API_KEY` | `""` | String | API key for Gemini or OpenAI (validated on boot). |
| `LLM_MODEL` | `gemini-2.5-flash` | String | Model identifier for virality ranking and hook scoring. |
| `REDIS_URL` | `redis://redis:6379/0` | Redis URI | Connection string for job state, versioned caches, and distributed locks. |
| `CELERY_BROKER_URL` | `None` | Redis URI | Celery broker connection string (defaults to `REDIS_URL` if omitted). |
| `STORAGE_BACKEND` | `local` | `local`, `s3` | Storage destination for rendered MP4 clips. |
| `PUBLIC_BASE_URL` | `http://localhost:8000` | Absolute URL | Base URL used to construct clip delivery links (loopback rejected in production). |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Absolute URL | Base URL consumed by the Next.js frontend to reach the FastAPI gateway. |

For detailed configuration reference, S3 bucket settings, retention window relationships, and operator remediation procedures, consult [RUNBOOK.md](RUNBOOK.md).

---

## API Reference

| Method & Route | Request Body | Response Status | Description |
| :--- | :--- | :--- | :--- |
| `POST /api/jobs` | `JobCreateRequest` | `202 Accepted` | Submits a media URL for clipping. Accepts `url`, `layout_mode` (`speaker` or `split`), `max_clips` (1-10), `caption_preset`, and `highlight_color`. Returns initial `JobStatusResponse` with `stage="queued"`. |
| `GET /api/jobs/{job_id}` | None | `200 OK` / `404 Not Found` | Retrieves job progress, stage, error details, warnings, and clip list. Returns `status="expired"` if the retention window has elapsed. Detects stale worker heartbeats and marks `WORKER_LOST`. |
| `GET /api/jobs/{job_id}/download-all` | None | `200 OK` / `410 Gone` | Streams all rendered MP4 clips and a JSON manifest as a single ZIP archive (`Content-Disposition: attachment`). Uses bounded memory spooling. |
| `GET /api/jobs/{job_id}/clips/{clip_id}/download` | None | `200 OK` / `404 Not Found` | Downloads an individual rendered clip directly with attachment headers for cross-origin browser compatibility. |
| `GET /health` | None | `200 OK` / `503 Unhealthy` | Deep health probe inspecting Redis ping latency, Celery queue depth, count and IDs of active workers, and storage writability. Available also at `/api/health`. |

---

## Testing & Quality Evaluation

### Backend Unit & Integration Tests

Run the full pytest suite:

```bash
backend/.venv/bin/pytest backend/tests/ -v
```

Run fast unit tests excluding slow FFmpeg rendering passes:

```bash
backend/.venv/bin/pytest backend/tests/ -m "not slow"
```

### Python Bytecode Compilation

Validate Python syntax across all modules and scripts:

```bash
python -m compileall backend/app backend/tests backend/scripts scripts
```

### Frontend Typecheck & Build

Verify TypeScript interfaces and produce a production Next.js build:

```bash
cd frontend
npx tsc --noEmit
npm run build
```

### Load & Soak Testing

Run concurrent synthetic jobs to verify queue throughput, peak memory usage, temporary directory cleanup, and worker loss recovery:

```bash
python backend/scripts/load_soak_test.py --concurrency 4 --total-jobs 8 --test-worker-lost --offline
```

### Pipeline Quality Evaluation Harness

Run the automated quality benchmark across 8 standard video categories (single speaker, two-person podcast, screen share, gameplay with webcam, vertical source, long video, multilingual, and noisy audio) using [`scripts/eval/run_eval.py`](scripts/eval/run_eval.py) to validate duration, sentence boundary cuts, face tracking ratios, loudness bounds, and contact sheet frames:

```bash
python scripts/eval/run_eval.py --generate-missing
```

Detailed results are written to `eval_results/summary.md` and `eval_results/report.json`.
