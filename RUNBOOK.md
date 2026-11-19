# Production Runbook & Operator Guide: Vynlo

This operational runbook documents the architecture, configuration parameters, failure codes, cache invalidation procedures, retention policies, health monitoring, test execution, and quality evaluation harness for the Vynlo video processing system.

---

## 1. System Architecture Overview

The system processes long-form videos (e.g. YouTube, Vimeo) and produces vertical 9:16 short clips with synchronized captions, dynamic face tracking, and loudness normalization.

```
                  ┌────────────────┐
                  │ Next.js 14 Web │
                  │    Frontend    │
                  └───────┬────────┘
                          │ HTTP REST / Polling
                          ▼
                  ┌────────────────┐
                  │  FastAPI API   │ ◄─── Pre-flight Startup Validation
                  │     Server     │ ◄─── /health (Queue/Worker/Redis)
                  └───────┬────────┘
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
      ┌─────────────┐           ┌─────────────┐
      │ Redis (DB0) │           │ Redis (DB1) │ (Optional separation)
      │  Job State  │           │   Broker    │
      └──────┬──────┘           └──────┬──────┘
             │                         │
             │                         ▼
             │                  ┌─────────────┐
             │                  │Celery Worker│ ◄─── Heartbeat & Single-Pass
             │                  │  Processes  │      Rendering Loop
             │                  └──────┬──────┘
             │                         │
             ▼                         ▼
      ┌─────────────┐           ┌─────────────┐
      │ Local Static│    OR     │  S3 / R2    │
      │  Directory  │           │   Bucket    │
      └─────────────┘           └─────────────┘
```

- **API Server (`FastAPI`)**: Validates job requests, enqueues tasks, tracks job status with stale worker heartbeat detection, handles single & batch zip exports, and serves `/health`.
- **Worker Pool (`Celery`)**: Executes asynchronous pipeline stages: metadata, download (`yt-dlp`), audio extraction, speech transcription (`faster-whisper`), LLM viral analysis (`Gemini`/`OpenAI`), YuNet neural face detection & tracking (`OpenCV`), ASS subtitle generation, single-pass FFmpeg rendering, and upload.
- **Beat Scheduler (`Celery Beat`)**: Periodically triggers maintenance tasks: hourly `/tmp/clipper` temp directory cleanup and static storage sweepers.
- **Cache & State (`Redis`)**: Houses job status records (`job:<job_id>`), transcript caches (`cache:v1:transcript:...`), candidate caches (`cache:v1:candidates:...`), and concurrency single-flight distributed locks.
- **Storage Layer**: Local volume (`/app/static`) or S3/R2 object storage with faststart web-optimized MP4 container files.

---

## 2. Configuration Reference

All settings can be specified via environment variables or a `.env` file loaded by `backend/app/config.py`.

### 2.1 Core Environment & Networking
| Variable | Default | Allowed Values | Description |
| :--- | :--- | :--- | :--- |
| `ENVIRONMENT` | `development` | `development`, `staging`, `production` | Deployment mode. Production enforces strict non-loopback URLs, mandatory S3 keys, and strict health checks. |
| `PUBLIC_BASE_URL` | `http://localhost:8000` | Absolute URL | Base URL used to construct clip and download links. Must **not** be `localhost` or `127.0.0.1` in production. |
| `REDIS_URL` | `redis://redis:6379/0` | Valid Redis URL | Connection string for Redis job state, locks, and caches. |
| `CELERY_BROKER_URL` | None | Valid Redis/AMQP URL | Celery broker URL (defaults to `REDIS_URL` if omitted). |

### 2.2 Storage Backend & S3 Configuration
| Variable | Default | Allowed Values | Description |
| :--- | :--- | :--- | :--- |
| `STORAGE_BACKEND` | `local` | `local`, `s3` | Storage destination. In `s3` mode, uploads are verified with head_object size match and retry backoff. |
| `STATIC_DIR` | `/app/static` | Path | Local mount point for rendered video clips. Must be writable and shared between API and workers. |
| `S3_BUCKET_NAME` | None | String | S3 / Cloudflare R2 bucket name (mandatory in `s3` mode). |
| `S3_ENDPOINT_URL` | None | Absolute URL | Custom S3 API endpoint (required for Cloudflare R2 or MinIO). |
| `S3_ACCESS_KEY` | None | String | AWS / R2 access key ID (mandatory in `s3` mode). |
| `S3_SECRET_KEY` | None | String | AWS / R2 secret access key (mandatory in `s3` mode). |
| `S3_URL_STYLE` | `public` | `public`, `presigned` | URL generation strategy. Presigned URLs expire according to `S3_PRESIGNED_EXPIRY_SECONDS`. |
| `S3_PUBLIC_BASE_URL` | None | Absolute URL | Custom CDN base URL (e.g. `https://cdn.example.com`). |

### 2.3 Retention Windows & Lifecycle Management
The lifetime of job records, static clip files, temp folders, and caches are governed by strict documented relationships:
$$\text{S3 Retention} \ge \text{Static Clip Retention} \ge \text{Job Record TTL} > \text{Temp Scratch Cleanup}$$

| Setting | Default | Unit | Documented Operational Relationship |
| :--- | :--- | :--- | :--- |
| `JOB_RECORD_TTL_SECONDS` | `86400` (24h) | seconds | Lifetime of Redis `job:<id>` status record. Defines the user return window. |
| `STATIC_RETENTION_HOURS` | `48.0` (48h) | hours | Local static clip retention. **Must exceed or equal Job TTL** so clips never expire before the record. |
| `TEMP_DIR_CLEANUP_HOURS` | `4.0` | hours | Scratch directory retention in `/tmp/clipper`. Must exceed max task runtime (30m) but remain shorter than job record TTL. |
| `CACHE_TRANSCRIPT_TTL_SECONDS` | `604800` (7d) | seconds | Raw Whisper speech transcript cache. Independent of job TTL; shared across repeat requests for the same source URL. |
| `CACHE_CANDIDATES_TTL_SECONDS` | `604800` (7d) | seconds | LLM viral candidate clip extractions cache. Independent of job TTL. |
| `S3_RETENTION_DAYS` | `7` | days | S3 bucket lifecycle expiration. Must cover the Job Record TTL. |
| `S3_PRESIGNED_EXPIRY_SECONDS` | `604800` (7d) | seconds | Presigned URL validity duration. Aligned with S3 retention. |

### 2.4 Processing, AI & Thread Budgets
| Variable | Default | Description |
| :--- | :--- | :--- |
| `LLM_PROVIDER` | `gemini` | AI provider (`gemini` or `openai`). |
| `LLM_API_KEY` | `""` | API key for Gemini or OpenAI (mandatory in production/staging). |
| `LLM_MODEL` | `gemini-2.5-flash` | Model identifier for viral clip extraction. |
| `YUNET_MODEL_PATH` | `/app/models/...` | Path to baked YuNet ONNX neural face detector model. |
| `WHISPER_CPU_THREADS` | `4` | Thread ceiling per worker for faster-whisper compute to avoid oversubscription. |
| `RENDER_MAX_CONCURRENCY` | `4` | Maximum concurrent FFmpeg clip renders per worker job. |
| `WORKER_HEARTBEAT_TIMEOUT_SECONDS`| `300` (5m) | Inactivity threshold after which a stale job is marked as `WORKER_LOST`. |

---

## 3. Pipeline Failure Code Reference & Operator Actions

When a job or clip encounters an error, the system records a typed `PipelineErrorCode`.

| Error Code | Severity | User Message / Impact | Likely Root Cause | Operator Action & Remediation |
| :--- | :---: | :--- | :--- | :--- |
| `INVALID_URL` | Low | Unsupported or invalid video URL. | User submitted unsupported domain or malformed URL. | None (user error). Confirm URL matches `ALLOWED_VIDEO_DOMAINS` (YouTube, Vimeo, Twitch). |
| `VIDEO_UNAVAILABLE` | Low | Video is private, deleted, or geoblocked. | DRM, copyright takedown, private video, or bot check. | Verify URL manually in browser. If bot blocked, update `yt-dlp` package or refresh cookies file. |
| `VIDEO_TOO_LONG` | Low | Video duration exceeds maximum configured limit. | Input video duration $> 2$ hours (`MAX_INPUT_DURATION_SECONDS`). | User error. Operator can increase `MAX_INPUT_DURATION_SECONDS` if system resources allow. |
| `DOWNLOAD_FAILED` | Medium | Failed to download media or extract metadata. | Network timeout, transient CDN error, or storage full. | Check worker outbound connectivity to video CDN. Check disk space on `/tmp/clipper`. Retry is safe (`retryable=True`). |
| `NO_SPEECH` | Low | No spoken words detected in audio. | Video contains music only, silence, or non-vocal audio. | Inspect audio track with `ffprobe`. System automatically flags as non-retryable user error. |
| `TRANSCRIPTION_FAILED`| High | Speech transcription pipeline failed. | Whisper model corrupted, memory exhaustion (OOM), or thread deadlock. | Check worker logs. Verify CPU memory (`free -m`). Restart worker container if ctranslate2 crashed. |
| `LLM_CONFIG` | High | LLM provider or API key misconfigured. | Missing `LLM_API_KEY`, incorrect `LLM_PROVIDER`, or invalid model. | Check environment variables. Set valid `LLM_API_KEY` in `.env` and restart service. |
| `LLM_UNAVAILABLE` | High | LLM service unreachable or rate-limited. | Gemini/OpenAI API outage, HTTP 429 rate limit, or network block. | Check API provider status page. System automatically falls back to heuristic selection if enabled. |
| `LLM_BLOCKED` | Low | Analysis response blocked by content safety filters. | Source transcript triggered safety filter (hate speech, violence, etc.). | Non-retryable. Pipeline falls back to heuristic boundary selection. |
| `ANALYSIS_INVALID` | Medium | LLM response could not be parsed as valid JSON. | Model generated malformed JSON or markdown code fences. | Transient model error. Safe to retry (`retryable=True`). |
| `NO_QUALIFYING_CLIPS` | Medium | No clip candidates met minimum virality score. | Video content had low engagement scores below `MIN_VIRALITY_SCORE`. | Adjust `MIN_VIRALITY_SCORE` in config or allow heuristic fallback clips. |
| `RENDER_FAILED` | High | FFmpeg failed to crop, re-encode, or burn captions. | Missing codec, bad crop filter syntax, corrupted source frame, or disk full. | Check FFmpeg stderr logs in worker task log. Inspect `TEMP_DIR` disk space. |
| `UPLOAD_FAILED` | High | Failed to upload rendered clip to local/S3 storage. | S3 credentials invalid, bucket unreachable, or local static directory not writable. | Verify `STORAGE_BACKEND`. For S3, verify bucket policy and IAM permissions. For local, check volume permissions. |
| `WORKER_LOST` | High | Worker process crashed or killed mid-job. | OOM kill by OS kernel, node restart, or hard worker segfault. | Check kernel dmesg for OOM (`dmesg -T \| grep -i oom`). Increase worker container memory limit. Job is marked `retryable=True`. |
| `EXPIRED` | Low | Job record or clip files have elapsed retention window. | User returned after `JOB_RECORD_TTL_SECONDS` or `STATIC_RETENTION_HOURS`. | Normal expiration. User should re-submit the source URL to re-render. |
| `INTERNAL` | High | Unexpected internal unhandled exception. | Uncaught exception in worker or API loop. | Inspect full stack trace in `job_status.error.detail` and worker logs. File issue for bug fix. |

---

## 4. Cache Management & Invalidation Procedures

The pipeline caches expensive intermediate stages in Redis to avoid re-transcribing or re-analyzing repeat URLs.

### 4.1 Cache Keys & Structure
- **Transcript Cache**: `cache:v1:transcript:<video_id_or_hash>:<whisper_model>:<lang>:<schema_ver>`
- **Candidates Cache**: `cache:v1:candidates:<video_id_or_hash>:<transcript_hash>:<prompt_ver>:<llm_model>:<schema_ver>`
- **Single-Flight Lock**: `lock:single_flight:<stage>:<cache_key>`

### 4.2 Invalidation Commands (via `redis-cli`)

1. **Inspect Cached Keys for a Video ID**:
   ```bash
   redis-cli --scan --pattern "*<video_id>*"
   ```

2. **Invalidate Transcript for a Specific Video**:
   ```bash
   redis-cli --scan --pattern "cache:v1:transcript:*<video_id>*" | xargs redis-cli del
   ```

3. **Invalidate LLM Candidates for a Specific Video**:
   ```bash
   redis-cli --scan --pattern "cache:v1:candidates:*<video_id>*" | xargs redis-cli del
   ```

4. **Clear All Stuck Single-Flight Concurrency Locks**:
   ```bash
   redis-cli --scan --pattern "lock:single_flight:*" | xargs redis-cli del
   ```

5. **Flush Entire Cache (Emergency / Schema Upgrade)**:
   ```bash
   # CAUTION: Flushes all keys in the current DB
   redis-cli flushdb
   ```

---

## 5. Retention Sweepers & S3 Lifecycle

### 5.1 Celery Beat Scheduled Sweepers
The Celery Beat scheduler automatically runs two maintenance tasks:
- **`sweep_stale_temp_dirs_task` (Hourly)**: Sweeps `/tmp/clipper/<job_id>` folders older than `TEMP_DIR_CLEANUP_HOURS` (4 hours) that are not currently active in Redis.
- **`sweep_stale_static_files_task` (Hourly)**: Sweeps local static clip directories in `/app/static/<job_id>` older than `STATIC_RETENTION_HOURS` (48 hours) or whose Redis job record has been deleted.

### 5.2 Manual Sweeper Invocation
To force an immediate manual cleanup:
```bash
# In backend directory or container:
python -c "
from app.services.sweeper import cleanup_stale_temp_dirs, cleanup_stale_static_files
from app.main import get_redis_client
print('Cleaned temp:', cleanup_stale_temp_dirs())
print('Cleaned static:', cleanup_stale_static_files(redis_client=get_redis_client()))
"
```

### 5.3 Configuring S3 / R2 Lifecycle Policy
When running in `STORAGE_BACKEND=s3` mode, apply the 7-day expiration lifecycle policy using the bundled script:
```bash
python backend/scripts/configure_s3_lifecycle.py --bucket $S3_BUCKET_NAME --days 7
```

---

## 6. Health & Operational Monitoring

### 6.1 The `/health` Endpoint
Querying `GET /health` or `GET /api/health` returns the health status of all subsystems:

```json
{
  "status": "ok",
  "service": "vynlo-api",
  "environment": "production",
  "timestamp": "2026-11-19T18:50:00.000000Z",
  "redis": {
    "status": "ok",
    "latency_ms": 0.42,
    "detail": null
  },
  "queue": {
    "status": "ok",
    "depth": 3,
    "active_jobs": 2,
    "detail": null
  },
  "workers": {
    "status": "ok",
    "active_workers": 4,
    "worker_names": ["celery@worker-1", "celery@worker-2"],
    "detail": "4 worker(s) active"
  },
  "storage": {
    "status": "ok",
    "latency_ms": null,
    "detail": "Local static directory writable (/app/static)"
  }
}
```

### 6.2 Health Status Codes & Alerting Thresholds
- **HTTP 200 `status: "ok"`**: All components responsive, workers active, queue backlog healthy.
- **HTTP 200 `status: "degraded"`**: System functioning but warnings present:
  - Queue depth $> 100$ tasks pending (alert: `QueueBacklogWarning`).
  - 0 active Celery workers in production (alert: `NoActiveWorkersAlert`).
- **HTTP 503 `status: "unhealthy"`**: Critical failure (Redis unreachable). Automated orchestrator should restart or failover.

---

## 7. How to Run Tests & Verification

### 7.1 Backend Unit & Integration Tests
```bash
# Run full unit and integration test suite
backend/.venv/bin/pytest backend/tests/ -v

# Run fast tests excluding slow FFmpeg video renders
backend/.venv/bin/pytest backend/tests/ -m "not slow"

# Run only slow real FFmpeg rendering / audio tests
backend/.venv/bin/pytest backend/tests/ -m "slow"
```

### 7.2 Python Syntax & Bytecode Compilation
```bash
python -m compileall backend/app backend/tests backend/scripts scripts
```

### 7.3 Frontend Typecheck & Production Build
```bash
cd frontend
npx tsc --noEmit
npm run build
```

### 7.4 Load & Soak Test Script
Verify queue throughput, memory bounds, temp directory cleanup, and worker loss recovery under concurrency:
```bash
# Run 8 jobs across 4 concurrent threads in offline mode
python backend/scripts/load_soak_test.py --concurrency 4 --total-jobs 8 --test-worker-lost --offline
```

---

## 8. How to Run the Pipeline Quality Evaluation Harness

The evaluation harness in [`scripts/eval/`](scripts/eval/) runs the real pipeline against 8 video categories and validates 10 automated quality checks per clip:

```bash
# Run full evaluation across all 8 benchmark categories (generating fixtures if missing)
python scripts/eval/run_eval.py --generate-missing

# Run evaluation on specific categories
python scripts/eval/run_eval.py --categories single_speaker,podcast_two_person

# Fail immediately on first hard check failure
python scripts/eval/run_eval.py --fail-fast

# Output machine-readable JSON to stdout
python scripts/eval/run_eval.py --json
```

### 8.1 Inspecting Evaluation Results
- **Markdown Report**: [`eval_results/summary.md`](eval_results/summary.md) provides an overview table and check matrix.
- **Structured JSON**: [`eval_results/report.json`](eval_results/report.json) provides raw metrics and timestamps.
- **Contact Sheets**: Check [`eval_results/contact_sheets/*.jpg`](eval_results/contact_sheets/) for visual confirmation of start, mid, and end frames with timestamp overlays.
