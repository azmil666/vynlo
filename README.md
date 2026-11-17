# Clipper MVP — YouTube/Stream to Shorts Generator

An automated AI pipeline that transforms long YouTube videos, podcasts, and streams into high-retention, 9:16 vertical shorts (YouTube Shorts, TikTok, Instagram Reels) with smart face tracking, split-screen gaming layout, and animated karaoke captions.

---

## Architecture Overview

```
                      +-------------------+
                      | Next.js Frontend  | (port 3000)
                      +---------+---------+
                                |
                     REST API   |
                                v
                      +-------------------+
                      |  FastAPI Gateway  | (port 8000)
                      +---------+---------+
                                |
                     Celery     | Redis (port 6379)
                                v
                      +-------------------+
                      |   Celery Worker   |
                      +---------+---------+
                                |
        +-----------+-----------+-----------+-----------+
        |           |           |           |           |
     yt-dlp      Whisper      LLM (AI)   MediaPipe   FFmpeg
   (Download)  (Transcribe)  (Viral Hook) (Face Crop) (Subtitles)
```

---

## Quickstart Guide

### 1. Configure Environment Variables
Copy the example environment configuration into `.env` at the root of the project:

```bash
cp .env.example .env
```

Open `.env` and set your LLM API Key:
```env
LLM_API_KEY=your_gemini_or_openai_api_key_here
LLM_MODEL=gemini-2.5-flash
```
*(You can use `gemini-2.5-flash` or OpenAI models such as `gpt-4o-mini`)*

---

### 2. Start the Backend Infrastructure (Docker)
Ensure Docker is installed and running, then start the Redis broker, FastAPI gateway, and Celery worker:

```bash
docker compose up --build
```

- **FastAPI API & Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)
- **Redis Server:** `localhost:6379`
- **Static Media Mount:** `http://localhost:8000/static/...`

---

### 3. Start the Next.js Frontend
In a new terminal window:

```bash
cd frontend
npm install
npm run dev
```

Open your browser at **[http://localhost:3000](http://localhost:3000)**.

---

## Features & Usage

1. **Enter Video URL:** Paste any public YouTube or stream URL.
2. **Choose Layout Mode:**
   - **Smart Face Crop (`speaker`):** Uses MediaPipe Face Detection to calculate median speaker positions and crops 9:16 vertical frames centering on the active speaker.
   - **Split Screen (`split`):** Stacks centered screen recording (top 50%) and webcam feed (bottom 50%) for gaming and reaction streams.
3. **Choose Max Clips:** Generate 1, 3, or 5 viral candidate shorts per run.
4. **Choose Highlight Color:** Select subtitle karaoke highlight colors (Classic Yellow, Neon Green, Vibrant Cyan, or Hot Pink).
5. **Live Pipeline Stepper:** Watch real-time progress through downloading, transcribing with Whisper, LLM viral hook identification, and FFmpeg rendering.
6. **Preview & Download:** Watch generated vertical clips directly in the studio gallery and download MP4 files.

---

## Troubleshooting

### YouTube Bot Protection & Rate Limiting (`yt-dlp`)
If YouTube restricts unauthenticated requests or presents bot verification:
1. Export cookies from your browser (e.g. using the "Get cookies.txt locally" extension).
2. Save the cookies file as `cookies.txt` in the root repository.
3. In `.env`, set:
   ```env
   YTDLP_COOKIES_PATH=/app/cookies.txt
   ```
4. Mount the cookies file in `docker-compose.yml` under `api` and `worker`:
   ```yaml
   volumes:
     - ./cookies.txt:/app/cookies.txt:ro
   ```
5. Re-run `docker compose up --build`.

### Custom S3 / Cloudflare R2 Storage (Optional)
By default, clips are saved to local volume storage (`/app/static`) and served over FastAPI at `http://localhost:8000/static/{job_id}/clip_{i}.mp4`.
To use Cloudflare R2 or AWS S3, configure these in `.env`:
```env
S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
S3_ACCESS_KEY=your_key
S3_SECRET_KEY=your_secret
S3_BUCKET_NAME=my-clips
S3_PUBLIC_BASE_URL=https://pub-<hash>.r2.dev
```
