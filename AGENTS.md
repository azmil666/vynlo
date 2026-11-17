STANDARD RULES (apply to every task)
- Repo: clipping-mvp (FastAPI + Celery + Redis backend; Next.js 14 frontend; faster-whisper, YuNet/OpenCV, FFmpeg, Gemini).
- The task descriptions come from an audit by a tool that saw the code. Treat every file name, line number and behaviour in the prompt as an UNVERIFIED hypothesis. First locate and read the real code, trace callers and callees, and understand the current behaviour. If reality differs from the prompt, follow the intent, adapt, and report the difference.
- Preserve existing working behaviour and the current architecture. Do not add services, microservices or new infrastructure. Do not refactor unrelated code.
- Never swallow errors. Every fallback must either produce valid, useful output or raise a typed error. Anything degraded must be recorded as a warning in the job result.
- Add or update tests (pytest for backend, tsc/build plus component tests for frontend). Tests must run offline, using fakes for Gemini/Redis/S3 and ffmpeg-generated fixtures.
- Verify: run the backend test suite, python -m compileall, and for frontend changes npx tsc --noEmit && npm run build. If something cannot be run, say so explicitly.
- Final report: files modified, behaviour changes, tests added and run (with results), assumptions that turned out false, and any follow-ups.