"""Cache management service for transcripts and viral candidate supersets.

Provides versioned cache keys, metadata validation against freshly probed media duration,
zlib compression, ranked superset slicing, and Redis SET NX single-flight locks to ensure
concurrent jobs share computations without redundant processing or stale hits.
"""

import base64
import hashlib
import json
import logging
import time
import zlib
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from app.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# Cache Key Construction & Versioning
# ============================================================================

def compute_transcript_hash(words: list[dict]) -> str:
    """Compute a deterministic hash of transcript words and timestamps."""
    if not words:
        return "empty"
    normalized = [
        {
            "start": round(float(w.get("start", 0.0)), 2),
            "end": round(float(w.get("end", 0.0)), 2),
            "word": str(w.get("word", "")).strip().lower(),
        }
        for w in words
    ]
    raw_str = json.dumps(normalized, sort_keys=True)
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:16]


def get_transcript_cache_key(
    video_id: str,
    asr_model: Optional[str] = None,
    language: Optional[str] = None,
    schema_version: Optional[str] = None,
) -> str:
    """Construct versioned transcript cache key:
    video_id + asr_model + language setting + TRANSCRIPT_SCHEMA_VERSION.
    """
    model = asr_model or settings.WHISPER_MODEL_SIZE
    lang = language or settings.WHISPER_LANGUAGE or "auto"
    version = schema_version or settings.TRANSCRIPT_SCHEMA_VERSION
    global_prefix = settings.GLOBAL_CACHE_VERSION
    return f"cache:{global_prefix}:transcript:{video_id}:{model}:{lang}:{version}"


def get_candidates_cache_key(
    video_id: str,
    words: list[dict],
    prompt_version: Optional[str] = None,
    llm_model: Optional[str] = None,
    schema_version: Optional[str] = None,
) -> str:
    """Construct versioned candidate cache key:
    video_id + transcript content hash + prompt version + LLM model + CANDIDATE_SCHEMA_VERSION.
    """
    t_hash = compute_transcript_hash(words)
    p_ver = prompt_version or settings.PROMPT_VERSION
    model = llm_model or settings.LLM_MODEL
    c_ver = schema_version or settings.CANDIDATE_SCHEMA_VERSION
    global_prefix = settings.GLOBAL_CACHE_VERSION
    return f"cache:{global_prefix}:candidates:{video_id}:{t_hash}:{p_ver}:{model}:{c_ver}"


# ============================================================================
# Compression Helpers
# ============================================================================

def compress_payload(payload: Any) -> str:
    """Compress JSON-serializable payload with zlib and base64 for safe string/Redis storage."""
    json_bytes = json.dumps(payload).encode("utf-8")
    if settings.CACHE_COMPRESSION_ENABLED:
        compressed = zlib.compress(json_bytes, level=6)
        return "ZLIB:" + base64.b64encode(compressed).decode("ascii")
    return json_bytes.decode("utf-8")


def decompress_payload(raw: bytes | str) -> Any:
    """Decompress payload, handling ZLIB: base64, raw zlib bytes, and legacy uncompressed JSON."""
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            decompressed = zlib.decompress(raw)
            return json.loads(decompressed.decode("utf-8"))

    if isinstance(raw, str):
        if raw.startswith("ZLIB:"):
            b64_str = raw[5:]
            compressed = base64.b64decode(b64_str.encode("ascii"))
            decompressed = zlib.decompress(compressed)
            return json.loads(decompressed.decode("utf-8"))
        try:
            return json.loads(raw)
        except Exception:
            try:
                compressed = base64.b64decode(raw.encode("ascii"))
                decompressed = zlib.decompress(compressed)
                return json.loads(decompressed.decode("utf-8"))
            except Exception:
                pass

    return json.loads(raw)


# ============================================================================
# Transcript Cache Operations
# ============================================================================

def get_cached_transcript(
    redis_client: Any,
    video_id: str,
    probed_duration: Optional[float] = None,
    tolerance: Optional[float] = None,
    asr_model: Optional[str] = None,
    language: Optional[str] = None,
) -> tuple[Optional[list[dict]], Optional[dict], Optional[str]]:
    """Retrieve and validate cached transcript.

    Validates:
      1. Entry exists and decodes properly.
      2. If freshly probed duration is provided:
         - Last word end does not exceed probed duration (+ tolerance).
         - Cached media duration does not differ materially from probed duration.

    Returns:
        (words, validation_report, rejection_reason)
        If valid: (words, validation, None)
        If cache miss or rejected: (None, None, reason)
    """
    key = get_transcript_cache_key(video_id, asr_model=asr_model, language=language)
    raw = redis_client.get(key)
    if not raw:
        return None, None, "cache_miss"

    try:
        data = decompress_payload(raw)
    except Exception as e:
        logger.warning("Corrupt cached transcript payload for %s: %s", video_id, e)
        return None, None, f"corrupt_payload: {e}"

    if isinstance(data, dict) and "words" in data:
        words = data["words"]
        cached_media_dur = data.get("media_duration")
        validation = data.get("validation", {})
    elif isinstance(data, list):
        # Legacy format
        words = data
        cached_media_dur = None
        validation = {}
    else:
        return None, None, "invalid_transcript_structure"

    if not isinstance(words, list) or not words:
        return None, None, "empty_words_list"

    # Validate against freshly probed media duration if provided
    if probed_duration is not None and probed_duration > 0:
        tol = tolerance if tolerance is not None else settings.CACHE_DURATION_TOLERANCE_SECONDS

        # 1. Reject if last word end exceeds probed duration + tolerance
        try:
            last_word_end = max(float(w.get("end", 0.0)) for w in words)
        except Exception:
            last_word_end = 0.0

        if last_word_end > (probed_duration + tol):
            reason = (
                f"last word end ({last_word_end:.2f}s) exceeds probed duration "
                f"({probed_duration:.2f}s + {tol:.2f}s tolerance)"
            )
            logger.warning("Rejecting cached transcript for %s: %s", video_id, reason)
            return None, None, reason

        # 2. Reject if cached media duration differs materially
        if cached_media_dur is not None:
            try:
                cached_dur_float = float(cached_media_dur)
                if abs(cached_dur_float - probed_duration) > tol:
                    reason = (
                        f"cached media duration ({cached_dur_float:.2f}s) differs from probed duration "
                        f"({probed_duration:.2f}s + {tol:.2f}s tolerance)"
                    )
                    logger.warning("Rejecting cached transcript for %s: %s", video_id, reason)
                    return None, None, reason
            except (ValueError, TypeError):
                pass

    return words, validation, None


def set_cached_transcript(
    redis_client: Any,
    video_id: str,
    words: list[dict],
    media_duration: Optional[float] = None,
    validation_report: Optional[dict] = None,
    asr_model: Optional[str] = None,
    language: Optional[str] = None,
    ttl_seconds: Optional[int] = None,
) -> None:
    """Store transcript with metadata and compression in Redis."""
    if not words:
        return

    key = get_transcript_cache_key(video_id, asr_model=asr_model, language=language)
    ttl = ttl_seconds if ttl_seconds is not None else settings.CACHE_TRANSCRIPT_TTL_SECONDS
    payload = {
        "version": settings.TRANSCRIPT_SCHEMA_VERSION,
        "global_version": settings.GLOBAL_CACHE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "video_id": video_id,
        "media_duration": media_duration,
        "validation": validation_report or {},
        "words": words,
    }
    compressed = compress_payload(payload)
    redis_client.set(key, compressed, ex=ttl)
    logger.info("Cached transcript for %s (key: %s, ttl: %ds)", video_id, key, ttl)


# ============================================================================
# Candidate Superset Cache Operations
# ============================================================================

def get_cached_candidates(
    redis_client: Any,
    video_id: str,
    words: list[dict],
    max_clips: int,
    prompt_version: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> tuple[Optional[list[dict]], bool]:
    """Retrieve and slice cached ranked candidate superset.

    Returns:
        (sliced_candidates, cache_hit)
    """
    key = get_candidates_cache_key(
        video_id, words, prompt_version=prompt_version, llm_model=llm_model
    )
    raw = redis_client.get(key)
    if not raw:
        return None, False

    try:
        data = decompress_payload(raw)
    except Exception as e:
        logger.warning("Corrupt cached candidates payload for %s: %s", video_id, e)
        return None, False

    candidates: list[dict] = []
    if isinstance(data, dict) and "candidates" in data:
        candidates = data["candidates"]
    elif isinstance(data, list):
        candidates = data
    else:
        return None, False

    if not isinstance(candidates, list) or not candidates:
        return None, False

    # Never serve heuristic or fallback clips from candidate cache
    if any(
        not isinstance(c, dict) or c.get("is_fallback") or c.get("selection_method") == "heuristic"
        for c in candidates
    ):
        logger.info("Bypassing candidate cache for %s: invalid elements or contains heuristic/fallback clips", video_id)
        return None, False

    # Reuse the ranked superset by slicing to requested max_clips
    logger.info(
        "Using cached candidates for %s (superset has %d, requested %d)",
        video_id,
        len(candidates),
        max_clips,
    )
    return candidates[:max_clips], True


def set_cached_candidates(
    redis_client: Any,
    video_id: str,
    words: list[dict],
    candidates_superset: list[dict],
    prompt_version: Optional[str] = None,
    llm_model: Optional[str] = None,
    ttl_seconds: Optional[int] = None,
) -> bool:
    """Store ranked candidate superset in Redis if all candidates are LLM-selected.

    Heuristic / fallback clips are NEVER cached.
    """
    if not candidates_superset:
        return False

    # Guard: Cache ONLY selection_method == "llm" candidates
    has_fallback = any(
        not isinstance(c, dict) or c.get("is_fallback") or c.get("selection_method") == "heuristic"
        for c in candidates_superset
    )
    if has_fallback:
        logger.info(
            "Skipping candidate caching for %s: candidates contain heuristic/fallback clips",
            video_id,
        )
        return False

    key = get_candidates_cache_key(
        video_id, words, prompt_version=prompt_version, llm_model=llm_model
    )
    ttl = ttl_seconds if ttl_seconds is not None else settings.CACHE_CANDIDATES_TTL_SECONDS
    payload = {
        "version": settings.CANDIDATE_SCHEMA_VERSION,
        "global_version": settings.GLOBAL_CACHE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "video_id": video_id,
        "prompt_version": prompt_version or settings.PROMPT_VERSION,
        "llm_model": llm_model or settings.LLM_MODEL,
        "candidates": candidates_superset,
    }
    compressed = compress_payload(payload)
    redis_client.set(key, compressed, ex=ttl)
    logger.info(
        "Cached %d candidate clips for %s (key: %s, ttl: %ds)",
        len(candidates_superset),
        video_id,
        key,
        ttl,
    )
    return True


# ============================================================================
# Single-Flight Locking
# ============================================================================

def acquire_single_flight_lock(
    redis_client: Any,
    lock_name: str,
    owner_id: str,
    ttl_seconds: Optional[int] = None,
) -> bool:
    """Acquire single-flight lock via Redis SET NX."""
    ttl = ttl_seconds if ttl_seconds is not None else settings.SINGLE_FLIGHT_LOCK_TTL_SECONDS
    lock_key = f"lock:single_flight:{lock_name}"
    acquired = redis_client.set(lock_key, owner_id, nx=True, ex=ttl)
    return bool(acquired)


def release_single_flight_lock(
    redis_client: Any,
    lock_name: str,
    owner_id: str,
) -> bool:
    """Release single-flight lock only if still owned by owner_id."""
    lock_key = f"lock:single_flight:{lock_name}"
    try:
        current_owner = redis_client.get(lock_key)
        if isinstance(current_owner, bytes):
            current_owner = current_owner.decode("utf-8")
        if current_owner == owner_id:
            redis_client.delete(lock_key)
            return True
    except Exception as e:
        logger.warning("Error releasing single flight lock %s: %s", lock_key, e)
    return False


def wait_for_single_flight_completion(
    redis_client: Any,
    lock_name: str,
    check_cache_fn: Callable[[], Any],
    timeout_seconds: Optional[float] = None,
    poll_interval: Optional[float] = None,
) -> Any:
    """Poll for completion of a concurrent job holding the single-flight lock.

    Returns the cached result if available before timeout, or None if the lock was
    released without producing a cached entry (e.g. failure).
    """
    timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else settings.SINGLE_FLIGHT_WAIT_TIMEOUT_SECONDS
    )
    interval = (
        poll_interval
        if poll_interval is not None
        else settings.SINGLE_FLIGHT_POLL_INTERVAL_SECONDS
    )
    lock_key = f"lock:single_flight:{lock_name}"

    start_time = time.time()
    while (time.time() - start_time) < timeout:
        cached_result = check_cache_fn()
        if cached_result is not None:
            return cached_result

        # Check if the lock was released
        if not redis_client.exists(lock_key):
            # Final check in case cache was written right before release
            final_check = check_cache_fn()
            if final_check is not None:
                return final_check
            return None

        time.sleep(interval)

    return check_cache_fn()


# ============================================================================
# Cache Invalidation Utility
# ============================================================================

def invalidate_video_cache(redis_client: Any, video_id: str) -> int:
    """Explicitly delete all cache keys (transcripts and candidates) for a video_id."""
    patterns = [
        f"cache:*:transcript:{video_id}:*",
        f"cache:*:candidates:{video_id}:*",
    ]
    deleted_count = 0
    for pattern in patterns:
        keys = redis_client.keys(pattern)
        if keys:
            deleted_count += redis_client.delete(*keys)
    logger.info("Invalidated %d cache keys for %s", deleted_count, video_id)
    return deleted_count
