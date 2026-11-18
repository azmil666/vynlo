import json
import threading
import time
from unittest.mock import patch
import pytest

from app.config import settings
from app.services.cache import (
    acquire_single_flight_lock,
    compress_payload,
    compute_transcript_hash,
    decompress_payload,
    get_cached_candidates,
    get_cached_transcript,
    get_candidates_cache_key,
    get_transcript_cache_key,
    invalidate_video_cache,
    release_single_flight_lock,
    set_cached_candidates,
    set_cached_transcript,
    wait_for_single_flight_completion,
)
from tests.fixtures.fake_llm import FakeLLM


class TestCacheVersioningAndInvalidation:
    """Tests for versioned cache keys and invalidation behavior."""

    def test_transcript_cache_version_change_invalidates(self, fake_redis, transcript_normal):
        video_id = "vid_trans_ver"
        set_cached_transcript(
            fake_redis,
            video_id,
            transcript_normal,
            media_duration=30.0,
        )

        # Hit with original version
        words, val, err = get_cached_transcript(fake_redis, video_id, probed_duration=30.0)
        assert words is not None
        assert err is None
        assert len(words) == len(transcript_normal)

        # Invalidate via TRANSCRIPT_SCHEMA_VERSION change
        with patch.object(settings, "TRANSCRIPT_SCHEMA_VERSION", "v2_new_schema"):
            words_v2, _, err_v2 = get_cached_transcript(fake_redis, video_id, probed_duration=30.0)
            assert words_v2 is None
            assert err_v2 == "cache_miss"

        # Invalidate via GLOBAL_CACHE_VERSION change
        with patch.object(settings, "GLOBAL_CACHE_VERSION", "v2_global_bump"):
            words_global, _, err_global = get_cached_transcript(fake_redis, video_id, probed_duration=30.0)
            assert words_global is None
            assert err_global == "cache_miss"

    def test_candidate_cache_version_change_invalidates(self, fake_redis, transcript_normal):
        video_id = "vid_cand_ver"
        candidates_superset = [
            {"start_time": 1.0, "end_time": 8.0, "title": "Clip 1", "selection_method": "llm", "is_fallback": False, "virality_score": 90},
            {"start_time": 10.0, "end_time": 18.0, "title": "Clip 2", "selection_method": "llm", "is_fallback": False, "virality_score": 85},
        ]
        set_cached_candidates(fake_redis, video_id, transcript_normal, candidates_superset)

        # Hit with current versions
        clips, hit = get_cached_candidates(fake_redis, video_id, transcript_normal, max_clips=2)
        assert hit is True
        assert len(clips) == 2

        # Invalidate via CANDIDATE_SCHEMA_VERSION bump
        with patch.object(settings, "CANDIDATE_SCHEMA_VERSION", "v2_cand"):
            clips_v2, hit_v2 = get_cached_candidates(fake_redis, video_id, transcript_normal, max_clips=2)
            assert hit_v2 is False
            assert clips_v2 is None

        # Invalidate via PROMPT_VERSION bump
        with patch.object(settings, "PROMPT_VERSION", "v2_prompt"):
            clips_p2, hit_p2 = get_cached_candidates(fake_redis, video_id, transcript_normal, max_clips=2)
            assert hit_p2 is False
            assert clips_p2 is None

        # Invalidate via LLM_MODEL change
        with patch.object(settings, "LLM_MODEL", "gemini-3.0-ultra"):
            clips_m2, hit_m2 = get_cached_candidates(fake_redis, video_id, transcript_normal, max_clips=2)
            assert hit_m2 is False
            assert clips_m2 is None

    def test_explicit_video_cache_invalidation(self, fake_redis, transcript_normal):
        video_id = "vid_purge_target"
        set_cached_transcript(fake_redis, video_id, transcript_normal, media_duration=30.0)
        set_cached_candidates(fake_redis, video_id, transcript_normal, [
            {"start_time": 1.0, "end_time": 6.0, "title": "Highlight", "selection_method": "llm", "is_fallback": False, "virality_score": 88}
        ])

        deleted = invalidate_video_cache(fake_redis, video_id)
        assert deleted >= 2

        words, _, _ = get_cached_transcript(fake_redis, video_id)
        assert words is None
        clips, hit = get_cached_candidates(fake_redis, video_id, transcript_normal, max_clips=1)
        assert hit is False


class TestCacheDurationMismatchRejection:
    """Tests for validating cached transcripts against freshly probed media duration."""

    def test_transcript_rejected_when_last_word_exceeds_probed_duration(self, fake_redis):
        video_id = "vid_word_exceeds"
        words = [
            {"word": "Hello", "start": 1.0, "end": 2.0},
            {"word": "world", "start": 2.5, "end": 35.0},  # ends at 35.0s
        ]
        set_cached_transcript(fake_redis, video_id, words, media_duration=35.0)

        # Freshly probed video is only 30.0s long (tolerance default 2.0s -> max allowed 32.0s)
        cached_words, val, reason = get_cached_transcript(
            fake_redis, video_id, probed_duration=30.0, tolerance=2.0
        )

        assert cached_words is None
        assert reason is not None
        assert "exceeds probed duration" in reason
        assert "35.00s" in reason

    def test_transcript_rejected_when_media_duration_differs_materially(self, fake_redis):
        video_id = "vid_dur_differs"
        words = [
            {"word": "Short", "start": 0.5, "end": 2.0},
            {"word": "clip", "start": 2.2, "end": 4.0},
        ]
        # Cached media duration recorded as 50.0s
        set_cached_transcript(fake_redis, video_id, words, media_duration=50.0)

        # Freshly probed duration is 60.0s (> 50.0s + 2.0s tolerance)
        cached_words, val, reason = get_cached_transcript(
            fake_redis, video_id, probed_duration=60.0, tolerance=2.0
        )

        assert cached_words is None
        assert reason is not None
        assert "differs from probed duration" in reason

    def test_transcript_accepted_within_duration_tolerance(self, fake_redis):
        video_id = "vid_tolerance_pass"
        words = [
            {"word": "Within", "start": 0.5, "end": 2.0},
            {"word": "tolerance", "start": 2.2, "end": 29.5},
        ]
        set_cached_transcript(fake_redis, video_id, words, media_duration=30.0)

        # Probed duration 30.5s (difference 0.5s <= 2.0s tolerance)
        cached_words, val, reason = get_cached_transcript(
            fake_redis, video_id, probed_duration=30.5, tolerance=2.0
        )

        assert cached_words is not None
        assert reason is None
        assert len(cached_words) == 2


class TestCandidateSupersetReuseAndGuards:
    """Tests for ranked candidate superset caching, slicing, and heuristic exclusion."""

    def test_different_max_clips_reuse_same_superset(self, fake_redis, transcript_normal):
        video_id = "vid_superset_reuse"
        superset = [
            {"start_time": 1.0, "end_time": 6.0, "title": f"Clip {i}", "selection_method": "llm", "is_fallback": False, "virality_score": 95 - i}
            for i in range(1, 6)
        ]  # 5 clips in superset
        set_cached_candidates(fake_redis, video_id, transcript_normal, superset)

        # 1. Request max_clips = 2
        clips_2, hit_2 = get_cached_candidates(fake_redis, video_id, transcript_normal, max_clips=2)
        assert hit_2 is True
        assert len(clips_2) == 2
        assert clips_2[0]["title"] == "Clip 1"
        assert clips_2[1]["title"] == "Clip 2"

        # 2. Request max_clips = 4 (reuses same superset without bypass)
        clips_4, hit_4 = get_cached_candidates(fake_redis, video_id, transcript_normal, max_clips=4)
        assert hit_4 is True
        assert len(clips_4) == 4
        assert clips_4[3]["title"] == "Clip 4"

        # 3. Request max_clips = 10 (returns all 5 available in superset without len(cached)<max_clips bypass)
        clips_10, hit_10 = get_cached_candidates(fake_redis, video_id, transcript_normal, max_clips=10)
        assert hit_10 is True
        assert len(clips_10) == 5

    def test_heuristic_candidates_never_cached(self, fake_redis, transcript_normal):
        video_id = "vid_no_cache_heuristic"
        heuristic_clips = [
            {"start_time": 1.0, "end_time": 6.0, "title": "Heuristic Clip", "selection_method": "heuristic", "is_fallback": True, "virality_score": 75}
        ]

        cached = set_cached_candidates(fake_redis, video_id, transcript_normal, heuristic_clips)
        assert cached is False

        clips, hit = get_cached_candidates(fake_redis, video_id, transcript_normal, max_clips=1)
        assert hit is False
        assert clips is None


class TestSingleFlightLockingAndConcurrency:
    """Tests for Redis SET NX single-flight locking for transcription and analysis."""

    def test_concurrent_transcription_shares_single_computation(self, fake_redis, transcript_normal):
        video_id = "vid_concur_trans"
        worker_1 = "worker_task_1"
        worker_2 = "worker_task_2"

        # 1. Worker 1 acquires lock
        lock_acquired = acquire_single_flight_lock(fake_redis, f"transcription:{video_id}", worker_1, ttl_seconds=60)
        assert lock_acquired is True

        # 2. Worker 2 tries to acquire lock and fails
        lock_acquired_w2 = acquire_single_flight_lock(fake_redis, f"transcription:{video_id}", worker_2, ttl_seconds=60)
        assert lock_acquired_w2 is False

        # 3. In background, Worker 1 completes transcription, writes to cache, and releases lock
        def worker_1_finish():
            time.sleep(0.1)
            set_cached_transcript(fake_redis, video_id, transcript_normal, media_duration=30.0)
            release_single_flight_lock(fake_redis, f"transcription:{video_id}", worker_1)

        t = threading.Thread(target=worker_1_finish)
        t.start()

        # 4. Worker 2 waits for completion and reuses the cached transcript
        def check_transcript():
            w, _, _ = get_cached_transcript(fake_redis, video_id, probed_duration=30.0)
            return w

        waited_words = wait_for_single_flight_completion(
            fake_redis,
            f"transcription:{video_id}",
            check_transcript,
            timeout_seconds=2.0,
            poll_interval=0.05,
        )
        t.join()

        assert waited_words is not None
        assert len(waited_words) == len(transcript_normal)

    def test_single_flight_lock_releases_cleanly_on_failure(self, fake_redis):
        video_id = "vid_lock_fail"
        worker_1 = "worker_task_crash"

        assert acquire_single_flight_lock(fake_redis, f"analysis:{video_id}", worker_1, ttl_seconds=60) is True

        # Released cleanly by owner
        assert release_single_flight_lock(fake_redis, f"analysis:{video_id}", worker_1) is True

        # Another worker can now acquire it immediately
        assert acquire_single_flight_lock(fake_redis, f"analysis:{video_id}", "worker_task_next", ttl_seconds=60) is True


class TestCompressionAndSerialization:
    """Tests for zlib payload compression and backwards compatibility."""

    def test_compression_round_trip_and_size_reduction(self):
        large_transcript = [
            {"word": f"word_{i}", "start": i * 0.4, "end": (i + 1) * 0.4}
            for i in range(1000)
        ]
        raw_json_bytes = json.dumps(large_transcript).encode("utf-8")
        compressed_bytes = compress_payload(large_transcript)

        assert len(compressed_bytes) < len(raw_json_bytes) * 0.4  # At least 60% compression ratio

        decompressed = decompress_payload(compressed_bytes)
        assert len(decompressed) == 1000
        assert decompressed[500]["word"] == "word_500"

    def test_decompression_handles_legacy_uncompressed_json_string(self):
        legacy_data = [{"word": "legacy", "start": 1.0, "end": 2.0}]
        raw_str = json.dumps(legacy_data)

        decompressed = decompress_payload(raw_str)
        assert decompressed == legacy_data
