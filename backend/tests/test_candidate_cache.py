import json
import pytest

from app.services.analyzer import find_viral_clips
from app.services.cache import (
    get_cached_candidates,
    get_candidates_cache_key,
    set_cached_candidates,
)
from tests.fixtures.fake_llm import FakeLLM


def _simulate_candidate_cache_flow(
    redis_client,
    video_id: str,
    max_clips: int,
    words: list[dict],
    fake_llm: FakeLLM,
) -> tuple[list[dict], bool]:
    """Simulate the candidate cache read/write guard logic from worker.py.

    Returns: (selected_candidates, cache_hit)
    """
    candidates, hit = get_cached_candidates(redis_client, video_id, words, max_clips=max_clips)
    if hit and candidates is not None:
        return candidates, True

    all_candidates = find_viral_clips(words, max_clips=max_clips, llm_client=fake_llm)
    set_cached_candidates(redis_client, video_id, words, all_candidates)
    return all_candidates[:max_clips], False


class TestCandidateCacheGuards:
    """Characterization tests for candidate cache read/write guards."""

    def test_cache_miss_writes_to_redis_when_no_fallback(
        self, fake_redis, transcript_normal, fake_llm
    ):
        video_id = "vid_123"
        fake_llm.add_response(
            {
                "clips": [
                    {
                        "start_time": 2.0,
                        "end_time": 10.0,
                        "title": "Clip 1",
                        "virality_score": 90,
                    },
                    {
                        "start_time": 11.0,
                        "end_time": 19.0,
                        "title": "Clip 2",
                        "virality_score": 85,
                    },
                ]
            }
        )

        candidates, cache_hit = _simulate_candidate_cache_flow(
            fake_redis, video_id, max_clips=2, words=transcript_normal, fake_llm=fake_llm
        )

        assert cache_hit is False
        assert len(candidates) == 2
        key = get_candidates_cache_key(video_id, transcript_normal)
        cached_raw = fake_redis.get(key)
        assert cached_raw is not None
        ttl = fake_redis.ttl(key)
        assert 0 < ttl <= 604800

    def test_cache_hit_returns_sliced_candidates(self, fake_redis, transcript_normal, fake_llm):
        video_id = "vid_hit"
        cached_clips = [
            {"start_time": 1.0, "end_time": 5.0, "title": "Clip 1", "selection_method": "llm", "is_fallback": False, "virality_score": 95},
            {"start_time": 6.0, "end_time": 10.0, "title": "Clip 2", "selection_method": "llm", "is_fallback": False, "virality_score": 90},
            {"start_time": 11.0, "end_time": 15.0, "title": "Clip 3", "selection_method": "llm", "is_fallback": False, "virality_score": 85},
        ]
        set_cached_candidates(fake_redis, video_id, transcript_normal, cached_clips)

        # Request 2 clips when cache has 3
        candidates, cache_hit = _simulate_candidate_cache_flow(
            fake_redis, video_id, max_clips=2, words=transcript_normal, fake_llm=fake_llm
        )

        assert cache_hit is True
        assert len(candidates) == 2
        assert candidates[0]["title"] == "Clip 1"
        assert candidates[1]["title"] == "Clip 2"
        # Verify LLM was not called
        assert len(fake_llm.calls) == 0

    def test_cache_reuse_when_requesting_more_clips_than_cached(self, fake_redis, transcript_normal, fake_llm):
        """P4: Deleted len(cached)<max_clips bypass. Reuses cached superset even if fewer than requested."""
        video_id = "vid_insufficient"
        cached_clips = [
            {"start_time": 1.0, "end_time": 5.0, "title": "Clip 1", "selection_method": "llm", "is_fallback": False, "virality_score": 95}
        ]
        set_cached_candidates(fake_redis, video_id, transcript_normal, cached_clips)

        # Request 2 clips when cache only has 1
        candidates, cache_hit = _simulate_candidate_cache_flow(
            fake_redis, video_id, max_clips=2, words=transcript_normal, fake_llm=fake_llm
        )

        # Reuses the cached clip without calling LLM
        assert cache_hit is True
        assert len(candidates) == 1
        assert len(fake_llm.calls) == 0

    def test_cache_bypass_when_cached_contains_fallback(
        self, fake_redis, transcript_normal, fake_llm
    ):
        video_id = "vid_fallback"
        cached_clips = [
            {"start_time": 1.0, "end_time": 5.0, "title": "Fallback 1", "selection_method": "heuristic", "is_fallback": True},
            {"start_time": 6.0, "end_time": 10.0, "title": "Fallback 2", "selection_method": "llm", "is_fallback": False},
        ]
        key = get_candidates_cache_key(video_id, transcript_normal)
        fake_redis.set(key, json.dumps({"candidates": cached_clips}))

        fake_llm.add_response(
            {
                "clips": [
                    {"start_time": 2.0, "end_time": 8.0, "title": "LLM 1", "virality_score": 90},
                    {"start_time": 9.0, "end_time": 15.0, "title": "LLM 2", "virality_score": 85},
                ]
            }
        )

        candidates, cache_hit = _simulate_candidate_cache_flow(
            fake_redis, video_id, max_clips=2, words=transcript_normal, fake_llm=fake_llm
        )

        assert cache_hit is False
        assert len(fake_llm.calls) == 1

    def test_cache_bypass_on_corrupt_json(self, fake_redis, transcript_normal, fake_llm):
        video_id = "vid_corrupt"
        key = get_candidates_cache_key(video_id, transcript_normal)
        fake_redis.set(key, "corrupted_non_json_string{")

        fake_llm.add_response(
            {
                "clips": [
                    {"start_time": 1.0, "end_time": 6.0, "title": "Clean", "virality_score": 90}
                ]
            }
        )

        candidates, cache_hit = _simulate_candidate_cache_flow(
            fake_redis, video_id, max_clips=1, words=transcript_normal, fake_llm=fake_llm
        )

        assert cache_hit is False
        assert len(candidates) == 1

    def test_skips_writing_cache_when_result_has_fallback(
        self, fake_redis, transcript_normal, fake_llm
    ):
        video_id = "vid_heuristic_fallback"
        # LLM failure triggers heuristic fallback clips
        fake_llm.add_response(Exception("429 Resource has been exhausted"))
        fake_llm.add_response(Exception("429 Resource has been exhausted"))
        fake_llm.add_response(Exception("429 Resource has been exhausted"))
        candidates, cache_hit = _simulate_candidate_cache_flow(
            fake_redis, video_id, max_clips=1, words=transcript_normal, fake_llm=fake_llm
        )

        assert cache_hit is False
        assert len(candidates) >= 1
        assert any(c.get("is_fallback") or c.get("selection_method") == "heuristic" for c in candidates)
        # Verify nothing was saved to Redis
        key = get_candidates_cache_key(video_id, transcript_normal)
        assert fake_redis.get(key) is None

    def test_corrupt_cached_elements_no_crash(self, fake_redis, transcript_normal, fake_llm):
        video_id = "vid_str_list"
        key = get_candidates_cache_key(video_id, transcript_normal)
        # Redis holds a JSON list of strings instead of dicts
        fake_redis.set(key, json.dumps(["clip1", "clip2"]))

        fake_llm.add_response(
            {
                "clips": [
                    {"start_time": 1.0, "end_time": 6.0, "title": "Safe", "virality_score": 90}
                ]
            }
        )

        # Does not crash with AttributeError
        candidates, cache_hit = _simulate_candidate_cache_flow(
            fake_redis, video_id, max_clips=1, words=transcript_normal, fake_llm=fake_llm
        )
        assert cache_hit is False
        assert len(candidates) == 1
