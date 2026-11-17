import json
import pytest

from app.services.analyzer import find_viral_clips
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
    candidates_cache_key = f"cache:candidates:{video_id}"
    cached_candidates_raw = redis_client.get(candidates_cache_key)
    cached_candidates = None
    if cached_candidates_raw:
        try:
            cached_candidates = json.loads(cached_candidates_raw)
        except Exception:
            cached_candidates = None

    if (
        cached_candidates
        and isinstance(cached_candidates, list)
        and len(cached_candidates) >= max_clips
        and not any(c.get("is_fallback") for c in cached_candidates)
    ):
        return cached_candidates[:max_clips], True
    else:
        candidates = find_viral_clips(words, max_clips=max_clips, llm_client=fake_llm)
        has_fallback = any(c.get("is_fallback") for c in candidates)
        if not has_fallback:
            redis_client.set(candidates_cache_key, json.dumps(candidates), ex=604800)
        return candidates, False


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
        cached_raw = fake_redis.get(f"cache:candidates:{video_id}")
        assert cached_raw is not None
        cached_data = json.loads(cached_raw)
        assert len(cached_data) == 2
        ttl = fake_redis.ttl(f"cache:candidates:{video_id}")
        assert 0 < ttl <= 604800

    def test_cache_hit_returns_sliced_candidates(self, fake_redis, transcript_normal, fake_llm):
        video_id = "vid_hit"
        cached_clips = [
            {"start_time": 1.0, "end_time": 5.0, "title": "Clip 1", "virality_score": 95, "is_fallback": False},
            {"start_time": 6.0, "end_time": 10.0, "title": "Clip 2", "virality_score": 90, "is_fallback": False},
            {"start_time": 11.0, "end_time": 15.0, "title": "Clip 3", "virality_score": 85, "is_fallback": False},
        ]
        fake_redis.set(f"cache:candidates:{video_id}", json.dumps(cached_clips))

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

    def test_cache_bypass_when_insufficient_clips(self, fake_redis, transcript_normal, fake_llm):
        video_id = "vid_insufficient"
        cached_clips = [
            {"start_time": 1.0, "end_time": 5.0, "title": "Clip 1", "virality_score": 95, "is_fallback": False}
        ]
        fake_redis.set(f"cache:candidates:{video_id}", json.dumps(cached_clips))

        # Request 2 clips when cache only has 1
        fake_llm.add_response(
            {
                "clips": [
                    {"start_time": 1.0, "end_time": 6.5, "title": "New 1", "virality_score": 90},
                    {"start_time": 12.0, "end_time": 18.0, "title": "New 2", "virality_score": 85},
                ]
            }
        )

        candidates, cache_hit = _simulate_candidate_cache_flow(
            fake_redis, video_id, max_clips=2, words=transcript_normal, fake_llm=fake_llm
        )

        assert cache_hit is False
        assert len(candidates) == 2
        assert len(fake_llm.calls) == 1

    def test_cache_bypass_when_cached_contains_fallback(
        self, fake_redis, transcript_normal, fake_llm
    ):
        video_id = "vid_fallback"
        cached_clips = [
            {"start_time": 1.0, "end_time": 5.0, "title": "Fallback 1", "is_fallback": True},
            {"start_time": 6.0, "end_time": 10.0, "title": "Fallback 2", "is_fallback": False},
        ]
        fake_redis.set(f"cache:candidates:{video_id}", json.dumps(cached_clips))

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
        fake_redis.set(f"cache:candidates:{video_id}", "corrupted_non_json_string{")

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
        self, fake_redis, transcript_empty, fake_llm
    ):
        video_id = "vid_empty_fallback"
        # Empty words causes find_viral_clips to return fallback clips
        candidates, cache_hit = _simulate_candidate_cache_flow(
            fake_redis, video_id, max_clips=1, words=transcript_empty, fake_llm=fake_llm
        )

        assert cache_hit is False
        assert any(c.get("is_fallback") for c in candidates)
        # Verify nothing was saved to Redis
        assert fake_redis.get(f"cache:candidates:{video_id}") is None

    @pytest.mark.xfail(
        reason="Bug: non-dict items in cached list crash any(c.get('is_fallback')) with AttributeError",
        strict=True,
    )
    def test_corrupt_cached_elements_crash_bug(self, fake_redis, transcript_normal, fake_llm):
        video_id = "vid_str_list"
        # Redis holds a JSON list of strings instead of dicts
        fake_redis.set(f"cache:candidates:{video_id}", json.dumps(["clip1", "clip2"]))

        # Current worker implementation raises AttributeError without try/except
        _simulate_candidate_cache_flow(
            fake_redis, video_id, max_clips=2, words=transcript_normal, fake_llm=fake_llm
        )
