import json
from unittest.mock import patch
import pytest

from app.config import settings
from app.schemas import PipelineError, PipelineErrorCode
from app.services.analyzer import (
    compute_rubric_score,
    compute_text_similarity,
    compute_time_overlap,
    candidate_rank_key,
    _dedupe_and_rank_candidates,
    find_viral_clips,
)
from tests.fixtures.fake_llm import FakeLLM


def _make_synthetic_transcript(duration: float, words_per_sec: float = 2.5) -> list[dict]:
    """Generate synthetic word-level transcript spanning given duration in seconds."""
    words: list[dict] = []
    current_time = 0.0
    idx = 0
    while current_time < duration:
        word_dur = 1.0 / words_per_sec
        is_period = (idx % 12 == 11)
        word_text = f"word{idx}." if is_period else f"word{idx}"
        words.append({
            "word": word_text,
            "start": round(current_time, 2),
            "end": round(min(duration, current_time + word_dur), 2),
        })
        current_time += word_dur
        idx += 1
    return words


def test_rubric_deterministic_scoring_formula():
    """Verify rubric computes deterministic weighted scores from sub-scores."""
    # 0.35 * 10 + 0.25 * 8 + 0.25 * 8 + 0.15 * 8 = 3.5 + 2.0 + 2.0 + 1.2 = 8.7 -> 87
    score, breakdown = compute_rubric_score(
        hook_strength=10.0,
        self_contained=8.0,
        payoff=8.0,
        emotional_intensity=8.0,
    )
    assert score == 87
    assert breakdown["hook_strength"] == 10.0
    assert breakdown["self_contained"] == 8.0
    assert breakdown["payoff"] == 8.0
    assert breakdown["emotional_intensity"] == 8.0


def test_rubric_tie_breaking():
    """Verify deterministic tie breaking: hook_strength > payoff > self_contained > -start_time."""
    clip1 = {
        "title": "Clip 1",
        "virality_score": 85,
        "score_breakdown": {"hook_strength": 9.0, "payoff": 7.0, "self_contained": 8.0, "emotional_intensity": 7.0},
        "start_time": 50.0,
        "end_time": 75.0,
    }
    clip2 = {
        "title": "Clip 2",
        "virality_score": 85,
        "score_breakdown": {"hook_strength": 8.0, "payoff": 9.0, "self_contained": 8.0, "emotional_intensity": 7.0},
        "start_time": 10.0,
        "end_time": 35.0,
    }
    # clip1 has higher hook_strength, so it must rank above clip2 despite later start_time
    ranked = sorted([clip2, clip1], key=candidate_rank_key, reverse=True)
    assert ranked[0]["title"] == "Clip 1"

    # When hook_strength is equal, payoff breaks the tie
    clip3 = {
        "title": "Clip 3",
        "virality_score": 85,
        "score_breakdown": {"hook_strength": 9.0, "payoff": 8.0, "self_contained": 7.0, "emotional_intensity": 7.0},
        "start_time": 20.0,
        "end_time": 45.0,
    }
    ranked = sorted([clip1, clip3], key=candidate_rank_key, reverse=True)
    assert ranked[0]["title"] == "Clip 3"


def test_deduplication_by_time_overlap():
    """Verify overlapping candidate clips (IoU >= threshold) deduplicate, keeping the higher-ranked clip."""
    # Clip A: [10.0, 30.0], score 90
    clip_a = {
        "title": "Alpha Highlight",
        "start_time": 10.0,
        "end_time": 30.0,
        "virality_score": 90,
        "score_breakdown": {"hook_strength": 9.0, "payoff": 9.0},
    }
    # Clip B: [15.0, 35.0], score 80 (Overlap is 15s, Union is 25s -> IoU = 0.6)
    clip_b = {
        "title": "Beta Highlight",
        "start_time": 15.0,
        "end_time": 35.0,
        "virality_score": 80,
        "score_breakdown": {"hook_strength": 8.0, "payoff": 8.0},
    }
    # Clip C: [50.0, 70.0], score 75 (No overlap)
    clip_c = {
        "title": "Gamma Highlight",
        "start_time": 50.0,
        "end_time": 70.0,
        "virality_score": 75,
        "score_breakdown": {"hook_strength": 7.5, "payoff": 7.5},
    }

    selected = _dedupe_and_rank_candidates([clip_a, clip_b, clip_c], max_clips=3)
    assert len(selected) == 2
    assert selected[0]["title"] == "Alpha Highlight"
    assert selected[1]["title"] == "Gamma Highlight"


def test_deduplication_by_near_duplicate_text():
    """Verify clips with near-identical text deduplicate even without time overlap."""
    clip_a = {
        "title": "Secret Python Optimization Tips for Developers",
        "hook": "Learn how to speed up your Python code instantly",
        "description": "Essential performance tricks for python developers",
        "start_time": 10.0,
        "end_time": 30.0,
        "virality_score": 88,
    }
    # Clip B is at a completely different time, but virtually identical text
    clip_b = {
        "title": "Secret Python Optimization Tips for All Developers",
        "hook": "Learn how to speed up your Python code quickly",
        "description": "Essential performance tricks for python devs",
        "start_time": 300.0,
        "end_time": 320.0,
        "virality_score": 78,
    }
    clip_c = {
        "title": "How to Brew Pour-over Coffee Properly",
        "hook": "The best ratio and water temp for coffee",
        "description": "Guide to brewing coffee at home",
        "start_time": 100.0,
        "end_time": 125.0,
        "virality_score": 75,
    }

    selected = _dedupe_and_rank_candidates([clip_a, clip_b, clip_c], max_clips=3)
    assert len(selected) == 2
    assert selected[0]["title"] == clip_a["title"]
    assert selected[1]["title"] == clip_c["title"]


def test_minimum_score_threshold_and_partial_return():
    """Verify candidates below threshold are excluded; return fewer clips rather than weak ones."""
    clip_strong = {
        "title": "Strong Moment",
        "start_time": 10.0,
        "end_time": 30.0,
        "virality_score": 85,
    }
    clip_weak = {
        "title": "Weak Boring Moment",
        "start_time": 40.0,
        "end_time": 60.0,
        "virality_score": 45,  # Below default MIN_VIRALITY_SCORE (60)
    }

    # Asking for 2 clips only returns 1 because weak clip is below threshold
    selected = _dedupe_and_rank_candidates([clip_strong, clip_weak], max_clips=2, min_score=60)
    assert len(selected) == 1
    assert selected[0]["title"] == "Strong Moment"


def test_all_below_threshold_raises_no_qualifying_clips():
    """Verify if all candidates are below threshold, find_viral_clips raises NO_QUALIFYING_CLIPS."""
    words = _make_synthetic_transcript(duration=100.0)
    # LLM returns candidates all with virality_score < 60
    fake_llm = FakeLLM(
        responses=[
            {
                "clips": [
                    {
                        "start_segment_id": 0,
                        "end_segment_id": 2,
                        "title": "Boring intro",
                        "virality_score": 40,
                    },
                    {
                        "start_segment_id": 3,
                        "end_segment_id": 5,
                        "title": "More boring talk",
                        "virality_score": 45,
                    },
                ]
            }
        ]
    )

    with pytest.raises(PipelineError) as exc_info:
        find_viral_clips(words, max_clips=2, llm_client=fake_llm)

    assert exc_info.value.code == PipelineErrorCode.NO_QUALIFYING_CLIPS


def test_boundary_straddling_clip_contained_fully():
    """Verify windowing overlap captures a boundary-straddling clip fully in the window that contains it."""
    # Create 1200s transcript (20 minutes). Window size = 600s, overlap = 75s (step = 525s).
    # Window 0: [0.0, 600.0]
    # Window 1: [525.0, 1125.0]
    # Straddling clip: [580.0, 620.0] (40s duration)
    # Window 0 ends at 600.0s, so clip is NOT fully contained in Window 0.
    # Window 1 begins at 525.0s and ends at 1125.0s, so clip IS fully contained in Window 1!
    words = _make_synthetic_transcript(duration=1200.0)

    # Fake LLM responses:
    # Window 0 response: clip [580, 620] (will be dropped as it exceeds 600.0s) + normal clip [100, 130]
    # Window 1 response: clip [580, 620] (fully contained: 580 >= 525 and 620 <= 1125)
    # Window 2 response: clip [1130, 1160]
    fake_llm = FakeLLM(
        responses=[
            # Window 0
            {
                "clips": [
                    {"start_time": 100.0, "end_time": 130.0, "title": "Early Win", "virality_score": 85},
                    {"start_time": 580.0, "end_time": 620.0, "title": "Straddling Win W0", "virality_score": 90},
                ]
            },
            # Window 1
            {
                "clips": [
                    {"start_time": 580.0, "end_time": 620.0, "title": "Straddling Win W1", "virality_score": 92},
                ]
            },
            # Window 2
            {
                "clips": [
                    {"start_time": 1130.0, "end_time": 1160.0, "title": "Late Win", "virality_score": 80},
                ]
            },
        ]
    )

    with patch.object(settings, "ANALYSIS_USE_WINDOWING", True), \
         patch.object(settings, "WINDOWING_THRESHOLD_SECONDS", 900.0), \
         patch.object(settings, "WINDOW_SIZE_SECONDS", 600.0), \
         patch.object(settings, "WINDOW_OVERLAP_SECONDS", 75.0):

        clips = find_viral_clips(words, max_clips=3, llm_client=fake_llm)

        titles = [c["title"] for c in clips]
        # Verify the straddling clip was captured from Window 1 where it was fully contained!
        assert any("Straddling Win" in t for t in titles)
        straddle_clip = next(c for c in clips if "Straddling Win" in c["title"])
        assert 575.0 <= straddle_clip["start_time"] <= 585.0
        assert 615.0 <= straddle_clip["end_time"] <= 625.0


def test_single_window_vs_multi_window_equivalence():
    """Verify single-pass and multi-window execution on a synthetic transcript produce consistent results."""
    words = _make_synthetic_transcript(duration=1000.0)

    # 1. Single-pass run (default: ANALYSIS_USE_WINDOWING = False)
    single_pass_resp = {
        "clips": [
            {"start_time": 120.0, "end_time": 150.0, "title": "Key Highlight 1", "virality_score": 90},
            {"start_time": 700.0, "end_time": 730.0, "title": "Key Highlight 2", "virality_score": 85},
        ]
    }
    fake_single = FakeLLM(responses=[single_pass_resp])

    with patch.object(settings, "ANALYSIS_USE_WINDOWING", False):
        clips_single = find_viral_clips(words, max_clips=2, llm_client=fake_single)

    assert len(clips_single) == 2
    assert clips_single[0]["title"] == "Key Highlight 1"
    assert clips_single[1]["title"] == "Key Highlight 2"

    # 2. Multi-window run (ANALYSIS_USE_WINDOWING = True)
    # Window 0 ([0, 600]) returns clip 1; Window 1 ([525, 1000]) returns clip 2
    fake_multi = FakeLLM(
        responses=[
            {"clips": [{"start_time": 120.0, "end_time": 150.0, "title": "Key Highlight 1", "virality_score": 90}]},
            {"clips": [{"start_time": 700.0, "end_time": 730.0, "title": "Key Highlight 2", "virality_score": 85}]},
        ]
    )

    with patch.object(settings, "ANALYSIS_USE_WINDOWING", True), \
         patch.object(settings, "WINDOWING_THRESHOLD_SECONDS", 900.0), \
         patch.object(settings, "WINDOW_SIZE_SECONDS", 600.0), \
         patch.object(settings, "WINDOW_OVERLAP_SECONDS", 75.0):

        clips_multi = find_viral_clips(words, max_clips=2, llm_client=fake_multi)

    assert len(clips_multi) == 2
    assert clips_multi[0]["title"] == "Key Highlight 1"
    assert clips_multi[1]["title"] == "Key Highlight 2"
    # Check boundary proximity (within boundary snapping tolerance)
    assert abs(clips_single[0]["start_time"] - clips_multi[0]["start_time"]) < 2.0
    assert abs(clips_single[1]["start_time"] - clips_multi[1]["start_time"]) < 2.0


def test_ranked_superset_cache_serving(fake_redis):
    """Verify that generating a superset of candidates allows cache to serve larger max_clips requests."""
    words = _make_synthetic_transcript(duration=100.0)
    video_id = "test_superset_vid"
    cache_key = f"cache:candidates:{video_id}"

    # Fake LLM returning 5 ranked clips at distinct time intervals
    fake_llm = FakeLLM(
        responses=[
            {
                "clips": [
                    {
                        "start_time": float(i * 15),
                        "end_time": float(i * 15 + 10),
                        "title": f"Clip {i}",
                        "virality_score": 95 - i * 5,
                    }
                    for i in range(1, 6)
                ]
            }
        ]
    )

    # Generate superset with superset_size=5
    all_candidates = find_viral_clips(words, max_clips=5, llm_client=fake_llm, superset_size=5)
    assert len(all_candidates) == 5

    # Simulate worker saving the superset to Redis
    fake_redis.set(cache_key, json.dumps(all_candidates))

    # Job 1 requests max_clips=2: served from cache without calling LLM
    cached_raw = json.loads(fake_redis.get(cache_key))
    assert len(cached_raw) >= 2
    job1_clips = cached_raw[:2]
    assert len(job1_clips) == 2
    assert job1_clips[0]["title"] == "Clip 1"
    assert job1_clips[1]["title"] == "Clip 2"

    # Job 2 later requests max_clips=4: also served directly from cache
    assert len(cached_raw) >= 4
    job2_clips = cached_raw[:4]
    assert len(job2_clips) == 4
    assert job2_clips[3]["title"] == "Clip 4"
