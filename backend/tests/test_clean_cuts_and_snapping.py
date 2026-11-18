import logging
import pytest

from app.config import settings
from app.services.analyzer import _snap_boundaries, find_viral_clips
from tests.fixtures.fake_llm import FakeLLM


def test_gap_smaller_than_tail_no_bleed():
    """Verify that when silence gap after word is smaller than tail, tail is capped at min(tail, gap/2)."""
    # Word 0: [2.0, 5.0] ("Hello.")
    # Word 1: [5.10, 8.0] ("World.") -> gap is 5.10 - 5.00 = 0.10s
    # Tail config is 0.35s. With min(tail, gap/2) = min(0.35, 0.05) = 0.05s.
    # Snapped end should be 5.00 + 0.05 = 5.05s, strictly < 5.10s (no bleed into "World.").
    words = [
        {"word": "Hello.", "start": 2.0, "end": 5.0},
        {"word": "World.", "start": 5.10, "end": 8.0},
    ]
    s, e = _snap_boundaries(2.0, 5.0, words)
    assert e == 5.05
    assert e < 5.10, "Cut must never reach the next word's start!"


def test_lead_in_gap_smaller_than_lead_in_no_bleed():
    """Verify that when gap before word is smaller than lead-in, lead-in is capped at gap/2."""
    # Word 0: [1.0, 3.90] ("Before.")
    # Word 1: [4.00, 7.00] ("After.") -> gap is 4.00 - 3.90 = 0.10s
    # Lead-in config is 0.20s. With min(lead_in, gap/2) = min(0.20, 0.05) = 0.05s.
    # Snapped start should be 4.00 - 0.05 = 3.95s, strictly > 3.90s (no bleed into "Before.").
    words = [
        {"word": "Before.", "start": 1.0, "end": 3.90},
        {"word": "After.", "start": 4.00, "end": 7.00},
    ]
    s, e = _snap_boundaries(4.0, 7.0, words)
    assert s == 3.95
    assert s > 3.90, "Cut must never enter the previous word!"


def test_no_punctuation_pause_based_snapping():
    """Verify pause-based snapping works cleanly when transcript has no punctuation."""
    # Words with no punctuation, but with a natural pause between phrase 1 and phrase 2
    words = [
        {"word": "welcome", "start": 1.0, "end": 1.5},
        {"word": "everyone", "start": 1.55, "end": 2.0},
        {"word": "today", "start": 2.05, "end": 2.5},
        # Pause: 3.5 - 2.5 = 1.0s (> SNAP_PAUSE_THRESHOLD)
        {"word": "now", "start": 3.5, "end": 3.8},
        {"word": "we", "start": 3.85, "end": 4.0},
        {"word": "begin", "start": 4.05, "end": 4.5},
        # Pause: 6.0 - 4.5 = 1.5s
        {"word": "thanks", "start": 6.0, "end": 6.5},
    ]

    # Snap near "now" (start 3.5s) and "begin" (end 4.5s)
    s, e = _snap_boundaries(3.4, 4.6, words)
    # Start: "now" starts at 3.5, gap is 1.0s, lead_in=0.20 -> 3.5 - 0.20 = 3.30s
    assert s == 3.30
    # End: "begin" ends at 4.5, next word "thanks" starts at 6.0, gap=1.5s, tail=0.35 -> 4.5 + 0.35 = 4.85s
    assert e == 4.85


def test_fast_speech_tiny_gaps_never_bleeds():
    """Verify extremely fast speech with 0.02s gaps never bleeds into adjacent words."""
    words = [
        {"word": "fast.", "start": 1.0, "end": 1.28},
        {"word": "talking.", "start": 1.30, "end": 1.58},
        {"word": "speaker.", "start": 1.60, "end": 1.88},
    ]
    # Snapping around "talking." [1.30, 1.58]
    s, e = _snap_boundaries(1.30, 1.58, words)
    # Lead-in: gap = 1.30 - 1.28 = 0.02s -> gap/2 = 0.01s -> s = 1.29s (> 1.28)
    assert s == 1.29
    assert s > 1.28
    # Tail: gap = 1.60 - 1.58 = 0.02s -> gap/2 = 0.01s -> e = 1.59s (< 1.60)
    assert e == 1.59
    assert e < 1.60


def test_search_window_widened_once_then_word_boundary_kept_with_warning():
    """Verify widening search window once, keeping word boundary with a warning when no natural boundary exists."""
    # Monologue with NO punctuation and tiny 0.05s gaps (no pause >= 0.40s)
    words = [
        {"word": f"word{i}", "start": float(i * 0.5), "end": float(i * 0.5 + 0.45)}
        for i in range(20)  # 0.0 to 10.0s
    ]
    warnings: list[str] = []
    # Target cut right in the middle at 5.2s (word10 is 5.0-5.45)
    s, e = _snap_boundaries(5.2, 8.2, words, warnings=warnings)
    # Must keep word boundary, never cut inside a word!
    assert 4.95 <= s <= 5.0
    assert any("widening" in w for w in warnings)


def test_clip_at_zero_and_media_end():
    """Verify clips at 0.0s and at total media duration are clamped safely."""
    words = [
        {"word": "First.", "start": 0.0, "end": 2.0},
        {"word": "Middle.", "start": 2.5, "end": 5.0},
        {"word": "Last.", "start": 5.5, "end": 10.0},
    ]
    # At 0.0s start
    s, e = _snap_boundaries(0.0, 5.0, words)
    assert s == 0.0
    assert s >= 0.0

    # At media end (10.0s)
    s2, e2 = _snap_boundaries(2.5, 10.0, words)
    assert e2 == 10.0
    assert e2 <= 10.0


def test_snap_pushing_duration_over_max_adjusted_with_warning():
    """Verify that when snapping pushes duration over MAX_CLIP_DURATION_SECONDS, it is adjusted with a warning."""
    # 70s transcript with punctuation
    # Word 0: [1.0, 3.0]
    # Word 1: [4.0, 57.0]
    # Word 2: [58.0, 60.5]
    # Word 3: [62.0, 65.0]
    words = [
        {"word": "Start.", "start": 1.0, "end": 3.0},
        {"word": "Filler.", "start": 4.0, "end": 57.0},
        {"word": "Ending.", "start": 58.0, "end": 60.5},
        {"word": "Next.", "start": 62.0, "end": 65.0},
    ]
    # LLM selected [1.0, 60.5] (initial duration = 59.5s <= 60.0s)
    # Snapping:
    # Start: word "Start." starts at 1.0s, gap=1.0s, lead_in=0.20s -> snapped_start = 0.80s
    # End: word "Ending." ends at 60.5s, next word starts at 62.0s, gap=1.5s, tail=0.35s -> snapped_end = 60.85s
    # Snapped duration: 60.85 - 0.80 = 60.05s (> 60.0s max duration!)
    # Post-snap re-validation should adjust snapped_end and record warning.
    fake_llm = FakeLLM(
        responses=[
            {
                "clips": [
                    {
                        "start_time": 1.0,
                        "end_time": 60.5,
                        "title": "Long Clip",
                        "virality_score": 90,
                    }
                ]
            }
        ]
    )

    clips = find_viral_clips(words, max_clips=1, llm_client=fake_llm)
    assert len(clips) == 1
    clip = clips[0]
    dur = round(clip["end_time"] - clip["start_time"], 2)
    assert dur <= settings.MAX_CLIP_DURATION_SECONDS
    assert any("exceeded max duration" in w for w in clip.get("warnings", []))


def test_overlap_created_by_snapping_resolved_or_dropped():
    """Verify that if snapping causes two adjacent clips to overlap, the overlap is resolved or dropped with warning."""
    # Words spanning 0.0s to 30.0s
    words = [
        {"word": "Clip", "start": 1.0, "end": 2.0},
        {"word": "one.", "start": 2.1, "end": 10.0},
        {"word": "Clip", "start": 10.1, "end": 11.0},
        {"word": "two.", "start": 11.1, "end": 20.0},
    ]

    # Two candidate clips close together:
    # Candidate 1: [1.0, 10.0]
    # Candidate 2: [10.1, 20.0]
    fake_llm = FakeLLM(
        responses=[
            {
                "clips": [
                    {"start_time": 1.0, "end_time": 10.0, "title": "Clip A", "virality_score": 90},
                    {"start_time": 10.1, "end_time": 20.0, "title": "Clip B", "virality_score": 85},
                ]
            }
        ]
    )

    clips = find_viral_clips(words, max_clips=2, llm_client=fake_llm)
    assert len(clips) >= 1
    # Check that clips never overlap in the final valid_clips
    for i in range(len(clips) - 1):
        assert clips[i]["end_time"] <= clips[i + 1]["start_time"]
