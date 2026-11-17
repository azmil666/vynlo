import pytest

from app.services.analyzer import _snap_boundaries


class TestSnapBoundariesCharacterization:
    """Characterization tests for natural boundary snapping in analyzer.py."""

    def test_empty_transcript_handling(self):
        # Empty words: returns (max(0.0, start), max(start + 1.0, end))
        assert _snap_boundaries(10.0, 35.0, []) == (10.0, 35.0)
        assert _snap_boundaries(-5.0, 20.0, []) == (0.0, 20.0)
        assert _snap_boundaries(30.0, 20.0, []) == (30.0, 31.0)

    def test_start_snap_to_sentence_punctuation(self, transcript_normal):
        # In transcript_normal, "Today" (start: 2.60) is preceded by "podcast." (end: 2.30)
        # Search window [0.8, 4.8] includes "Today".
        # Snapped start should be 2.60 - 0.25 = 2.35.
        start_snapped, _ = _snap_boundaries(2.80, 15.0, transcript_normal)
        assert start_snapped == 2.35

    def test_start_snap_to_silence_gap(self, transcript_long_pauses):
        # In transcript_long_pauses:
        # "here." ends at 2.70, "Now" starts at 5.00 (pause = 2.30s > 0.65s)
        # If start is 4.80, "Now" starts a new sentence. Snapped start = 5.00 - 0.25 = 4.75.
        start_snapped, _ = _snap_boundaries(4.80, 10.0, transcript_long_pauses)
        assert start_snapped == 4.75

    def test_end_snap_characterization_picks_last_candidate(self, transcript_normal):
        # In transcript_normal:
        # "videos." ends at 5.95, "pop?" ends at 8.90, "hook!" ends at 10.75.
        # If end_time is 7.50, window is [5.0, 10.0].
        # Sentence ends in window: "videos." (5.95) and "pop?" (8.90).
        # Current implementation picks sentence_ends[-1] ("pop?"), snapped_end = 8.90 + 0.35 = 9.25.
        _, end_snapped = _snap_boundaries(2.0, 7.50, transcript_normal)
        assert end_snapped == 9.25

    def test_clamping_to_max_video_time(self, transcript_normal):
        max_time = max(float(w["end"]) for w in transcript_normal)  # 20.70
        _, end_snapped = _snap_boundaries(10.0, 20.50, transcript_normal)
        # 20.70 + 0.35 = 21.05 clamped to max_video_time (20.70)
        assert end_snapped == max_time

    def test_fallback_when_snapped_end_less_than_start(self, transcript_normal):
        # When end_time is configured so snapped_end <= snapped_start:
        # line 154: snapped_end = min(max_video_time, snapped_start + 15.0)
        s, e = _snap_boundaries(5.0, 5.0, transcript_normal)
        assert e > s
        assert e <= 20.70

    @pytest.mark.xfail(
        reason="Bug: start_time >= max_video_time with equal end_time causes snapped_end < snapped_start (inverted range)",
        strict=True,
    )
    def test_start_beyond_max_video_time_inversion_bug(self, transcript_normal):
        # When start_time == end_time >= max_video_time (e.g. 25.0):
        # snapped_start remains 25.0
        # snapped_end triggers line 155: min(20.70, 25.0 + 15.0) = 20.70
        # Leading to snapped_start (25.0) > snapped_end (20.70)
        s, e = _snap_boundaries(25.0, 25.0, transcript_normal)
        assert s < e, f"Expected snapped_start ({s}) < snapped_end ({e})"
