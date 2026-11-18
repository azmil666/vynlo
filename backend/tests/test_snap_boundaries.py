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
        # In transcript_normal, "Today" (start: 2.60) is preceded by "podcast." (end: 2.30, gap: 0.30s)
        # Lead-in capped at min(lead_in=0.20, gap/2=0.15) = 0.15s
        # Snapped start: 2.60 - 0.15 = 2.45s (never bleeds into "podcast.")
        start_snapped, _ = _snap_boundaries(2.80, 15.0, transcript_normal)
        assert start_snapped == 2.45

    def test_start_snap_to_silence_gap(self, transcript_long_pauses):
        # In transcript_long_pauses:
        # "here." ends at 2.70, "Now" starts at 5.00 (pause = 2.30s > 0.40s)
        # Lead-in capped at min(lead_in=0.20, gap/2=1.15) = 0.20s
        # Snapped start = 5.00 - 0.20 = 4.80s
        start_snapped, _ = _snap_boundaries(4.80, 10.0, transcript_long_pauses)
        assert start_snapped == 4.80

    def test_end_snap_characterization_picks_last_candidate(self, transcript_normal):
        # In transcript_normal:
        # "pop?" ends at 8.90, next word "It" starts at 9.20 (gap: 0.30s)
        # Tail capped at min(tail=0.35, gap/2=0.15) = 0.15s
        # Snapped end = 8.90 + 0.15 = 9.05s (never bleeds into "It")
        _, end_snapped = _snap_boundaries(2.0, 7.50, transcript_normal)
        assert end_snapped == 9.05

    def test_clamping_to_max_video_time(self, transcript_normal):
        max_time = max(float(w["end"]) for w in transcript_normal)  # 20.70
        _, end_snapped = _snap_boundaries(10.0, 20.50, transcript_normal)
        assert end_snapped == max_time

    def test_fallback_when_snapped_end_less_than_start(self, transcript_normal):
        # When end_time is configured so snapped_end <= snapped_start:
        s, e = _snap_boundaries(5.0, 5.0, transcript_normal)
        assert e > s
        assert e <= 20.70

    def test_start_beyond_max_video_time_inversion_bug(self, transcript_normal):
        # When start_time == end_time >= max_video_time (e.g. 25.0):
        # Handled cleanly without inversion: s < e
        s, e = _snap_boundaries(25.0, 25.0, transcript_normal)
        assert s < e, f"Expected snapped_start ({s}) < snapped_end ({e})"
