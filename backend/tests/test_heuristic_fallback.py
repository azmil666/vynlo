import pytest
from app.config import settings
from app.schemas import ClipResult, PipelineError, PipelineErrorCode
from app.services.analyzer import (
    build_transcript_segments,
    find_viral_clips,
    score_heuristic_window,
    select_heuristic_clips,
)
from tests.fixtures.fake_llm import FakeLLM


class TestHeuristicFallback:
    """Tests for transcript-based heuristic clip selector and error class boundaries."""

    def test_llm_config_error_does_not_trigger_heuristic(self, transcript_normal):
        """LLM_CONFIG must raise immediately without attempting heuristic fallback."""
        fake_llm = FakeLLM([Exception("API_KEY_INVALID: User has not enabled the API")])

        with pytest.raises(PipelineError) as exc_info:
            find_viral_clips(
                transcript_normal,
                max_clips=2,
                llm_client=fake_llm,
                allow_heuristic_fallback=True,
            )

        assert exc_info.value.code == PipelineErrorCode.LLM_CONFIG
        assert exc_info.value.retryable is False

    def test_llm_blocked_error_does_not_trigger_heuristic(self, transcript_normal):
        """LLM_BLOCKED must raise immediately without attempting heuristic fallback."""
        fake_llm = FakeLLM([Exception("Safety block: HarmCategory sexually explicit detected")])

        with pytest.raises(PipelineError) as exc_info:
            find_viral_clips(
                transcript_normal,
                max_clips=2,
                llm_client=fake_llm,
                allow_heuristic_fallback=True,
            )

        assert exc_info.value.code == PipelineErrorCode.LLM_BLOCKED
        assert exc_info.value.retryable is False

    def test_llm_unavailable_triggers_heuristic_selector(self, transcript_normal):
        """LLM_UNAVAILABLE triggers select_heuristic_clips when transcript is valid."""
        # 3 failures to exhaust retries
        fake_llm = FakeLLM([
            Exception("429 Resource has been exhausted"),
            Exception("429 Resource has been exhausted"),
            Exception("429 Resource has been exhausted"),
        ])

        clips = find_viral_clips(
            transcript_normal,
            max_clips=2,
            llm_client=fake_llm,
            allow_heuristic_fallback=True,
        )

        assert len(clips) >= 1
        for clip in clips:
            assert clip["selection_method"] == "heuristic"
            assert clip["is_fallback"] is True
            assert "selection_reason" in clip
            assert clip["virality_score"] >= settings.HEURISTIC_QUALITY_FLOOR

    def test_heuristic_output_obeys_sentence_boundaries_and_floor(self, transcript_normal):
        """Heuristic clips must align with sentence boundaries and pass the quality floor."""
        clips = select_heuristic_clips(transcript_normal, max_clips=3, min_score=65)
        assert len(clips) >= 1

        segments = build_transcript_segments(transcript_normal)
        seg_starts = {round(s.start_time, 2) for s in segments}
        seg_ends = {round(s.end_time, 2) for s in segments}

        for c in clips:
            assert c["start_time"] in seg_starts
            assert c["end_time"] in seg_ends
            assert c["virality_score"] >= 65
            assert c["selection_method"] == "heuristic"
            assert c["is_fallback"] is True

    def test_heuristic_candidate_below_floor_fails_with_no_qualifying_clips(self):
        """If transcript heuristics cannot meet the quality floor, raise NO_QUALIFYING_CLIPS."""
        # Transcript with extreme filler, low speech density, and no punctuation
        low_quality_words = [
            {"word": "um", "start": 0.0, "end": 2.0},
            {"word": "uh", "start": 3.0, "end": 5.0},
            {"word": "and", "start": 6.0, "end": 8.0},
            {"word": "so", "start": 9.0, "end": 11.0},
            {"word": "yeah", "start": 12.0, "end": 14.0},
            {"word": "uh", "start": 15.0, "end": 18.0},
        ]
        # Quality floor set high
        clips = select_heuristic_clips(low_quality_words, max_clips=2, min_score=80)
        assert len(clips) == 0

        fake_llm = FakeLLM([
            Exception("503 Service Unavailable"),
            Exception("503 Service Unavailable"),
            Exception("503 Service Unavailable"),
        ])
        with pytest.raises(PipelineError) as exc_info:
            find_viral_clips(
                low_quality_words,
                max_clips=2,
                llm_client=fake_llm,
                allow_heuristic_fallback=True,
            )

        assert exc_info.value.code == PipelineErrorCode.NO_QUALIFYING_CLIPS

    def test_short_video_with_no_speech_fails(self, transcript_empty):
        """Empty transcript must immediately raise NO_QUALIFYING_CLIPS without padding."""
        fake_llm = FakeLLM()
        with pytest.raises(PipelineError) as exc_info:
            find_viral_clips(transcript_empty, max_clips=3, llm_client=fake_llm)

        assert exc_info.value.code == PipelineErrorCode.NO_QUALIFYING_CLIPS
        assert len(fake_llm.calls) == 0

    def test_llm_shortfall_is_not_padded_with_heuristics(self, transcript_normal):
        """When LLM returns fewer clips than requested, do not pad with heuristic clips."""
        fake_llm = FakeLLM()
        # LLM returns exactly 1 clip when 3 are requested
        fake_llm.add_response(
            {
                "clips": [
                    {
                        "start_time": 0.0,
                        "end_time": 8.0,
                        "title": "Sole LLM Highlight",
                        "hook": "Check this out",
                        "virality_score": 92,
                    }
                ]
            }
        )

        clips = find_viral_clips(
            transcript_normal,
            max_clips=3,
            llm_client=fake_llm,
            allow_heuristic_fallback=True,
        )

        assert len(clips) == 1
        assert clips[0]["title"] == "Sole LLM Highlight"
        assert clips[0]["selection_method"] == "llm"
        assert clips[0]["is_fallback"] is False

    def test_clip_result_schema_backward_compatibility(self):
        """ClipResult pydantic model synchronizes is_fallback and selection_method."""
        # 1. From legacy payload with only is_fallback: True
        legacy_fb = ClipResult(
            clip_id="clip_1",
            title="Clip",
            virality_score=80,
            duration=15.0,
            video_url="http://localhost:8000/clip_1.mp4",
            is_fallback=True,
        )
        assert legacy_fb.selection_method == "heuristic"
        assert legacy_fb.is_fallback is True

        # 2. From legacy payload with only is_fallback: False
        legacy_llm = ClipResult(
            clip_id="clip_2",
            title="Clip",
            virality_score=90,
            duration=15.0,
            video_url="http://localhost:8000/clip_2.mp4",
            is_fallback=False,
        )
        assert legacy_llm.selection_method == "llm"
        assert legacy_llm.is_fallback is False

        # 3. From new payload with selection_method="heuristic"
        new_fb = ClipResult(
            clip_id="clip_3",
            title="Clip",
            virality_score=75,
            duration=12.0,
            video_url="http://localhost:8000/clip_3.mp4",
            selection_method="heuristic",
            selection_reason="Transcript heuristic density",
        )
        assert new_fb.is_fallback is True
        assert new_fb.selection_method == "heuristic"
        assert new_fb.selection_reason == "Transcript heuristic density"
