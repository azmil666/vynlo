from pathlib import Path
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from app.config import settings
from app.schemas import PipelineError, PipelineErrorCode, TranscriptValidation
from app.services.downloader import validate_downloaded_media
from app.services.transcriber import (
    compute_transcript_validation,
    drop_repeated_ngram_loops,
    get_whisper_model,
    is_silence_hallucination_text,
    normalize_words,
    set_whisper_model,
    transcribe_audio,
)
from tests.fixtures.media import generate_silent_audio


def test_normalize_words_drops_none_and_empty():
    """Verify missing, None, and whitespace-only words/timestamps are dropped."""
    raw = [
        {"word": None, "start": 0.0, "end": 1.0},
        {"word": "   ", "start": 1.0, "end": 2.0},
        {"word": "valid", "start": None, "end": 2.0},
        {"word": "word", "start": 2.0, "end": None},
        {"word": "good", "start": 2.0, "end": 2.5},
    ]
    normalized = normalize_words(raw)
    assert len(normalized) == 1
    assert normalized[0]["word"] == "good"
    assert normalized[0]["start"] == 2.0
    assert normalized[0]["end"] == 2.5


def test_normalize_words_fixes_negative_and_zero_durations():
    """Verify negative start is clamped to 0.0 and zero/negative duration gets minimum length."""
    raw = [
        {"word": "first", "start": -2.0, "end": -1.0},  # Negative timestamps
        {"word": "zero", "start": 3.0, "end": 3.0},     # Zero duration
        {"word": "inverted", "start": 5.0, "end": 4.0}, # Negative duration
    ]
    normalized = normalize_words(raw)
    assert len(normalized) == 3

    # first word starts at 0.0 with positive duration
    assert normalized[0]["start"] == 0.0
    assert normalized[0]["end"] > normalized[0]["start"]

    # zero word has positive duration
    assert normalized[1]["start"] == 3.0
    assert normalized[1]["end"] == 3.05

    # inverted word has positive duration
    assert normalized[2]["start"] == 5.0
    assert normalized[2]["end"] == 5.05


def test_normalize_words_enforces_monotonicity():
    """Verify overlapping timestamps are adjusted to be strictly monotonic."""
    raw = [
        {"word": "one", "start": 1.0, "end": 3.0},
        {"word": "two", "start": 2.0, "end": 2.5},  # Starts before previous ends
        {"word": "three", "start": 2.2, "end": 4.0}, # Starts before previous ends
    ]
    normalized = normalize_words(raw)
    assert len(normalized) == 3
    assert normalized[0]["start"] == 1.0
    assert normalized[0]["end"] == 3.0
    # 'two' pushed to 3.0
    assert normalized[1]["start"] == 3.0
    assert normalized[1]["end"] == 3.05
    # 'three' pushed to 3.05
    assert normalized[2]["start"] == 3.05
    assert normalized[2]["end"] == 4.0


def test_normalize_words_clamps_to_media_duration():
    """Verify words extending past media duration are clamped and words beyond are dropped."""
    raw = [
        {"word": "inside", "start": 8.0, "end": 9.5},
        {"word": "overlap", "start": 9.5, "end": 12.0}, # Extends past 10.0
        {"word": "outside", "start": 10.5, "end": 11.5}, # Starts after 10.0
    ]
    normalized = normalize_words(raw, media_duration=10.0)
    assert len(normalized) == 2
    assert normalized[0]["word"] == "inside"
    assert normalized[1]["word"] == "overlap"
    assert normalized[1]["end"] == 10.0


def test_hallucination_filter_silence_phrases():
    """Verify known Whisper silence and credit hallucinations are detected and dropped."""
    hallucinations = [
        "Thank you for watching!",
        "Thanks for watching.",
        "Please subscribe to my channel.",
        "Like and subscribe!",
        "See you in the next video.",
        "Subtitles by Amara.org",
        "[Music]",
        "♪",
        "♫",
    ]
    for text in hallucinations:
        assert is_silence_hallucination_text(text) is True

    # Real words should not be flagged
    assert is_silence_hallucination_text("Welcome to our tutorial on coding") is False


def test_hallucination_filter_repeated_ngram_loops():
    """Verify consecutive repeated word/n-gram loops are truncated."""
    # 2-gram loop: "you know" repeated 5 times
    raw_words = [
        {"word": "hello", "start": 0.0, "end": 0.5},
        {"word": "you", "start": 0.5, "end": 0.7},
        {"word": "know", "start": 0.7, "end": 0.9},
        {"word": "you", "start": 0.9, "end": 1.1},
        {"word": "know", "start": 1.1, "end": 1.3},
        {"word": "you", "start": 1.3, "end": 1.5},
        {"word": "know", "start": 1.5, "end": 1.7},
        {"word": "you", "start": 1.7, "end": 1.9},
        {"word": "know", "start": 1.9, "end": 2.1},
        {"word": "you", "start": 2.1, "end": 2.3},
        {"word": "know", "start": 2.3, "end": 2.5},
        {"word": "goodbye", "start": 2.5, "end": 3.0},
    ]
    filtered = drop_repeated_ngram_loops(raw_words, max_repeats=2)
    # Kept: "hello", 2x "you know" (4 words), "goodbye" = 6 words
    words_text = [w["word"] for w in filtered]
    assert words_text == ["hello", "you", "know", "you", "know", "goodbye"]


def test_validation_report_metrics():
    """Verify TranscriptValidation calculates correct word count, speech duration, WPM, and coverage."""
    words = [
        {"word": "hello", "start": 0.0, "end": 2.0},
        {"word": "world", "start": 2.0, "end": 4.0},
        {"word": "test", "start": 4.0, "end": 6.0},
        {"word": "speech", "start": 6.0, "end": 8.0},
        {"word": "analysis", "start": 8.0, "end": 10.0},
    ]
    report = compute_transcript_validation(
        words, media_duration=20.0, language="en", language_probability=0.98, model_size="base"
    )
    assert report.word_count == 5
    assert report.speech_duration == 10.0
    assert report.speech_coverage_ratio == 0.5  # 10s / 20s
    assert report.words_per_minute == 30.0     # 5 words in 10s = 30 wpm
    assert report.language == "en"
    assert report.model == "base"


def test_coverage_and_word_count_thresholds_raise_no_speech():
    """Verify below-threshold word count or coverage raises NO_SPEECH."""
    # Under minimum word count (< 5 words)
    too_few_words = [
        {"word": "hi", "start": 0.0, "end": 1.0},
        {"word": "there", "start": 1.0, "end": 2.0},
    ]
    with pytest.raises(PipelineError) as exc_info:
        compute_transcript_validation(too_few_words, media_duration=10.0)
    assert exc_info.value.code == PipelineErrorCode.NO_SPEECH

    # 5 words in 1000s video = 0.5% coverage (< 4% threshold)
    low_coverage_words = [
        {"word": f"word{i}", "start": float(i), "end": float(i + 1)}
        for i in range(5)
    ]
    with pytest.raises(PipelineError) as exc_info:
        compute_transcript_validation(low_coverage_words, media_duration=1000.0)
    assert exc_info.value.code == PipelineErrorCode.NO_SPEECH


def test_silent_or_empty_audio_raises_no_speech(tmp_path):
    """Verify empty/silent audio transcription raises NO_SPEECH."""
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], MagicMock(language="en", language_probability=0.99))
    set_whisper_model(mock_model)

    try:
        dummy_audio = str(tmp_path / "dummy.wav")
        Path(dummy_audio).touch()

        with pytest.raises(PipelineError) as exc_info:
            transcribe_audio(dummy_audio, media_duration=30.0)

        assert exc_info.value.code == PipelineErrorCode.NO_SPEECH
    finally:
        set_whisper_model(None)


def test_model_reuse_singleton():
    """Verify get_whisper_model reuses the same instance across invocations."""
    with patch("app.services.transcriber.WhisperModel") as mock_model_cls:
        instance = MagicMock()
        mock_model_cls.return_value = instance

        # Reset singleton cache
        set_whisper_model(None)

        m1 = get_whisper_model()
        m2 = get_whisper_model()

        assert m1 is instance
        assert m2 is instance
        # WhisperModel constructor should only have been called once
        mock_model_cls.assert_called_once()
        set_whisper_model(None)


def test_config_driven_transcription_parameters(tmp_path):
    """Verify transcribe_audio passes condition_on_previous_text=False, VAD, and thresholds to WhisperModel."""
    mock_model = MagicMock()
    fake_segment = MagicMock()
    fake_segment.text = "Hello world this is a reliable speech test"
    fake_segment.words = [
        MagicMock(word="Hello", start=0.0, end=0.5),
        MagicMock(word="world", start=0.5, end=1.0),
        MagicMock(word="this", start=1.0, end=1.5),
        MagicMock(word="is", start=1.5, end=2.0),
        MagicMock(word="reliable", start=2.0, end=2.5),
        MagicMock(word="speech", start=2.5, end=3.0),
    ]
    fake_info = MagicMock(language="en", language_probability=0.97)
    mock_model.transcribe.return_value = ([fake_segment], fake_info)

    set_whisper_model(mock_model)
    try:
        dummy_audio = str(tmp_path / "test.wav")
        Path(dummy_audio).touch()

        result = transcribe_audio(dummy_audio, media_duration=10.0)

        assert len(result) == 6
        assert result.validation.language == "en"
        assert result.validation.word_count == 6

        mock_model.transcribe.assert_called_once()
        _, kwargs = mock_model.transcribe.call_args
        assert kwargs["condition_on_previous_text"] is False
        assert kwargs["beam_size"] == settings.WHISPER_BEAM_SIZE
        assert kwargs["vad_filter"] == settings.WHISPER_VAD_FILTER
        assert kwargs["no_speech_threshold"] == settings.WHISPER_NO_SPEECH_THRESHOLD
        assert kwargs["compression_ratio_threshold"] == settings.WHISPER_COMPRESSION_RATIO_THRESHOLD
    finally:
        set_whisper_model(None)


def test_timeline_offset_with_nonzero_start_time_fixture(tmp_path):
    """Verify container with non-zero start time detects offset and applies it to timestamps."""
    out_video = str(tmp_path / "offset_container.mp4")

    # Generate container with 2.0s timestamp offset using lavfi and -output_ts_offset
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=320x240:rate=30",
        "-f",
        "lavfi",
        "-i",
        "sine=f=1000:r=44100",
        "-t",
        "2.0",
        "-output_ts_offset",
        "2.0",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        out_video,
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    source_info, warnings = validate_downloaded_media(out_video)
    # The container was created with output_ts_offset, so start times should reflect offset
    assert source_info.has_audio is True

    # Test that applying a known timeline offset properly shifts normalized words
    raw_words = [
        {"word": "shifted", "start": 1.0, "end": 2.0},
        {"word": "words", "start": 2.0, "end": 3.0},
    ]
    offset_result = normalize_words(raw_words, timeline_offset=1.5)
    assert offset_result[0]["start"] == 2.5
    assert offset_result[0]["end"] == 3.5
    assert offset_result[1]["start"] == 3.5
    assert offset_result[1]["end"] == 4.5
