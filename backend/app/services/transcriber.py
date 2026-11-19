import logging
import os
import re
from typing import Any, Optional

from faster_whisper import WhisperModel

from app.config import settings
from app.schemas import PipelineError, PipelineErrorCode, TranscriptValidation

logger = logging.getLogger(__name__)

_cached_model: Optional[WhisperModel] = None
_cached_model_key: Optional[tuple] = None

# Known silence, background music, or subtitle hallucination phrases
KNOWN_SILENCE_HALLUCINATIONS = {
    "thank you for watching",
    "thanks for watching",
    "thank you",
    "thank you very much",
    "please subscribe",
    "subscribe to my channel",
    "like and subscribe",
    "see you next time",
    "see you in the next video",
    "bye bye",
    "bye",
    "subtitles by",
    "subtitles created by",
    "translated by",
    "all rights reserved",
    "closed captions by",
    "amara.org",
    "mbc",
    "music",
    "[music]",
    "(music)",
    "♪",
    "♫",
    "[applause]",
    "(applause)",
    "[laughter]",
    "(laughter)",
    "[silence]",
    "[blank_audio]",
}


class TranscriptWordList(list):
    """Subclass of list holding word dicts with attached TranscriptValidation."""

    def __init__(self, words: list[dict], validation: TranscriptValidation):
        super().__init__(words)
        self.validation = validation


def get_whisper_model() -> WhisperModel:
    """Return a cached instance of WhisperModel loaded once per worker process."""
    global _cached_model, _cached_model_key
    if _cached_model is not None:
        return _cached_model

    cpu_threads = getattr(settings, "WHISPER_CPU_THREADS", 4)
    if cpu_threads <= 0:
        cpu_threads = max(1, min(4, (os.cpu_count() or 4) // 2))

    current_key = (
        settings.WHISPER_MODEL_SIZE,
        settings.WHISPER_DEVICE,
        settings.WHISPER_COMPUTE_TYPE,
        cpu_threads,
    )
    logger.info(
        "Loading WhisperModel(size=%s, device=%s, compute_type=%s, cpu_threads=%d)",
        settings.WHISPER_MODEL_SIZE,
        settings.WHISPER_DEVICE,
        settings.WHISPER_COMPUTE_TYPE,
        cpu_threads,
    )
    _cached_model = WhisperModel(
        settings.WHISPER_MODEL_SIZE,
        device=settings.WHISPER_DEVICE,
        compute_type=settings.WHISPER_COMPUTE_TYPE,
        cpu_threads=cpu_threads,
    )
    _cached_model_key = current_key
    return _cached_model


def set_whisper_model(model: Optional[Any]) -> None:
    """Inject a fake or mock model for offline unit tests."""
    global _cached_model, _cached_model_key
    _cached_model = model
    _cached_model_key = None


def is_silence_hallucination_text(text: str) -> bool:
    """Check if text matches known silence/music hallucination phrases."""
    cleaned = re.sub(r"[^\w\s]", "", text.strip().lower())
    if not cleaned:
        return True
    cleaned_lower = text.strip().lower()
    for pattern in KNOWN_SILENCE_HALLUCINATIONS:
        pat_clean = re.sub(r"[^\w\s]", "", pattern)
        if pat_clean and pat_clean in cleaned:
            return True
        if pattern in cleaned_lower:
            return True
    # Check if text is just music symbols or punctuation
    if all(ch in "♪♫[]() \t\n\r" for ch in text):
        return True
    return False


def drop_repeated_ngram_loops(words: list[dict], max_repeats: int = 2) -> list[dict]:
    """Detect and drop consecutive repeated n-gram loops (n in 1..4 repeating >= max_repeats)."""
    if len(words) <= 3:
        return words

    cleaned_words: list[dict] = []
    i = 0
    n_words = len(words)

    while i < n_words:
        loop_found = False
        # Check n-gram lengths from 4 down to 1
        for n in range(min(4, (n_words - i) // (max_repeats + 1)), 0, -1):
            pattern = [words[i + k].get("word", "").strip().lower() for k in range(n)]
            if not any(pattern):
                continue

            # Count consecutive matches
            repeat_count = 1
            idx = i + n
            while idx + n <= n_words:
                next_chunk = [words[idx + k].get("word", "").strip().lower() for k in range(n)]
                if next_chunk == pattern:
                    repeat_count += 1
                    idx += n
                else:
                    break

            if repeat_count > max_repeats:
                # Keep max_repeats instances and discard the rest of the repetition loop
                keep_len = n * max_repeats
                cleaned_words.extend(words[i : i + keep_len])
                i = idx
                loop_found = True
                break

        if not loop_found:
            cleaned_words.append(words[i])
            i += 1

    return cleaned_words


def normalize_words(
    words: list[dict],
    timeline_offset: float = 0.0,
    media_duration: Optional[float] = None,
) -> list[dict]:
    """Normalize word timestamps: drop invalid, fix zero/negative durations, enforce monotonicity, clamp to media."""
    valid_words: list[dict] = []

    for w in words:
        raw_text = str(w.get("word") or "").strip()
        if not raw_text:
            continue

        raw_start = w.get("start")
        raw_end = w.get("end")
        if raw_start is None or raw_end is None:
            continue

        try:
            start = float(raw_start) + timeline_offset
            end = float(raw_end) + timeline_offset
        except (ValueError, TypeError):
            continue

        # Drop music notes or punctuation-only tokens
        if raw_text in ("♪", "♫") or is_silence_hallucination_text(raw_text):
            continue

        # Fix negative starts
        start = max(0.0, start)

        # Fix zero or negative durations
        if end <= start:
            end = start + 0.05

        valid_words.append({
            "word": raw_text,
            "start": start,
            "end": end,
        })

    # Drop repeated n-gram hallucination loops
    de_looped = drop_repeated_ngram_loops(valid_words)

    # Enforce strictly monotonic non-overlapping timestamps
    normalized: list[dict] = []
    prev_end = 0.0

    for item in de_looped:
        start = item["start"]
        end = item["end"]

        if start < prev_end:
            start = prev_end

        if end <= start:
            end = start + 0.05

        # Clamp to media duration if known
        if media_duration is not None and media_duration > 0:
            if start >= media_duration:
                # Starts after media ended
                continue
            if end > media_duration:
                end = media_duration
                if end <= start:
                    start = max(0.0, media_duration - 0.05)

        prev_end = end
        normalized.append({
            "word": item["word"],
            "start": round(start, 2),
            "end": round(end, 2),
        })

    return normalized


def compute_transcript_validation(
    words: list[dict],
    media_duration: Optional[float] = None,
    language: str = "en",
    language_probability: float = 1.0,
    model_size: str = "base",
) -> TranscriptValidation:
    """Calculate quality validation metrics and enforce speech coverage thresholds."""
    word_count = len(words)
    speech_duration = round(sum(w["end"] - w["start"] for w in words), 2)

    if media_duration and media_duration > 0:
        coverage = round(min(1.0, speech_duration / media_duration), 4)
    else:
        coverage = 1.0 if word_count > 0 else 0.0

    wpm = round((word_count / (speech_duration / 60.0)), 1) if speech_duration > 0 else 0.0

    report = TranscriptValidation(
        word_count=word_count,
        speech_duration=speech_duration,
        speech_coverage_ratio=coverage,
        words_per_minute=wpm,
        language=language,
        language_probability=round(float(language_probability or 0.0), 3),
        model=model_size,
    )

    # Threshold guards
    if word_count < settings.WHISPER_MIN_WORD_COUNT:
        raise PipelineError(
            code=PipelineErrorCode.NO_SPEECH,
            stage="transcribing",
            user_message=f"Insufficient speech detected: only {word_count} word(s) transcribed.",
            retryable=False,
        )

    # If media is longer than 10s and speech coverage is below threshold
    if media_duration and media_duration > 10.0 and coverage < settings.WHISPER_MIN_SPEECH_COVERAGE:
        raise PipelineError(
            code=PipelineErrorCode.NO_SPEECH,
            stage="transcribing",
            user_message=(
                f"Speech coverage ({round(coverage * 100, 1)}%) below minimum "
                f"threshold ({round(settings.WHISPER_MIN_SPEECH_COVERAGE * 100, 1)}%)."
            ),
            retryable=False,
        )

    return report


def transcribe_audio(
    audio_path: str,
    timeline_offset: float = 0.0,
    media_duration: Optional[float] = None,
) -> TranscriptWordList:
    """Transcribe audio into normalized words with validation report, rejecting hallucinations."""
    model = get_whisper_model()

    vad_params = dict(
        min_silence_duration_ms=settings.WHISPER_VAD_MIN_SILENCE_DURATION_MS,
        speech_pad_ms=settings.WHISPER_VAD_SPEECH_PAD_MS,
    )
    transcribe_kwargs = dict(
        word_timestamps=True,
        vad_filter=settings.WHISPER_VAD_FILTER,
        vad_parameters=vad_params,
        condition_on_previous_text=False,
        beam_size=settings.WHISPER_BEAM_SIZE,
        no_speech_threshold=settings.WHISPER_NO_SPEECH_THRESHOLD,
        compression_ratio_threshold=settings.WHISPER_COMPRESSION_RATIO_THRESHOLD,
    )
    if settings.WHISPER_LANGUAGE:
        transcribe_kwargs["language"] = settings.WHISPER_LANGUAGE

    segments, info = model.transcribe(audio_path, **transcribe_kwargs)

    raw_words: list[dict] = []
    for segment in segments:
        seg_text = getattr(segment, "text", "") or ""
        # Drop segments matching known silence hallucinations
        if is_silence_hallucination_text(seg_text):
            continue

        segment_words = getattr(segment, "words", None)
        if not segment_words:
            continue
        for w in segment_words:
            word_str = getattr(w, "word", "")
            if not word_str or not word_str.strip():
                continue
            raw_words.append({
                "word": word_str.strip(),
                "start": getattr(w, "start", 0.0),
                "end": getattr(w, "end", 0.0),
            })

    normalized = normalize_words(
        raw_words,
        timeline_offset=timeline_offset,
        media_duration=media_duration,
    )

    detected_lang = getattr(info, "language", "en") if info else "en"
    lang_prob = getattr(info, "language_probability", 1.0) if info else 1.0

    validation = compute_transcript_validation(
        normalized,
        media_duration=media_duration,
        language=detected_lang,
        language_probability=lang_prob,
        model_size=settings.WHISPER_MODEL_SIZE,
    )

    return TranscriptWordList(normalized, validation=validation)
