from typing import Optional

from faster_whisper import WhisperModel

_model: Optional[WhisperModel] = None


def _get_model() -> WhisperModel:
    """Return a cached instance of WhisperModel to avoid reloading weights."""
    global _model
    if _model is None:
        _model = WhisperModel("base", device="cpu", compute_type="int8")
    return _model


def transcribe_audio(audio_path: str) -> list[dict]:
    """Transcribe audio file into a flattened list of words with timestamps."""
    model = _get_model()
    segments, _ = model.transcribe(audio_path, word_timestamps=True, vad_filter=True)

    words: list[dict] = []
    for segment in segments:
        segment_words = getattr(segment, "words", None)
        if not segment_words:
            continue
        for w in segment_words:
            cleaned_word = w.word.strip()
            if not cleaned_word:
                continue
            words.append(
                {
                    "word": cleaned_word,
                    "start": round(float(w.start), 2),
                    "end": round(float(w.end), 2),
                }
            )

    if not words:
        raise ValueError("No speech detected in audio")

    return words
