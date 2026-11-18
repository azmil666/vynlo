"""Core media, AI intelligence, computer vision, and storage services."""

from app.services.analyzer import (
    build_transcript_segments,
    candidate_rank_key,
    classify_llm_error,
    compute_rubric_score,
    compute_text_similarity,
    compute_time_overlap,
    find_viral_clips,
    sanitize_clip_text,
    score_heuristic_window,
    select_heuristic_clips,
    validate_llm_config,
    _dedupe_and_rank_candidates,
)
from app.services.captions import burn_subtitles, generate_ass_subtitles, render_clip_single_pass
from app.services.cropper import crop_to_vertical, compute_crop_filter
from app.services.downloader import (
    classify_ytdlp_error,
    download_media,
    get_format_selector,
    get_video_metadata,
    validate_downloaded_media,
)
from app.services.storage import upload_clip
from app.services.transcriber import (
    compute_transcript_validation,
    get_whisper_model,
    is_silence_hallucination_text,
    normalize_words,
    set_whisper_model,
    transcribe_audio,
)

__all__ = [
    "classify_ytdlp_error",
    "download_media",
    "get_format_selector",
    "get_video_metadata",
    "validate_downloaded_media",
    "transcribe_audio",
    "get_whisper_model",
    "set_whisper_model",
    "normalize_words",
    "is_silence_hallucination_text",
    "compute_transcript_validation",
    "find_viral_clips",
    "select_heuristic_clips",
    "score_heuristic_window",
    "crop_to_vertical",
    "compute_crop_filter",
    "generate_ass_subtitles",
    "burn_subtitles",
    "render_clip_single_pass",
    "upload_clip",
]
