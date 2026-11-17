import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)


class ClipCandidate(BaseModel):
    start_time: float = Field(
        description="Start time of the clip in seconds matching transcript timestamps"
    )
    end_time: float = Field(
        description="End time of the clip in seconds matching transcript timestamps"
    )
    title: str = Field(description="Punchy 4-8 word hook title for the clip")
    virality_score: int = Field(
        description="Virality score from 1 to 100", ge=1, le=100
    )


class ClipAnalysisResponse(BaseModel):
    clips: list[ClipCandidate] = Field(
        default_factory=list, description="List of identified viral clips"
    )


def _format_transcript_chunks(words: list[dict]) -> str:
    """Group words into sentence- or pause-based timestamped lines."""
    if not words:
        return ""

    chunks: list[str] = []
    current_words: list[str] = []
    chunk_start: float = float(words[0]["start"])
    prev_end: float = float(words[0]["end"])

    for idx, w in enumerate(words):
        word_text = str(w.get("word", "")).strip()
        if not word_text:
            continue

        start = float(w.get("start", prev_end))
        end = float(w.get("end", start))
        pause = start - prev_end

        if current_words and (pause > 1.2 or len(current_words) >= 20):
            chunks.append(
                f"[{chunk_start:.2f}s - {prev_end:.2f}s] {' '.join(current_words)}"
            )
            current_words = []
            chunk_start = start

        current_words.append(word_text)
        prev_end = end

        if word_text.endswith((".", "?", "!")):
            chunks.append(
                f"[{chunk_start:.2f}s - {end:.2f}s] {' '.join(current_words)}"
            )
            current_words = []
            if idx + 1 < len(words):
                chunk_start = float(words[idx + 1].get("start", end))

    if current_words:
        chunks.append(
            f"[{chunk_start:.2f}s - {prev_end:.2f}s] {' '.join(current_words)}"
        )

    return "\n".join(chunks)


def _snap_boundaries(
    start_time: float, end_time: float, words: list[dict]
) -> tuple[float, float]:
    """Snap start and end times to natural sentence or silence boundaries to prevent clipping cutoffs."""
    if not words:
        return max(0.0, start_time), max(start_time + 1.0, end_time)

    max_video_time = max(float(w.get("end", 0.0)) for w in words)

    # 1. Start Snap: Search words in [start_time - 2.0, start_time + 2.0]
    # for the first word that starts a sentence (preceded by ., ?, ! or silence gap > 0.65s)
    snapped_start = start_time
    start_window_words: list[tuple[int, dict]] = [
        (i, w)
        for i, w in enumerate(words)
        if (start_time - 2.0) <= float(w.get("start", 0.0)) <= (start_time + 2.0)
    ]

    sentence_starts: list[dict] = []
    for i, w in start_window_words:
        if i == 0:
            sentence_starts.append(w)
        else:
            prev_word = words[i - 1]
            prev_text = str(prev_word.get("word", "")).strip()
            pause = float(w.get("start", 0.0)) - float(prev_word.get("end", 0.0))
            if prev_text.endswith((".", "?", "!")) or pause > 0.65:
                sentence_starts.append(w)

    if sentence_starts:
        best_start_word = min(
            sentence_starts,
            key=lambda w: abs(float(w.get("start", 0.0)) - start_time),
        )
        snapped_start = max(0.0, float(best_start_word.get("start", 0.0)) - 0.25)
    elif start_window_words:
        closest_word = min(
            start_window_words,
            key=lambda item: abs(float(item[1].get("start", 0.0)) - start_time),
        )[1]
        snapped_start = max(0.0, float(closest_word.get("start", 0.0)) - 0.25)

    # 2. End Snap: Search words in [end_time - 2.5, end_time + 2.5]
    # for the last word ending in ., ?, ! or followed by silence gap > 0.65s
    snapped_end = end_time
    end_window_words: list[tuple[int, dict]] = [
        (i, w)
        for i, w in enumerate(words)
        if (end_time - 2.5) <= float(w.get("end", 0.0)) <= (end_time + 2.5)
    ]

    sentence_ends: list[dict] = []
    for i, w in end_window_words:
        word_text = str(w.get("word", "")).strip()
        is_last_word = (i == len(words) - 1)
        next_pause = 0.0
        if not is_last_word:
            next_word = words[i + 1]
            next_pause = float(next_word.get("start", 0.0)) - float(w.get("end", 0.0))

        if word_text.endswith((".", "?", "!")) or next_pause > 0.65 or is_last_word:
            sentence_ends.append(w)

    if sentence_ends:
        best_end_word = sentence_ends[-1]
        snapped_end = min(max_video_time, float(best_end_word.get("end", 0.0)) + 0.35)
    elif end_window_words:
        best_end_word = end_window_words[-1][1]
        snapped_end = min(max_video_time, float(best_end_word.get("end", 0.0)) + 0.35)

    if snapped_end <= snapped_start:
        snapped_end = min(max_video_time, snapped_start + 15.0)

    return round(snapped_start, 2), round(snapped_end, 2)


def _dedupe_non_overlapping(clips: list[dict], max_clips: int) -> list[dict]:
    """Greedily select highest-scoring non-overlapping clips."""
    sorted_clips = sorted(
        clips, key=lambda x: x.get("virality_score", 0), reverse=True
    )
    selected: list[dict] = []
    for c in sorted_clips:
        s_time = c["start_time"]
        e_time = c["end_time"]
        overlap = any(
            (s_time < s["end_time"] and e_time > s["start_time"])
            for s in selected
        )
        if not overlap:
            selected.append(c)
            if len(selected) >= max_clips:
                break
    return selected


def _call_llm(system_instruction: str, user_content: str) -> str:
    """Invoke either Gemini or OpenAI-compatible provider based on configuration."""
    model_name = settings.LLM_MODEL.lower()

    if "gemini" in model_name:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=settings.LLM_API_KEY)
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=ClipAnalysisResponse,
            temperature=0.3,
        )
        response = client.models.generate_content(
            model=settings.LLM_MODEL,
            contents=user_content,
            config=config,
        )
        return response.text or "{}"
    else:
        from openai import OpenAI

        client = OpenAI(api_key=settings.LLM_API_KEY)
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        return response.choices[0].message.content or "{}"


def _analyze_transcript_segment(words_slice: list[dict], max_clips: int = 3) -> list[dict]:
    """Analyze a single transcript window using the LLM."""
    if not words_slice:
        return []

    min_start = min(float(w["start"]) for w in words_slice)
    max_end = max(float(w["end"]) for w in words_slice)
    formatted_transcript = _format_transcript_chunks(words_slice)

    system_instruction = (
        "You are an expert short-form viral video editor specializing in YouTube Shorts, TikTok, and Instagram Reels.\n"
        "Your job is to analyze the provided timestamped transcript and select the highest-potential viral clips.\n"
        "Rules:\n"
        f"1. Select up to {max_clips} distinct, non-overlapping segments.\n"
        "2. Each clip MUST be between 25.0 and 58.0 seconds long (unless the available segment is shorter than 30.0 seconds, in which case select the best available range).\n"
        "3. Start right on a strong hook, bold claim, question, or high-energy moment—never mid-sentence.\n"
        "4. End on a complete punchline, resolution, or takeaway—never cut off mid-thought.\n"
        "5. Return a JSON object with a single key 'clips', where each item has:\n"
        "   - 'start_time' (float, matching transcript timestamps)\n"
        "   - 'end_time' (float, matching transcript timestamps)\n"
        "   - 'title' (str, punchy 4-8 word hook title)\n"
        "   - 'virality_score' (int, 1-100)"
    )

    user_content = (
        f"Timestamped Transcript Segment:\n{formatted_transcript}\n\n"
        f"Analyze and output the top {max_clips} viral clips in the requested JSON structure."
    )

    raw_json = "{}"
    try:
        raw_json = _call_llm(system_instruction, user_content)
    except Exception as e:
        logger.warning("LLM segment analysis error: %s", e)

    parsed_clips: list[dict] = []
    try:
        data: Any = json.loads(raw_json)
        raw_candidates = (
            data.get("clips", [])
            if isinstance(data, dict)
            else (data if isinstance(data, list) else [])
        )

        for item in raw_candidates:
            if not isinstance(item, dict):
                continue
            try:
                s_time = float(item.get("start_time", 0.0))
                e_time = float(item.get("end_time", 0.0))
                title = str(item.get("title", "Viral Highlight")).strip()
                score = int(item.get("virality_score", 70))
            except (ValueError, TypeError):
                continue

            s_time = max(min_start, s_time)
            e_time = min(max_end, e_time)

            if (e_time - s_time) < 5.0:
                continue

            parsed_clips.append(
                {
                    "start_time": round(s_time, 2),
                    "end_time": round(e_time, 2),
                    "title": title or "Viral Highlight",
                    "virality_score": max(1, min(100, score)),
                }
            )
    except Exception as e:
        logger.warning("Failed parsing LLM segment JSON: %s", e)

    parsed_clips.sort(key=lambda x: x["virality_score"], reverse=True)
    return parsed_clips[:max_clips]


def _rank_and_dedupe_clips(candidates: list[dict], max_clips: int) -> list[dict]:
    """Reduce pass: rank and deduplicate pooled candidate clips from multiple windows."""
    if not candidates:
        return []

    if len(candidates) <= max_clips:
        return _dedupe_non_overlapping(candidates, max_clips)

    candidate_text_lines = []
    for idx, c in enumerate(candidates, start=1):
        candidate_text_lines.append(
            f"Candidate {idx}: [{c['start_time']:.2f}s - {c['end_time']:.2f}s] "
            f"Score: {c.get('virality_score', 70)} | Title: \"{c.get('title', 'Highlight')}\""
        )
    candidate_summary = "\n".join(candidate_text_lines)

    system_instruction = (
        "You are an expert viral video editor. You are given a list of candidate short-form viral clips extracted from different segments of a long video.\n"
        f"Select the top {max_clips} best, highest-virality, distinct, and non-overlapping clips.\n"
        "Rules:\n"
        f"1. Select up to {max_clips} clips.\n"
        "2. The selected clips MUST NOT overlap with each other in time.\n"
        "3. Preserve the exact or slightly refined start_time, end_time, title, and virality_score.\n"
        "4. Return a JSON object with a single key 'clips'."
    )

    user_content = (
        f"Candidate Clips:\n{candidate_summary}\n\n"
        f"Select and output the top {max_clips} non-overlapping viral clips in JSON."
    )

    try:
        raw_json = _call_llm(system_instruction, user_content)
        data = json.loads(raw_json)
        raw_clips = data.get("clips", []) if isinstance(data, dict) else []
        parsed = []
        for item in raw_clips:
            if isinstance(item, dict):
                parsed.append(
                    {
                        "start_time": float(item.get("start_time", 0.0)),
                        "end_time": float(item.get("end_time", 0.0)),
                        "title": str(item.get("title", "Viral Highlight")).strip()
                        or "Viral Highlight",
                        "virality_score": int(item.get("virality_score", 70)),
                    }
                )
        if parsed:
            return _dedupe_non_overlapping(parsed, max_clips)
    except Exception as e:
        logger.warning(
            "Reduce pass LLM call failed: %s. Falling back to greedy selection.", e
        )

    return _dedupe_non_overlapping(candidates, max_clips)


def find_viral_clips(words: list[dict], max_clips: int = 3) -> list[dict]:
    """Analyze transcript with LLM (using Map-Reduce for >15min videos) and snap boundaries."""
    if not words:
        return [
            {
                "start_time": 0.0,
                "end_time": 0.0,
                "title": "Default Clip",
                "virality_score": 50,
                "is_fallback": True,
            }
        ]

    min_start = min(float(w["start"]) for w in words)
    total_duration = max(float(w["end"]) for w in words)

    if total_duration > 900.0:
        # Long video (> 15 min): Map-Reduce windowing with 10-minute (600s) windows and 30s overlap
        window_size = 600.0
        overlap = 30.0
        step = window_size - overlap  # 570.0s

        pooled_candidates: list[dict] = []
        win_start = 0.0

        while win_start < total_duration:
            win_end = min(total_duration, win_start + window_size)
            window_words = [
                w
                for w in words
                if float(w.get("end", 0.0)) > win_start
                and float(w.get("start", 0.0)) < win_end
            ]

            if len(window_words) >= 5:
                window_clips = _analyze_transcript_segment(
                    window_words, max_clips=max_clips
                )
                pooled_candidates.extend(window_clips)

            win_start += step
            if win_end >= total_duration:
                break

        selected_candidates = _rank_and_dedupe_clips(
            pooled_candidates, max_clips=max_clips
        )
    else:
        # Standard video (<= 15 min): Analyze full compressed transcript in one pass
        selected_candidates = _analyze_transcript_segment(
            words, max_clips=max_clips
        )

    # Apply sentence & silence boundary snapping to each selected clip
    valid_clips: list[dict] = []
    for clip in selected_candidates:
        snapped_start, snapped_end = _snap_boundaries(
            clip["start_time"], clip["end_time"], words
        )
        if (snapped_end - snapped_start) < 5.0:
            continue

        valid_clips.append(
            {
                "start_time": snapped_start,
                "end_time": snapped_end,
                "title": clip["title"],
                "virality_score": clip["virality_score"],
                "is_fallback": False,
            }
        )

    # Safe fallback if zero valid clips returned
    if not valid_clips:
        fallback_duration = min(45.0, total_duration - min_start)
        fallback_end = min_start + fallback_duration
        if fallback_end <= min_start:
            fallback_end = total_duration

        snapped_start, snapped_end = _snap_boundaries(
            min_start, fallback_end, words
        )
        valid_clips = [
            {
                "start_time": snapped_start,
                "end_time": snapped_end,
                "title": "Engaging Highlight",
                "virality_score": 85,
                "is_fallback": True,
            }
        ]

    return valid_clips
