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


def find_viral_clips(words: list[dict], max_clips: int = 3) -> list[dict]:
    """Analyze transcript with LLM to find the most viral, hook-driven video clips."""
    if not words:
        return [
            {
                "start_time": 0.0,
                "end_time": 0.0,
                "title": "Default Clip",
                "virality_score": 50,
            }
        ]

    min_start = min(float(w["start"]) for w in words)
    total_duration = max(float(w["end"]) for w in words)

    formatted_transcript = _format_transcript_chunks(words)

    system_instruction = (
        "You are an expert short-form viral video editor specializing in YouTube Shorts, TikTok, and Instagram Reels.\n"
        "Your job is to analyze the provided timestamped transcript and select the highest-potential viral clips.\n"
        "Rules:\n"
        f"1. Select up to {max_clips} distinct, non-overlapping segments.\n"
        "2. Each clip MUST be between 25.0 and 58.0 seconds long (unless the entire video is shorter than 30.0 seconds, in which case select the best available range).\n"
        "3. Start right on a strong hook, bold claim, question, or high-energy moment—never mid-sentence.\n"
        "4. End on a complete punchline, resolution, or takeaway—never cut off mid-thought.\n"
        "5. Return a JSON object with a single key 'clips', where each item has:\n"
        "   - 'start_time' (float, matching transcript timestamps)\n"
        "   - 'end_time' (float, matching transcript timestamps)\n"
        "   - 'title' (str, punchy 4-8 word hook title)\n"
        "   - 'virality_score' (int, 1-100)"
    )

    user_content = (
        f"Timestamped Transcript:\n{formatted_transcript}\n\n"
        f"Analyze and output the top {max_clips} viral clips in the requested JSON structure."
    )

    raw_json = "{}"
    try:
        raw_json = _call_llm(system_instruction, user_content)
    except Exception as e:
        logger.warning("LLM analysis encountered error: %s. Proceeding to fallback.", e)

    parsed_clips: list[dict] = []
    try:
        data: Any = json.loads(raw_json)
        raw_candidates = data.get("clips", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])

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

            s_time = max(0.0, s_time)
            e_time = min(total_duration, e_time)

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
        logger.warning("Failed parsing LLM response JSON: %s", e)

    parsed_clips.sort(key=lambda x: x["virality_score"], reverse=True)
    valid_clips = parsed_clips[:max_clips]

    # Safe fallback if zero valid clips returned
    if not valid_clips:
        fallback_duration = min(45.0, total_duration - min_start)
        fallback_end = min_start + fallback_duration
        if fallback_end <= min_start:
            fallback_end = total_duration

        valid_clips = [
            {
                "start_time": round(min_start, 2),
                "end_time": round(fallback_end, 2),
                "title": "Engaging Highlight",
                "virality_score": 85,
            }
        ]

    return valid_clips
