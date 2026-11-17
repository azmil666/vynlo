import os
import subprocess


def _format_ass_time(seconds: float) -> str:
    """Format seconds into ASS timestamp string: H:MM:SS.cc (centiseconds)."""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))

    if centis >= 100:
        secs += 1
        centis -= 100
        if secs >= 60:
            minutes += 1
            secs -= 60
            if minutes >= 60:
                hours += 1
                minutes -= 60

    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def generate_ass_subtitles(
    words: list[dict],
    start_time: float,
    end_time: float,
    output_ass_path: str,
    highlight_color: str = "&H0000FFFF",
) -> str:
    """Generate dynamic ASS karaoke subtitle file with highlighted active words."""
    out_dir = os.path.dirname(output_ass_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,78,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,2,2,60,60,380,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    color_tag = (
        highlight_color
        if highlight_color.endswith("&")
        else f"{highlight_color}&"
    )

    filtered_words: list[dict] = []
    for w in words:
        w_start = float(w.get("start", 0.0))
        w_end = float(w.get("end", 0.0))
        word_text = str(w.get("word", "")).strip().upper()

        if not word_text:
            continue
        if w_end > start_time and w_start < end_time:
            rel_start = max(0.0, w_start - start_time)
            rel_end = max(rel_start + 0.05, w_end - start_time)
            filtered_words.append(
                {
                    "word": word_text,
                    "rel_start": rel_start,
                    "rel_end": rel_end,
                }
            )

    dialogue_lines: list[str] = []

    if filtered_words:
        chunks: list[list[dict]] = []
        current_chunk: list[dict] = []

        for item in filtered_words:
            current_chunk.append(item)
            if len(current_chunk) >= 4 or item["word"].endswith(
                (".", "?", "!")
            ):
                chunks.append(current_chunk)
                current_chunk = []

        if current_chunk:
            chunks.append(current_chunk)

        for chunk in chunks:
            chunk_len = len(chunk)
            for idx in range(chunk_len):
                w_curr = chunk[idx]
                event_start = w_curr["rel_start"]

                if idx + 1 < chunk_len:
                    event_end = chunk[idx + 1]["rel_start"]
                else:
                    event_end = w_curr["rel_end"]

                if event_end <= event_start:
                    event_end = event_start + 0.08

                words_formatted = []
                for i, w_item in enumerate(chunk):
                    if i == idx:
                        words_formatted.append(
                            f"{{\\c{color_tag}\\fscx112\\fscy112}}{w_item['word']}{{\\r}}"
                        )
                    else:
                        words_formatted.append(w_item["word"])

                text_payload = " ".join(words_formatted)
                dialogue_lines.append(
                    f"Dialogue: 0,{_format_ass_time(event_start)},{_format_ass_time(event_end)},"
                    f"Default,,0,0,0,,{text_payload}"
                )

    content = header + "\n".join(dialogue_lines) + "\n"
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_ass_path


def burn_subtitles(
    input_video_path: str,
    ass_path: str,
    output_video_path: str,
) -> str:
    """Burn ASS subtitles into the video using FFmpeg."""
    out_dir = os.path.dirname(output_video_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    escaped_ass = (
        ass_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_video_path,
        "-vf",
        f"ass='{escaped_ass}'",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-c:a",
        "copy",
        output_video_path,
    ]

    subprocess.run(
        cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return output_video_path
