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


def _wrap_hook_title(title: str, max_chars: int = 28) -> str:
    """Wrap hook title with ASS newline tags if longer than max_chars."""
    words = title.strip().upper().split()
    if not words:
        return ""
    lines: list[str] = []
    current_line: list[str] = []
    current_len = 0
    for w in words:
        if current_line and (current_len + 1 + len(w)) > max_chars:
            lines.append(" ".join(current_line))
            current_line = [w]
            current_len = len(w)
        else:
            current_line.append(w)
            current_len += len(w) + (1 if len(current_line) > 1 else 0)
    if current_line:
        lines.append(" ".join(current_line))
    return r"\N".join(lines)


def generate_ass_subtitles(
    words: list[dict],
    start_time: float,
    end_time: float,
    output_ass_path: str,
    preset: str = "clean_white",
    safe_zone: str = "shorts",
    highlight_color: str = "&H00FFFFFF",
    hook_title: str | None = None,
    show_hook_banner: bool = True,
) -> str:
    """Generate dynamic ASS subtitle file with presets and safe zones."""
    out_dir = os.path.dirname(output_ass_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Safe zone margins: shorts -> 380, reels -> 420, tiktok -> 460
    safe_zone_margins = {
        "shorts": 380,
        "reels": 420,
        "tiktok": 460,
    }
    margin_v = safe_zone_margins.get(safe_zone.lower(), 380)

    # Preset configurations
    font_name = "Liberation Sans"
    primary_color = "&H00FFFFFF"
    outline_color = "&H00000000"
    back_color = "&H80000000"

    if preset == "boxed":
        font_size = 74
        border_style = 3  # Opaque background box
        outline = 6
        shadow = 0
        back_color = "&H99000000"
    elif preset == "minimal":
        font_size = 62
        border_style = 1
        outline = 3
        shadow = 1
    elif preset == "bold_yellow":
        font_size = 78
        border_style = 1
        outline = 6
        shadow = 2
    else:  # clean_white (default)
        font_size = 78
        border_style = 1
        outline = 5
        shadow = 2

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
        f"Style: Default,{font_name},{font_size},{primary_color},&H0000FFFF,{outline_color},{back_color},"
        f"-1,0,0,0,100,100,0,0,{border_style},{outline},{shadow},2,60,60,{margin_v},1\n"
        f"Style: HookBanner,{font_name},54,&H00FFFFFF,&H0000FFFF,&H00000000,&HCC000000,"
        "-1,0,0,0,100,100,0,0,3,14,0,8,80,80,190,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    # Determine highlight color tag
    active_color = highlight_color
    if preset == "bold_yellow" and (not active_color or active_color == "&H00FFFFFF"):
        active_color = "&H0000FFFF"

    color_tag = (
        active_color
        if active_color.endswith("&")
        else f"{active_color}&"
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
                        if preset == "bold_yellow":
                            words_formatted.append(
                                f"{{\\c{color_tag}\\fscx114\\fscy114}}{w_item['word']}{{\\r}}"
                            )
                        elif preset == "minimal":
                            # Minimal: no scale animation
                            words_formatted.append(w_item["word"])
                        elif preset == "boxed":
                            words_formatted.append(
                                f"{{\\fscx108\\fscy108}}{w_item['word']}{{\\r}}"
                            )
                        else:  # clean_white
                            # Clean white: subtle scale pop without neon color shift
                            words_formatted.append(
                                f"{{\\fscx108\\fscy108}}{w_item['word']}{{\\r}}"
                            )
                    else:
                        words_formatted.append(w_item["word"])

                text_payload = " ".join(words_formatted)
                dialogue_lines.append(
                    f"Dialogue: 0,{_format_ass_time(event_start)},{_format_ass_time(event_end)},"
                    f"Default,,0,0,0,,{text_payload}"
                )

    if show_hook_banner and hook_title and hook_title.strip():
        wrapped_hook = _wrap_hook_title(hook_title.strip(), max_chars=28)
        dialogue_lines.insert(
            0,
            f"Dialogue: 1,0:00:00.00,0:00:03.20,HookBanner,,0,0,0,,{wrapped_hook}",
        )

    content = header + "\n".join(dialogue_lines) + "\n"
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_ass_path


def render_clip_single_pass(
    video_path: str,
    start_time: float,
    end_time: float,
    crop_filter: str,
    ass_path: str,
    output_path: str,
    fonts_dir: str = "/usr/share/fonts",
) -> str:
    """Render 9:16 clip with crop, subtitles, and EBU R128 loudnorm in a single FFmpeg pass."""
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    duration = max(0.1, end_time - start_time)

    # Input seeking before -i to decode only the target slice
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_time:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        video_path,
    ]

    has_ass = bool(ass_path and os.path.exists(ass_path))
    if has_ass:
        escaped_ass = (
            ass_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        )
        ass_filter = f"ass='{escaped_ass}':fontsdir='{fonts_dir}'"
    else:
        ass_filter = ""

    # Chain crop filter and subtitle burning in a single pass
    is_complex = (
        ";[0:v]" in crop_filter
        or "[top]" in crop_filter
        or "[bottom]" in crop_filter
        or "[v]" in crop_filter
    )

    if is_complex:
        if has_ass:
            base = crop_filter.rstrip(";")
            if base.endswith("[v]"):
                base = base[:-3] + "[v_crop]"
            filter_complex = f"{base};[v_crop]{ass_filter}[v]"
        else:
            filter_complex = crop_filter
        cmd.extend(["-filter_complex", filter_complex, "-map", "[v]", "-map", "0:a?"])
    else:
        if has_ass:
            vf = f"{crop_filter},{ass_filter}" if crop_filter else ass_filter
        else:
            vf = crop_filter if crop_filter else "null"
        filter_complex = f"[0:v]{vf}[v]"
        cmd.extend(["-filter_complex", filter_complex, "-map", "[v]", "-map", "0:a?"])

    # EBU R128 loudness normalization and single-pass encoding
    cmd.extend([
        "-af",
        "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        output_path,
    ])

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
    )
    return output_path


def burn_subtitles(
    input_video_path: str,
    ass_path: str,
    output_video_path: str,
) -> str:
    """Legacy helper: burn subtitles into an existing video."""
    return render_clip_single_pass(
        video_path=input_video_path,
        start_time=0.0,
        end_time=999999.0,
        crop_filter="",
        ass_path=ass_path,
        output_path=output_video_path,
    )
