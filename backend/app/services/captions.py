import json
import os
import re
import signal
import subprocess
from typing import Any

from app.schemas import PipelineError, PipelineErrorCode

FONTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fonts"))

SUPPORTED_SCRIPTS = {
    "latin": {
        "font_name": "Noto Sans",
        "font_file": "NotoSans-Regular.ttf",
    },
    "devanagari": {
        "font_name": "Noto Sans Devanagari",
        "font_file": "NotoSansDevanagari-Regular.ttf",
    },
    "malayalam": {
        "font_name": "Noto Sans Malayalam",
        "font_file": "NotoSansMalayalam-Regular.ttf",
    },
    "arabic": {
        "font_name": "Noto Sans Arabic",
        "font_file": "NotoSansArabic-Regular.ttf",
    },
    "cjk": {
        "font_name": "Noto Sans CJK SC",
        "font_file": "NotoSansCJK-Regular.ttf",
    },
}

PLACEHOLDER_TITLES = {
    "viral highlight",
    "untitled",
    "untitled clip",
    "clip",
    "highlight",
    "video title",
    "placeholder",
    "hook title",
    "none",
    "null",
    "n/a",
}


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


def sanitize_ass_text(text: str) -> str:
    """Escape and strip ASS-significant characters (braces, backslashes, newlines)."""
    if not text:
        return ""
    # Strip ASS style override blocks e.g. {\b1}, {\c&H...&}, {\pos...}
    def _strip_or_unwrap_tag(m: re.Match) -> str:
        inner = m.group(1)
        if inner.startswith("\\"):
            return ""
        return inner

    cleaned = re.sub(r"\{([^{}]*)\}", _strip_or_unwrap_tag, text)
    cleaned = re.sub(r"\{([^{}]*)\}", _strip_or_unwrap_tag, cleaned)
    cleaned = cleaned.replace("{", "").replace("}", "")
    # Convert ASS newline and spacing escapes \N, \n, \h, \r to space
    cleaned = re.sub(r"\\([nNhHrR])", " ", cleaned)
    # Strip any remaining backslashes
    cleaned = cleaned.replace("\\", "")
    # Strip control characters and raw newlines
    cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", cleaned)
    # Normalize multiple whitespace
    return " ".join(cleaned.split()).strip()


def detect_script(text: str) -> str:
    """Detect dominant script in text: latin, devanagari, malayalam, arabic, cjk, or unsupported."""
    if not text:
        return "latin"

    counts = {
        "latin": 0,
        "devanagari": 0,
        "malayalam": 0,
        "arabic": 0,
        "cjk": 0,
        "unsupported": 0,
    }

    for ch in text:
        if ch.isspace() or ch.isdigit() or ch in ".,!?:;\"'()[]{}<>-–—_@#$%^&*+=/\\|~`":
            continue
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            counts["devanagari"] += 1
        elif 0x0D00 <= cp <= 0x0D7F:
            counts["malayalam"] += 1
        elif (0x0600 <= cp <= 0x06FF) or (0x0750 <= cp <= 0x077F) or (0x08A0 <= cp <= 0x08FF):
            counts["arabic"] += 1
        elif (
            (0x4E00 <= cp <= 0x9FFF)
            or (0x3040 <= cp <= 0x309F)
            or (0x30A0 <= cp <= 0x30FF)
            or (0xAC00 <= cp <= 0xD7AF)
        ):
            counts["cjk"] += 1
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F):
            counts["latin"] += 1
        else:
            counts["unsupported"] += 1

    best_script = "latin"
    max_count = 0
    for script, count in counts.items():
        if count > max_count:
            max_count = count
            best_script = script

    return best_script


def check_missing_glyphs(text: str, font_path: str) -> list[str]:
    """Return list of characters in text that have no glyph in the given TTF/OTF font."""
    if not os.path.exists(font_path):
        return [c for c in text if not c.isspace()]
    try:
        from fontTools.ttLib import TTFont
        tt = TTFont(font_path)
        cmap = tt.getBestCmap() or {}
        missing = []
        for ch in text:
            if not ch.isspace() and ord(ch) not in cmap:
                if ch not in missing:
                    missing.append(ch)
        return missing
    except Exception:
        return []


def verify_font_rendering(text: str, font_path: str, font_size: int = 36) -> tuple[bool, list[str]]:
    """Rasterize a sample string with PIL and verify glyphs render without missing glyphs."""
    missing = check_missing_glyphs(text, font_path)
    if missing:
        return False, missing

    if not os.path.exists(font_path):
        return False, ["<font_not_found>"]

    try:
        from PIL import Image, ImageFont, ImageDraw
        font = ImageFont.truetype(font_path, font_size)
        img = Image.new("L", (1080, 200), color=0)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), text, font=font, fill=255)
        bbox = img.getbbox()
        if bbox is None and text.strip():
            return False, ["<empty_raster>"]
        return True, []
    except Exception as e:
        return False, [str(e)]


def is_valid_hook_title(title: str | None, expected_language: str | None = None) -> bool:
    """Validate hook title: check length, language, and ensure no placeholder text."""
    if not title or not isinstance(title, str):
        return False

    cleaned = sanitize_ass_text(title)
    if len(cleaned) < 4 or len(cleaned) > 120:
        return False

    lower = cleaned.lower()
    if lower in PLACEHOLDER_TITLES:
        return False

    # Check regex for patterns like "clip 1", "clip 02", "highlight at 14s", "highlight 3"
    if re.match(r"^(clip\s*\d+|highlight\s*(at\s*\d+\.?\d*s?)?|untitled(\s*clip)?)$", lower):
        return False

    # Ensure title contains readable letters (not just digits/punctuation)
    if not any(c.isalpha() for c in cleaned):
        return False

    if expected_language:
        script = detect_script(cleaned)
        if expected_language.lower().startswith("en") and script != "latin":
            return False

    return True


def _wrap_hook_title(title: str, max_chars: int = 28) -> str:
    """Wrap hook title with ASS newline tags, safely handling long words and no spaces."""
    cleaned = sanitize_ass_text(title).upper()
    if not cleaned:
        return ""

    words = cleaned.split()
    if not words:
        return ""

    # Split words longer than max_chars into max_chars chunks
    tokens: list[str] = []
    for w in words:
        if len(w) <= max_chars:
            tokens.append(w)
        else:
            for i in range(0, len(w), max_chars):
                tokens.append(w[i : i + max_chars])

    lines: list[str] = []
    current_line: list[str] = []
    current_len = 0

    for token in tokens:
        needed = len(token) + (1 if current_line else 0)
        if current_line and (current_len + needed > max_chars):
            lines.append(" ".join(current_line))
            current_line = [token]
            current_len = len(token)
        else:
            current_line.append(token)
            current_len += needed

    if current_line:
        lines.append(" ".join(current_line))

    return r"\N".join(lines)


def filter_and_trim_words(
    words: list[dict],
    start_time: float,
    end_time: float,
) -> list[dict]:
    """Filter words for clip window, trimming or dropping boundary-straddling words."""
    clip_duration = max(0.1, end_time - start_time)
    filtered: list[dict] = []

    for w in words:
        w_start = float(w.get("start", 0.0))
        w_end = float(w.get("end", 0.0))
        raw_text = str(w.get("word", ""))
        word_text = sanitize_ass_text(raw_text).upper()

        if not word_text:
            continue

        # Completely outside clip window
        if w_end <= start_time or w_start >= end_time:
            continue

        w_dur = max(0.01, w_end - w_start)

        # Straddles start boundary
        if w_start < start_time < w_end:
            inside_dur = w_end - start_time
            if inside_dur < 0.1 or (inside_dur / w_dur < 0.4):
                continue
            rel_start = 0.0
            rel_end = min(clip_duration, max(0.05, w_end - start_time))
        # Straddles end boundary
        elif w_start < end_time < w_end:
            inside_dur = end_time - w_start
            if inside_dur < 0.1 or (inside_dur / w_dur < 0.4):
                continue
            rel_start = max(0.0, w_start - start_time)
            rel_end = clip_duration
        # Completely inside clip
        else:
            rel_start = max(0.0, w_start - start_time)
            rel_end = min(clip_duration, max(rel_start + 0.05, w_end - start_time))

        if rel_start >= clip_duration:
            continue
        if rel_end <= rel_start:
            rel_end = min(clip_duration, rel_start + 0.05)

        filtered.append({
            "word": word_text,
            "rel_start": rel_start,
            "rel_end": rel_end,
        })

    return filtered


def derive_max_chars_per_line(font_size: int, play_res_x: int = 1080, margin_h: int = 120) -> int:
    """Derive max characters per line from font size and PlayResX (~20-24 for default 78)."""
    usable_w = max(100, play_res_x - margin_h)
    return max(12, int(usable_w / (font_size * 0.55)))


def _estimate_lines(words: list[str], max_chars: int) -> list[str]:
    """Greedy pack words into lines."""
    lines: list[str] = []
    curr: list[str] = []
    curr_len = 0
    for w in words:
        needed = len(w) + (1 if curr else 0)
        if curr and (curr_len + needed > max_chars):
            lines.append(" ".join(curr))
            curr = [w]
            curr_len = len(w)
        else:
            curr.append(w)
            curr_len += needed
    if curr:
        lines.append(" ".join(curr))
    return lines


def group_words_into_captions(
    words: list[dict],
    max_chars_per_line: int,
    pause_threshold: float = 0.4,
    min_display_time: float = 0.8,
    clip_duration: float = 999999.0,
) -> list[list[dict]]:
    """Group words into caption blocks: max chars per line, max 2 lines, break at punctuation/pauses."""
    if not words:
        return []

    groups: list[list[dict]] = []
    current_group: list[dict] = []

    for item in words:
        if not current_group:
            current_group.append(item)
            if item["word"].endswith((".", "?", "!")):
                groups.append(current_group)
                current_group = []
            continue

        prev_item = current_group[-1]
        pause = item["rel_start"] - prev_item["rel_end"]

        # Check if pause threshold exceeded
        if pause >= pause_threshold:
            groups.append(current_group)
            current_group = [item]
            if item["word"].endswith((".", "?", "!")):
                groups.append(current_group)
                current_group = []
            continue

        # Check line count if we add this word
        test_words = [w["word"] for w in current_group] + [item["word"]]
        lines = _estimate_lines(test_words, max_chars_per_line)
        if len(lines) > 2:
            groups.append(current_group)
            current_group = [item]
        else:
            current_group.append(item)

        if item["word"].endswith((".", "?", "!")):
            groups.append(current_group)
            current_group = []

    if current_group:
        groups.append(current_group)

    return groups


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
    layout: str = "speaker",
    language: str | None = None,
    warnings: list | None = None,
    font_name: str | None = None,
) -> str:
    """Generate dynamic ASS subtitle file with presets, fonts, and safe zones."""
    out_dir = os.path.dirname(output_ass_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    clip_duration = max(0.1, end_time - start_time)

    # Safe zone bottom margins: shorts -> 380, reels -> 420, tiktok -> 460
    safe_zone_margins = {
        "shorts": 380,
        "reels": 420,
        "tiktok": 460,
    }
    # Hook banner top margins: shorts -> 190, reels -> 240, tiktok -> 280
    hook_banner_top_margins = {
        "shorts": 190,
        "reels": 240,
        "tiktok": 280,
    }

    # In split layout, anchor captions at the seam between panes (Y=960 out of 1920)
    if layout.lower() == "split":
        margin_v = 960
    else:
        margin_v = safe_zone_margins.get(safe_zone.lower(), 380)

    hook_margin_v = hook_banner_top_margins.get(safe_zone.lower(), 190)

    # Filter and trim words relative to clip window
    filtered_words = filter_and_trim_words(words, start_time, end_time)

    # Script detection and font selection
    sample_text = " ".join([w["word"] for w in filtered_words] + ([hook_title] if hook_title else []))
    detected_script = detect_script(sample_text)

    font_file = None
    if font_name:
        active_font_name = font_name
    elif detected_script in SUPPORTED_SCRIPTS:
        active_font_name = SUPPORTED_SCRIPTS[detected_script]["font_name"]
        font_file = os.path.join(FONTS_DIR, SUPPORTED_SCRIPTS[detected_script]["font_file"])
    else:
        # Unsupported script: record warning instead of rendering tofu
        active_font_name = "Noto Sans"
        font_file = os.path.join(FONTS_DIR, "NotoSans-Regular.ttf")
        if warnings is not None:
            warn_msg = f"Script '{detected_script}' is not supported by bundled fonts; captions may render as tofu or fallback."
            try:
                from app.schemas import JobWarning
                warnings.append(JobWarning(code="UNSUPPORTED_FONT_SCRIPT", message=warn_msg))
            except Exception:
                warnings.append({"code": "UNSUPPORTED_FONT_SCRIPT", "message": warn_msg})

    if font_file and os.path.exists(font_file):
        missing_glyphs = check_missing_glyphs(sample_text, font_file)
        if missing_glyphs and warnings is not None:
            warn_msg = f"Missing glyphs for characters '{''.join(missing_glyphs[:5])}' in font {active_font_name}."
            try:
                from app.schemas import JobWarning
                warnings.append(JobWarning(code="UNSUPPORTED_FONT_SCRIPT", message=warn_msg))
            except Exception:
                warnings.append({"code": "UNSUPPORTED_FONT_SCRIPT", "message": warn_msg})

    # Preset configurations
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
        f"Style: Default,{active_font_name},{font_size},{primary_color},&H0000FFFF,{outline_color},{back_color},"
        f"-1,0,0,0,100,100,0,0,{border_style},{outline},{shadow},2,60,60,{margin_v},1\n"
        f"Style: HookBanner,{active_font_name},54,&H00FFFFFF,&H0000FFFF,&H00000000,&HCC000000,"
        f"-1,0,0,0,100,100,0,0,3,14,0,8,80,80,{hook_margin_v},1\n\n"
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

    dialogue_lines: list[str] = []

    if filtered_words:
        max_chars_line = derive_max_chars_per_line(font_size)
        groups = group_words_into_captions(
            filtered_words,
            max_chars_line,
            pause_threshold=0.4,
            min_display_time=0.8,
            clip_duration=clip_duration,
        )

        for g_idx, chunk in enumerate(groups):
            chunk_len = len(chunk)
            group_start = chunk[0]["rel_start"]
            raw_end = chunk[-1]["rel_end"]

            # Minimum display time per group
            group_end = max(raw_end, group_start + 0.8)
            if g_idx + 1 < len(groups):
                group_end = min(group_end, groups[g_idx + 1][0]["rel_start"])
            group_end = min(group_end, clip_duration)

            # Determine static line wrapping for the chunk (max 2 lines)
            chunk_words = [w["word"] for w in chunk]
            chunk_lines = _estimate_lines(chunk_words, max_chars_line)
            # Find the split index between line 1 and line 2
            line1_word_count = len(chunk_lines[0].split()) if chunk_lines else chunk_len

            for idx in range(chunk_len):
                w_curr = chunk[idx]
                event_start = max(0.0, min(w_curr["rel_start"], clip_duration))

                # Highlight per word with no flicker: each word event ends where the next begins
                if idx + 1 < chunk_len:
                    event_end = max(event_start + 0.05, min(chunk[idx + 1]["rel_start"], clip_duration))
                else:
                    event_end = max(event_start + 0.05, min(group_end, clip_duration))

                # Format words for the dialogue payload
                line1_formatted: list[str] = []
                line2_formatted: list[str] = []

                for i, w_item in enumerate(chunk):
                    w_text = w_item["word"]
                    if i == idx:
                        if preset == "bold_yellow":
                            fmt_word = f"{{\\c{color_tag}\\fscx114\\fscy114}}{w_text}{{\\r}}"
                        elif preset == "minimal":
                            fmt_word = w_text
                        elif preset == "boxed":
                            fmt_word = f"{{\\fscx108\\fscy108}}{w_text}{{\\r}}"
                        else:  # clean_white
                            fmt_word = f"{{\\fscx108\\fscy108}}{w_text}{{\\r}}"
                    else:
                        fmt_word = w_text

                    if i < line1_word_count:
                        line1_formatted.append(fmt_word)
                    else:
                        line2_formatted.append(fmt_word)

                if line2_formatted:
                    text_payload = f"{' '.join(line1_formatted)}\\N{' '.join(line2_formatted)}"
                else:
                    text_payload = " ".join(line1_formatted)

                dialogue_lines.append(
                    f"Dialogue: 0,{_format_ass_time(event_start)},{_format_ass_time(event_end)},"
                    f"Default,,0,0,0,,{text_payload}"
                )

    if show_hook_banner and hook_title and is_valid_hook_title(hook_title, language):
        wrapped_hook = _wrap_hook_title(hook_title, max_chars=28)
        banner_end = min(3.20, clip_duration)
        dialogue_lines.insert(
            0,
            f"Dialogue: 1,0:00:00.00,{_format_ass_time(banner_end)},HookBanner,,0,0,0,,{wrapped_hook}",
        )

    content = header + "\n".join(dialogue_lines) + "\n"
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_ass_path


def check_faststart(mp4_path: str) -> bool:
    """Verify that 'moov' atom appears before 'mdat' atom in an MP4 file."""
    if not os.path.exists(mp4_path) or os.path.getsize(mp4_path) < 16:
        return False
    moov_pos = None
    mdat_pos = None
    with open(mp4_path, "rb") as f:
        pos = 0
        file_size = os.path.getsize(mp4_path)
        while pos < file_size:
            header = f.read(8)
            if len(header) < 8:
                break
            size = int.from_bytes(header[:4], "big")
            atom = header[4:8]
            if atom == b"moov" and moov_pos is None:
                moov_pos = pos
            elif atom == b"mdat" and mdat_pos is None:
                mdat_pos = pos

            if size == 1:
                ext = f.read(8)
                if len(ext) < 8:
                    break
                size = int.from_bytes(ext, "big")
                pos += size
                f.seek(pos)
            elif size == 0:
                # Atom extends to end of file
                break
            else:
                pos += size
                f.seek(pos)

            if moov_pos is not None and mdat_pos is not None:
                break

    return moov_pos is not None and mdat_pos is not None and moov_pos < mdat_pos


def probe_video_stream_info(video_path: str) -> dict:
    """Probe source video for resolution, duration, and stream presence."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,codec_name,width,height",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            video_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=15)
        data = json.loads(res.stdout)
        streams = data.get("streams", [])
        has_video = any(s.get("codec_type") == "video" for s in streams)
        has_audio = any(s.get("codec_type") == "audio" for s in streams)
        v_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
        width = int(v_stream.get("width") or 1920)
        height = int(v_stream.get("height") or 1080)
        raw_dur = data.get("format", {}).get("duration")
        duration = float(raw_dur) if raw_dur is not None else 0.0
        return {
            "has_video": has_video,
            "has_audio": has_audio,
            "width": width,
            "height": height,
            "duration": duration,
        }
    except Exception:
        return {
            "has_video": True,
            "has_audio": True,
            "width": 1920,
            "height": 1080,
            "duration": 0.0,
        }


def measure_audio_loudness(
    video_path: str,
    start_time: float,
    duration: float,
    timeout: float = 30.0,
) -> dict | None:
    """Run pass 1 of loudnorm to measure EBU R128 parameters."""
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_time:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        video_path,
        "-vn",
        "-af",
        "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        _, stderr = proc.communicate(timeout=timeout)
        if proc.returncode != 0:
            return None
        stderr_text = stderr.decode("utf-8", errors="replace")
        json_match = re.search(r"\{[\s\S]*?\"input_i\"[\s\S]*?\}", stderr_text)
        if json_match:
            data = json.loads(json_match.group(0))
            if "input_i" in data and "input_tp" in data:
                return data
    except Exception:
        pass
    return None


def validate_rendered_clip(
    output_path: str,
    expected_duration: float,
    duration_tolerance: float = 0.5,
) -> None:
    """Validate rendered MP4 with ffprobe for compliance and decodability."""
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise PipelineError(
            code=PipelineErrorCode.RENDER_FAILED,
            stage="rendering",
            user_message="Rendered clip is missing or empty.",
            retryable=True,
            detail=f"File {output_path} does not exist or has size 0.",
        )

    # 1. Probe format and streams
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size",
        "-show_entries",
        "stream=index,codec_type,codec_name,pix_fmt,width,height,sample_aspect_ratio",
        "-of",
        "json",
        output_path,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=15)
        probe_data = json.loads(res.stdout)
    except Exception as e:
        raise PipelineError(
            code=PipelineErrorCode.RENDER_FAILED,
            stage="rendering",
            user_message="Failed to inspect rendered clip with ffprobe.",
            retryable=True,
            detail=str(e),
        )

    streams = probe_data.get("streams", [])
    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    # Check video stream
    if not v_stream:
        raise PipelineError(
            code=PipelineErrorCode.RENDER_FAILED,
            stage="rendering",
            user_message="Rendered clip is missing video stream.",
            retryable=True,
            detail="No video stream found in rendered clip.",
        )
    if v_stream.get("codec_name") != "h264":
        raise PipelineError(
            code=PipelineErrorCode.RENDER_FAILED,
            stage="rendering",
            user_message="Rendered video stream is not H.264.",
            retryable=True,
            detail=f"Expected h264 codec, found {v_stream.get('codec_name')}.",
        )
    if v_stream.get("pix_fmt") != "yuv420p":
        raise PipelineError(
            code=PipelineErrorCode.RENDER_FAILED,
            stage="rendering",
            user_message="Rendered video pixel format is not yuv420p.",
            retryable=True,
            detail=f"Expected yuv420p, found {v_stream.get('pix_fmt')}.",
        )
    width = int(v_stream.get("width") or 0)
    height = int(v_stream.get("height") or 0)
    if width != 1080 or height != 1920:
        raise PipelineError(
            code=PipelineErrorCode.RENDER_FAILED,
            stage="rendering",
            user_message="Rendered video dimensions are not 1080x1920.",
            retryable=True,
            detail=f"Expected 1080x1920, got {width}x{height}.",
        )

    sar = v_stream.get("sample_aspect_ratio")
    if sar and sar not in ("1:1", "1/1", "0:1"):
        try:
            num, den = map(float, sar.replace(":", "/").split("/", 1))
            if abs(num / den - 1.0) > 0.02:
                raise ValueError()
        except Exception:
            raise PipelineError(
                code=PipelineErrorCode.RENDER_FAILED,
                stage="rendering",
                user_message="Rendered video does not have square pixels (SAR=1:1).",
                retryable=True,
                detail=f"Expected SAR=1:1, got {sar}.",
            )

    # Check audio stream
    if not a_stream:
        raise PipelineError(
            code=PipelineErrorCode.RENDER_FAILED,
            stage="rendering",
            user_message="Rendered clip is missing audio stream.",
            retryable=True,
            detail="No audio stream found in rendered clip.",
        )
    if a_stream.get("codec_name") != "aac":
        raise PipelineError(
            code=PipelineErrorCode.RENDER_FAILED,
            stage="rendering",
            user_message="Rendered audio stream is not AAC.",
            retryable=True,
            detail=f"Expected aac codec, found {a_stream.get('codec_name')}.",
        )

    # Check duration
    format_info = probe_data.get("format", {})
    actual_dur = float(format_info.get("duration") or 0.0)
    dur_diff = abs(actual_dur - expected_duration)
    if dur_diff > duration_tolerance:
        raise PipelineError(
            code=PipelineErrorCode.RENDER_FAILED,
            stage="rendering",
            user_message="Rendered clip duration deviates from requested duration.",
            retryable=True,
            detail=f"Expected duration {expected_duration:.2f}s, got {actual_dur:.2f}s (diff {dur_diff:.2f}s > {duration_tolerance:.2f}s).",
        )

    # Check faststart (moov before mdat)
    if not check_faststart(output_path):
        raise PipelineError(
            code=PipelineErrorCode.RENDER_FAILED,
            stage="rendering",
            user_message="Rendered clip is not faststart web-optimized (moov not before mdat).",
            retryable=True,
            detail="MP4 atom order check failed: 'moov' atom was not positioned before 'mdat'.",
        )

    # Decodability check of final frames
    check_ss = max(0.0, actual_dur - 0.5)
    decode_cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{check_ss:.3f}",
        "-i",
        output_path,
        "-f",
        "null",
        "-",
    ]
    try:
        dec_res = subprocess.run(decode_cmd, capture_output=True, text=True, check=True, timeout=15)
        if dec_res.stderr and "error" in dec_res.stderr.lower():
            raise PipelineError(
                code=PipelineErrorCode.RENDER_FAILED,
                stage="rendering",
                user_message="Final frames failed decodability validation.",
                retryable=True,
                detail=dec_res.stderr.strip(),
            )
    except subprocess.CalledProcessError as e:
        raise PipelineError(
            code=PipelineErrorCode.RENDER_FAILED,
            stage="rendering",
            user_message="Failed to decode final frames of rendered clip.",
            retryable=True,
            detail=e.stderr.strip() if e.stderr else str(e),
        )


def render_clip_single_pass(
    video_path: str,
    start_time: float,
    end_time: float,
    crop_filter: str,
    ass_path: str,
    output_path: str,
    fonts_dir: str = FONTS_DIR,
    target_fps: int = 30,
    timeout: float | None = None,
    timeout_multiplier: float = 4.0,
    base_timeout: float = 30.0,
    validate_output: bool = True,
    threads: int | None = None,
) -> str:
    """Render 9:16 clip with normalized frame rate, two-pass loudnorm, scaled timeout, and ffprobe validation."""
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    src_info = probe_video_stream_info(video_path)
    if src_info["duration"] > 0 and end_time > src_info["duration"]:
        effective_end = src_info["duration"]
    else:
        effective_end = end_time

    duration = max(0.1, effective_end - start_time)

    # Scale timeout to clip duration and source resolution
    res_factor = max(1.0, (src_info["width"] * src_info["height"]) / (1920.0 * 1080.0))
    calc_timeout = base_timeout + (duration * timeout_multiplier * res_factor)
    actual_timeout = timeout if timeout is not None else calc_timeout

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

    has_audio = src_info["has_audio"]
    if has_audio:
        # Two-pass loudnorm: measure then apply linearly
        meas = measure_audio_loudness(
            video_path,
            start_time,
            duration,
            timeout=min(30.0, actual_timeout / 2.0),
        )
        if meas:
            audio_filter = (
                f"loudnorm=I=-14:TP=-1.5:LRA=11:"
                f"measured_I={meas['input_i']}:"
                f"measured_TP={meas['input_tp']}:"
                f"measured_LRA={meas['input_lra']}:"
                f"measured_thresh={meas['input_thresh']}:"
                f"offset={meas['target_offset']}:"
                f"linear=true"
            )
        else:
            audio_filter = "loudnorm=I=-14:TP=-1.5:LRA=11"
    else:
        # Generate silent audio track matching clip duration
        cmd.extend([
            "-f",
            "lavfi",
            "-t",
            f"{duration:.3f}",
            "-i",
            "anullsrc=r=48000:cl=stereo",
        ])

    has_ass = bool(ass_path and os.path.exists(ass_path))
    if has_ass:
        escaped_ass = (
            ass_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        )
        ass_filter = f"ass='{escaped_ass}':fontsdir='{fonts_dir}'"
    else:
        ass_filter = ""

    crop_filter_str = str(crop_filter) if crop_filter else ""
    is_complex = (
        ";[0:v]" in crop_filter_str
        or "[top]" in crop_filter_str
        or "[bottom]" in crop_filter_str
        or "[v]" in crop_filter_str
    )

    if is_complex:
        base = crop_filter_str.rstrip(";")
        if base.endswith("[v]"):
            base = base[:-3] + "[v_crop]"
        if has_ass:
            filter_complex = f"{base};[v_crop]{ass_filter},fps=fps={target_fps},setsar=1,format=yuv420p[v]"
        else:
            filter_complex = f"{base};[v_crop]fps=fps={target_fps},setsar=1,format=yuv420p[v]"
    else:
        vf_parts = []
        if crop_filter_str:
            vf_parts.append(crop_filter_str)
        if has_ass:
            vf_parts.append(ass_filter)
        vf_parts.append(f"fps=fps={target_fps}")
        vf_parts.append("setsar=1")
        vf_parts.append("format=yuv420p")
        filter_complex = f"[0:v]{','.join(vf_parts)}[v]"

    cmd.extend(["-filter_complex", filter_complex, "-map", "[v]"])

    if has_audio:
        cmd.extend(["-map", "0:a:0?", "-af", audio_filter])
    else:
        cmd.extend(["-map", "1:a"])

    # Output encoding: CFR, H.264, yuv420p, web-optimized faststart, 48kHz stereo AAC
    cmd.extend([
        "-fps_mode",
        "cfr",
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
        "-ar",
        "48000",
        "-ac",
        "2",
    ])
    if threads is not None and threads > 0:
        cmd.extend(["-threads", str(threads)])
    cmd.append(output_path)

    # Run FFmpeg in a new process group to enable clean termination on timeout
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=actual_timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
        proc.communicate()
        raise PipelineError(
            code=PipelineErrorCode.TIMEOUT,
            stage="rendering",
            user_message=f"Rendering clip timed out after {actual_timeout:.1f}s.",
            retryable=True,
            detail=f"FFmpeg process group {proc.pid} killed after exceeding {actual_timeout:.1f}s timeout.",
        )
    except Exception as e:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
        proc.communicate()
        raise

    if proc.returncode != 0:
        stderr_text = stderr.decode("utf-8", errors="replace")
        stderr_tail = "\n".join(stderr_text.splitlines()[-25:])
        raise PipelineError(
            code=PipelineErrorCode.RENDER_FAILED,
            stage="rendering",
            user_message="FFmpeg failed to render clip.",
            retryable=True,
            detail=f"FFmpeg exited with code {proc.returncode}. Stderr tail:\n{stderr_tail}",
        )

    if validate_output:
        validate_rendered_clip(output_path, expected_duration=duration)

    return output_path


def burn_subtitles(
    input_video_path: str,
    ass_path: str,
    output_video_path: str,
) -> str:
    """Legacy helper: burn subtitles into an existing video."""
    src = probe_video_stream_info(input_video_path)
    end = src["duration"] if src["duration"] > 0 else 999999.0
    return render_clip_single_pass(
        video_path=input_video_path,
        start_time=0.0,
        end_time=end,
        crop_filter="",
        ass_path=ass_path,
        output_path=output_video_path,
    )
