import os
from pathlib import Path
import pytest

from app.services.captions import (
    _format_ass_time,
    _wrap_hook_title,
    check_missing_glyphs,
    derive_max_chars_per_line,
    detect_script,
    filter_and_trim_words,
    generate_ass_subtitles,
    group_words_into_captions,
    is_valid_hook_title,
    sanitize_ass_text,
    verify_font_rendering,
    FONTS_DIR,
)


class TestAssFormattingHelpers:
    """Tests for low-level ASS string and time formatters."""

    def test_format_ass_time(self):
        assert _format_ass_time(0.0) == "0:00:00.00"
        assert _format_ass_time(1.25) == "0:00:01.25"
        assert _format_ass_time(65.50) == "0:01:05.50"
        assert _format_ass_time(3665.12) == "1:01:05.12"
        # Negative clamp
        assert _format_ass_time(-5.0) == "0:00:00.00"

    def test_format_ass_time_centisecond_rollover(self):
        # 0.999 rounds centis to 100 -> secs += 1, centis = 0
        assert _format_ass_time(0.999) == "0:00:01.00"

    def test_wrap_hook_title(self):
        assert _wrap_hook_title("") == ""
        assert _wrap_hook_title("SHORT TITLE") == "SHORT TITLE"
        long_title = "THIS IS A VERY LONG HOOK TITLE THAT NEEDS WRAPPING"
        wrapped = _wrap_hook_title(long_title, max_chars=20)
        assert r"\N" in wrapped

    def test_wrap_hook_title_no_spaces_and_long_words(self):
        # Extremely long word with no spaces
        long_no_space = "SUPERLONGWORDWITHOUTANYSPACESWHATEVER"
        wrapped = _wrap_hook_title(long_no_space, max_chars=12)
        lines = wrapped.split(r"\N")
        assert len(lines) > 1
        for line in lines:
            assert len(line) <= 12

        # Mixed normal words with long tokens
        mixed = "WELCOME TO ANUNUSUALLYLONGWORDTHATDOESNOTFIT HERE"
        wrapped_mixed = _wrap_hook_title(mixed, max_chars=15)
        for line in wrapped_mixed.split(r"\N"):
            assert len(line) <= 15


class TestAssEscapingAndSanitization:
    """Tests for escaping/stripping ASS-significant characters."""

    def test_hostile_strings_sanitization(self):
        hostile_inputs = [
            (
                r"Hello {\b1\c&H0000FF&} evil \N tags \n and newlines\r\n{exploit}",
                "Hello evil tags and newlines exploit",
            ),
            (
                r"Test {\pos(100,100)} \clip(0,0,10,10) injection",
                "Test clip(0,0,10,10) injection",
            ),
            (
                "Line 1\r\nLine 2\nLine 3\rLine 4",
                "Line 1 Line 2 Line 3 Line 4",
            ),
            (
                r"Escaped \\ backslashes \\\ and {nested {braces}}",
                "Escaped backslashes and nested braces",
            ),
        ]
        for hostile, expected in hostile_inputs:
            cleaned = sanitize_ass_text(hostile)
            assert "{" not in cleaned
            assert "}" not in cleaned
            assert "\\" not in cleaned
            assert "\n" not in cleaned
            assert "\r" not in cleaned
            assert cleaned == expected

    def test_hostile_transcript_in_subtitles(self, tmp_path: Path):
        hostile_words = [
            {"word": r"{\b1}EVIL{\r}", "start": 0.5, "end": 1.0},
            {"word": r"WORDS\NWITH\TAGS", "start": 1.1, "end": 1.5},
            {"word": "NEWLINE\nINJECTION", "start": 1.6, "end": 2.0},
        ]
        ass_path = str(tmp_path / "hostile.ass")
        generate_ass_subtitles(hostile_words, 0.0, 3.0, ass_path, hook_title=r"HOOK {\b1}\NTITLE")

        content = Path(ass_path).read_text(encoding="utf-8")
        events = content.split("[Events]")[1]
        default_lines = [line for line in events.splitlines() if "Default" in line]
        assert len(default_lines) > 0
        joined = " ".join(default_lines)
        assert "EVIL" in joined
        assert "WORDS WITHTAGS" in joined
        assert "NEWLINE INJECTION" in joined
        assert r"\b1" not in joined


class TestHookBannerValidation:
    """Tests for hook banner validation."""

    def test_placeholder_rejection(self):
        placeholders = [
            "Viral Highlight",
            "untitled",
            "Untitled Clip",
            "clip 1",
            "clip 04",
            "highlight at 14s",
            "highlight at 30.5s",
            "Highlight",
            "Video Title",
            "Placeholder",
            "Hook Title",
            "None",
            "null",
            "N/A",
        ]
        for p in placeholders:
            assert not is_valid_hook_title(p), f"Should have rejected placeholder: {p}"

    def test_length_and_content_validation(self):
        # Too short
        assert not is_valid_hook_title("")
        assert not is_valid_hook_title("a")
        assert not is_valid_hook_title("no")
        # Too long (> 120 chars)
        assert not is_valid_hook_title("A" * 125)
        # Digits/symbols only
        assert not is_valid_hook_title("???!!!...")
        assert not is_valid_hook_title("1234567")

        # Valid titles
        assert is_valid_hook_title("The Secret to Success")
        assert is_valid_hook_title("Why AI is Changing Everything")


class TestGroupingAndTiming:
    """Tests for word grouping, line lengths, pause breaking, and flicker-free timing."""

    def test_derive_max_chars_per_line(self):
        # Font size 78 should derive ~20-24 chars
        chars_78 = derive_max_chars_per_line(78, play_res_x=1080, margin_h=120)
        assert 20 <= chars_78 <= 24

        chars_62 = derive_max_chars_per_line(62, play_res_x=1080, margin_h=120)
        assert chars_62 >= chars_78

    def test_grouping_max_two_lines_and_punctuation(self):
        words = [
            {"word": "FIRST", "rel_start": 0.0, "rel_end": 0.3},
            {"word": "SENTENCE.", "rel_start": 0.35, "rel_end": 0.8},
            {"word": "SECOND", "rel_start": 0.9, "rel_end": 1.2},
            {"word": "SENTENCE", "rel_start": 1.25, "rel_end": 1.6},
            {"word": "IS", "rel_start": 1.65, "rel_end": 1.8},
            {"word": "LONGER", "rel_start": 1.85, "rel_end": 2.2},
            {"word": "AND", "rel_start": 2.25, "rel_end": 2.4},
            {"word": "HAS", "rel_start": 2.45, "rel_end": 2.6},
            {"word": "MORE", "rel_start": 2.65, "rel_end": 2.8},
            {"word": "WORDS.", "rel_start": 2.85, "rel_end": 3.2},
        ]
        groups = group_words_into_captions(words, max_chars_per_line=20)
        # Group 1 should end after FIRST SENTENCE.
        assert [w["word"] for w in groups[0]] == ["FIRST", "SENTENCE."]
        # Remaining words split cleanly into groups with at most 2 lines each
        assert len(groups) >= 2

    def test_grouping_breaks_on_speech_pauses(self):
        words = [
            {"word": "HELLO", "rel_start": 0.0, "rel_end": 0.5},
            # 1.0s gap between 0.5 and 1.5 exceeds pause threshold 0.4s
            {"word": "WORLD", "rel_start": 1.5, "rel_end": 2.0},
        ]
        groups = group_words_into_captions(words, max_chars_per_line=22, pause_threshold=0.4)
        assert len(groups) == 2
        assert [w["word"] for w in groups[0]] == ["HELLO"]
        assert [w["word"] for w in groups[1]] == ["WORLD"]

    def test_no_flicker_highlight_timing(self, tmp_path: Path):
        words = [
            {"word": "One", "start": 0.2, "end": 0.5},
            {"word": "two", "start": 0.6, "end": 0.9},
            {"word": "three.", "start": 1.0, "end": 1.3},
        ]
        ass_path = str(tmp_path / "flicker.ass")
        generate_ass_subtitles(words, 0.0, 3.0, ass_path)

        content = Path(ass_path).read_text(encoding="utf-8")
        dialogues = [line for line in content.splitlines() if line.startswith("Dialogue: 0,")]
        assert len(dialogues) == 3

        # Parse timings: Start of word 2 must equal End of word 1 (no flicker)
        times = []
        for d in dialogues:
            parts = d.split(",")
            times.append((parts[1], parts[2]))

        # Word 0 end matches word 1 start
        assert times[0][1] == times[1][0]
        # Word 1 end matches word 2 start
        assert times[1][1] == times[2][0]

    def test_minimum_display_time_per_group(self, tmp_path: Path):
        # Single fast word of 0.2s duration (start 0.1, end 0.3)
        fast_words = [{"word": "Quick.", "start": 0.1, "end": 0.3}]
        ass_path = str(tmp_path / "min_time.ass")
        generate_ass_subtitles(fast_words, 0.0, 5.0, ass_path)

        content = Path(ass_path).read_text(encoding="utf-8")
        dialogue = [line for line in content.splitlines() if line.startswith("Dialogue: 0,")][0]
        parts = dialogue.split(",")
        start_str, end_str = parts[1], parts[2]
        # Group start is 0:00:00.10, end should be extended to at least 0.8s display time (0:00:00.90)
        assert start_str == "0:00:00.10"
        assert end_str == "0:00:00.90"


class TestBoundaryWordsAndRelativeTiming:
    """Tests for clip-relative time clamping and boundary word trimming/dropping."""

    def test_boundary_word_trim_and_drop(self):
        clip_start = 10.0
        clip_end = 20.0
        words = [
            # Word 1: mostly before clip (0.05s inside vs 0.5s total) -> DROP
            {"word": "Early", "start": 9.5, "end": 10.05},
            # Word 2: straddles start with majority inside (0.4s inside vs 0.5s total) -> TRIM
            {"word": "StraddleStart", "start": 9.9, "end": 10.4},
            # Word 3: completely inside -> KEEP
            {"word": "Inside", "start": 12.0, "end": 13.0},
            # Word 4: straddles end with majority inside -> TRIM
            {"word": "StraddleEnd", "start": 19.6, "end": 20.2},
            # Word 5: mostly after clip (0.05s inside vs 0.6s total) -> DROP
            {"word": "Late", "start": 19.95, "end": 20.55},
            # Word 6: completely outside -> DROP
            {"word": "FarOut", "start": 21.0, "end": 22.0},
        ]
        filtered = filter_and_trim_words(words, clip_start, clip_end)
        words_kept = [w["word"] for w in filtered]

        assert "EARLY" not in words_kept
        assert "STRADDLESTART" in words_kept
        assert "INSIDE" in words_kept
        assert "STRADDLEEND" in words_kept
        assert "LATE" not in words_kept
        assert "FAROUT" not in words_kept

        # Verify relative times are bounded within [0, clip_duration = 10.0]
        for w in filtered:
            assert 0.0 <= w["rel_start"] < w["rel_end"] <= 10.0

        # StraddleStart should start at 0.0
        assert filtered[0]["rel_start"] == 0.0
        # StraddleEnd should end at 10.0 (duration)
        assert filtered[-1]["rel_end"] == 10.0


class TestFontsAndScriptGlyphs:
    """Tests for multi-script detection, font selection, glyph checking, and rasterization."""

    @pytest.mark.parametrize(
        ("text", "expected_script"),
        [
            ("Hello world and welcome to the show", "latin"),
            ("नमस्ते आप कैसे हैं", "devanagari"),
            ("നമസ്കാരം സുഖമാണോ", "malayalam"),
            ("مرحبا بكم في هذا الفيديو", "arabic"),
            ("你好世界欢迎大家", "cjk"),
        ],
    )
    def test_script_detection(self, text: str, expected_script: str):
        assert detect_script(text) == expected_script

    def test_bundled_fonts_rasterization_verification(self):
        scripts_samples = [
            ("NotoSans-Regular.ttf", "Hello World 123"),
            ("NotoSansDevanagari-Regular.ttf", "\u0905\u0906\u0915\u0916"),
            ("NotoSansMalayalam-Regular.ttf", "\u0D05\u0D06\u0D15\u0D16"),
            ("NotoSansArabic-Regular.ttf", "\u0627\u0644\u0639\u0631\u0628\u064A\u0629"),
            ("NotoSansCJK-Regular.ttf", "\u4E00\u4E01\u3042\u3044"),
        ]
        for font_file, sample_text in scripts_samples:
            font_path = os.path.join(FONTS_DIR, font_file)
            assert os.path.exists(font_path), f"Bundled font missing: {font_file}"
            success, missing = verify_font_rendering(sample_text, font_path)
            assert success, f"Rendering failed for {font_file} with text {sample_text}: missing {missing}"

    def test_unsupported_script_warning(self, tmp_path: Path):
        # Thai text is not supported by our bundled fonts
        thai_words = [{"word": "สวัสดี", "start": 0.5, "end": 1.5}]
        ass_path = str(tmp_path / "thai.ass")
        warnings = []
        generate_ass_subtitles(thai_words, 0.0, 3.0, ass_path, warnings=warnings)

        assert len(warnings) > 0
        warning_codes = [getattr(w, "code", w.get("code") if isinstance(w, dict) else "") for w in warnings]
        assert "UNSUPPORTED_FONT_SCRIPT" in warning_codes


class TestPlacementAndLayout:
    """Tests for placement margins in split layout and platform safe zones."""

    def test_split_layout_seam_margin(self, transcript_normal, tmp_path: Path):
        ass_path = str(tmp_path / "split.ass")
        generate_ass_subtitles(transcript_normal, 0.0, 5.0, ass_path, layout="split")

        content = Path(ass_path).read_text(encoding="utf-8")
        # Split layout anchors captions at seam between panes (MarginV = 960)
        assert ",2,60,60,960,1" in content

    @pytest.mark.parametrize(
        ("safe_zone", "expected_hook_margin"),
        [("shorts", 190), ("reels", 240), ("tiktok", 280), ("unknown", 190)],
    )
    def test_hook_banner_platform_aware_margins(
        self, safe_zone: str, expected_hook_margin: int, transcript_normal, tmp_path: Path
    ):
        ass_path = str(tmp_path / f"hook_{safe_zone}.ass")
        generate_ass_subtitles(
            transcript_normal,
            0.0,
            5.0,
            ass_path,
            safe_zone=safe_zone,
            hook_title="Valid Viral Title",
            show_hook_banner=True,
        )

        content = Path(ass_path).read_text(encoding="utf-8")
        # Style: HookBanner,...,MarginV,...
        assert f",8,80,80,{expected_hook_margin},1" in content


class TestGenerateAssSubtitles:
    """Preset and style generation characterization tests."""

    def test_script_info_resolution_headers(self, transcript_normal, tmp_path: Path):
        ass_path = str(tmp_path / "test.ass")
        generate_ass_subtitles(transcript_normal, 0.0, 5.0, ass_path)

        content = Path(ass_path).read_text(encoding="utf-8")
        assert "PlayResX: 1080" in content
        assert "PlayResY: 1920" in content
        assert "[V4+ Styles]" in content
        assert "[Events]" in content

    @pytest.mark.parametrize(
        ("safe_zone", "expected_margin"),
        [("shorts", 380), ("reels", 420), ("tiktok", 460), ("unknown_platform", 380)],
    )
    def test_safe_zone_vertical_margins(
        self, safe_zone: str, expected_margin: int, transcript_normal, tmp_path: Path
    ):
        ass_path = str(tmp_path / f"margin_{safe_zone}.ass")
        generate_ass_subtitles(transcript_normal, 0.0, 5.0, ass_path, safe_zone=safe_zone)

        content = Path(ass_path).read_text(encoding="utf-8")
        # Style: Default,...,MarginV,...
        assert f",2,60,60,{expected_margin},1" in content

    def test_preset_boxed_style(self, transcript_normal, tmp_path: Path):
        ass_path = str(tmp_path / "boxed.ass")
        generate_ass_subtitles(transcript_normal, 0.0, 5.0, ass_path, preset="boxed")

        content = Path(ass_path).read_text(encoding="utf-8")
        assert "Style: Default,Noto Sans,74" in content
        assert "&H99000000" in content

    def test_preset_bold_yellow_highlight_tags(self, transcript_normal, tmp_path: Path):
        ass_path = str(tmp_path / "yellow.ass")
        generate_ass_subtitles(transcript_normal, 0.0, 5.0, ass_path, preset="bold_yellow")

        content = Path(ass_path).read_text(encoding="utf-8")
        assert "Style: Default,Noto Sans,78" in content
        assert r"{\c&H0000FFFF&\fscx114\fscy114}" in content

    def test_preset_minimal_no_scale_animation(self, transcript_normal, tmp_path: Path):
        ass_path = str(tmp_path / "minimal.ass")
        generate_ass_subtitles(transcript_normal, 0.0, 5.0, ass_path, preset="minimal")

        content = Path(ass_path).read_text(encoding="utf-8")
        assert "Style: Default,Noto Sans,62" in content
        events_section = content.split("[Events]")[1]
        assert r"\fscx" not in events_section

    def test_hook_banner_generation_toggle(self, transcript_normal, tmp_path: Path):
        # With valid hook banner enabled
        ass_with = str(tmp_path / "with_hook.ass")
        generate_ass_subtitles(
            transcript_normal,
            0.0,
            5.0,
            ass_with,
            hook_title="Viral Secret Revealed",
            show_hook_banner=True,
        )
        content_with = Path(ass_with).read_text(encoding="utf-8")
        assert "HookBanner" in content_with
        assert "VIRAL SECRET REVEALED" in content_with

        # With hook banner disabled
        ass_without = str(tmp_path / "without_hook.ass")
        generate_ass_subtitles(
            transcript_normal,
            0.0,
            5.0,
            ass_without,
            hook_title="Viral Secret Revealed",
            show_hook_banner=False,
        )
        content_without = Path(ass_without).read_text(encoding="utf-8")
        assert "Dialogue: 1," not in content_without

    def test_empty_words_handling(self, transcript_empty, tmp_path: Path):
        ass_path = str(tmp_path / "empty.ass")
        generate_ass_subtitles(transcript_empty, 0.0, 5.0, ass_path)

        content = Path(ass_path).read_text(encoding="utf-8")
        assert "PlayResX: 1080" in content
        # No Default dialogue events generated
        assert "Default,,0,0,0,," not in content
