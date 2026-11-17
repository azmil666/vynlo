from pathlib import Path
import pytest

from app.services.captions import _format_ass_time, _wrap_hook_title, generate_ass_subtitles


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


class TestGenerateAssSubtitles:
    """Characterization tests for generate_ass_subtitles preset and style generation."""

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
        assert "Style: Default,Liberation Sans,74" in content
        # border_style=3 (opaque box), back_color=&H99000000
        assert "&H99000000" in content

    def test_preset_bold_yellow_highlight_tags(self, transcript_normal, tmp_path: Path):
        ass_path = str(tmp_path / "yellow.ass")
        generate_ass_subtitles(transcript_normal, 0.0, 5.0, ass_path, preset="bold_yellow")

        content = Path(ass_path).read_text(encoding="utf-8")
        assert "Style: Default,Liberation Sans,78" in content
        assert r"{\c&H0000FFFF&\fscx114\fscy114}" in content

    def test_preset_minimal_no_scale_animation(self, transcript_normal, tmp_path: Path):
        ass_path = str(tmp_path / "minimal.ass")
        generate_ass_subtitles(transcript_normal, 0.0, 5.0, ass_path, preset="minimal")

        content = Path(ass_path).read_text(encoding="utf-8")
        assert "Style: Default,Liberation Sans,62" in content
        # Minimal preset omits scale tags \fscx
        events_section = content.split("[Events]")[1]
        assert r"\fscx" not in events_section

    def test_hook_banner_generation_toggle(self, transcript_normal, tmp_path: Path):
        # With hook banner enabled
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
