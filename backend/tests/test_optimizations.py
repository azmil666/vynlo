from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings, get_render_concurrency, settings
from app.services.captions import render_clip_single_pass
from app.services.transcriber import get_whisper_model, set_whisper_model


class TestPerformanceOptimizations:
    """Tests verifying Whisper thread limits, render concurrency derivation, and FFmpeg thread budgeting."""

    def test_get_render_concurrency_auto_derivation(self):
        # Simulated 8-core CPU
        with patch("os.cpu_count", return_value=8):
            cfg = Settings(RENDER_CONCURRENCY=0)
            # 2 candidate clips on 8 cores -> concurrency 2, 4 threads each
            concurrency, threads = get_render_concurrency(2, cfg=cfg)
            assert concurrency == 2
            assert threads == 4
            assert concurrency * threads <= 8

            # 1 candidate clip on 8 cores -> concurrency 1, 8 threads
            concurrency, threads = get_render_concurrency(1, cfg=cfg)
            assert concurrency == 1
            assert threads == 8

        # Simulated 2-core CPU
        with patch("os.cpu_count", return_value=2):
            cfg = Settings(RENDER_CONCURRENCY=0)
            # On 2 cores -> concurrency capped at 1 to prevent thrashing
            concurrency, threads = get_render_concurrency(3, cfg=cfg)
            assert concurrency == 1
            assert threads == 2

    def test_get_render_concurrency_explicit_overrides(self):
        with patch("os.cpu_count", return_value=12):
            cfg = Settings(RENDER_CONCURRENCY=3, FFMPEG_THREADS_PER_CLIP=3)
            concurrency, threads = get_render_concurrency(5, cfg=cfg)
            assert concurrency == 3
            assert threads == 3

    def test_whisper_thread_budgeting(self):
        set_whisper_model(None)
        mock_whisper_cls = MagicMock()

        with patch("app.services.transcriber.WhisperModel", mock_whisper_cls):
            with patch.object(settings, "WHISPER_CPU_THREADS", 3):
                model = get_whisper_model()
                assert model is not None
                mock_whisper_cls.assert_called_once()
                call_kwargs = mock_whisper_cls.call_args[1]
                assert call_kwargs["cpu_threads"] == 3
                assert call_kwargs["compute_type"] == settings.WHISPER_COMPUTE_TYPE

        set_whisper_model(None)

    def test_render_clip_passes_thread_budget_to_ffmpeg(self, tmp_path):
        dummy_in = str(tmp_path / "in.mp4")
        dummy_out = str(tmp_path / "out.mp4")
        dummy_ass = str(tmp_path / "in.ass")
        with open(dummy_in, "w") as f:
            f.write("fake")
        with open(dummy_ass, "w") as f:
            f.write("fake")

        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0

        with patch("app.services.captions.probe_video_stream_info", return_value={
            "width": 1920,
            "height": 1080,
            "duration": 10.0,
            "has_audio": True,
        }), patch("app.services.captions.measure_audio_loudness", return_value=None), \
           patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
           patch("app.services.captions.validate_rendered_clip"):

            render_clip_single_pass(
                video_path=dummy_in,
                start_time=0.0,
                end_time=5.0,
                crop_filter="crop=w=607:h=1080",
                ass_path=dummy_ass,
                output_path=dummy_out,
                threads=4,
            )

            mock_popen.assert_called_once()
            cmd_args = mock_popen.call_args[0][0]
            assert "-threads" in cmd_args
            thread_idx = cmd_args.index("-threads")
            assert cmd_args[thread_idx + 1] == "4"
