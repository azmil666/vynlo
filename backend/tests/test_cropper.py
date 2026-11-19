import os
from pathlib import Path
import pytest

from app.services.cropper import compute_crop_filter, crop_to_vertical


class TestComputeCropFilterOutputStrings:
    """Characterization tests for FFmpeg crop filter string generation."""

    def test_split_screen_mode_returns_hardcoded_filter(self):
        # When video file does not exist, split mode returns the 9:8 undistorted fallback filter
        filter_str = compute_crop_filter("non_existent_video.mp4", 0.0, 10.0, mode="split")
        assert filter_str == (
            "[0:v]crop=min(iw\\,ih*9/8):min(ih\\,iw*8/9):0:0,scale=1080:960:flags=lanczos,setsar=1[top];"
            "[0:v]crop=ih*0.36*9/8:ih*0.36:iw-ih*0.36*9/8:ih-ih*0.36,scale=1080:960:flags=lanczos,setsar=1[bottom];"
            "[top][bottom]vstack=inputs=2[v]"
        )

    def test_speaker_mode_missing_video_raises_value_error(self):
        with pytest.raises(ValueError) as exc:
            compute_crop_filter("/invalid/path/non_existent.mp4", 0.0, 5.0, mode="speaker")
        assert "Failed to open video file" in str(exc.value)

    def test_speaker_mode_vertical_video_exact_9_16_scaled_no_pad(self, video_vertical: str):
        # Exact 9:16 video (1080x1920) passes through scaled with no distortion or black padding
        filter_str = compute_crop_filter(video_vertical, 0.0, 2.0, mode="speaker")
        assert filter_str == "scale=1080:1920:flags=lanczos,setsar=1"
        assert filter_str.layout_used == "speaker"

    def test_speaker_mode_landscape_16_9_no_faces_falls_back_to_fit_blur(self, video_16_9_30s: str):
        # 16:9 test pattern without detected human faces falls back to fit_blur instead of dead-center crop
        filter_str = compute_crop_filter(video_16_9_30s, 0.0, 3.0, mode="speaker")
        assert filter_str.layout_used == "fit_blur"
        assert filter_str.coverage == 0.0
        assert "boxblur" in filter_str
        assert "overlay" in filter_str

        # When force_crop=True is passed, it forces the center crop (0.5)
        forced_str = compute_crop_filter(video_16_9_30s, 0.0, 3.0, mode="speaker", force_crop=True)
        assert forced_str.startswith("crop=606:1080:")
        assert "scale=1080:1920:flags=lanczos,setsar=1" in forced_str

    def test_speaker_mode_square_video_no_faces_falls_back_to_fit_blur(self, video_square: str):
        # Square video without faces falls back to fit_blur
        filter_str = compute_crop_filter(video_square, 0.0, 2.0, mode="speaker")
        assert filter_str.layout_used == "fit_blur"
        assert filter_str.coverage == 0.0
        assert "boxblur" in filter_str

        # With force_crop=True, forces crop filter
        forced_str = compute_crop_filter(video_square, 0.0, 2.0, mode="speaker", force_crop=True)
        assert forced_str.startswith("crop=606:1080:")
        assert "scale=1080:1920:flags=lanczos,setsar=1" in forced_str

    @pytest.mark.slow
    def test_crop_to_vertical_full_pipeline(self, video_vertical: str, tmp_path: Path):
        output_file = str(tmp_path / "cropped.mp4")
        result_path = crop_to_vertical(
            video_path=video_vertical,
            start_time=0.0,
            end_time=1.0,
            output_path=output_file,
            mode="speaker",
        )
        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 0
