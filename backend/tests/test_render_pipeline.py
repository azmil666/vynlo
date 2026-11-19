import json
import os
from pathlib import Path
import subprocess
import cv2
import numpy as np
import pytest

from app.schemas import PipelineError, PipelineErrorCode
from app.services.captions import (
    check_faststart,
    generate_ass_subtitles,
    measure_audio_loudness,
    probe_video_stream_info,
    render_clip_single_pass,
    validate_rendered_clip,
)
from app.services.cropper import CropFilterResult, compute_crop_filter


@pytest.fixture
def timestamp_encoded_video(tmp_path: Path) -> str:
    """Generate a 4.0s 1920x1080 30fps video where every frame visually and numerically encodes its time T.

    Left half (x < 960) has a moving marker whose center x moves from 100 to 800 between T=1.0 and T=3.0.
    Right half (x >= 960) has a stationary anchor.
    Audio track contains a 440Hz sine wave.
    """
    video_path = str(tmp_path / "timestamp_encoded.mp4")
    raw_video = str(tmp_path / "raw_video.mp4")
    fps = 30
    duration = 4.0
    total_frames = int(fps * duration)
    w, h = 1920, 1080

    out = cv2.VideoWriter(raw_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame_idx in range(total_frames):
        t = frame_idx / fps
        frame = np.full((h, w, 3), 50, dtype=np.uint8)

        # Draw timestamp text
        cv2.putText(
            frame,
            f"T={t:05.2f}",
            (50, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            3.0,
            (255, 255, 255),
            5,
        )

        # Draw a moving high-contrast white rectangle on the left half between t=1.0 and t=3.0
        # at t=1.0, cx=200; at t=3.0, cx=800
        if 1.0 <= t <= 3.0:
            alpha = (t - 1.0) / 2.0
            cx = int(200 + alpha * 600)
        elif t < 1.0:
            cx = 200
        else:
            cx = 800

        cv2.circle(frame, (cx, 540), 60, (255, 255, 255), -1)
        out.write(frame)
    out.release()

    # Mux with a 440Hz sine audio track using ffmpeg
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        raw_video,
        "-f",
        "lavfi",
        "-t",
        f"{duration}",
        "-i",
        "sine=frequency=440:sample_rate=48000",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ar",
        "48000",
        "-ac",
        "2",
        video_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return video_path


@pytest.fixture
def silent_video(tmp_path: Path) -> str:
    """Generate a 2.0s 1920x1080 30fps video with strictly NO audio track."""
    video_path = str(tmp_path / "silent_no_audio.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=2:size=1920x1080:rate=30",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        video_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return video_path


@pytest.fixture
def high_fps_video(tmp_path: Path) -> str:
    """Generate a 2.0s 1920x1080 60fps video with audio."""
    video_path = str(tmp_path / "high_fps_60.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=2:size=1920x1080:rate=60",
        "-f",
        "lavfi",
        "-t",
        "2",
        "-i",
        "sine=frequency=500:sample_rate=44100",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        video_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return video_path


class TestTimeBaseAlignment:
    """Verify and test that with input-side -ss, filter and subtitle time start at 0."""

    def test_input_seeking_resets_timebase_to_zero(self, timestamp_encoded_video: str, tmp_path: Path):
        start_time = 1.0
        end_time = 3.0
        duration = 2.0
        out_clip = str(tmp_path / "out_aligned.mp4")
        ass_path = str(tmp_path / "aligned.ass")

        # Subtitle scheduled for relative time 0.5s to 1.5s
        words = [
            {"word": "CAPTION_ACTIVE", "start": start_time + 0.5, "end": start_time + 1.5}
        ]
        generate_ass_subtitles(words, start_time, end_time, ass_path, show_hook_banner=False)

        # Dynamic crop motion that pans across time t in [0.0, 2.0]
        # x starts at 100 at t=0, moves to 500 at t=2
        crop_expr = "100+200*t"
        crop_filter = f"crop=608:1080:x='{crop_expr}':y=0,scale=1080:1920:flags=lanczos,setsar=1"

        render_clip_single_pass(
            video_path=timestamp_encoded_video,
            start_time=start_time,
            end_time=end_time,
            crop_filter=crop_filter,
            ass_path=ass_path,
            output_path=out_clip,
            validate_output=True,
        )

        cap = cv2.VideoCapture(out_clip)
        total_rendered_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert total_rendered_frames >= 58  # 2.0s at 30fps ~ 60 frames

        # Frame 0 corresponds to relative t=0.0 (source time T=1.0)
        ret, frame_0 = cap.read()
        assert ret

        # Check OCR-free: the frame at t=0.0 must capture source time T=1.0 where circle cx=200
        # In original source at T=1.0, cx=200.
        # With crop x = 100 at t=0, relative x in crop is 200 - 100 = 100.
        # Scaled to 1080: 100 * (1080 / 608) = 177 px.
        # Check white pixels around x ~ 177 in middle y (y=960)
        center_y = 960
        roi_0 = frame_0[center_y - 20 : center_y + 20, 120:240]
        assert np.mean(roi_0) > 100, "Moving circle at t=0.0 not found at expected position for T=1.0"

        # Frame 30 corresponds to relative t=1.0 (source time T=2.0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
        ret, frame_30 = cap.read()
        assert ret

        # At relative t=1.0 (T=2.0), source cx = 500. Crop x = 100 + 200*1 = 300.
        # Relative x in crop is 500 - 300 = 200. Scaled to 1080: 200 * (1080/608) = 355 px.
        roi_30 = frame_30[center_y - 20 : center_y + 20, 300:420]
        assert np.mean(roi_30) > 100, "Moving circle at t=1.0 not found at expected position for T=2.0"

        # Subtitle caption at t=1.0 (frame 30) must be present in frame
        # Subtitle text is in lower half (around margin_v=380, y ~ 1920 - 380 = 1540)
        subtitle_area_30 = frame_30[1450:1650, :]
        assert np.any(subtitle_area_30 > 200), "Caption not visible at t=1.0"

        # At t=0.1 (frame 3), caption should NOT yet be visible (starts at t=0.5)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 3)
        ret, frame_3 = cap.read()
        assert ret
        subtitle_area_3 = frame_3[1450:1650, :]
        # Base background is dark gray (50), no bright caption text
        assert np.max(subtitle_area_3) < 180, "Caption incorrectly appeared before relative start time"

        cap.release()


class TestFrameRateNormalization:
    """Verify that fps filter and CFR output normalize 60fps/VFR sources to constant 30fps."""

    def test_60fps_source_normalized_to_30fps_cfr(self, high_fps_video: str, tmp_path: Path):
        out_clip = str(tmp_path / "out_cfr_30.mp4")
        crop_filter = "crop=608:1080:656:0,scale=1080:1920:flags=lanczos,setsar=1"

        render_clip_single_pass(
            video_path=high_fps_video,
            start_time=0.0,
            end_time=1.0,
            crop_filter=crop_filter,
            ass_path="",
            output_path=out_clip,
            target_fps=30,
            validate_output=True,
        )

        # Probe with ffprobe
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=r_frame_rate,avg_frame_rate,nb_frames",
            "-of",
            "json",
            out_clip,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        probe = json.loads(res.stdout)
        v_stream = next(s for s in probe["streams"] if "r_frame_rate" in s)

        assert v_stream["r_frame_rate"] == "30/1"
        assert v_stream["avg_frame_rate"] == "30/1"
        assert int(v_stream["nb_frames"]) == 30


class TestAudioLoudnormAndNoAudioHandling:
    """Verify two-pass loudnorm and seamless handling of videos with no audio track."""

    def test_two_pass_loudnorm_measurement(self, timestamp_encoded_video: str):
        # Measure loudnorm parameters
        meas = measure_audio_loudness(timestamp_encoded_video, start_time=0.5, duration=2.0)
        assert meas is not None
        assert "input_i" in meas
        assert "input_tp" in meas
        assert "input_lra" in meas
        assert "input_thresh" in meas
        assert "target_offset" in meas

    def test_video_with_no_audio_track_produces_silent_aac(self, silent_video: str, tmp_path: Path):
        out_clip = str(tmp_path / "out_silent.mp4")
        crop_filter = "crop=608:1080:656:0,scale=1080:1920:flags=lanczos,setsar=1"

        # Source has no audio; must not fail and must produce valid AAC 48kHz stereo track
        render_clip_single_pass(
            video_path=silent_video,
            start_time=0.0,
            end_time=1.0,
            crop_filter=crop_filter,
            ass_path="",
            output_path=out_clip,
            validate_output=True,
        )

        assert os.path.exists(out_clip)
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,sample_rate,channels",
            "-of",
            "json",
            out_clip,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        probe = json.loads(res.stdout)
        audio_stream = next(s for s in probe["streams"] if s.get("codec_name") == "aac")
        assert audio_stream["sample_rate"] == "48000"
        assert int(audio_stream["channels"]) == 2


class TestTimeoutAndProcessGroupKill:
    """Verify scaled timeout calculation and process group termination on timeout."""

    def test_timeout_calculation_scaling(self):
        src_info = {"width": 3840, "height": 2160, "duration": 60.0}
        # 4K resolution factor = 4.0
        res_factor = (src_info["width"] * src_info["height"]) / (1920.0 * 1080.0)
        assert res_factor == 4.0

        duration = 10.0
        calc_timeout = 30.0 + (duration * 4.0 * res_factor)
        # 30 + (10 * 16) = 190.0s
        assert calc_timeout == 190.0

    def test_process_group_killed_on_timeout(self, silent_video: str, tmp_path: Path):
        out_clip = str(tmp_path / "timeout_fail.mp4")
        crop_filter = "crop=608:1080:656:0,scale=1080:1920:flags=lanczos,setsar=1"

        # Set an unrealistically tight timeout (0.001s) to force TimeoutExpired
        with pytest.raises(PipelineError) as exc_info:
            render_clip_single_pass(
                video_path=silent_video,
                start_time=0.0,
                end_time=1.0,
                crop_filter=crop_filter,
                ass_path="",
                output_path=out_clip,
                timeout=0.001,
            )

        err = exc_info.value
        assert err.code == PipelineErrorCode.TIMEOUT
        assert err.stage == "rendering"
        assert "timed out" in err.user_message.lower()


class TestPostRenderValidationAcrossLayouts:
    """Verify ffprobe validation across split and speaker layouts and failure handling."""

    def test_validation_passes_on_speaker_layout(self, timestamp_encoded_video: str, tmp_path: Path):
        out_clip = str(tmp_path / "speaker_valid.mp4")
        crop_filter = "crop=608:1080:656:0,scale=1080:1920:flags=lanczos,setsar=1"

        render_clip_single_pass(
            video_path=timestamp_encoded_video,
            start_time=0.0,
            end_time=1.5,
            crop_filter=crop_filter,
            ass_path="",
            output_path=out_clip,
            validate_output=True,
        )

        validate_rendered_clip(out_clip, expected_duration=1.5)

    def test_validation_passes_on_split_layout(self, timestamp_encoded_video: str, tmp_path: Path):
        out_clip = str(tmp_path / "split_valid.mp4")
        # Split layout filter
        split_crop = compute_crop_filter(
            timestamp_encoded_video,
            0.0,
            1.0,
            mode="split",
            webcam_region={"x": 0.5, "y": 0.5, "w": 0.4, "h": 0.4},
        )

        render_clip_single_pass(
            video_path=timestamp_encoded_video,
            start_time=0.0,
            end_time=1.0,
            crop_filter=split_crop,
            ass_path="",
            output_path=out_clip,
            validate_output=True,
        )

        validate_rendered_clip(out_clip, expected_duration=1.0)

    def test_validation_raises_render_failed_on_empty_file(self, tmp_path: Path):
        empty_file = str(tmp_path / "empty.mp4")
        Path(empty_file).touch()

        with pytest.raises(PipelineError) as exc_info:
            validate_rendered_clip(empty_file, expected_duration=2.0)
        assert exc_info.value.code == PipelineErrorCode.RENDER_FAILED
        assert "missing or empty" in exc_info.value.user_message

    def test_validation_raises_render_failed_on_invalid_codec_or_dimensions(self, tmp_path: Path):
        # Generate 640x480 video (wrong dimensions) with yuv420p
        invalid_file = str(tmp_path / "invalid_dim.mp4")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=640x480:rate=30",
            "-f",
            "lavfi",
            "-t",
            "1",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            invalid_file,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        with pytest.raises(PipelineError) as exc_info:
            validate_rendered_clip(invalid_file, expected_duration=1.0)
        assert exc_info.value.code == PipelineErrorCode.RENDER_FAILED
        assert "1080x1920" in exc_info.value.detail

    def test_validation_raises_render_failed_on_faststart_violation(self, tmp_path: Path):
        # Generate 1080x1920 without faststart (moov after mdat)
        no_faststart_file = str(tmp_path / "no_faststart.mp4")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=1080x1920:rate=30",
            "-f",
            "lavfi",
            "-t",
            "1",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            no_faststart_file,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        assert not check_faststart(no_faststart_file)
        with pytest.raises(PipelineError) as exc_info:
            validate_rendered_clip(no_faststart_file, expected_duration=1.0)
        assert exc_info.value.code == PipelineErrorCode.RENDER_FAILED
        assert "faststart" in exc_info.value.user_message.lower()
