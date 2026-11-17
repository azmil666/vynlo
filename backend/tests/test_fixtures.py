import pytest

from tests.fixtures.media import probe


class TestMediaFixturesAndProbe:
    """Verifies that all ffmpeg-generated fixtures and the probe helper work offline."""

    def test_probe_vertical_fixture(self, video_vertical: str):
        info = probe(video_vertical)
        assert "streams" in info
        assert "format" in info

        v_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
        assert int(v_stream["width"]) == 1080
        assert int(v_stream["height"]) == 1920

        a_stream = next(s for s in info["streams"] if s["codec_type"] == "audio")
        assert a_stream["codec_name"] == "aac"

    def test_probe_square_fixture(self, video_square: str):
        info = probe(video_square)
        v_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
        assert int(v_stream["width"]) == 1080
        assert int(v_stream["height"]) == 1080

    def test_probe_vfr_fixture(self, video_vfr: str):
        info = probe(video_vfr)
        v_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
        assert v_stream["codec_type"] == "video"

    def test_probe_no_audio_fixture(self, video_no_audio: str):
        info = probe(video_no_audio)
        audio_streams = [s for s in info["streams"] if s["codec_type"] == "audio"]
        assert len(audio_streams) == 0

    def test_probe_silent_audio_fixture(self, video_silent_audio: str):
        info = probe(video_silent_audio)
        audio_streams = [s for s in info["streams"] if s["codec_type"] == "audio"]
        assert len(audio_streams) == 1
        assert audio_streams[0]["codec_name"] == "aac"

    @pytest.mark.slow
    def test_probe_16_9_30s_slow_fixture(self, video_16_9_30s: str):
        info = probe(video_16_9_30s)
        v_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
        assert int(v_stream["width"]) == 1920
        assert int(v_stream["height"]) == 1080
        duration = float(info["format"]["duration"])
        assert 29.0 <= duration <= 31.0
