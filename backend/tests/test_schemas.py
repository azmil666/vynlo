import pytest
from pydantic import ValidationError

from app.schemas import ALLOWED_VIDEO_DOMAINS, ClipResult, JobCreateRequest, JobStatusResponse, WebcamRegion


class TestJobCreateRequestValidation:
    """Characterization tests for JobCreateRequest schema validation."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://youtube.com/watch?v=abc123",
            "https://www.youtube.com/watch?v=abc123",
            "https://m.youtube.com/watch?v=abc123",
            "https://youtu.be/abc123",
            "https://vimeo.com/123456",
            "https://www.vimeo.com/123456",
            "https://twitch.tv/videos/123456",
            "https://www.twitch.tv/videos/123456",
            "https://clips.twitch.tv/AwesomeClip",
            "HTTPS://WWW.YOUTUBE.COM/watch?v=uppercase",
        ],
    )
    def test_valid_allowed_video_domains(self, url: str):
        req = JobCreateRequest(url=url)
        assert req.url is not None
        assert req.layout_mode == "speaker"
        assert req.max_clips == 3
        assert req.caption_preset == "clean_white"
        assert req.platform_safe_zone == "shorts"
        assert req.highlight_color == "&H00FFFFFF"
        assert req.show_hook_banner is True

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/video.mp4",
            "https://evil-youtube.com/watch?v=123",
            "https://tiktok.com/@user/video/123",
            "https://instagram.com/reel/123",
            "https://not-youtube.com",
            "ftp://youtube.com/video",
        ],
    )
    def test_rejected_video_domains(self, url: str):
        with pytest.raises(ValidationError) as exc_info:
            JobCreateRequest(url=url)
        assert "Unsupported video domain" in str(exc_info.value) or "URL" in str(exc_info.value)

    def test_invalid_url_format(self):
        with pytest.raises(ValidationError):
            JobCreateRequest(url="not_a_valid_url")

    @pytest.mark.parametrize("layout", ["speaker", "split"])
    def test_valid_layout_modes(self, layout: str):
        req = JobCreateRequest(url="https://youtube.com/watch?v=123", layout_mode=layout)
        assert req.layout_mode == layout

    def test_invalid_layout_mode(self):
        with pytest.raises(ValidationError):
            JobCreateRequest(url="https://youtube.com/watch?v=123", layout_mode="picture_in_picture")

    def test_valid_webcam_region(self):
        region = WebcamRegion(x=0.1, y=0.2, w=0.3, h=0.4)
        req = JobCreateRequest(url="https://youtube.com/watch?v=123", webcam_region=region)
        assert req.webcam_region is not None
        assert req.webcam_region.x == 0.1
        assert req.webcam_region.w == 0.3

    def test_invalid_webcam_region(self):
        with pytest.raises(ValidationError):
            WebcamRegion(x=-0.1, y=0.2, w=0.3, h=0.4)
        with pytest.raises(ValidationError):
            WebcamRegion(x=0.1, y=1.2, w=0.3, h=0.4)
        with pytest.raises(ValidationError):
            WebcamRegion(x=0.1, y=0.2, w=0.0, h=0.4)

    @pytest.mark.parametrize("count", [1, 5, 10])
    def test_valid_max_clips_bounds(self, count: int):
        req = JobCreateRequest(url="https://youtube.com/watch?v=123", max_clips=count)
        assert req.max_clips == count

    @pytest.mark.parametrize("count", [0, -1, 11, 100])
    def test_invalid_max_clips_bounds(self, count: int):
        with pytest.raises(ValidationError):
            JobCreateRequest(url="https://youtube.com/watch?v=123", max_clips=count)

    @pytest.mark.parametrize("preset", ["clean_white", "bold_yellow", "boxed", "minimal"])
    def test_valid_caption_presets(self, preset: str):
        req = JobCreateRequest(url="https://youtube.com/watch?v=123", caption_preset=preset)
        assert req.caption_preset == preset

    def test_invalid_caption_preset(self):
        with pytest.raises(ValidationError):
            JobCreateRequest(url="https://youtube.com/watch?v=123", caption_preset="neon_glow")

    @pytest.mark.parametrize("safe_zone", ["shorts", "reels", "tiktok"])
    def test_valid_platform_safe_zones(self, safe_zone: str):
        req = JobCreateRequest(url="https://youtube.com/watch?v=123", platform_safe_zone=safe_zone)
        assert req.platform_safe_zone == safe_zone

    def test_invalid_platform_safe_zone(self):
        with pytest.raises(ValidationError):
            JobCreateRequest(url="https://youtube.com/watch?v=123", platform_safe_zone="square_post")

    @pytest.mark.parametrize("color", ["&H00FFFFFF", "&H0000FFFF", "&H00ffffff", "&H12345678"])
    def test_valid_highlight_colors(self, color: str):
        req = JobCreateRequest(url="https://youtube.com/watch?v=123", highlight_color=color)
        assert req.highlight_color == color

    @pytest.mark.parametrize("color", ["&h00ffffff", "#FFFFFF", "red", "&HFFF", "&H00FFFFFF00", "&HGGGGGGGG"])
    def test_invalid_highlight_colors(self, color: str):
        with pytest.raises(ValidationError):
            JobCreateRequest(url="https://youtube.com/watch?v=123", highlight_color=color)


class TestJobStatusResponseValidation:
    """Characterization tests for JobStatusResponse and ClipResult schemas."""

    @pytest.mark.parametrize(
        "status",
        ["queued", "downloading", "transcribing", "analyzing", "rendering", "completed", "failed"],
    )
    def test_valid_statuses(self, status: str):
        resp = JobStatusResponse(job_id="job_123", status=status, progress=50)
        assert resp.status == status
        assert resp.progress == 50
        assert resp.clips == []
        assert resp.error is None

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            JobStatusResponse(job_id="job_123", status="processing", progress=50)

    @pytest.mark.parametrize("progress", [0, 50, 100])
    def test_valid_progress_bounds(self, progress: int):
        resp = JobStatusResponse(job_id="job_123", status="queued", progress=progress)
        assert resp.progress == progress

    @pytest.mark.parametrize("progress", [-1, 101])
    def test_invalid_progress_bounds(self, progress: int):
        with pytest.raises(ValidationError):
            JobStatusResponse(job_id="job_123", status="queued", progress=progress)

    def test_clip_result_serialization(self):
        clip = ClipResult(
            clip_id="clip_1",
            title="Epic Moment",
            virality_score=85,
            duration=32.5,
            video_url="http://localhost:8000/static/job_1/clip_1.mp4",
            is_fallback=False,
            description="A great highlight.",
            hashtags=["gaming", "viral"],
        )
        resp = JobStatusResponse(job_id="job_1", status="completed", progress=100, clips=[clip])
        assert len(resp.clips) == 1
        assert resp.clips[0].clip_id == "clip_1"
        assert resp.clips[0].virality_score == 85
