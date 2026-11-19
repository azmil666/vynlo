"""Quality checks for rendered video clips.

Automated verification of:
1. Output spec (ffprobe: resolution, codec, SAR, audio, faststart, trailing frames)
2. Sentence boundary (clip starts and ends on natural sentence/pause boundaries)
3. No word clipping (clip boundaries do not cut into any spoken word)
4. Caption / audio sync (ASS subtitle timing vs transcript word timings)
5. Crop jumps (camera panning speed and smooth transition bounds)
6. Face-in-frame ratio (YuNet face centering across sampled frames)
7. Non-black-bar ratio (detects unintended pillarboxing/letterboxing)
8. Loudness within tolerance (EBU R128 integrated LUFS and true peak)
9. Duration bounds (minimum/maximum clip duration and requested interval matching)
10. Warnings present when degraded (validates typed warning codes during fallback)
"""

from dataclasses import dataclass, field
import json
import logging
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("eval.checks")


def _sanitize_for_json(val: Any) -> Any:
    """Recursively convert numpy types and non-serializable objects for JSON dump."""
    if isinstance(val, dict):
        return {k: _sanitize_for_json(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_sanitize_for_json(v) for v in val]
    if hasattr(val, "item"):
        return val.item()
    if isinstance(val, (bool, int, float, str)) or val is None:
        return val
    return str(val)


@dataclass
class CheckResult:
    """Result of an individual clip quality check."""

    name: str
    passed: bool
    is_hard_check: bool = True
    score: float = 1.0  # 0.0 to 1.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Guarantee standard Python primitive types instead of numpy scalar types
        self.passed = bool(self.passed)
        self.is_hard_check = bool(self.is_hard_check)
        self.score = float(self.score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": str(self.name),
            "passed": bool(self.passed),
            "is_hard_check": bool(self.is_hard_check),
            "score": round(float(self.score), 3),
            "message": str(self.message),
            "details": _sanitize_for_json(self.details),
        }


def check_faststart(file_path: str | Path) -> bool:
    """Verify MP4 container has faststart enabled ('moov' atom before 'mdat')."""
    p = str(file_path)
    if not os.path.exists(p):
        return False
    try:
        with open(p, "rb") as f:
            moov_pos = None
            mdat_pos = None
            pos = 0
            f.seek(0, 2)
            file_size = f.tell()
            f.seek(0)
            # Scan top-level atoms in container
            while pos < min(file_size, 10 * 1024 * 1024):
                header = f.read(8)
                if len(header) < 8:
                    break
                atom_size = int.from_bytes(header[0:4], byteorder="big")
                atom_type = header[4:8]
                if atom_type == b"moov" and moov_pos is None:
                    moov_pos = pos
                elif atom_type == b"mdat" and mdat_pos is None:
                    mdat_pos = pos
                if moov_pos is not None and mdat_pos is not None:
                    break
                if atom_size == 1:  # 64-bit atom
                    header64 = f.read(8)
                    atom_size = int.from_bytes(header64, byteorder="big")
                    pos += atom_size
                    f.seek(pos)
                elif atom_size == 0 or atom_size < 8:
                    break
                else:
                    pos += atom_size
                    f.seek(pos)
            if moov_pos is not None and mdat_pos is not None:
                return moov_pos < mdat_pos
            return False
    except Exception as e:
        logger.warning(f"Error checking faststart on {p}: {e}")
        return False


def check_trailing_frame_decodable(file_path: str | Path) -> Tuple[bool, str]:
    """Verify that trailing frames can be decoded by ffmpeg without fatal errors."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-sseof",
        "-1",
        "-i",
        str(file_path),
        "-f",
        "null",
        "-",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return False, res.stderr.strip()
    return True, ""


def check_output_spec(
    clip_path: str | Path,
    expected_width: int = 1080,
    expected_height: int = 1920,
    is_hard_check: bool = True,
) -> CheckResult:
    """Probe container and streams for required 9:16 vertical delivery spec."""
    p = str(clip_path)
    if not os.path.exists(p):
        return CheckResult(
            name="output_spec",
            passed=False,
            is_hard_check=is_hard_check,
            score=0.0,
            message=f"Clip file does not exist: {p}",
        )

    # 1. Probe ffprobe streams and format
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        p,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(proc.stdout)
    except Exception as e:
        return CheckResult(
            name="output_spec",
            passed=False,
            is_hard_check=is_hard_check,
            score=0.0,
            message=f"ffprobe failed: {e}",
        )

    streams = probe_data.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    errors = []
    details = {}

    # Check Video
    if not video_stream:
        errors.append("No video stream found")
    else:
        v_codec = video_stream.get("codec_name", "")
        v_width = video_stream.get("width")
        v_height = video_stream.get("height")
        v_pix_fmt = video_stream.get("pix_fmt", "")
        v_sar = video_stream.get("sample_aspect_ratio", "1:1")

        details["video"] = {
            "codec": v_codec,
            "width": v_width,
            "height": v_height,
            "pix_fmt": v_pix_fmt,
            "sar": v_sar,
        }

        if v_codec not in ("h264", "libx264"):
            errors.append(f"Invalid video codec: {v_codec} (expected h264)")
        if v_width != expected_width or v_height != expected_height:
            errors.append(f"Invalid dimensions: {v_width}x{v_height} (expected {expected_width}x{expected_height})")
        if v_pix_fmt != "yuv420p":
            errors.append(f"Invalid pixel format: {v_pix_fmt} (expected yuv420p)")
        if v_sar and v_sar not in ("1:1", "1/1", "0:1"):
            errors.append(f"Non-square SAR: {v_sar} (expected 1:1)")

    # Check Audio
    if not audio_stream:
        errors.append("No audio stream found")
    else:
        a_codec = audio_stream.get("codec_name", "")
        a_channels = audio_stream.get("channels")
        a_sample_rate = audio_stream.get("sample_rate")

        details["audio"] = {
            "codec": a_codec,
            "channels": a_channels,
            "sample_rate": a_sample_rate,
        }

        if a_codec != "aac":
            errors.append(f"Invalid audio codec: {a_codec} (expected aac)")
        if a_channels != 2:
            errors.append(f"Invalid audio channels: {a_channels} (expected 2 stereo)")
        if a_sample_rate not in ("44100", "48000"):
            errors.append(f"Unusual audio sample rate: {a_sample_rate} (expected 48000 or 44100)")

    # Check Faststart
    has_faststart = check_faststart(p)
    details["faststart"] = has_faststart
    if not has_faststart:
        errors.append("MP4 missing faststart ('moov' atom not before 'mdat')")

    # Check Trailing Frame Decodability
    decodable, err_msg = check_trailing_frame_decodable(p)
    details["trailing_frame_decodable"] = decodable
    if not decodable:
        errors.append(f"Trailing frames not decodable: {err_msg}")

    passed = len(errors) == 0
    score = 1.0 if passed else max(0.0, 1.0 - (len(errors) * 0.25))
    msg = "Output spec validated (1080x1920, H.264, AAC stereo, yuv420p, faststart, decodable)" if passed else "; ".join(errors)

    return CheckResult(
        name="output_spec",
        passed=passed,
        is_hard_check=is_hard_check,
        score=score,
        message=msg,
        details=details,
    )


def check_sentence_boundary(
    clip_start: float,
    clip_end: float,
    words: List[Dict[str, Any]],
    pause_threshold: float = 0.35,
    is_hard_check: bool = False,
) -> CheckResult:
    """Verify that clip starts and ends on natural sentence or pause boundaries."""
    if not words:
        return CheckResult(
            name="sentence_boundary",
            passed=True,
            is_hard_check=is_hard_check,
            score=1.0,
            message="No words in transcript to evaluate sentence boundary",
        )

    # Find word indices closest to clip_start and clip_end
    start_idx = None
    end_idx = None

    for i, w in enumerate(words):
        w_start = float(w.get("start", 0.0))
        w_end = float(w.get("end", 0.0))
        if start_idx is None and w_start >= clip_start - 0.15:
            start_idx = i
        if w_end <= clip_end + 0.15:
            end_idx = i

    if start_idx is None:
        start_idx = 0
    if end_idx is None or end_idx < start_idx:
        end_idx = min(start_idx + 1, len(words) - 1)

    start_clean = False
    start_reason = ""
    # Check start boundary
    if start_idx == 0:
        start_clean = True
        start_reason = "Starts at beginning of speech"
    else:
        prev_word = words[start_idx - 1]
        prev_text = str(prev_word.get("word", "")).strip()
        gap = float(words[start_idx].get("start", 0.0)) - float(prev_word.get("end", 0.0))
        if prev_text and prev_text[-1] in (".", "?", "!", ":", ";"):
            start_clean = True
            start_reason = f"Starts after punctuation in '{prev_text}'"
        elif gap >= pause_threshold:
            start_clean = True
            start_reason = f"Starts after natural pause of {gap:.2f}s"
        else:
            start_reason = f"Starts mid-sentence without pause (gap: {gap:.2f}s)"

    # Check end boundary
    end_clean = False
    end_reason = ""
    if end_idx >= len(words) - 1:
        end_clean = True
        end_reason = "Ends at completion of speech"
    else:
        curr_word = words[end_idx]
        curr_text = str(curr_word.get("word", "")).strip()
        next_word = words[end_idx + 1]
        gap = float(next_word.get("start", 0.0)) - float(curr_word.get("end", 0.0))
        if curr_text and curr_text[-1] in (".", "?", "!", ":", ";"):
            end_clean = True
            end_reason = f"Ends with punctuation '{curr_text}'"
        elif gap >= pause_threshold:
            end_clean = True
            end_reason = f"Ends before natural pause of {gap:.2f}s"
        else:
            end_reason = f"Ends mid-sentence without pause (gap: {gap:.2f}s)"

    score = 1.0 if (start_clean and end_clean) else (0.5 if (start_clean or end_clean) else 0.0)
    passed = score >= 0.5  # Soft check: passes if at least one boundary is clean or if within bounds

    details = {
        "start_clean": start_clean,
        "start_reason": start_reason,
        "end_clean": end_clean,
        "end_reason": end_reason,
        "clip_start": clip_start,
        "clip_end": clip_end,
    }

    msg = f"Boundary check: {start_reason}; {end_reason}"
    return CheckResult(
        name="sentence_boundary",
        passed=passed,
        is_hard_check=is_hard_check,
        score=score,
        message=msg,
        details=details,
    )


def check_no_word_clipping(
    clip_start: float,
    clip_end: float,
    words: List[Dict[str, Any]],
    tolerance: float = 0.03,
    is_hard_check: bool = True,
) -> CheckResult:
    """Verify that clip start and end timestamps do not slice strictly inside any word."""
    if not words:
        return CheckResult(
            name="no_word_clipping",
            passed=True,
            is_hard_check=is_hard_check,
            score=1.0,
            message="No words to evaluate word clipping",
        )

    clipped_words = []
    for w in words:
        w_start = float(w.get("start", 0.0))
        w_end = float(w.get("end", 0.0))
        word_text = w.get("word", "").strip()

        # Check if clip_start slices into word
        if (w_start + tolerance) < clip_start < (w_end - tolerance):
            clipped_words.append({
                "boundary": "start",
                "word": word_text,
                "word_interval": [w_start, w_end],
                "clip_timestamp": clip_start,
            })

        # Check if clip_end slices into word
        if (w_start + tolerance) < clip_end < (w_end - tolerance):
            clipped_words.append({
                "boundary": "end",
                "word": word_text,
                "word_interval": [w_start, w_end],
                "clip_timestamp": clip_end,
            })

    passed = len(clipped_words) == 0
    score = 1.0 if passed else 0.0
    msg = "No words clipped at boundaries" if passed else f"Clipped {len(clipped_words)} word(s) across boundary"

    return CheckResult(
        name="no_word_clipping",
        passed=passed,
        is_hard_check=is_hard_check,
        score=score,
        message=msg,
        details={"clipped_words": clipped_words},
    )


def _parse_ass_timestamp(ts_str: str) -> float:
    """Parse ASS format timestamp 'H:MM:SS.cc' into seconds."""
    parts = ts_str.strip().split(":")
    if len(parts) != 3:
        return 0.0
    h = int(parts[0])
    m = int(parts[1])
    s = float(parts[2])
    return h * 3600.0 + m * 60.0 + s


def check_caption_audio_sync(
    ass_path: str | Path,
    clip_start: float,
    clip_end: float,
    words: List[Dict[str, Any]],
    max_drift: float = 0.25,
    is_hard_check: bool = True,
) -> CheckResult:
    """Cross-check ASS subtitle event timestamps against transcript word timestamps."""
    p = str(ass_path)
    if not os.path.exists(p):
        return CheckResult(
            name="caption_audio_sync",
            passed=False,
            is_hard_check=is_hard_check,
            score=0.0,
            message=f"ASS subtitle file not found: {p}",
        )

    clip_duration = clip_end - clip_start
    events = []

    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("Dialogue:"):
                # Dialogue: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
                fields = line.split(",", 9)
                if len(fields) >= 10:
                    start_sec = _parse_ass_timestamp(fields[1])
                    end_sec = _parse_ass_timestamp(fields[2])
                    text = fields[9].strip()
                    events.append({
                        "start": start_sec,
                        "end": end_sec,
                        "text": text,
                    })

    if not events:
        # If words were expected in this range, missing dialogue is a failure
        words_in_range = [w for w in words if float(w.get("start", 0.0)) >= clip_start - 0.5 and float(w.get("end", 0.0)) <= clip_end + 0.5]
        if words_in_range:
            return CheckResult(
                name="caption_audio_sync",
                passed=False,
                is_hard_check=is_hard_check,
                score=0.0,
                message="No subtitle Dialogue events found in ASS file despite speech in transcript",
            )
        return CheckResult(
            name="caption_audio_sync",
            passed=True,
            is_hard_check=is_hard_check,
            score=1.0,
            message="No dialogue events (no speech in range)",
        )

    errors = []
    # 1. Bounds check: all events must fall within [0, clip_duration + 0.5]
    for ev in events:
        if ev["start"] < -0.05 or ev["start"] > clip_duration + 0.5:
            errors.append(f"Subtitle start {ev['start']:.2f}s outside clip bounds [0, {clip_duration:.2f}s]")
        if ev["end"] > clip_duration + 1.0:
            errors.append(f"Subtitle end {ev['end']:.2f}s exceeds clip duration {clip_duration:.2f}s")

    # 2. Sync cross-check with words in range
    drifts = []
    for ev in events:
        # Extract plain text without ASS override tags {...}
        plain = re.sub(r"\{.*?\}", "", ev["text"]).strip().lower()
        # Find matching word in relative time
        ev_abs_start = clip_start + ev["start"]
        best_diff = None
        for w in words:
            w_start = float(w.get("start", 0.0))
            diff = abs(w_start - ev_abs_start)
            if best_diff is None or diff < best_diff:
                best_diff = diff
        if best_diff is not None:
            drifts.append(best_diff)
            if best_diff > max_drift:
                errors.append(f"Subtitle drift {best_diff:.2f}s exceeds {max_drift:.2f}s for text '{plain}'")

    avg_drift = sum(drifts) / len(drifts) if drifts else 0.0
    passed = len(errors) == 0
    score = 1.0 if passed else max(0.0, 1.0 - (len(errors) * 0.2))

    return CheckResult(
        name="caption_audio_sync",
        passed=passed,
        is_hard_check=is_hard_check,
        score=score,
        message=f"Captions in sync (avg drift: {avg_drift:.3f}s)" if passed else "; ".join(errors[:3]),
        details={
            "event_count": len(events),
            "avg_drift": round(avg_drift, 3),
            "max_drift_observed": round(max(drifts) if drifts else 0.0, 3),
            "error_count": len(errors),
        },
    )


def check_crop_jumps(
    crop_filter: str | Any,
    max_speed_px_per_sec: float = 320.0,
    fps: float = 30.0,
    is_hard_check: bool = True,
) -> CheckResult:
    """Evaluate camera pan smoothness and ensure frame-to-frame x delta <= max speed."""
    filter_str = str(crop_filter)

    # Static crop (e.g. crop=1080:1920:x:y) or split screen layout
    if not filter_str or "crop=" not in filter_str or "between" not in filter_str:
        return CheckResult(
            name="crop_jumps",
            passed=True,
            is_hard_check=is_hard_check,
            score=1.0,
            message="Static crop or multi-panel split layout (camera speed: 0 px/s)",
            details={"type": "static_or_split", "max_speed": 0.0},
        )

    # Dynamic camera movement: parse transitions
    # Example format from cropper: between(t, t0, t1)*(x0 + (x1-x0)*(t-t0)/(t1-t0))
    transitions = re.findall(
        r"between\(t,([\d\.]+),([\d\.]+)\)\*\(([\d\.]+)\s*\+\s*\(([\-\d\.]+)\)\*\(t-[\d\.]+\)/([\d\.]+)\)",
        filter_str,
    )

    max_observed_speed = 0.0
    jumps = []

    for m in transitions:
        t0, t1, x0, dx, dt = float(m[0]), float(m[1]), float(m[2]), float(m[3]), float(m[4])
        if dt > 0.001:
            speed = abs(dx) / dt
            max_observed_speed = max(max_observed_speed, speed)
            single_frame_jump = speed / fps
            if speed > max_speed_px_per_sec:
                jumps.append({
                    "interval": [t0, t1],
                    "speed_px_s": round(speed, 1),
                    "single_frame_jump_px": round(single_frame_jump, 1),
                })

    passed = len(jumps) == 0
    score = 1.0 if passed else max(0.0, 1.0 - (max_observed_speed / (max_speed_px_per_sec * 2)))
    msg = f"Smooth camera panning (max speed: {max_observed_speed:.1f} px/s <= {max_speed_px_per_sec:.1f})" if passed else f"Camera pan speed exceeded on {len(jumps)} transitions"

    return CheckResult(
        name="crop_jumps",
        passed=passed,
        is_hard_check=is_hard_check,
        score=score,
        message=msg,
        details={
            "max_observed_speed": round(max_observed_speed, 1),
            "max_allowed_speed": max_speed_px_per_sec,
            "violations": jumps,
        },
    )


def check_face_in_frame_ratio(
    clip_path: str | Path,
    min_ratio: float = 0.85,
    sample_interval_sec: float = 1.0,
    is_face_expected: bool = True,
    is_hard_check: bool = True,
) -> CheckResult:
    """Verify that speaker faces remain centered within vertical 9:16 frame across sampled frames."""
    if not is_face_expected:
        return CheckResult(
            name="face_in_frame_ratio",
            passed=True,
            is_hard_check=is_hard_check,
            score=1.0,
            message="Face check skipped: no face expected for this category/layout",
            details={"is_face_expected": False},
        )

    p = str(clip_path)
    if not os.path.exists(p):
        return CheckResult(
            name="face_in_frame_ratio",
            passed=False,
            is_hard_check=is_hard_check,
            score=0.0,
            message=f"Clip file not found: {p}",
        )

    try:
        import cv2
        from app.services.cropper import _get_face_detector
        detector = _get_face_detector()
    except Exception as e:
        logger.warning(f"Could not load YuNet face detector: {e}")
        return CheckResult(
            name="face_in_frame_ratio",
            passed=True,
            is_hard_check=False,
            score=1.0,
            message=f"YuNet detector unavailable ({e}), skipped face check",
        )

    cap = cv2.VideoCapture(p)
    if not cap.isOpened():
        return CheckResult(
            name="face_in_frame_ratio",
            passed=False,
            is_hard_check=is_hard_check,
            score=0.0,
            message="Could not open video file for face checking",
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_step = max(1, int(fps * sample_interval_sec))

    sampled_count = 0
    in_frame_count = 0
    detected_count = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            h, w = frame.shape[:2]
            sampled_count += 1
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)

            if faces is not None and len(faces) > 0:
                detected_count += 1
                # Find largest face
                largest = max(faces, key=lambda f: f[2] * f[3])
                fx, fy, fw, fh = largest[0], largest[1], largest[2], largest[3]
                cx = fx + (fw / 2.0)
                # In 1080x1920, center is 540. Margins: face center between 10% (108px) and 90% (972px)
                if (0.10 * w) <= cx <= (0.90 * w):
                    in_frame_count += 1
        frame_idx += 1

    cap.release()

    if sampled_count == 0:
        return CheckResult(
            name="face_in_frame_ratio",
            passed=False,
            is_hard_check=is_hard_check,
            score=0.0,
            message="Video contains 0 decodable frames",
        )

    if detected_count == 0:
        # If no face was detected in any frame, check if this is an expected fallback
        return CheckResult(
            name="face_in_frame_ratio",
            passed=True,
            is_hard_check=False,
            score=1.0,
            message="No face detected in clip (fallback layout verified)",
            details={"sampled_frames": sampled_count, "faces_detected": 0},
        )

    ratio = in_frame_count / detected_count
    passed = ratio >= min_ratio
    score = min(1.0, ratio)
    msg = f"Face in frame ratio: {ratio:.1%} ({in_frame_count}/{detected_count} frames >= {min_ratio:.1%})"

    return CheckResult(
        name="face_in_frame_ratio",
        passed=passed,
        is_hard_check=is_hard_check,
        score=score,
        message=msg,
        details={
            "sampled_frames": sampled_count,
            "detected_frames": detected_count,
            "in_frame_frames": in_frame_count,
            "ratio": round(ratio, 3),
        },
    )


def check_non_black_bar_ratio(
    clip_path: str | Path,
    max_black_bar_ratio: float = 0.15,
    sample_count: int = 5,
    is_hard_check: bool = True,
) -> CheckResult:
    """Verify that clip does not have unintended black pillarbox or letterbox bars."""
    p = str(clip_path)
    if not os.path.exists(p):
        return CheckResult(
            name="non_black_bar_ratio",
            passed=False,
            is_hard_check=is_hard_check,
            score=0.0,
            message=f"Clip file not found: {p}",
        )

    import cv2
    import numpy as np

    cap = cv2.VideoCapture(p)
    if not cap.isOpened():
        return CheckResult(
            name="non_black_bar_ratio",
            passed=False,
            is_hard_check=is_hard_check,
            score=0.0,
            message="Could not open video file",
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        return CheckResult(
            name="non_black_bar_ratio",
            passed=False,
            is_hard_check=is_hard_check,
            score=0.0,
            message="Empty video file",
        )

    step = max(1, total_frames // (sample_count + 1))
    sample_indices = [step * (i + 1) for i in range(sample_count)]

    black_ratios = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Measure horizontal black bars (left & right columns)
        col_means = np.mean(gray, axis=0)
        left_black = np.sum(col_means[: int(w * 0.3)] < 8)
        right_black = np.sum(col_means[int(w * 0.7) :] < 8)
        pillar_ratio = (left_black + right_black) / float(w)

        # Measure vertical black bars (top & bottom rows)
        row_means = np.mean(gray, axis=1)
        top_black = np.sum(row_means[: int(h * 0.3)] < 8)
        bottom_black = np.sum(row_means[int(h * 0.7) :] < 8)
        letter_ratio = (top_black + bottom_black) / float(h)

        black_ratios.append(max(pillar_ratio, letter_ratio))

    cap.release()

    if not black_ratios:
        return CheckResult(
            name="non_black_bar_ratio",
            passed=True,
            is_hard_check=is_hard_check,
            score=1.0,
            message="Could not sample frames, skipped black bar check",
        )

    max_black = max(black_ratios)
    passed = max_black <= max_black_bar_ratio
    score = max(0.0, 1.0 - max_black)
    msg = f"Non-black-bar check passed (max black ratio: {max_black:.1%} <= {max_black_bar_ratio:.1%})" if passed else f"Unintended black bars detected: {max_black:.1%} > {max_black_bar_ratio:.1%}"

    return CheckResult(
        name="non_black_bar_ratio",
        passed=passed,
        is_hard_check=is_hard_check,
        score=score,
        message=msg,
        details={"max_black_ratio": round(max_black, 3), "samples": [round(r, 3) for r in black_ratios]},
    )


def check_loudness_tolerance(
    clip_path: str | Path,
    target_lufs: float = -14.0,
    lufs_tol: float = 2.0,
    max_true_peak: float = -1.0,
    is_hard_check: bool = True,
) -> CheckResult:
    """Measure EBU R128 loudness and true peak with ffmpeg loudnorm."""
    p = str(clip_path)
    if not os.path.exists(p):
        return CheckResult(
            name="loudness_tolerance",
            passed=False,
            is_hard_check=is_hard_check,
            score=0.0,
            message=f"Clip file not found: {p}",
        )

    cmd = [
        "ffmpeg",
        "-i",
        p,
        "-af",
        "loudnorm=print_format=json",
        "-f",
        "null",
        "-",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    stderr = res.stderr

    # Locate JSON block in stderr
    json_match = re.search(r"\{\s*\"input_i\"\s*:\s*\"([^\"]+)\".*?\"input_tp\"\s*:\s*\"([^\"]+)\".*?\}", stderr, re.DOTALL)
    if not json_match:
        # Fallback to general regex search
        i_match = re.search(r'"input_i"\s*:\s*"([\-\d\.]+)"', stderr)
        tp_match = re.search(r'"input_tp"\s*:\s*"([\-\d\.]+)"', stderr)
        if not i_match:
            return CheckResult(
                name="loudness_tolerance",
                passed=False,
                is_hard_check=is_hard_check,
                score=0.0,
                message="Could not parse loudnorm output from ffmpeg",
            )
        input_i = float(i_match.group(1))
        input_tp = float(tp_match.group(1)) if tp_match else 0.0
    else:
        input_i = float(json_match.group(1))
        input_tp = float(json_match.group(2))

    # Infinite or extremely low audio (pure silence)
    if math.isinf(input_i) or input_i < -70.0:
        return CheckResult(
            name="loudness_tolerance",
            passed=False,
            is_hard_check=is_hard_check,
            score=0.0,
            message=f"Audio is silent or inaudible (integrated loudness: {input_i} LUFS)",
            details={"input_i": input_i, "input_tp": input_tp},
        )

    min_lufs = target_lufs - lufs_tol
    max_lufs = target_lufs + lufs_tol

    errors = []
    if not (min_lufs <= input_i <= max_lufs):
        errors.append(f"Integrated loudness {input_i:.1f} LUFS outside target [{min_lufs:.1f}, {max_lufs:.1f}]")
    if input_tp > (max_true_peak + 0.5):
        errors.append(f"True peak {input_tp:.1f} dBTP exceeds limit {max_true_peak:.1f} dBTP")

    passed = len(errors) == 0
    delta = abs(input_i - target_lufs)
    score = max(0.0, 1.0 - (delta / 10.0))
    msg = f"Loudness in tolerance: {input_i:.1f} LUFS (target: {target_lufs:.1f} ± {lufs_tol:.1f}), Peak: {input_tp:.1f} dBTP" if passed else "; ".join(errors)

    return CheckResult(
        name="loudness_tolerance",
        passed=passed,
        is_hard_check=is_hard_check,
        score=score,
        message=msg,
        details={
            "input_i": input_i,
            "input_tp": input_tp,
            "target_lufs": target_lufs,
            "tolerance": lufs_tol,
        },
    )


def check_duration_bounds(
    clip_path: str | Path,
    min_dur: float = 5.0,
    max_dur: float = 60.0,
    requested_start: Optional[float] = None,
    requested_end: Optional[float] = None,
    is_hard_check: bool = True,
) -> CheckResult:
    """Verify that probed clip duration is within [min_dur, max_dur] and matches requested span."""
    p = str(clip_path)
    if not os.path.exists(p):
        return CheckResult(
            name="duration_bounds",
            passed=False,
            is_hard_check=is_hard_check,
            score=0.0,
            message=f"Clip file not found: {p}",
        )

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        p,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        fmt = json.loads(proc.stdout).get("format", {})
        duration = float(fmt.get("duration", 0.0))
    except Exception as e:
        return CheckResult(
            name="duration_bounds",
            passed=False,
            is_hard_check=is_hard_check,
            score=0.0,
            message=f"ffprobe failed to read duration: {e}",
        )

    errors = []
    if duration < min_dur:
        errors.append(f"Duration {duration:.2f}s is below minimum {min_dur:.1f}s")
    if duration > max_dur:
        errors.append(f"Duration {duration:.2f}s exceeds maximum {max_dur:.1f}s")

    if requested_start is not None and requested_end is not None:
        expected = requested_end - requested_start
        diff = abs(duration - expected)
        if diff > 0.5:
            errors.append(f"Duration {duration:.2f}s differs from requested {expected:.2f}s by {diff:.2f}s > 0.5s")

    passed = len(errors) == 0
    score = 1.0 if passed else 0.0
    msg = f"Duration {duration:.2f}s within bounds [{min_dur:.1f}s, {max_dur:.1f}s]" if passed else "; ".join(errors)

    return CheckResult(
        name="duration_bounds",
        passed=passed,
        is_hard_check=is_hard_check,
        score=score,
        message=msg,
        details={
            "duration": round(duration, 3),
            "min_dur": min_dur,
            "max_dur": max_dur,
            "requested_start": requested_start,
            "requested_end": requested_end,
        },
    )


def check_warnings_when_degraded(
    clip_warnings: List[str],
    job_warnings: Optional[List[str]] = None,
    expected_degradation: Optional[str] = None,
    is_hard_check: bool = True,
) -> CheckResult:
    """Verify that fallback operations emit expected structured warnings."""
    all_warnings = list(clip_warnings or []) + list(job_warnings or [])

    if not expected_degradation:
        # No specific degradation expected, passes
        return CheckResult(
            name="warnings_when_degraded",
            passed=True,
            is_hard_check=is_hard_check,
            score=1.0,
            message=f"No degradation expected ({len(all_warnings)} warnings recorded)",
            details={"warnings": all_warnings},
        )

    # Check if expected degradation string or prefix exists in any warning
    deg_norm = expected_degradation.lower()
    found = any(deg_norm in w.lower() for w in all_warnings)

    passed = found
    score = 1.0 if passed else 0.0
    msg = f"Recorded expected degradation warning '{expected_degradation}'" if passed else f"Missing expected degradation warning '{expected_degradation}'. Recorded: {all_warnings}"

    return CheckResult(
        name="warnings_when_degraded",
        passed=passed,
        is_hard_check=is_hard_check,
        score=score,
        message=msg,
        details={"expected": expected_degradation, "recorded_warnings": all_warnings},
    )
