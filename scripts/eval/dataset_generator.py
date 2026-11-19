"""Synthetic dataset generator for evaluation harness.

Generates offline test videos and accompanying transcripts using FFmpeg lavfi filters
for all 8 evaluation categories when local media files are not provided:
1. single-speaker talk
2. two-person podcast
3. screen share / slides
4. gameplay with webcam
5. vertical source
6. long (60+ min)
7. non-English
8. noisy / music
"""

import json
import logging
import os
from pathlib import Path
import subprocess
from typing import Dict, List, Tuple

logger = logging.getLogger("eval.dataset_generator")


def _generate_synthetic_transcript(category_id: str, duration: float = 15.0) -> List[Dict]:
    """Generate realistic transcript words with punctuation and word intervals."""
    transcripts = {
        "single_speaker": [
            ("Today", 0.5, 0.9),
            ("we", 0.95, 1.1),
            ("are", 1.15, 1.3),
            ("going", 1.35, 1.6),
            ("to", 1.65, 1.8),
            ("explore", 1.85, 2.3),
            ("the", 2.35, 2.5),
            ("future", 2.55, 3.0),
            ("of", 3.05, 3.2),
            ("computing.", 3.25, 4.0),
            ("It", 4.5, 4.7),
            ("is", 4.75, 4.9),
            ("transforming", 4.95, 5.7),
            ("every", 5.75, 6.1),
            ("single", 6.15, 6.5),
            ("industry.", 6.55, 7.2),
            ("First,", 7.8, 8.2),
            ("let's", 8.25, 8.5),
            ("examine", 8.55, 9.0),
            ("neural", 9.05, 9.4),
            ("networks.", 9.45, 10.1),
            ("The", 10.6, 10.8),
            ("results", 10.85, 11.3),
            ("are", 11.35, 11.5),
            ("truly", 11.55, 11.9),
            ("remarkable.", 11.95, 12.8),
        ],
        "podcast_two_person": [
            ("Welcome", 0.5, 1.0),
            ("back", 1.05, 1.3),
            ("to", 1.35, 1.5),
            ("the", 1.55, 1.7),
            ("show,", 1.75, 2.2),
            ("everyone.", 2.25, 2.8),
            ("Thanks", 3.4, 3.8),
            ("for", 3.85, 4.0),
            ("having", 4.05, 4.4),
            ("me", 4.45, 4.6),
            ("here", 4.65, 4.9),
            ("today.", 4.95, 5.5),
            ("What", 6.0, 6.3),
            ("did", 6.35, 6.5),
            ("you", 6.55, 6.7),
            ("think", 6.75, 7.0),
            ("of", 7.05, 7.2),
            ("the", 7.25, 7.4),
            ("announcement?", 7.45, 8.4),
            ("I", 9.0, 9.2),
            ("was", 9.25, 9.5),
            ("completely", 9.55, 10.2),
            ("blown", 10.25, 10.6),
            ("away.", 10.65, 11.2),
            ("Same", 11.8, 12.1),
            ("here.", 12.15, 12.6),
        ],
        "screen_share": [
            ("In", 0.5, 0.7),
            ("this", 0.75, 1.0),
            ("dashboard,", 1.05, 1.7),
            ("you", 1.8, 2.0),
            ("can", 2.05, 2.2),
            ("observe", 2.25, 2.7),
            ("system", 2.75, 3.2),
            ("throughput.", 3.25, 4.0),
            ("Notice", 4.6, 5.1),
            ("the", 5.15, 5.3),
            ("drop", 5.35, 5.7),
            ("in", 5.75, 5.9),
            ("latency", 5.95, 6.6),
            ("after", 6.7, 7.0),
            ("caching.", 7.05, 7.7),
            ("All", 8.3, 8.6),
            ("metrics", 8.65, 9.2),
            ("are", 9.25, 9.4),
            ("now", 9.45, 9.7),
            ("within", 9.75, 10.2),
            ("nominal", 10.25, 10.8),
            ("limits.", 10.85, 11.5),
        ],
        "gameplay_webcam": [
            ("Check", 0.5, 0.9),
            ("this", 0.95, 1.2),
            ("out,", 1.25, 1.6),
            ("we", 1.8, 2.0),
            ("are", 2.05, 2.2),
            ("entering", 2.25, 2.8),
            ("the", 2.85, 3.0),
            ("final", 3.05, 3.5),
            ("zone.", 3.55, 4.1),
            ("Watch", 4.7, 5.1),
            ("my", 5.15, 5.3),
            ("flank", 5.35, 5.8),
            ("on", 5.85, 6.0),
            ("the", 6.05, 6.2),
            ("left!", 6.25, 6.8),
            ("That", 7.4, 7.7),
            ("was", 7.75, 8.0),
            ("an", 8.05, 8.2),
            ("incredible", 8.25, 8.9),
            ("play.", 8.95, 9.5),
            ("Let's", 10.1, 10.5),
            ("go!", 10.55, 11.2),
        ],
        "vertical_source": [
            ("Welcome", 0.5, 1.0),
            ("to", 1.05, 1.2),
            ("our", 1.25, 1.5),
            ("quick", 1.55, 1.9),
            ("vertical", 1.95, 2.5),
            ("tip", 2.55, 2.8),
            ("of", 2.85, 3.0),
            ("the", 3.05, 3.2),
            ("day.", 3.25, 3.8),
            ("Always", 4.4, 4.9),
            ("render", 4.95, 5.4),
            ("in", 5.45, 5.6),
            ("proper", 5.65, 6.1),
            ("nine", 6.15, 6.4),
            ("sixteen", 6.45, 6.9),
            ("format.", 6.95, 7.6),
            ("It", 8.2, 8.4),
            ("maximizes", 8.45, 9.2),
            ("engagement.", 9.25, 10.1),
        ],
        "long_video": [
            ("Over", 0.5, 0.8),
            ("the", 0.85, 1.0),
            ("course", 1.05, 1.4),
            ("of", 1.45, 1.6),
            ("this", 1.65, 1.9),
            ("extended", 1.95, 2.5),
            ("discussion,", 2.55, 3.3),
            ("we", 3.8, 4.0),
            ("will", 4.05, 4.3),
            ("cover", 4.35, 4.7),
            ("architectural", 4.75, 5.6),
            ("principles.", 5.65, 6.4),
            ("Stability", 7.0, 7.6),
            ("under", 7.65, 8.0),
            ("load", 8.05, 8.4),
            ("is", 8.45, 8.6),
            ("essential.", 8.65, 9.4),
            ("Memory", 10.0, 10.5),
            ("must", 10.55, 10.9),
            ("remain", 10.95, 11.4),
            ("bounded.", 11.45, 12.1),
        ],
        "multilingual": [
            ("Bienvenidos", 0.5, 1.3),
            ("a", 1.35, 1.5),
            ("nuestra", 1.55, 2.0),
            ("conferencia", 2.05, 2.9),
            ("internacional.", 2.95, 3.9),
            ("Hoy", 4.5, 4.8),
            ("vamos", 4.85, 5.2),
            ("a", 5.25, 5.4),
            ("analizar", 5.45, 6.1),
            ("la", 6.15, 6.3),
            ("calidad", 6.35, 6.9),
            ("del", 6.95, 7.2),
            ("video.", 7.25, 7.9),
            ("Los", 8.5, 8.8),
            ("resultados", 8.85, 9.6),
            ("son", 9.65, 9.9),
            ("excelentes.", 9.95, 10.8),
        ],
        "noisy_music": [
            ("Even", 0.5, 0.8),
            ("in", 0.85, 1.0),
            ("a", 1.05, 1.2),
            ("noisy", 1.25, 1.6),
            ("environment,", 1.65, 2.5),
            ("audio", 3.0, 3.4),
            ("normalization", 3.45, 4.4),
            ("maintains", 4.45, 5.0),
            ("clarity.", 5.05, 5.7),
            ("Loudness", 6.3, 6.9),
            ("is", 6.95, 7.1),
            ("carefully", 7.15, 7.7),
            ("metered", 7.75, 8.3),
            ("at", 8.35, 8.5),
            ("fourteen", 8.55, 9.1),
            ("LUFS.", 9.15, 9.8),
        ],
    }

    raw = transcripts.get(category_id, transcripts["single_speaker"])
    return [{"word": w, "start": s, "end": e} for (w, s, e) in raw if e <= duration]


def generate_synthetic_video(
    category_id: str,
    output_video_path: Path,
    output_transcript_path: Path,
    duration: float = 14.0,
) -> Tuple[Path, Path]:
    """Generate offline test video and transcript for one of the 8 categories."""
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_transcript_path.parent.mkdir(parents=True, exist_ok=True)

    v_out = str(output_video_path)
    t_out = str(output_transcript_path)

    # 1. Generate video stream filter
    if category_id == "vertical_source":
        # Native 1080x1920
        v_filter = "testsrc=size=1080x1920:rate=30,drawbox=x=340:y=600:w=400:h=400:color=blue@0.8:t=fill"
    elif category_id == "podcast_two_person":
        # Two boxes (left speaker, right speaker)
        v_filter = (
            "testsrc=size=1920x1080:rate=30,"
            "drawbox=x=300:y=300:w=360:h=480:color=blue@0.8:t=fill,"
            "drawbox=x=1260:y=300:w=360:h=480:color=green@0.8:t=fill"
        )
    elif category_id == "screen_share":
        # Slides / Presentation pattern
        v_filter = (
            "smptebars=size=1920x1080:rate=30,"
            "drawbox=x=100:y=100:w=1720:h=880:color=black@0.6:t=fill,"
            "drawgrid=width=100:height=100:thickness=2:color=white@0.2"
        )
    elif category_id == "gameplay_webcam":
        # Dynamic gameplay background + webcam inset box at bottom right
        v_filter = (
            "testsrc=size=1920x1080:rate=30,"
            "drawbox=x=1440:y=680:w=400:h=320:color=red@0.8:t=fill"
        )
    else:
        # Standard 16:9 talk with speaker box
        v_filter = "testsrc=size=1920x1080:rate=30,drawbox=x=760:y=300:w=400:h=500:color=blue@0.8:t=fill"

    # 2. Audio filter
    if category_id == "noisy_music":
        # Sine tone + noise source mixed
        a_filter = "sine=f=440:r=48000[tone];anoisesrc=d=15:c=pink:r=48000:a=0.08[noise];[tone][noise]amix=inputs=2"
    else:
        a_filter = "sine=f=880:r=48000"

    # 3. Assemble ffmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        v_filter,
        "-f",
        "lavfi",
        "-i",
        a_filter,
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-ac",
        "2",
        v_out,
    ]

    subprocess.run(cmd, capture_output=True, check=True)

    # 4. Write transcript JSON
    words = _generate_synthetic_transcript(category_id, duration)
    with open(t_out, "w", encoding="utf-8") as f:
        json.dump(words, f, indent=2)

    return output_video_path, output_transcript_path
