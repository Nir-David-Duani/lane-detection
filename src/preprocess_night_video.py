"""
Pre-processing script for the night-time video
==============================================

Goal
----
This script performs **only pre-processing** for the night-time enhancement:

1. Download **only a specific 3-minute segment** from a long YouTube video
   using `yt-dlp` + `ffmpeg` (without downloading the entire many-hours file).
2. Save that segment into `data/processed/` to be used as the night clip.

The idea is that this script is **standalone** and does *not* change the
lane detection algorithm – it only prepares the input clip that will later
be used by the night-time lane detection code.

Requirements
-----------
- `yt-dlp` must be installed and accessible in your PATH.
- `ffmpeg` must be installed and accessible in your PATH.

You can install them for example via:

- `pip install yt-dlp`
- Download/Install ffmpeg from the official site or your package manager.
"""

import subprocess
from pathlib import Path
from datetime import datetime, timedelta


# ============================================================
#                CONFIGURATION
# ============================================================

# 1) YouTube URL of the full night-time video
YOUTUBE_URL = "https://youtu.be/obMbyVhYED8?si=DlYfkEN5BoU7f12T"

# 2) Local paths – always resolved relative to the project root
# Detect project root as the parent directory of this file's folder (i.e. "..")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Name for the 3-minute clipped video (within PROCESSED_DIR)
CLIPPED_VIDEO_NAME = "night_clip_3min.mp4"

# 3) Time configuration for clipping (HH:MM:SS)
#    Start time in the original YouTube video
CLIP_START = "02:33:30"
#    Duration of the clip – here: 3 minutes
CLIP_DURATION = "00:03:00"


# ============================================================
#                HELPER FUNCTIONS
# ============================================================

def run_cmd(cmd: list[str]) -> None:
    """Run a subprocess command and print it nicely."""
    print("\n[CMD]", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def _add_time(start: str, duration: str) -> str:
    """
    Helper: given start and duration in HH:MM:SS, return end time in HH:MM:SS.
    """
    t_start = datetime.strptime(start, "%H:%M:%S")
    h, m, s = map(int, duration.split(":"))
    end = t_start + timedelta(hours=h, minutes=m, seconds=s)
    return end.strftime("%H:%M:%S")


def download_clip_direct(url: str, output_path: Path, start: str, duration: str) -> None:
    """
    Download only a specific segment of a YouTube video using yt-dlp's
    `--download-sections` feature (internally uses ffmpeg).

    Args:
        url: Full YouTube URL.
        output_path: Target file path (including filename, typically .mp4).
        start: Start time in HH:MM:SS.
        duration: Duration in HH:MM:SS.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # yt-dlp uses a section syntax "*start-end" with absolute times.
    end = _add_time(start, duration)
    section = f"*{start}-{end}"

    template = str(output_path.with_suffix(""))

    cmd = [
        "yt-dlp",
        "-f",
        "mp4",
        "--download-sections",
        section,
        "-o",
        f"{template}.%(ext)s",
        url,
    ]
    run_cmd(cmd)


# ============================================================
#                     MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=== NIGHT VIDEO PRE-PROCESSING STARTED ===")

    clipped_video_path = PROCESSED_DIR / CLIPPED_VIDEO_NAME

    # Download ONLY the required 3-minute segment directly
    print(
        f"[INFO] Downloading 3-minute clip from {CLIP_START} for {CLIP_DURATION} "
        f"into {clipped_video_path}"
    )
    download_clip_direct(YOUTUBE_URL, clipped_video_path, CLIP_START, CLIP_DURATION)

    print("\n=== NIGHT VIDEO PRE-PROCESSING FINISHED SUCCESSFULLY ===")


