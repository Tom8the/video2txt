from __future__ import annotations

import subprocess
from pathlib import Path

from video2txt.config import OCRSettings


def extract_subtitle_frames(
    source: Path,
    output_dir: Path,
    settings: OCRSettings,
    *,
    ffmpeg_path: str = "ffmpeg",
) -> list[Path]:
    """Sample and crop the configured subtitle band with FFmpeg."""
    if not source.is_file():
        raise FileNotFoundError(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    for previous in output_dir.glob("frame-*.jpg"):
        previous.unlink()

    crop_height = settings.crop_bottom - settings.crop_top
    filters = (
        f"crop=iw:trunc(ih*{crop_height:.6f}/2)*2:0:trunc(ih*{settings.crop_top:.6f}/2)*2,"
        f"scale=iw*{settings.scale}:ih*{settings.scale}:flags=lanczos,"
        f"fps=1/{settings.sample_interval:.6f}"
    )
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-an",
        "-sn",
        "-dn",
        "-vf",
        filters,
        "-frames:v",
        str(settings.max_frames),
        "-q:v",
        "2",
        str(output_dir / "frame-%06d.jpg"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        detail = result.stderr.strip() or "FFmpeg failed to extract OCR frames"
        raise RuntimeError(detail)
    return sorted(output_dir.glob("frame-*.jpg"))
