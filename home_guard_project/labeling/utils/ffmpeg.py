"""FFmpeg detection, probe, and video re-encoding to H.264."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from tqdm import tqdm

log = logging.getLogger("labeling.ffmpeg")


# ---------------------------------------------------------------------------
# FFmpeg / FFprobe discovery
# ---------------------------------------------------------------------------

def detect_ffmpeg(explicit_path: Optional[str] = None) -> Optional[str]:
    """
    Try to find the ``ffmpeg`` executable.

    Search order: *explicit_path* -> ``PATH`` -> common Windows locations.
    """
    if explicit_path:
        p = os.path.expandvars(os.path.expanduser(explicit_path))
        if os.path.isfile(p):
            return p

    which = shutil.which("ffmpeg")
    if which:
        return which

    candidates: List[str] = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\FFmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\Gyan\FFmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\Gyan\FFmpeg\bin\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
    ]
    user = os.environ.get("USERPROFILE")
    if user:
        candidates.append(os.path.join(user, "scoop", "shims", "ffmpeg.exe"))

    for c in candidates:
        if os.path.isfile(c):
            return c

    local_app = os.environ.get("LOCALAPPDATA")
    if local_app:
        pkg_root = os.path.join(local_app, "Microsoft", "WinGet", "Packages")
        if os.path.isdir(pkg_root):
            try:
                from pathlib import Path

                for p in Path(pkg_root).rglob("ffmpeg.exe"):
                    if "bin" in {part.lower() for part in p.parts}:
                        return str(p)
                for p in Path(pkg_root).rglob("ffmpeg.exe"):
                    return str(p)
            except Exception:
                pass
    return None


def derive_ffprobe(ffmpeg_path: str) -> Optional[str]:
    """Derive the ``ffprobe`` path from a known ``ffmpeg`` path."""
    dirname = os.path.dirname(ffmpeg_path)
    basename = os.path.basename(ffmpeg_path)
    probe_name = basename.replace("ffmpeg", "ffprobe")
    candidate = os.path.join(dirname, probe_name)
    if os.path.isfile(candidate):
        return candidate
    return shutil.which("ffprobe")


# ---------------------------------------------------------------------------
# Codec detection
# ---------------------------------------------------------------------------

def is_h264(filepath: str, ffprobe: Optional[str] = None) -> bool:
    """Return *True* if *filepath* is already encoded with H.264 (AVC)."""
    probe = ffprobe
    if not probe:
        ffmpeg = detect_ffmpeg()
        if ffmpeg:
            probe = derive_ffprobe(ffmpeg)
        else:
            probe = shutil.which("ffprobe")
    if not probe:
        return False

    cmd = [
        probe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "csv=p=0",
        filepath,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout.strip().lower() == "h264"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Re-encoding
# ---------------------------------------------------------------------------

def _reencode_one(
    src: str,
    ffmpeg: str,
    ffprobe: Optional[str],
    skip_reencoded: bool,
) -> str:
    """Re-encode a single clip. Returns ``"ok"``, ``"skipped"``, or ``"failed"``."""
    if skip_reencoded and is_h264(src, ffprobe):
        return "skipped"

    tmp = src.replace(".mp4", "_h264.mp4")
    cmd = [
        ffmpeg, "-y", "-i", src,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-movflags", "+faststart",
        "-an",
        tmp,
    ]
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        os.replace(tmp, src)
        return "ok"
    except subprocess.CalledProcessError:
        if os.path.exists(tmp):
            os.remove(tmp)
        return "failed"


def reencode_videos(
    clip_paths: List[str],
    *,
    ffmpeg_path: Optional[str] = None,
    skip_reencoded: bool = True,
    workers: Optional[int] = None,
) -> Dict[str, int]:
    """
    Re-encode mp4v clips to H.264 in-place using ffmpeg.

    Returns ``{encoded, skipped, failed}``.
    """
    ffmpeg = detect_ffmpeg(ffmpeg_path)
    if not ffmpeg:
        log.error("ffmpeg not found. Install it or pass --ffmpeg /path/to/ffmpeg")
        return {"encoded": 0, "skipped": 0, "failed": 0}

    ffprobe: Optional[str] = derive_ffprobe(ffmpeg)
    total = len(clip_paths)
    if total == 0:
        log.info("No MP4 files found to re-encode.")
        return {"encoded": 0, "skipped": 0, "failed": 0}

    max_workers = workers or min(os.cpu_count() or 4, 8)
    log.info(
        "Re-encoding %d clips (%d workers, %s)",
        total, max_workers,
        "skip already H.264" if skip_reencoded else "force all",
    )

    skipped = encoded = failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_reencode_one, src, ffmpeg, ffprobe, skip_reencoded): src
            for src in clip_paths
        }
        pbar = tqdm(as_completed(futures), total=total, desc="Re-encoding", unit="clip")
        for future in pbar:
            result = future.result()
            if result == "ok":
                encoded += 1
            elif result == "skipped":
                skipped += 1
            else:
                failed += 1
            pbar.set_postfix(encoded=encoded, skipped=skipped, failed=failed)

    log.info(
        "Re-encode complete: %d encoded, %d skipped (already H.264), %d failed",
        encoded, skipped, failed,
    )
    return {"encoded": encoded, "skipped": skipped, "failed": failed}
