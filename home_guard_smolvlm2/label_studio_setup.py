"""
Label Studio setup for security camera data annotation.

Generates:
  1. A labeling config XML  (dataset_multi/label_studio_config.xml)
  2. A tasks JSON file       (dataset_multi/label_studio_tasks.json)

Each task corresponds to one video clip and includes:
  - Video playback
  - VideoRectangle pre-annotations from YOLO weak labels
  - TextArea pre-filled with existing VLM responses

Usage:
    python -m home_guard_smolvlm2.label_studio_setup
    python -m home_guard_smolvlm2.label_studio_setup --dataset-dir ./dataset_multi --limit 50
    python -m home_guard_smolvlm2.label_studio_setup --cameras main_door,back_door --kinds trigger
    python -m home_guard_smolvlm2.label_studio_setup --reencode   # re-encode mp4v -> H.264
    python -m home_guard_smolvlm2.label_studio_setup --serve      # start file server on port 8081
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
from typing import Any, Dict, List, Optional

# Default port for the file server (separate from Label Studio on 8080)
FILE_SERVER_PORT = 8081


# ---------------------------------------------------------------------------
# COCO class-id -> label name (subset relevant for security cameras)
# ---------------------------------------------------------------------------
COCO_LABELS: Dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    14: "bird",
    15: "cat",
    16: "dog",
}

# All label names used in the XML config (keep in sync with COCO_LABELS values)
ALL_LABEL_NAMES: List[str] = sorted(set(COCO_LABELS.values()))

# ---------------------------------------------------------------------------
# Label Studio XML config
# ---------------------------------------------------------------------------
LABEL_CONFIG_XML = """\
<View>
  <Header value="$header_info"/>
  <Video name="video" value="$video_url" frameRate="$fps"/>

  <VideoRectangle name="bbox" toName="video">
    <Labels name="label" toName="bbox">
{label_tags}
    </Labels>
  </VideoRectangle>

  <Header value="Scene Description (for VLM training)"/>
  <TextArea name="vlm_description" toName="video"
            rows="4" editable="true"
            placeholder="Describe what happens in the video..."/>
</View>
"""


def _build_label_config() -> str:
    """Return the complete Label Studio XML labeling config."""
    tags = "\n".join(
        f'    <Label value="{name}"/>' for name in ALL_LABEL_NAMES
    )
    return LABEL_CONFIG_XML.format(label_tags=tags)


# ---------------------------------------------------------------------------
# YOLO -> Label Studio coordinate conversion
# ---------------------------------------------------------------------------

def _yolo_to_ls_rect(
    xc: float, yc: float, w: float, h: float,
) -> Dict[str, float]:
    """
    Convert YOLO normalized (0-1) center-xywh to Label Studio
    percentage (0-100) top-left-xywh.
    """
    return {
        "x": max(0.0, (xc - w / 2.0)) * 100.0,
        "y": max(0.0, (yc - h / 2.0)) * 100.0,
        "width": w * 100.0,
        "height": h * 100.0,
    }


def _read_yolo_label_file(path: str) -> List[Dict[str, Any]]:
    """
    Parse a YOLO label .txt file.

    Returns list of dicts:  {class_id: int, xc, yc, w, h}
    """
    detections: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                detections.append({
                    "class_id": int(parts[0]),
                    "xc": float(parts[1]),
                    "yc": float(parts[2]),
                    "w": float(parts[3]),
                    "h": float(parts[4]),
                })
    except FileNotFoundError:
        pass
    return detections


# ---------------------------------------------------------------------------
# Build predictions for one task from meta.json data
# ---------------------------------------------------------------------------

def _build_predictions(
    meta: Dict[str, Any],
    dataset_dir: str,
) -> List[Dict[str, Any]]:
    """
    Build a Label Studio ``predictions`` list for a single task.

    Includes:
      - VideoRectangle results from YOLO exported frames
      - TextArea result from VLM model_response
    """
    results: List[Dict[str, Any]] = []
    result_idx = 0

    # --- YOLO box predictions (VideoRectangle) ---
    yolo_export = meta.get("yolo_export")
    if yolo_export and yolo_export.get("enabled"):
        exported_frames: List[Dict[str, Any]] = yolo_export.get("exported_frames", [])
        for ef in exported_frames:
            frame_index: int = ef.get("frame_index", 0)
            label_rel: str = ef.get("label_path", "")
            if not label_rel:
                continue

            # Resolve label file on disk
            label_path = os.path.join(dataset_dir, label_rel.replace("\\", "/"))
            detections = _read_yolo_label_file(label_path)

            for det in detections:
                class_id = det["class_id"]
                label_name = COCO_LABELS.get(class_id)
                if label_name is None:
                    # Unknown class – skip (or use generic name)
                    label_name = f"class_{class_id}"

                rect = _yolo_to_ls_rect(det["xc"], det["yc"], det["w"], det["h"])

                # Video frame is 1-indexed in Label Studio
                video_frame = frame_index + 1
                time_offset = ef.get("approx_time_offset_sec", 0.0)

                results.append({
                    "id": f"bbox_{result_idx}",
                    "type": "videorectangle",
                    "from_name": "bbox",
                    "to_name": "video",
                    "value": {
                        "labels": [label_name],
                        "sequence": [
                            {
                                "frame": video_frame,
                                "x": round(rect["x"], 4),
                                "y": round(rect["y"], 4),
                                "width": round(rect["width"], 4),
                                "height": round(rect["height"], 4),
                                "time": round(time_offset, 4),
                                "enabled": True,
                                "rotation": 0,
                            }
                        ],
                    },
                })
                result_idx += 1

    # --- VLM text prediction (TextArea) ---
    vlm_text = meta.get("model_response", "")
    if vlm_text:
        results.append({
            "id": f"vlm_text_{result_idx}",
            "type": "textarea",
            "from_name": "vlm_description",
            "to_name": "video",
            "value": {
                "text": [vlm_text],
            },
        })

    return results


# ---------------------------------------------------------------------------
# Scan dataset and build tasks
# ---------------------------------------------------------------------------

def _iter_meta_entries(
    dataset_dir: str,
    *,
    cameras: Optional[List[str]] = None,
    kinds: Optional[List[str]] = None,
    limit: Optional[int] = None,
):
    """
    Yield (meta_path, meta_dict) for meta files under dataset_dir/meta,
    filtered by cameras/kinds and truncated by limit.
    """
    meta_root = os.path.join(dataset_dir, "meta")
    if not os.path.isdir(meta_root):
        return

    seen = 0
    for dirpath, _dirnames, filenames in os.walk(meta_root):
        for fname in sorted(filenames):
            if not fname.endswith(".meta.json"):
                continue

            meta_path = os.path.join(dirpath, fname)
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta: Dict[str, Any] = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"[WARN] Skipping {meta_path}: {exc}", file=sys.stderr)
                continue

            camera_name: str = meta.get("camera_name", "unknown")
            kind: str = meta.get("kind", "unknown")

            if cameras and camera_name not in cameras:
                continue
            if kinds and kind not in kinds:
                continue

            yield meta_path, meta

            seen += 1
            if limit and seen >= limit:
                return


def scan_and_build_tasks(
    dataset_dir: str,
    *,
    cameras: Optional[List[str]] = None,
    kinds: Optional[List[str]] = None,
    limit: Optional[int] = None,
    include_predictions: bool = True,
    video_base_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Walk ``dataset_dir/meta/`` and build Label Studio tasks.

    Args:
        video_base_url: If provided, video URLs use this as the base
            (e.g. ``http://localhost:8081``).  Otherwise falls back to
            Label Studio local-files serving (``/data/local-files/?d=…``).

    Returns a list of task dicts ready for JSON export.
    """
    tasks: List[Dict[str, Any]] = []
    meta_root = os.path.join(dataset_dir, "meta")
    if not os.path.isdir(meta_root):
        print(f"[ERROR] Meta directory not found: {meta_root}", file=sys.stderr)
        return []

    for meta_path, meta in _iter_meta_entries(dataset_dir, cameras=cameras, kinds=kinds, limit=limit):
        camera_name: str = meta.get("camera_name", "unknown")
        kind: str = meta.get("kind", "unknown")

        # clip_path in meta is relative to dataset_dir with OS separators.
        clip_rel = meta.get("clip_path", "").replace("\\", "/")
        if not clip_rel:
            continue

        # Verify clip file exists on disk
        clip_abs = os.path.join(dataset_dir, clip_rel)
        if not os.path.isfile(clip_abs):
            print(f"[WARN] Clip not found, skipping: {clip_abs}", file=sys.stderr)
            continue

        if video_base_url:
            video_url = f"{video_base_url.rstrip('/')}/{clip_rel}"
        else:
            video_url = f"/data/local-files/?d={clip_rel}"

        fps = meta.get("fps_estimated", meta.get("buffer", {}).get("store_fps", 10.0))
        start_local = meta.get("clip_start_local", "?")
        end_local = meta.get("clip_end_local", "?")
        # Only show the time portion if dates are the same
        end_display = end_local.split(" ")[-1] if " " in end_local else end_local

        header = f"{camera_name} | {kind} | {start_local} - {end_display}"

        task: Dict[str, Any] = {
            "data": {
                "video_url": video_url,
                "fps": fps,
                "header_info": header,
                # Extra fields for filtering / reference in LS
                "camera_name": camera_name,
                "kind": kind,
                "clip_start_local": start_local,
                "clip_end_local": end_local,
                "duration_sec": meta.get("duration_sec", 0),
                "meta_path": os.path.relpath(meta_path, start=dataset_dir).replace("\\", "/"),
            },
        }

        if include_predictions:
            preds = _build_predictions(meta, dataset_dir)
            if preds:
                task["predictions"] = [
                    {
                        "model_version": "yolov8n-weak-labels",
                        "result": preds,
                    }
                ]

        tasks.append(task)

    return tasks


# ---------------------------------------------------------------------------
# Re-encode videos to H.264
# ---------------------------------------------------------------------------

def _detect_ffmpeg(explicit_path: Optional[str] = None) -> Optional[str]:
    """
    Try to find ffmpeg executable.
    - Uses explicit_path if provided
    - Then checks PATH (shutil.which)
    - Then checks common Windows install locations (winget/choco/manual)
    """
    if explicit_path:
        p = os.path.expandvars(os.path.expanduser(explicit_path))
        if os.path.isfile(p):
            return p

    which = shutil.which("ffmpeg")
    if which:
        return which

    candidates: List[str] = []
    # Common manual installs
    candidates += [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\FFmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\Gyan\FFmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\Gyan\FFmpeg\bin\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
    ]
    # Scoop default shim
    user = os.environ.get("USERPROFILE")
    if user:
        candidates.append(os.path.join(user, "scoop", "shims", "ffmpeg.exe"))

    for c in candidates:
        if os.path.isfile(c):
            return c

    # winget often installs to:
    #   %LOCALAPPDATA%\\Microsoft\\WinGet\\Packages\\...\\bin\\ffmpeg.exe
    local_app = os.environ.get("LOCALAPPDATA")
    if local_app:
        pkg_root = os.path.join(local_app, "Microsoft", "WinGet", "Packages")
        if os.path.isdir(pkg_root):
            try:
                from pathlib import Path

                for p in Path(pkg_root).rglob("ffmpeg.exe"):
                    # Prefer binaries under a bin/ directory
                    if "bin" in {part.lower() for part in p.parts}:
                        return str(p)
                # Fallback to any ffmpeg.exe found
                for p in Path(pkg_root).rglob("ffmpeg.exe"):
                    return str(p)
            except Exception:
                pass
    return None


def _collect_clip_abs_paths_for_subset(
    dataset_dir: str,
    *,
    cameras: Optional[List[str]] = None,
    kinds: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> List[str]:
    """
    Collect clip absolute paths for the same subset implied by cameras/kinds/limit.
    This makes --reencode consistent with the tasks you generate.
    """
    clips: List[str] = []
    for _meta_path, meta in _iter_meta_entries(dataset_dir, cameras=cameras, kinds=kinds, limit=limit):
        clip_rel = meta.get("clip_path", "").replace("\\", "/")
        if not clip_rel:
            continue
        clip_abs = os.path.join(dataset_dir, clip_rel)
        if os.path.isfile(clip_abs):
            clips.append(clip_abs)
    return clips


def reencode_videos(
    dataset_dir: str,
    *,
    cameras: Optional[List[str]] = None,
    kinds: Optional[List[str]] = None,
    limit: Optional[int] = None,
    ffmpeg_path: Optional[str] = None,
) -> None:
    """
    Re-encode mp4v clips to H.264 in-place using ffmpeg.

    The original file is replaced.  A temporary file with suffix
    ``_h264.mp4`` is written first.
    """
    ffmpeg = _detect_ffmpeg(ffmpeg_path)
    if not ffmpeg:
        print("[ERROR] ffmpeg not found.", file=sys.stderr)
        print(
            "Install ffmpeg and ensure it's on PATH, or pass --ffmpeg C:\\path\\to\\ffmpeg.exe",
            file=sys.stderr,
        )
        return

    mp4_files = _collect_clip_abs_paths_for_subset(
        dataset_dir,
        cameras=cameras,
        kinds=kinds,
        limit=limit,
    )

    total = len(mp4_files)
    print(f"[reencode] Found {total} MP4 files to process (subset).")

    for i, src in enumerate(mp4_files, 1):
        tmp = src.replace(".mp4", "_h264.mp4")
        cmd = [
            ffmpeg, "-y", "-i", src,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-an",  # no audio in security cam clips
            tmp,
        ]
        print(f"  [{i}/{total}] {os.path.basename(src)} ... ", end="", flush=True)
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            os.replace(tmp, src)
            print("OK")
        except subprocess.CalledProcessError:
            print("FAILED")
            # Clean up temp file
            if os.path.exists(tmp):
                os.remove(tmp)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_tasks(tasks: List[Dict[str, Any]], output_path: str) -> None:
    """Write the tasks list as a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    print(f"[export] Wrote {len(tasks)} tasks to {output_path}")


def write_config(output_path: str) -> None:
    """Write the Label Studio XML labeling config."""
    config = _build_label_config()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(config)
    print(f"[config] Wrote labeling config to {output_path}")


# ---------------------------------------------------------------------------
# Simple CORS-enabled file server (bypasses Label Studio local-files issues)
# ---------------------------------------------------------------------------

class _CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that adds CORS headers so Label Studio can fetch videos."""

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        # Quieter logging — only show errors
        if args and str(args[0]).startswith("2"):
            return
        super().log_message(format, *args)


def start_file_server(
    dataset_dir: str,
    port: int = FILE_SERVER_PORT,
    *,
    background: bool = False,
) -> None:
    """
    Start a simple HTTP server rooted at *dataset_dir* with CORS headers.

    When *background* is True the server runs in a daemon thread and
    this function returns immediately.
    """
    abs_dir = os.path.abspath(dataset_dir)
    handler = functools.partial(_CORSRequestHandler, directory=abs_dir)
    server = http.server.HTTPServer(("0.0.0.0", port), handler)

    def _run() -> None:
        print(f"[serve] File server running at http://localhost:{port}")
        print(f"[serve] Serving files from: {abs_dir}")
        print(f"[serve] Test URL: http://localhost:{port}/clips/")
        server.serve_forever()

    if background:
        t = threading.Thread(target=_run, daemon=True)
        t.start()
    else:
        _run()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_setup_instructions(dataset_dir: str, *, use_file_server: bool = False, file_server_port: int = FILE_SERVER_PORT) -> None:
    abs_dataset = os.path.abspath(dataset_dir)
    print()
    print("=" * 70)
    print("  LABEL STUDIO SETUP INSTRUCTIONS")
    print("=" * 70)
    print()
    print("1. Install Label Studio (if not already):")
    print("     pip install label-studio")
    print()

    if use_file_server:
        print(f"2. Video files are served via the built-in file server on port {file_server_port}.")
        print(f"   Task URLs already point to http://localhost:{file_server_port}/...")
        print()
        print("   To start the file server (keep this running in a separate terminal):")
        print(f"     py -m home_guard_smolvlm2.label_studio_setup --serve --dataset-dir {dataset_dir}")
        print()
        print("3. Start Label Studio (in another terminal):")
        print("     label-studio start")
    else:
        print("2. Set environment variables (before starting LS):")
        print()
        if sys.platform == "win32":
            print("   Windows cmd.exe / PowerShell:")
            print("     set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true")
            print(f"     set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={abs_dataset}")
            print()
            print("   Git Bash / MSYS2:")
            print("     export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true")
            print(f"     export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=\"{abs_dataset.replace(os.sep, '/')}\"")
        else:
            print("     export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true")
            print(f"     export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={abs_dataset}")
        print()
        print("3. Start Label Studio:")
        print("     label-studio start")

    print()
    print("4. Create a new project in the Label Studio UI.")
    print()
    print("5. Go to project Settings -> Labeling Interface -> Code,")
    print(f"   and paste the contents of:")
    print(f"     {os.path.join(abs_dataset, 'label_studio_config.xml')}")
    print()
    print("6. Import the tasks file:")
    print(f"   - Go to the project, click 'Import'")
    print(f"   - Upload: {os.path.join(abs_dataset, 'label_studio_tasks.json')}")
    print()
    print("7. Start annotating! Each task shows:")
    print("   - Video player for the clip")
    print("   - Pre-loaded YOLO bounding boxes (VideoRectangle)")
    print("   - Pre-filled VLM description (editable TextArea)")
    print()
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Label Studio config and tasks from security camera dataset.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="./dataset_multi",
        help="Path to dataset_multi directory (default: ./dataset_multi)",
    )
    parser.add_argument(
        "--reencode",
        action="store_true",
        help="Re-encode MP4 clips from mp4v to H.264 (requires ffmpeg)",
    )
    parser.add_argument(
        "--ffmpeg",
        default=None,
        help="Optional path to ffmpeg executable (e.g. C:\\\\ffmpeg\\\\bin\\\\ffmpeg.exe)",
    )
    parser.add_argument(
        "--cameras",
        default=None,
        help="Comma-separated camera names to include (e.g. main_door,back_door)",
    )
    parser.add_argument(
        "--kinds",
        default=None,
        help="Comma-separated clip kinds to include (e.g. trigger,random)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tasks to generate (for testing)",
    )
    parser.add_argument(
        "--no-predictions",
        action="store_true",
        help="Skip generating pre-annotations (YOLO boxes + VLM text)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start a CORS-enabled file server for video files (recommended)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=FILE_SERVER_PORT,
        help=f"Port for the file server (default: {FILE_SERVER_PORT})",
    )

    args = parser.parse_args()

    dataset_dir: str = args.dataset_dir
    if not os.path.isdir(dataset_dir):
        print(f"[ERROR] Dataset directory not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    # If --serve only, just start the file server and block
    if args.serve:
        start_file_server(dataset_dir, port=args.port, background=False)
        return

    camera_filter: Optional[List[str]] = None
    if args.cameras:
        camera_filter = [c.strip() for c in args.cameras.split(",") if c.strip()]

    kind_filter: Optional[List[str]] = None
    if args.kinds:
        kind_filter = [k.strip() for k in args.kinds.split(",") if k.strip()]

    # Determine video URL base.
    # Always use external file server — avoids Label Studio local-files issues.
    video_base_url = f"http://localhost:{args.port}"

    # Step 1: Write labeling config
    config_path = os.path.join(dataset_dir, "label_studio_config.xml")
    write_config(config_path)

    # Step 2: Optionally re-encode videos
    if args.reencode:
        reencode_videos(
            dataset_dir,
            cameras=camera_filter,
            kinds=kind_filter,
            limit=args.limit,
            ffmpeg_path=args.ffmpeg,
        )

    # Step 3: Scan and build tasks
    print("[scan] Scanning meta files...")
    tasks = scan_and_build_tasks(
        dataset_dir,
        cameras=camera_filter,
        kinds=kind_filter,
        limit=args.limit,
        include_predictions=not args.no_predictions,
        video_base_url=video_base_url,
    )
    print(f"[scan] Built {len(tasks)} tasks.")

    if not tasks:
        print("[WARN] No tasks generated. Check filters and dataset directory.", file=sys.stderr)
        sys.exit(0)

    # Step 4: Export tasks JSON
    tasks_path = os.path.join(dataset_dir, "label_studio_tasks.json")
    export_tasks(tasks, tasks_path)

    # Step 5: Print instructions
    _print_setup_instructions(dataset_dir, use_file_server=True, file_server_port=args.port)


if __name__ == "__main__":
    main()
