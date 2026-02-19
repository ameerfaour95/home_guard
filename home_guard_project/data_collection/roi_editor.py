"""
Interactive ROI polygon editor.

Opens a live camera frame and lets the user draw a polygon by clicking
vertices.  The result is saved to zones.yaml as normalised (0-1)
coordinates.

Controls:
  Left-click   — add a vertex
  Right-click  — undo last vertex
  'r'          — reset all vertices
  Enter        — confirm polygon
  Escape       — skip / cancel this camera

Runnable standalone:  py home_guard_project/data_collection/roi_editor.py
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

log = logging.getLogger(__name__)

_DIR = os.path.dirname(os.path.abspath(__file__))
_ZONES_PATH = os.path.join(_DIR, "zones.yaml")

_HEADER = (
    "# ──────────────────────────────────────────────────────────────────────────────\n"
    "#  ROI zones — normalised polygon vertices per camera.\n"
    "#  Cameras without entries use full-frame detection.\n"
    "#  DO NOT COMMIT (deployment-specific)\n"
    "# ──────────────────────────────────────────────────────────────────────────────\n\n"
)


# ─────────────────────────────────────────────────────────────────────────────
# YAML helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_zones(path: str = _ZONES_PATH) -> Dict[str, List[List[float]]]:
    """Load zones.yaml and return {camera_name: [[x,y], ...]}."""
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("zones", {})
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, list)}


def save_zones(
    zones: Dict[str, List[List[float]]], path: str = _ZONES_PATH,
) -> None:
    """Write zones dict to zones.yaml."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(_HEADER)
        yaml.dump({"zones": zones}, f, default_flow_style=None, allow_unicode=True)
    log.info("Wrote %d zone(s) to %s", len(zones), path)


# ─────────────────────────────────────────────────────────────────────────────
# OpenCV polygon editor
# ─────────────────────────────────────────────────────────────────────────────

def _draw_overlay(
    base: np.ndarray,
    points_px: List[Tuple[int, int]],
    closed: bool = False,
) -> np.ndarray:
    """Draw current polygon state on a copy of *base*."""
    vis = base.copy()
    if not points_px:
        cv2.putText(vis, "Click to add vertices | R=reset | Enter=confirm | Esc=skip",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return vis

    pts = np.array(points_px, dtype=np.int32)

    if closed and len(pts) >= 3:
        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

    color = (0, 255, 0) if closed else (0, 200, 255)
    if len(pts) >= 2:
        cv2.polylines(vis, [pts], isClosed=closed, color=color, thickness=2)

    for i, (px, py) in enumerate(points_px):
        cv2.circle(vis, (px, py), 5, (0, 0, 255), -1)
        cv2.putText(vis, str(i + 1), (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    label = "Enter to confirm" if len(pts) >= 3 else f"Need {3 - len(pts)} more point(s)"
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return vis


def edit_zone_on_frame(
    camera_name: str, frame: np.ndarray,
) -> Optional[List[List[float]]]:
    """
    Open an OpenCV window for drawing a polygon on *frame*.

    Returns normalised [[x,y], ...] or None if cancelled.
    """
    h, w = frame.shape[:2]
    points_px: List[Tuple[int, int]] = []
    confirmed = False
    window = f"ROI Editor — {camera_name}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        nonlocal points_px
        if event == cv2.EVENT_LBUTTONDOWN:
            points_px.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points_px:
            points_px.pop()

    cv2.setMouseCallback(window, on_mouse)

    while True:
        vis = _draw_overlay(frame, points_px, closed=False)
        cv2.imshow(window, vis)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # Escape
            break
        elif key == ord("r"):
            points_px.clear()
        elif key in (13, 10):  # Enter
            if len(points_px) >= 3:
                preview = _draw_overlay(frame, points_px, closed=True)
                cv2.imshow(window, preview)
                cv2.waitKey(1500)
                confirmed = True
                break

    cv2.destroyWindow(window)
    cv2.waitKey(1)

    if not confirmed:
        return None

    return [[round(px / w, 6), round(py / h, 6)] for px, py in points_px]


# ─────────────────────────────────────────────────────────────────────────────
# Interactive CLI flow
# ─────────────────────────────────────────────────────────────────────────────

def _prompt(msg: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"{msg}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)
    return val or default


def _prompt_yn(msg: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    val = _prompt(f"{msg} [{hint}]")
    if not val:
        return default
    return val.lower().startswith("y")


def interactive_roi_setup(cameras: Dict[str, str]) -> Dict[str, List[List[float]]]:
    """
    Full interactive flow: grab one frame per camera, let user draw ROI.
    Returns the zones dict (camera_name -> normalised polygon).
    """
    zones: Dict[str, List[List[float]]] = {}

    cam_names = list(cameras.keys())
    print(f"\nConfigured cameras ({len(cam_names)}):")
    for i, name in enumerate(cam_names, 1):
        print(f"  [{i}] {name}")
    print(f"  [a] All cameras")

    sel = _prompt("Select cameras to define ROI for", "a")
    if sel.lower() == "a":
        selected = cam_names
    else:
        indices = [s.strip() for s in sel.replace(",", " ").split()]
        selected = []
        for s in indices:
            try:
                selected.append(cam_names[int(s) - 1])
            except (IndexError, ValueError):
                pass
        if not selected:
            selected = cam_names

    for name in selected:
        rtsp = cameras[name]
        print(f"\n  Opening {name}...")

        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp|stimeout;5000000",
        )
        cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
        frame = None
        for _ in range(60):
            ret, f = cap.read()
            if ret and f is not None:
                frame = f
                break
        cap.release()

        if frame is None:
            print(f"  Could not read frame from {name} — skipping.")
            continue

        if max(frame.shape[:2]) > 900:
            scale = 900.0 / max(frame.shape[:2])
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        polygon = edit_zone_on_frame(name, frame)
        if polygon is not None:
            zones[name] = polygon
            print(f"  {name}: {len(polygon)} vertices saved.")
        else:
            print(f"  {name}: skipped (no ROI).")

    return zones


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cameras_path = os.path.join(_DIR, "cameras.yaml")
    if not os.path.isfile(cameras_path):
        print("cameras.yaml not found. Run camera discovery first.")
        sys.exit(1)

    with open(cameras_path, "r", encoding="utf-8") as f:
        cam_data = yaml.safe_load(f) or {}
    cameras = cam_data.get("cameras", {})
    if not cameras:
        print("No cameras in cameras.yaml.")
        sys.exit(1)

    zones = interactive_roi_setup(cameras)
    if not zones:
        print("\nNo zones defined.")
        return

    print(f"\n── Summary: {len(zones)} zone(s) ──")
    for name, pts in zones.items():
        print(f"  {name}: {len(pts)} vertices")

    if _prompt_yn("\nSave to zones.yaml?"):
        save_zones(zones)
        print("Done.")
    else:
        print("Cancelled.")


if __name__ == "__main__":
    main()
