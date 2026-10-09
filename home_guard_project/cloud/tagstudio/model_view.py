"""What the AI saw of a clip: the exact pictures the box sent the vision model, for Tag · AI ("tag what the AI sees").

In this order, the first that exists:

1. **sent**: the frames the box sent, saved as ``vlm_crops/<camera>/<day>/<stem>_f<i>.jpg`` (box alert_clips.py
   ``teacher_record``; the meta's ``teacher.input_frames``). Exactly the bytes the model got.
2. **recipe**: the meta's ``model_input`` (or ``teacher.model_input``) record names the frames: ``frame_indices``
   into the saved crop clip (``vlm_input: crop``) or the whole clip (``whole_frame_fallback``).
3. **rendered**: the saved crop clip through the box's own ``render_model_input`` (fleet_contract/model_input.py,
   vendored verbatim) at the crop clip's fps (``vlm_crop.fps``) and 1 frame a second.
4. **whole**: the whole clip the same way at ``fps_estimated`` (a clip without a crop).

The Set-of-Mark overlay (P1, P2 drawn on the frames) is never drawn: the box does not draw it yet
(``render_model_input(overlay=None)``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

SENT, RECIPE, RENDERED, WHOLE = "sent", "recipe", "rendered", "whole"
SOURCE_TITLES = {SENT: "the frames the box sent", RECIPE: "the box's frames, picked from the saved crop",
                 RENDERED: "rendered like the box", WHOLE: "whole frames: no crop was saved"}
SAMPLE_FPS = 1.0   # config vlm.sample_fps, the box's default
MAX_FRAMES = 60


@dataclass
class ModelView:
    source: str                         # SENT / RECIPE / RENDERED / WHOLE
    frames: List[bytes]                 # JPEG bytes, in the order the model got them
    vlm_input: str                      # "crop" or "whole_frame_fallback"
    times: List[Optional[float]] = field(default_factory=list)
    size: Optional[Sequence[int]] = None
    sample_fps: float = SAMPLE_FPS
    record: Dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """"What the AI sees: crop · 1 fps · 10 frames · 384×384" (+ where the frames came from)."""
        kind = "crop" if self.vlm_input == "crop" else "whole frame"
        fps = f"{self.sample_fps:g} fps"
        bits = [kind, fps, f"{len(self.frames)} frame{'s' if len(self.frames) != 1 else ''}"]
        if self.size:
            bits.append(f"{int(self.size[0])}×{int(self.size[1])}")
        return "What the AI sees: " + " · ".join(bits)

    def as_dict(self) -> Dict[str, Any]:
        import base64  # noqa: PLC0415

        return {"label": self.label, "source": self.source, "source_title": SOURCE_TITLES[self.source],
                "vlm_input": self.vlm_input, "times": list(self.times), "size": list(self.size) if self.size else None,
                "sample_fps": self.sample_fps, "record": self.record,
                "frames": [base64.b64encode(f).decode("ascii") for f in self.frames]}


def recipe(meta: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The meta's model-input record (top level, else the teacher's), when it names frames."""
    for rec in (meta.get("model_input"), (meta.get("teacher") or {}).get("model_input")):
        if isinstance(rec, dict) and isinstance(rec.get("frame_indices"), list):
            return rec
    return None


def read_frames(path: str) -> List[Any]:
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while len(frames) < 100_000:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def _jpeg_size(data: bytes) -> Optional[Sequence[int]]:
    try:
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return (int(img.shape[1]), int(img.shape[0])) if img is not None else None
    except Exception:  # noqa: BLE001 - the size is a caption only
        return None


def _number(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else None


def build(meta: Mapping[str, Any], sent: Sequence[bytes] = (), crop_path: Optional[str] = None,
          clip_path: Optional[str] = None, frames_of: Callable[[str], List[Any]] = read_frames) -> Optional[ModelView]:
    """The model's view of one clip (see the module docstring), or None when nothing can be shown."""
    from ...fleet_contract.model_input import (VLM_INPUT_CROP, VLM_INPUT_WHOLE, ModelInputConfig,  # noqa: PLC0415
                                               encode_jpeg, render_model_input)

    rec = recipe(meta) or {}
    sample_fps = _number(rec.get("sample_fps")) or SAMPLE_FPS
    vlm_input = str(rec.get("vlm_input") or meta.get("vlm_input") or VLM_INPUT_CROP)
    sent = [bytes(b) for b in sent if b][:MAX_FRAMES]
    if sent:
        size = rec.get("size") or _jpeg_size(sent[0])
        return ModelView(SENT, sent, vlm_input, list(rec.get("times") or [None] * len(sent))[:len(sent)], size,
                         sample_fps, rec)
    crop_ok = bool(crop_path) and os.path.isfile(crop_path)
    clip_ok = bool(clip_path) and os.path.isfile(clip_path)
    if rec:
        source_path = crop_path if vlm_input == VLM_INPUT_CROP else clip_path
        if source_path and os.path.isfile(source_path):
            frames = frames_of(source_path)
            picked = [frames[i] for i in rec["frame_indices"] if isinstance(i, int) and 0 <= i < len(frames)]
            if picked:
                data = [d for d in (encode_jpeg(f) for f in picked[:MAX_FRAMES]) if d]
                size = rec.get("size") or (picked[0].shape[1], picked[0].shape[0])
                return ModelView(RECIPE, data, vlm_input, list(rec.get("times") or [])[:len(data)], size,
                                 sample_fps, rec)
    if crop_ok:
        fps = _number((meta.get("vlm_crop") or {}).get("fps")) or _number(meta.get("fps_estimated")) or 5.0
        rendered = render_model_input(frames_of(crop_path), {"vlm_input": VLM_INPUT_CROP, "fps": fps},
                                      ModelInputConfig(sample_fps))
        if rendered.frames:
            return ModelView(RENDERED, rendered.jpegs()[:MAX_FRAMES], rendered.vlm_input, rendered.times,
                             rendered.size, sample_fps, {k: v for k, v in rendered.record().items() if k != "crops"})
    if clip_ok:
        fps = _number(meta.get("fps_estimated")) or 7.0
        rendered = render_model_input(frames_of(clip_path), {"vlm_input": VLM_INPUT_WHOLE, "fps": fps},
                                      ModelInputConfig(sample_fps))
        if rendered.frames:
            return ModelView(WHOLE, rendered.jpegs()[:MAX_FRAMES], rendered.vlm_input, rendered.times,
                             rendered.size, sample_fps, {k: v for k, v in rendered.record().items() if k != "crops"})
    return None


def local_sent_frames(meta: Mapping[str, Any], meta_path: str) -> List[bytes]:
    """The saved ``_f<i>.jpg`` frames a local copy's meta lists (``teacher.input_frames``, relative to the copy's root:
    ``<root>/meta/<camera>/<day>/<stem>.meta.json``); [] when any is missing (never a partial set)."""
    rels = (meta.get("teacher") or {}).get("input_frames") or []
    if not rels or not meta_path:
        return []
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(meta_path)))))
    out = []
    for rel in rels:
        path = os.path.join(root, *str(rel).replace("\\", "/").split("/"))
        try:
            with open(path, "rb") as f:
                out.append(f.read())
        except OSError:
            return []
    return out
