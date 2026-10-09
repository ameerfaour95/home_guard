"""What the vision model sees: one pure function, the single source of truth.

:func:`render_model_input` turns a clip's frames into the exact pictures the box
sends the VLM, and a small record of how (which input, which frames, which
crop boxes, the rate, the size, :data:`MODEL_INPUT_VERSION`). The box calls it
on every alert (inference._prepare_alert), data collection's crop cuts its
frames with the same :func:`cut_crop`, and the admin labeling studio vendors
this file verbatim so labelers see exactly the model's input - so it imports
nothing but numpy and cv2, and a change to what it returns bumps the version.

Moved here unchanged from vlm_crop.py and inference.py (2026-10-08); a
characterization test pins every byte (tests/box/test_model_input_characterization.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

MODEL_INPUT_VERSION = "model-input-1"

VLM_INPUT_CROP = "crop"                     # the main-stream crop around the trigger detections
VLM_INPUT_WHOLE = "whole_frame_fallback"    # whole sub-stream frames: no main stream or no usable crop
JPEG_QUALITY = 85                           # what the backend encodes each frame with

Crop = Tuple[int, int, int, int]            # x1, y1, x2, y2 in the clip frames' pixels


@dataclass(frozen=True)
class ModelInputConfig:
    sample_fps: float = 1.0     # config vlm.sample_fps: frames a second sent to the model


def config_from(cfg: Any) -> ModelInputConfig:
    """The model-input settings of a data_collection Config (the box reads the same one)."""
    return ModelInputConfig(sample_fps=float(cfg.VLM_SAMPLE_FPS))


@dataclass
class ModelInput:
    frames: List[np.ndarray]            # exactly the pictures sent, in order (BGR)
    vlm_input: str                      # VLM_INPUT_CROP or VLM_INPUT_WHOLE (the alert meta's "vlm_input")
    frame_indices: List[int]            # each sent frame's index in the clip frames
    times: List[Optional[float]]        # its offset from the clip's first frame, seconds (None without a rate)
    crops: List[Optional[Crop]]         # its crop box in the clip frames' pixels; None: the frame as given
    union_crop: Optional[Crop]          # the box around every crop sent; None when nothing was cut
    fps: float                          # rate of the clip frames
    sample_fps: float
    step: int                           # every step-th usable frame, from the first
    size: Optional[Tuple[int, int]]     # (width, height) of the sent frames; None when nothing is sent
    version: str = MODEL_INPUT_VERSION

    def jpegs(self) -> List[bytes]:
        """The frames as the JPEG bytes the backend sends; a frame that cannot be encoded is skipped."""
        return [d for d in (encode_jpeg(f) for f in self.frames) if d]

    def record(self) -> Dict[str, Any]:
        """Everything but the pixels, JSON-safe."""
        return {
            "version": self.version,
            "vlm_input": self.vlm_input,
            "frame_indices": list(self.frame_indices),
            "times": list(self.times),
            "crops": [list(c) if c is not None else None for c in self.crops],
            "union_crop": list(self.union_crop) if self.union_crop is not None else None,
            "fps": self.fps,
            "sample_fps": self.sample_fps,
            "step": self.step,
            "size": list(self.size) if self.size is not None else None,
        }


def encode_jpeg(frame_bgr: Any) -> bytes:
    """JPEG-encode a BGR frame to raw bytes, as the backend sends it; b"" when it cannot be encoded."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return buf.tobytes() if ok else b""


def sample_step(fps: float, sample_fps: float) -> int:
    """Take every round(fps / sample_fps)-th frame (at least every one)."""
    return max(1, int(round(float(fps) / max(1e-6, float(sample_fps)))))


def median_crop_size(crops: Sequence[Optional[Crop]]) -> Optional[Tuple[int, int]]:
    """The (width, height) every cropped frame is resized to: the median of the boxes' sizes; None without boxes."""
    sizes = sorted((c[2] - c[0], c[3] - c[1]) for c in crops if c is not None)
    if not sizes:
        return None
    return sizes[len(sizes) // 2]


def cut_crop(frame: np.ndarray, crop: Crop, size: Tuple[int, int]) -> Optional[np.ndarray]:
    """*frame* cut to *crop* and resized to *size* (width, height) when it differs; None when the cut is empty."""
    x1, y1, x2, y2 = crop
    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return None
    if cropped.shape[1] != size[0] or cropped.shape[0] != size[1]:
        cropped = cv2.resize(cropped, size)
    return cropped


def render_model_input(
    clip_frames: Sequence[np.ndarray],
    meta: Mapping[str, Any],
    config: ModelInputConfig,
    overlay: Any = None,
) -> ModelInput:
    """The pictures the vision model is sent for one clip, and how they were made.

    *meta*:
      ``fps``        rate of *clip_frames* (required).
      ``crops``      optional, one box per clip frame (or None): each frame is cut to its own box and resized
                     to ``crop_size`` (default the median box size). Frames without a usable box are dropped
                     before sampling, as the saved crop clip drops them.
      ``crop_size``  optional (width, height) for ``crops``.
      ``vlm_input``  optional, VLM_INPUT_CROP or VLM_INPUT_WHOLE; default crop when ``crops`` is given.

    Without ``crops`` the frames are sent as they are: whole sub-stream frames, or a saved crop clip
    (``vlm_crops/*.mp4``, with ``vlm_input: crop`` and the meta's ``vlm_crop.fps``). Either way every
    round(fps / sample_fps)-th usable frame is sent, starting with the first.

    *overlay* is reserved for the Set-of-Mark marks (P1, P2, CAR1 drawn on the sent frames). Not drawn yet:
    anything but None raises NotImplementedError, so None never changes the output, and drawing will bump
    MODEL_INPUT_VERSION.
    """
    if overlay is not None:
        raise NotImplementedError("Set-of-Mark overlays are not drawn yet; pass overlay=None")
    fps = float(meta["fps"])
    crops = meta.get("crops")
    vlm_input = str(meta.get("vlm_input") or (VLM_INPUT_CROP if crops is not None else VLM_INPUT_WHOLE))
    step = sample_step(fps, config.sample_fps)

    size: Optional[Tuple[int, int]] = None
    if crops is None:
        boxes: List[Optional[Crop]] = [None] * len(clip_frames)
        usable = list(range(len(clip_frames)))
    else:
        if len(crops) != len(clip_frames):
            raise ValueError(f"{len(crops)} crop boxes for {len(clip_frames)} frames")
        boxes = [tuple(int(v) for v in c) if c is not None else None for c in crops]  # type: ignore[misc]
        wanted = meta.get("crop_size")
        size = (int(wanted[0]), int(wanted[1])) if wanted is not None else median_crop_size(boxes)
        usable = [i for i, c in enumerate(boxes)
                  if c is not None and size is not None and clip_frames[i][c[1]:c[3], c[0]:c[2]].size > 0]

    indices = usable[::step]
    if crops is None:
        frames = [clip_frames[i] for i in indices]
    else:
        frames = [cut_crop(clip_frames[i], boxes[i], size) for i in indices]  # type: ignore[arg-type]
    sent = [boxes[i] for i in indices]
    cut = [c for c in sent if c is not None]
    union = (min(c[0] for c in cut), min(c[1] for c in cut), max(c[2] for c in cut), max(c[3] for c in cut)) \
        if cut else None
    if frames and crops is None:
        size = (int(frames[0].shape[1]), int(frames[0].shape[0]))
    return ModelInput(
        frames=frames,
        vlm_input=vlm_input,
        frame_indices=indices,
        times=[i / fps if fps > 0 else None for i in indices],
        crops=sent,
        union_crop=union,
        fps=fps,
        sample_fps=float(config.sample_fps),
        step=step,
        size=size if frames else None,
        version=MODEL_INPUT_VERSION,
    )
