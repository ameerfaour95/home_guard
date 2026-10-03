"""The YOLO detector, on the Intel graphics chip when the machine has one, else the CPU.

Shared by the collector (data_collection) and the box's inference mode. On the
Beelink's N150 the graphics chip (through OpenVINO) runs yolo11s in ~65 ms a
picture, the CPU (PyTorch) in ~550 ms. The OpenVINO copy of the model is made
once, next to the .pt file. Anything that goes wrong (no openvino, no graphics
driver, a broken copy) leaves the detector on the CPU.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from typing import Any, Optional

log = logging.getLogger("detector")


class Detector:
    """A YOLO model and the device it runs on, called like the model itself.

    One call at a time: the collector also uses the detector from its crop
    thread, and an OpenVINO model cannot run two pictures at once.
    """

    def __init__(self, model: Any, device: Optional[str] = None) -> None:
        self.model = model
        self.device = device          # "intel:gpu", or None for ultralytics' default (the CPU here)
        self._lock = threading.Lock()

    @property
    def on_graphics_chip(self) -> bool:
        return self.device is not None

    def predict(self, source: Any, **kwargs: Any) -> Any:
        if self.device:
            kwargs.setdefault("device", self.device)
        with self._lock:
            return self.model(source, **kwargs)

    __call__ = predict


def load_detector(model_path: str, device: str = "auto") -> Detector:
    """The detector for *model_path*; *device* "auto" tries the Intel graphics chip first.

    "cpu" never does: plain PyTorch, which runs on an NVIDIA card when there is one.
    """
    from ultralytics import YOLO  # noqa: PLC0415

    if str(device).strip().lower() == "auto" and model_path.endswith(".pt"):
        stem = os.path.splitext(os.path.basename(model_path))[0]
        ov_dir = os.path.join(os.path.dirname(model_path), stem + "_openvino_model")
        copied = False   # the failure came from the OpenVINO copy itself (not a missing chip or package)
        try:
            import numpy as np  # noqa: PLC0415
            import openvino as ov  # noqa: PLC0415

            if "GPU" not in ov.Core().available_devices:   # ultralytics' "intel:gpu" opens exactly "GPU"
                raise RuntimeError("no single Intel graphics chip")
            copied = True
            if not os.path.isfile(os.path.join(ov_dir, "metadata.yaml")):   # written last by the export
                log.info("Preparing %s for the Intel graphics chip (once, about 10 s)...", model_path)
                YOLO(model_path).export(format="openvino", imgsz=640, verbose=False)
            detector = Detector(YOLO(ov_dir, task="detect"), "intel:gpu")
            detector.predict(np.zeros((576, 704, 3), dtype=np.uint8), verbose=False)   # compile + warm up
            log.info("Detector: %s on the Intel graphics chip (OpenVINO)", model_path)
            return detector
        except Exception as exc:  # noqa: BLE001 - the CPU always works
            log.warning("Intel graphics chip not used (%s); the detector runs on the CPU", exc)
            if copied and os.path.isdir(ov_dir):
                shutil.rmtree(ov_dir, ignore_errors=True)   # a broken copy: made again at the next start
    log.info("Detector: %s on the CPU", model_path)
    return Detector(YOLO(model_path))
