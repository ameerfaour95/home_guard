"""Isolated preview adapter for the current inference stream interface.

Keeps inference.py untouched. Uses its existing reader and never adds detection.
Inference has no annotated display image, so these previews are camera pictures.
"""

import os
from pathlib import Path
import yaml
from .preview import PreviewWriter
from .boxconfig import LOG_DIR


def preview_enabled():
    base = Path(__file__).parents[1] / "data_collection" / "config.yaml"
    enabled = False
    for path in (base, os.environ.get("HOME_GUARD_CONFIG_OVERLAY")):
        if not path:
            continue
        try:
            with open(path, encoding="utf-8") as stream:
                display = (yaml.safe_load(stream) or {}).get("display", {})
            enabled = bool(display.get("preview_enabled", enabled))
        except OSError:
            pass
    return enabled


def adapt_stream(stream_class, writer):
    names = []

    class PreviewStream(stream_class):
        def __init__(self, name, url, **kwargs):
            # inference.py also passes its clip ring; hand on whatever it gives.
            super().__init__(name, url, **kwargs)
            names.append(name)
            writer.set_cameras(names)

        def read(self):
            frame = super().read()
            if frame is None or not writer.enabled:
                return frame
            # The underlying frame object changes only when capture succeeds.
            # Keeping its identity prevents publishing a frozen camera forever.
            source = self._frame
            if writer.wanted(self.name, source):
                import cv2

                disp = frame
                if max(frame.shape[:2]) > 1280:
                    scale = 1280 / max(frame.shape[:2])
                    disp = cv2.resize(
                        frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                    )
                writer.publish(self.name, disp, source=source)
            return frame

    return PreviewStream


def main():
    from . import inference

    writer = PreviewWriter(Path(LOG_DIR) / "preview", enabled=preview_enabled())
    if writer.enabled:
        inference._Stream = adapt_stream(inference._Stream, writer)
    inference.main()


if __name__ == "__main__":
    main()
