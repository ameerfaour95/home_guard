"""Isolated preview adapter for the current inference stream interface.

Keeps inference.py untouched. Uses its existing reader and never adds detection.
Inference has no annotated display image, so these previews are camera pictures.
"""

from pathlib import Path
from .preview import PreviewWriter, preview_settings
from .boxconfig import LOG_DIR


def preview_enabled():
    return bool(preview_settings().get("preview_enabled", False))


def adapt_stream(stream_class, writer):
    names = []
    writer.detector_instrumented = True

    class PreviewStream(stream_class):
        def __init__(self, name, url, **kwargs):
            # inference.py also passes its clip ring; hand on whatever it gives.
            super().__init__(name, url, **kwargs)
            names.append(name)
            writer.set_cameras(names)

        def read(self):
            # Inference visits each stream exactly once per outer detector loop.
            # Count at its first stream without changing inference or its status.
            if names and self.name == names[0]:
                writer.detector_loops += 1
            frame = super().read()
            # Real capture uses _ingest below, independently of YOLO's loop.
            # Keep the original adapter contract for older stream interfaces.
            if hasattr(self, "_running"):
                return frame
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

        def _ingest(self, frame, now):
            super()._ingest(frame, now)
            # _frame is already zone-masked and replaced, never mutated, by capture.
            # One bounded latest-frame slot per camera; encoding is on one worker.
            writer.offer(self.name, self._frame)

    return PreviewStream


def main():
    from . import inference

    writer = PreviewWriter(Path(LOG_DIR) / "preview", enabled=preview_enabled())
    if writer.enabled:
        inference._Stream = adapt_stream(inference._Stream, writer)
    try:
        inference.main()
    finally:
        writer.close()


if __name__ == "__main__":
    main()
