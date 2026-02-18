"""CORS-enabled HTTP file server for serving video clips to Label Studio."""

from __future__ import annotations

import functools
import http.server
import logging
import os
import threading

log = logging.getLogger("labeling.file_server")


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
        if args and str(args[0]).startswith("2"):
            return
        super().log_message(format, *args)


def start_file_server(
    dataset_dir: str,
    port: int = 8081,
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
        log.info("File server running at http://localhost:%d", port)
        log.info("Serving files from: %s", abs_dir)
        log.debug("Test URL: http://localhost:%d/clips/", port)
        server.serve_forever()

    if background:
        t = threading.Thread(target=_run, daemon=True)
        t.start()
    else:
        _run()
