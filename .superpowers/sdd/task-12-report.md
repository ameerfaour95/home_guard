# Task 12 report
Added cloud/media.py (ensure_media, process_pending), S3.download_file/upload_file (upload restricted to writable prefixes), tests/cloud/test_media_worker.py (5 tests).
Failures: IndexProblem keyed admin_cache/thumbs/<event_id>.jpg, reason "media: ...", etag = clip etag; skipped until clip etag changes. No schema change.
Rendition when codec != h264, pix_fmt != yuv420p, or moov after mdat (own box parser). Filmstrip capped at 60 tiles (fps=60/duration recorded).
Full suite: 254 passed.

## Fix round
RED: 4 new tests failed first (rendition failure kept rows, replaced clip, limit, huge-box moov ValueError). Now green.
- Each artifact committed on upload; rendition failure is its own IndexProblem (admin_cache/renditions/<id>.mp4, clip etag), thumb/filmstrip kept.
- Every cloud artifact has detail.src_etag; thumbnail detail has needs_rendition. Pending selection is SQL (LIMIT, preferred clip, NOT EXISTS open media problem per key); limit bounds attempts. Replaced clip regenerates; stale rendition set available=False.
- moov_first: OSError/Overflow/ValueError -> False, 10,000 box cap. ffmpeg runs cwd=work with relative names; _problem_key reused. Added S3.upload_file rejected-key test (s3.py untouched). #4 media-once deferred to Task 14.
