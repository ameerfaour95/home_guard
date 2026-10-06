"""Score the cloud VLM's prompt against our human tags.

Build the set on the laptop, run it wherever the model is reachable:

    # Laptop: 5 frames from every reviewed clip of home_guard_dataset (a folder, or s3://bucket/prefix).
    python -m home_guard_project.box.eval_prompt prepare --out eval_set --dataset <home_guard_dataset>
    # Laptop: clips from outside the dataset, framed inside their annotated segment (picks.jsonl).
    python -m home_guard_project.box.eval_prompt add --dir eval_set --picks picks.jsonl
    # Freeze it: FROZEN.json holds the sha256 of the manifest and of every frame.
    python -m home_guard_project.box.eval_prompt freeze --dir eval_set
    # Ask the model (--strict-frozen refuses a set that changed since it was frozen):
    python -m home_guard_project.box.eval_prompt run --dir eval_set --strict-frozen
    python -m home_guard_project.box.eval_prompt run --dir eval_set --prompt-file my_prompt.txt
    python -m home_guard_project.box.eval_prompt summary --dir eval_set --tag <tag>

``prepare`` writes ``frames/<clip_id>_<i>.jpg`` and ``manifest.jsonl`` (one row per
clip with our label and text). ``run`` asks the model about every clip with 5 frames,
writes ``results/<tag>.jsonl`` / ``.csv`` / ``.summary.json`` and prints the score.
Both steps resume where they stopped. Results match clips by ``clip_id``.

Caveat: the box sends the model the last 5 buffered frames, 1 second apart, up to the
trigger, with the camera's zone mask applied. The eval takes 5 frames evenly across the
whole tagged clip, unmasked. The eval sees the whole clip, so alert recall reads somewhat
optimistic compared with the box. Frames are JPEG-encoded twice (quality 90 on disk, 85
when sent); the effect is negligible.

Results are never deleted silently: the default results name is ``<prompt id>__<model>``,
and a results file that holds answers from another prompt, wording or model is refused
(exit 2) unless ``--overwrite`` is given. Answers are only appended: a clip is asked again
when its last answer was an error or was given for other inputs (its camera, time of day or
frame bytes changed), and answers for clips no longer in the manifest stay in the file.
One run at a time per results file (``<tag>.lock``; a second run exits 3).

The score is always recomputed from the last answer per clip and the CURRENT manifest's
truth, so re-tagging a clip and re-running (or ``summary``) needs no new model call.

Our label comes from the dataset (``dataset_row``, the one adapter): ``alert`` true (or an
``[alert]``/``[alet]`` tag in the text) is ``alert``, no text or "No special activity" is
``empty``, anything else ``normal``. Clips nobody reviewed or marked ``[delete]`` are left out.
A manifest row may also carry ``source``, ``category`` (taxonomy id), ``day_night`` and
``subset`` (``misses``: the hard cases); the score is split by them.

The prompt is told the clip's own time of day (from the epoch in its name), not
the time the evaluation runs, so a score does not change with the hour it ran.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import logging
import math
import os
import re
import sys
import tempfile
import time
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from . import inference, providers

log = logging.getLogger("box.eval_prompt")

BUCKET = "security-camera-project-v1"
HOME_BATCH_PREFIX = "ameer_house"
DATASET_ENV = "HOME_GUARD_DATASET"        # default --dataset: a folder or s3://bucket/prefix
ANNOTATIONS = "annotations/clips.jsonl"   # inside the dataset
TRUTH_LABELS = ("alert", "normal", "empty")
# The eval's own truth on a clip, on top of the dataset's; prepare keeps it per clip_id.
# hard: why an alert is in the misses set (night, occlusion, subtle, loitering).
EVAL_KEYS = ("category", "subset", "day_night", "hard")
MISSES = "misses"

CONFIRM_OVER = 20          # a real run of more clips than this names the model and waits
CONFIRM_SECONDS = 5
FRAME_COUNT = 5            # what the box sends (inference.AlertSettings.clip_frames)
MAX_SIDE = 1280
JPEG_QUALITY = 90

MANIFEST = "manifest.jsonl"
FROZEN = "FROZEN.json"
FRAMES_DIR = "frames"
RESULTS_DIR = "results"
# What a results .jsonl line holds: the model's answer and what it answered (prompt, model,
# input fingerprint). Our truth is not stored: it is read from the current manifest when scoring.
USAGE_COLUMNS = ("prompt_tokens", "completion_tokens", "cost_usd", "latency_s")
ANSWER_COLUMNS = ("clip_id", "ai_label", "ai_summary", "ai_people", "ai_animals", "ai_vehicle_moving",
                  "raw", "error", "prompt_id", "prompt_sha12", "model", "input_sha12") + USAGE_COLUMNS
# The scored table (.csv): the answer next to the manifest's current camera and truth.
TRUTH_COLUMNS = ("source", "category", "subset", "day_night")
RESULT_COLUMNS = ("clip_id", "camera", "ours_label", "ours_text", "ai_label", "ai_summary", "ai_people", "ai_animals",
                  "ai_vehicle_moving", "raw", "error", "prompt_id", "prompt_sha12", "model", "input_sha12"
                  ) + USAGE_COLUMNS + ("local_time", "batch") + TRUTH_COLUMNS
NIGHT_FROM, NIGHT_UNTIL = 19, 6          # local hours: 19:00-05:59 is night
CALLS_PER_DAY = (150, 300)               # a typical and a busy house (plan page section 2)

NO_ACTIVITY = "No special activity."
PADDING_WORDS = ("without", "no one", "visible", "background", "parked")
CAUGHT_LABELS = ("suspicious", "escalation")

_TAG_RE = re.compile(r"\[[^\]]*\]")
_ALERT_RE = re.compile(r"\[(alert|alet)\]", re.IGNORECASE)
_EPOCH_RE = re.compile(r"_(\d{10})(?:_|$)")

S3_HELP = ("boto3 on the laptop needs the antivirus workaround: run with "
           "env -u SSLKEYLOGFILE -u PYTHONSTARTUP AWS_CA_BUNDLE=<bundle.pem> (see box/README.md, "
           "'Scoring the AI's prompt against our tags').")
EXIT_FROZEN = 4


# ----------------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------------
def parse_truth(description: str) -> Tuple[str, str]:
    """Our label (``alert`` | ``empty`` | ``normal``) and the text without its ``[...]`` tag."""
    text = re.sub(r"\s+", " ", _TAG_RE.sub("", description or "")).strip()
    if _ALERT_RE.search(description or ""):
        return "alert", text
    if text.rstrip(".").strip().lower() == "no special activity":
        return "empty", text
    return "normal", text


def clip_stem(clip_id_or_key: str) -> str:
    return os.path.splitext(os.path.basename(str(clip_id_or_key)))[0]


def segment_range(n: int, fps: float, start_sec: Optional[float] = None,
                  end_sec: Optional[float] = None) -> Tuple[int, int]:
    """Frames ``[lo, hi)`` of an ``n``-frame clip whose time (index / fps) lies in ``[start_sec, end_sec)``;
    the whole clip when no segment is given, nothing when the fps is unknown."""
    if start_sec is None and end_sec is None:
        return 0, n
    if fps <= 0:
        return 0, 0
    lo = 0 if start_sec is None else min(n, max(0, math.ceil(start_sec * fps - 1e-6)))
    hi = n if end_sec is None else min(n, math.ceil(end_sec * fps - 1e-6))
    return lo, max(lo, hi)


def sample_indices(n: int, k: int = FRAME_COUNT) -> List[int]:
    """``k`` evenly spaced indices over ``n`` frames (first and last included)."""
    if n <= 0:
        return []
    if k == 1:
        return [0]
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def fit_long_side(frame: Any, max_side: int = MAX_SIDE) -> Any:
    """Shrink so the long side is at most ``max_side``; never enlarge (the same object comes back)."""
    h, w = frame.shape[:2]
    if max(h, w) <= max_side:
        return frame
    import cv2  # noqa: PLC0415

    scale = max_side / float(max(h, w))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def clip_local_time(clip_id: str) -> Optional[str]:
    """``HH:MM:SS`` (this machine's time zone) from the epoch in a collector clip's name."""
    m = _EPOCH_RE.search(clip_id or "")
    if not m:
        return None
    return datetime.fromtimestamp(int(m.group(1))).strftime("%H:%M:%S")


def clip_hash(clip_id: str) -> int:
    """A stable hash (Python's own hash() changes between runs)."""
    return int(hashlib.sha256(clip_id.encode("utf-8")).hexdigest(), 16)


def prompt_id_of(prompt_text: Optional[str]) -> str:
    if prompt_text is None:
        return inference.PROMPT_VERSION
    return f"file-{hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()[:12]}"


def default_tag(prompt_text: Optional[str], model: str, fake: bool = False) -> str:
    """Results file name ``<prompt id>__<model>``; a fake run gets its own ``fake-`` file."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(model or "unknown"))
    tag = f"{prompt_id_of(prompt_text)}__{safe}"
    return f"fake-{tag}" if fake else tag


class ResultsConflict(Exception):
    """The results file already holds answers from another prompt, wording or model."""


def _words(text: str) -> int:
    return len((text or "").split())


def _ratio(count: int, total: int) -> Optional[float]:
    return count / total if total else None


def is_night(local_time: Optional[str]) -> Optional[bool]:
    if not local_time:
        return None
    try:
        hour = int(str(local_time).split(":")[0])
    except ValueError:
        return None
    return hour >= NIGHT_FROM or hour < NIGHT_UNTIL


def _counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    ok = [r for r in rows if not r.get("error")]
    alerts = [r for r in ok if r.get("ours_label") == "alert"]
    normals = [r for r in ok if r.get("ours_label") == "normal"]
    return {"alerts_caught": sum(r.get("ai_label") in CAUGHT_LABELS for r in alerts), "alerts_total": len(alerts),
            "normal_flagged": sum(r.get("ai_label") != "normal" for r in normals), "normal_total": len(normals),
            "errors": len(rows) - len(ok)}


def _mean(values: Iterable[Any]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


def _is_home(row: Dict[str, Any]) -> bool:
    """Filmed by our own cameras (the external imports in a home batch are not)."""
    return source_of(row) == "house"


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The score. Rows with an error are counted in ``errors`` and left out of everything else."""
    ok = [r for r in rows if not r.get("error")]
    alerts = [r for r in ok if r.get("ours_label") == "alert"]
    caught = [r for r in alerts if r.get("ai_label") in CAUGHT_LABELS]
    escalated = [r for r in caught if r.get("ai_label") == "escalation"]
    normals = [r for r in ok if r.get("ours_label") == "normal"]
    flagged = [r for r in normals if r.get("ai_label") != "normal"]
    empties = [r for r in ok if r.get("ours_label") == "empty"]
    exact = [r for r in empties if str(r.get("ai_summary") or "").strip() == NO_ACTIVITY]
    padded = [r for r in ok if any(w in str(r.get("ai_summary") or "").lower() for w in PADDING_WORDS)]
    both = [r for r in ok if str(r.get("ai_summary") or "").strip() and str(r.get("ours_text") or "").strip()]
    cost = _mean(r.get("cost_usd") for r in rows)          # errored calls were paid for too
    extra = {
        "missed_alerts": sorted(str(r.get("clip_id")) for r in alerts if r not in caught),
        "false_alarms": sorted(str(r.get("clip_id")) for r in flagged),
        "tokens_in_mean": _mean(r.get("prompt_tokens") for r in rows),
        "tokens_out_mean": _mean(r.get("completion_tokens") for r in rows),
        "cost_per_call": cost,
        "latency_mean": _mean(r.get("latency_s") for r in rows),
        "per_month_150": None if cost is None else cost * CALLS_PER_DAY[0] * 30,
        "per_month_300": None if cost is None else cost * CALLS_PER_DAY[1] * 30,
        "day": _counts([r for r in rows if day_night_of(r) == "day"]),
        "night": _counts([r for r in rows if day_night_of(r) == "night"]),
        "unknown_time": sum(day_night_of(r) is None for r in rows),
        "home": _counts([r for r in rows if _is_home(r)]),
        "external": _counts([r for r in rows if not _is_home(r)]),
        "misses_set": _counts([r for r in rows if r.get("subset") == MISSES]),
    }
    return {**extra, **{
        "rows": len(rows),
        "alerts_caught": len(caught),
        "alerts_total": len(alerts),
        "alerts_caught_ratio": _ratio(len(caught), len(alerts)),
        "escalation_share": _ratio(len(escalated), len(caught)),
        "normal_flagged": len(flagged),
        "normal_total": len(normals),
        "normal_flagged_ratio": _ratio(len(flagged), len(normals)),
        "empty_exact": len(exact),
        "empty_total": len(empties),
        "padding": len(padded),
        "padding_total": len(ok),
        "words_ai_mean": (sum(_words(r["ai_summary"]) for r in both) / len(both)) if both else None,
        "words_ours_mean": (sum(_words(r["ours_text"]) for r in both) / len(both)) if both else None,
        "errors": len(rows) - len(ok),
    }}


def _pct(value: Optional[float]) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


def _num(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.1f}"


def _split_line(name: str, c: Optional[Dict[str, int]]) -> str:
    if not c:
        return ""
    return (f"{name:<18}alerts {c['alerts_caught']}/{c['alerts_total']}, "
            f"normal flagged {c['normal_flagged']}/{c['normal_total']}, errors {c['errors']}")


def format_summary(s: Dict[str, Any]) -> str:
    head = []
    if s.get("tag"):
        head.append(f"tag {s['tag']}  prompt {s.get('prompt_id', '?')}  model {s.get('model', '?')}")
    if s.get("frozen"):
        head.append(f"eval set          {s['frozen']}")
    lines = head + [
        f"alerts caught     {s['alerts_caught']}/{s['alerts_total']} ({_pct(s['alerts_caught_ratio'])})"
        "   ours [alert] -> AI suspicious or escalation",
        f"escalation share  {_pct(s['escalation_share'])}   of the caught alerts, labelled escalation",
        f"normal flagged    {s['normal_flagged']}/{s['normal_total']} ({_pct(s['normal_flagged_ratio'])})"
        "   ours normal -> AI not normal (false alarms)",
        f"empty exact       {s['empty_exact']}/{s['empty_total']}   "
        f"ours empty -> AI wrote exactly \"{NO_ACTIVITY}\"",
        f"padding           {s['padding']}/{s['padding_total']}   AI summary mentions {', '.join(PADDING_WORDS)}",
        f"words per summary AI {_num(s['words_ai_mean'])} vs ours {_num(s['words_ours_mean'])}",
        f"errors            {s['errors']} of {s['rows']} rows",
        _split_line("day", s.get("day")),
        _split_line("night", s.get("night")),
        _split_line("home clips", s.get("home")),
        _split_line("external clips", s.get("external")),
        _split_line("misses set", s.get("misses_set")) if (s.get("misses_set") or {}).get("alerts_total") else "",
        f"missed alerts     {', '.join(s['missed_alerts']) or '-'}" if "missed_alerts" in s else "",
        f"false alarms      {', '.join(s['false_alarms']) or '-'}" if "false_alarms" in s else "",
        (f"tokens per call   in {_num(s.get('tokens_in_mean'))} out {_num(s.get('tokens_out_mean'))}   "
         f"latency {_num(s.get('latency_mean'))} s"),
        ("cost              unknown (local or unpriced model)" if s.get("cost_per_call") is None else
         f"cost              ${s['cost_per_call']:.5f} per call; per box per month "
         f"${s['per_month_150']:.2f} at 150 calls/day, ${s['per_month_300']:.2f} at 300"),
    ] + ([f"outdated          {s['outdated']} answers were for other frames, camera or time; left out "
          "(run again to ask)"] if s.get("outdated") else [])
    return "\n".join(line for line in lines if line)


# ----------------------------------------------------------------------------
# Frames
# ----------------------------------------------------------------------------
def _read_at(path: str, wanted: Sequence[int]) -> Dict[int, Any]:
    """Decode the clip in order and keep only the frames at ``wanted`` indices."""
    import cv2  # noqa: PLC0415

    want = set(wanted)
    got: Dict[int, Any] = {}
    cap = cv2.VideoCapture(path)
    try:
        i = 0
        last = max(want) if want else -1
        while i <= last and cap.grab():
            if i in want:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    got[i] = frame
            i += 1
    finally:
        cap.release()
    return got


def _count_frames(path: str) -> Tuple[int, int]:
    """(frame count the file claims, frames that actually decode)."""
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return 0, 0
        claimed = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        decoded = 0
        while cap.grab():
            decoded += 1
        return claimed, decoded
    finally:
        cap.release()


def sample_frames(path: str, k: int = FRAME_COUNT, start_sec: Optional[float] = None,
                  end_sec: Optional[float] = None) -> List[Any]:
    """``k`` evenly spaced frames of the clip (or of its ``[start_sec, end_sec)`` segment), long side
    at most MAX_SIDE; ``[]`` if it will not decode or the segment holds no frame."""
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(path)
    opened = cap.isOpened()
    claimed = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if opened else 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) if opened else 0.0
    cap.release()

    def pick(n: int) -> List[int]:
        lo, hi = segment_range(n, fps, start_sec, end_sec)
        return [lo + i for i in sample_indices(hi - lo, k)]

    wanted = pick(claimed)
    got = _read_at(path, wanted) if wanted else {}
    if not wanted or any(i not in got for i in wanted):
        # The header's count was missing or wrong: count what really decodes, then pick again.
        _, decoded = _count_frames(path)
        wanted = pick(decoded)
        got = _read_at(path, wanted) if wanted else {}
        if any(i not in got for i in wanted):
            return []
    return [fit_long_side(got[i]) for i in wanted]


def _write_jpeg(path: str, frame: Any) -> None:
    import cv2  # noqa: PLC0415

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError(f"could not encode {path}")
    _atomic_write(path, lambda f: f.write(buf.tobytes()), binary=True)


def frame_paths(clip_id: str) -> List[str]:
    """The frame files of a clip, relative to the eval folder (forward slashes, as in the manifest)."""
    return [f"{FRAMES_DIR}/{clip_id}_{i}.jpg" for i in range(FRAME_COUNT)]


def _frames_exist(out_dir: str, clip_id: str) -> bool:
    return all(os.path.isfile(os.path.join(out_dir, rel)) for rel in frame_paths(clip_id))


# ----------------------------------------------------------------------------
# prepare (laptop, reads home_guard_dataset)
# ----------------------------------------------------------------------------
def dataset_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The one place that reads a line of home_guard_dataset's ``annotations/clips.jsonl``, so a field
    renamed there touches only this function.

    Returns ``clip_id, source, batch, camera, clip, ours_text, ours_label``, or None for a clip
    nobody reviewed (``alert`` null) or one marked ``[delete]``. ``alert`` true, or an ``[alert]`` /
    ``[alet]`` tag left in the text, is ``alert``; an empty description or "No special activity" is
    ``empty``; anything else ``normal``.
    """
    description = str(row.get("description") or "")
    if row.get("alert") is None or "[delete]" in description.lower():
        return None
    label, text = parse_truth(description)
    if row.get("alert") is True:
        label = "alert"
    elif not text:
        label = "empty"
    return {"clip_id": clip_stem(row.get("clip_id") or row.get("clip") or ""),
            "source": str(row.get("source") or ""), "batch": str(row.get("batch") or ""),
            "camera": str(row.get("camera") or ""), "clip": str(row.get("clip") or ""),
            "ours_text": text, "ours_label": label}


class Dataset:
    """home_guard_dataset in a local folder, or under ``s3://<bucket>/<prefix>`` with the same relative
    paths (read only)."""

    def __init__(self, root: str, client: Any = None) -> None:
        self.root = str(root).rstrip("/\\")
        self.client = client
        self.bucket = self.prefix = ""
        if self.root.startswith("s3://"):
            self.bucket, _, prefix = self.root[len("s3://"):].partition("/")
            self.prefix = prefix.strip("/")
            if client is None:
                raise ValueError(f"{root} is on S3: a boto3 client is needed")

    @property
    def on_s3(self) -> bool:
        return bool(self.bucket)

    def _key(self, rel: str) -> str:
        return f"{self.prefix}/{rel}" if self.prefix else rel

    def rows(self) -> List[Dict[str, Any]]:
        if not self.on_s3:
            return read_jsonl(os.path.join(self.root, ANNOTATIONS))
        with _s3_temp(self.client, self.bucket, self._key(ANNOTATIONS), ".jsonl") as path:
            return read_jsonl(path)

    @contextlib.contextmanager
    def local_clip(self, rel: str) -> Iterator[str]:
        """A local path of the clip ``rel`` while the block runs (an S3 clip is a temp download)."""
        if not rel:
            raise FileNotFoundError("the dataset row names no clip")
        if not self.on_s3:
            path = os.path.join(self.root, rel)
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            yield path
            return
        with _s3_temp(self.client, self.bucket, self._key(rel), ".mp4") as path:
            yield path


@contextlib.contextmanager
def _s3_temp(client: Any, bucket: str, key: str, suffix: str) -> Iterator[str]:
    """Download ``key`` to a temp file in the system temp folder; removed when the block ends."""
    fd, tmp = tempfile.mkstemp(suffix=suffix, dir=tempfile.gettempdir())
    os.close(fd)
    try:
        client.download_file(Bucket=bucket, Key=key, Filename=tmp)
        yield tmp
    finally:
        with contextlib.suppress(OSError):
            os.remove(tmp)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """The rows of a jsonl file. A cut-off last line (a run killed mid-write, possibly inside a
    multibyte character) is skipped with a warning; a bad line anywhere else still raises."""
    with open(path, "rb") as f:
        lines = [line for line in f.read().split(b"\n") if line.strip()]
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line.decode("utf-8")))
        except ValueError:                       # UnicodeDecodeError or JSONDecodeError
            if i != len(lines) - 1:
                raise
            log.warning("%s: the last line is cut off; skipping it", path)
    return rows


def _atomic_write(path: str, write: Callable[[Any], Any], binary: bool = False) -> None:
    """Write through a uniquely named temp file next to *path*, then swap it in: a reader never
    sees half a file, and two writers never share a temp file."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=directory)
    try:
        if binary:
            with os.fdopen(fd, "wb") as f:
                write(f)
        else:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                write(f)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)

    def write(f: Any) -> None:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    _atomic_write(path, write)


def _old_manifest(out_dir: str) -> Dict[str, Dict[str, Any]]:
    path = os.path.join(out_dir, MANIFEST)
    return {r["clip_id"]: r for r in read_jsonl(path)} if os.path.isfile(path) else {}


def _write_clip_frames(path: str, out_dir: str, clip_id: str, segment: Optional[Sequence[float]] = None) -> bool:
    start, end = (segment[0], segment[1]) if segment else (None, None)
    frames = sample_frames(path, start_sec=start, end_sec=end)
    if len(frames) != FRAME_COUNT:
        log.warning("%s: could not read %d frames%s", clip_id, FRAME_COUNT,
                    f" between {start} and {end} s" if segment else "")
        return False
    for rel, frame in zip(frame_paths(clip_id), frames):
        _write_jpeg(os.path.join(out_dir, rel), frame)
    log.info("prepared %s", clip_id)
    return True


def prepare(out_dir: str, dataset: Dataset, sources: Optional[Sequence[str]] = None,
            batches: Optional[Sequence[str]] = None, limit: Optional[int] = None) -> Dict[str, Any]:
    """Fetch 5 frames, evenly across the clip, for every reviewed clip of *dataset* into *out_dir*.

    *sources* / *batches* narrow the clips (default: all). *limit* caps how many clips are newly
    read this time; clips already in ``frames/`` are kept and never read again. The eval's own
    truth on a clip (``category``, ``subset``, ``day_night``) is kept from the old manifest, and
    clips that ``add`` put in (they carry a ``segment``) stay as they are. A frozen set is refused.
    """
    _refuse_if_frozen(out_dir)
    os.makedirs(os.path.join(out_dir, FRAMES_DIR), exist_ok=True)
    old = _old_manifest(out_dir)
    raw = dataset.rows()
    reviewed = [r for r in (dataset_row(x) for x in raw) if r is not None]
    chosen = [r for r in reviewed
              if (not sources or r["source"] in sources) and (not batches or r["batch"] in batches)]
    chosen.sort(key=lambda r: (r["batch"], r["clip_id"]))

    counts: Dict[str, Any] = {"rows": len(raw), "dropped": len(raw) - len(reviewed), "chosen": len(chosen),
                              "new": 0, "cached": 0, "failed": [], "skipped_by_limit": 0, "duplicates": [],
                              "added_kept": 0}
    manifest: List[Dict[str, Any]] = []
    seen: set = set()
    for row in chosen:
        clip_id = row["clip_id"]
        if clip_id in seen:
            counts["duplicates"].append(clip_id)     # listed twice: the first (by batch) wins
            continue
        seen.add(clip_id)
        if _frames_exist(out_dir, clip_id):
            counts["cached"] += 1
        elif limit is not None and counts["new"] + len(counts["failed"]) >= limit:
            counts["skipped_by_limit"] += 1
            continue
        else:
            try:
                with dataset.local_clip(row["clip"]) as path:
                    ok = _write_clip_frames(path, out_dir, clip_id)
            except Exception as exc:  # noqa: BLE001 - one bad clip must not stop the rest
                log.warning("%s: %s", clip_id, exc)
                ok = False
            if not ok:
                counts["failed"].append(clip_id)
                continue
            counts["new"] += 1
        local_time = clip_local_time(clip_id)
        entry = {**row, "frames": frame_paths(clip_id), "local_time": local_time}
        night = is_night(local_time)
        if night is not None:
            entry["day_night"] = "night" if night else "day"
        entry.update({k: old[clip_id][k] for k in EVAL_KEYS if k in old.get(clip_id, {})})
        manifest.append(entry)
    for clip_id, row in old.items():
        if row.get("segment") is not None and clip_id not in seen:
            manifest.append(row)
            counts["added_kept"] += 1
    _write_jsonl(os.path.join(out_dir, MANIFEST), manifest)
    return counts


def format_prepare_counts(c: Dict[str, Any]) -> str:
    lines = [
        f"rows read         {c['rows']}  ({c['dropped']} not reviewed or marked [delete], left out)",
        f"chosen            {c['chosen']}",
        f"newly prepared    {c['new']}",
        f"cached            {c['cached']}",
    ]
    if c["failed"]:
        lines.append(f"failed            {len(c['failed'])}: {', '.join(c['failed'])}")
    if c["skipped_by_limit"]:
        lines.append(f"left for later    {c['skipped_by_limit']} (--limit)")
    if c["duplicates"]:
        lines.append(f"listed twice      {len(c['duplicates'])} (first kept): {', '.join(c['duplicates'])}")
    if c["added_kept"]:
        lines.append(f"added clips kept  {c['added_kept']} (from add, not in the dataset)")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# add (laptop): clips outside the dataset, framed inside their annotated segment
# ----------------------------------------------------------------------------
PICK_REQUIRED = ("clip_id", "source", "batch", "camera", "ours_text", "ours_label", "segment")


def check_pick(pick: Dict[str, Any]) -> Optional[str]:
    """Why *pick* cannot be added, or None."""
    missing = [k for k in PICK_REQUIRED if k not in pick]
    if missing:
        return f"missing {', '.join(missing)}"
    if pick["ours_label"] not in TRUTH_LABELS:
        return f"ours_label {pick['ours_label']!r} is not one of {', '.join(TRUTH_LABELS)}"
    seg = pick["segment"]
    if not (isinstance(seg, (list, tuple)) and len(seg) == 2
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in seg) and 0 <= seg[0] < seg[1]):
        return f"segment {seg!r} is not [start, end] seconds with start < end"
    if not (pick.get("s3_key") or pick.get("clip")):
        return "names neither s3_key nor clip"
    return None


def add_picks(out_dir: str, picks: Sequence[Dict[str, Any]], client: Any = None, bucket: str = BUCKET,
              dataset: Optional[Dataset] = None) -> Dict[str, Any]:
    """Add *picks* to the manifest, each with 5 frames evenly spaced inside its ``segment``
    (``[start, end]`` seconds of the clip), the same sampling and size as ``prepare``.

    A pick names its mp4 by ``s3_key`` (in *bucket*, read with *client*) or ``clip`` (in *dataset*).
    Its other fields (``category``, ``subset``, ``day_night``, ``local_time``, ``note``...) go into
    the manifest row as given. A pick whose clip_id the manifest already holds from the dataset is
    refused; one added before is replaced. A frozen set is refused.
    """
    _refuse_if_frozen(out_dir)
    os.makedirs(os.path.join(out_dir, FRAMES_DIR), exist_ok=True)
    manifest = list(_old_manifest(out_dir).values())
    by_id = {r["clip_id"]: i for i, r in enumerate(manifest)}
    counts: Dict[str, Any] = {"picks": len(picks), "new": 0, "cached": 0, "invalid": [], "conflicts": [],
                              "failed": []}
    for pick in picks:
        why = check_pick(pick)
        clip_id = str(pick.get("clip_id", ""))
        if why:
            counts["invalid"].append(f"{clip_id or '?'}: {why}")
            continue
        at = by_id.get(clip_id)
        if at is not None and manifest[at].get("segment") is None:
            counts["conflicts"].append(clip_id)
            continue
        same = (at is not None and list(manifest[at]["segment"]) == [float(x) for x in pick["segment"]]
                and manifest[at].get("s3_key") == pick.get("s3_key") and manifest[at].get("clip") == pick.get("clip"))
        if same and _frames_exist(out_dir, clip_id):
            counts["cached"] += 1
        else:
            try:
                if pick.get("s3_key"):
                    if client is None:
                        raise ValueError("an S3 pick needs a boto3 client")
                    source = _s3_temp(client, bucket, pick["s3_key"], ".mp4")
                else:
                    if dataset is None:
                        raise ValueError("a dataset pick needs --dataset")
                    source = dataset.local_clip(pick["clip"])
                with source as path:
                    ok = _write_clip_frames(path, out_dir, clip_id, pick["segment"])
            except Exception as exc:  # noqa: BLE001 - one bad clip must not stop the rest
                log.warning("%s: %s", clip_id, exc)
                ok = False
            if not ok:
                counts["failed"].append(clip_id)
                continue
            counts["new"] += 1
        row = {**pick, "segment": [float(pick["segment"][0]), float(pick["segment"][1])],
               "frames": frame_paths(clip_id), "local_time": pick.get("local_time")}
        if at is None:
            by_id[clip_id] = len(manifest)
            manifest.append(row)
        else:
            manifest[at] = row
    _write_jsonl(os.path.join(out_dir, MANIFEST), manifest)
    return counts


def format_add_counts(c: Dict[str, Any]) -> str:
    lines = [f"picks             {c['picks']}", f"newly added       {c['new']}", f"cached            {c['cached']}"]
    for key, name in (("invalid", "invalid"), ("conflicts", "already from dataset"), ("failed", "failed")):
        if c[key]:
            lines.append(f"{name:<18}{len(c[key])}: {'; '.join(c[key])}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# freeze: the set a score was made on, checked before every run
# ----------------------------------------------------------------------------
class FrozenSet(Exception):
    """The eval folder is frozen (FROZEN.json); prepare and add would change it."""


class FrozenChanged(Exception):
    """run --strict-frozen: the set is not frozen, or changed since it was."""


def _refuse_if_frozen(out_dir: str) -> None:
    path = os.path.join(out_dir, FROZEN)
    if os.path.isfile(path):
        raise FrozenSet(f"{out_dir} is frozen ({path}). Build a new folder (copy frames/ to reuse them); "
                        f"delete {FROZEN} only if you mean to change the frozen set.")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_sha256(manifest_sha: str, frames: Dict[str, str]) -> str:
    """One hash for the whole set: the manifest's hash and every frame's, in path order."""
    h = hashlib.sha256(f"manifest {manifest_sha}\n".encode("ascii"))
    for rel in sorted(frames):
        h.update(f"{rel} {frames[rel]}\n".encode("utf-8"))
    return h.hexdigest()


def source_of(row: Dict[str, Any]) -> str:
    """``house`` | ``external`` | ``uca`` | ``smarthome`` | ``other``. Rows from before the dataset
    have no ``source`` and are told by batch (the external imports sit in a home batch)."""
    if row.get("source"):
        return str(row["source"])
    batch, clip_id = str(row.get("batch") or ""), str(row.get("clip_id") or "")
    if batch.startswith(HOME_BATCH_PREFIX):
        return "external" if clip_id.startswith("external_") else "house"
    for name in ("uca", "smarthome"):
        if batch.startswith(name):
            return name
    return "other"


def day_night_of(row: Dict[str, Any]) -> Optional[str]:
    """``day`` | ``night`` from the manifest's ``day_night``, else from the clip's time; None if unknown."""
    if row.get("day_night") in ("day", "night"):
        return str(row["day_night"])
    night = is_night(row.get("local_time"))
    return None if night is None else ("night" if night else "day")


def set_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """How the set is made up: all clips and the alerts, by source, label, category, day/night, subset."""
    def by(key: Callable[[Dict[str, Any]], Any], subset: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in subset:
            name = str(key(r) or "none")
            out[name] = out.get(name, 0) + 1
        return dict(sorted(out.items()))

    alerts = [r for r in rows if r.get("ours_label") == "alert"]
    keys: Dict[str, Callable[[Dict[str, Any]], Any]] = {
        "source": source_of, "label": lambda r: r.get("ours_label"), "category": lambda r: r.get("category"),
        "day_night": lambda r: day_night_of(r) or "unknown", "subset": lambda r: r.get("subset") or "main"}
    counts = {f"by_{name}": by(key, rows) for name, key in keys.items()}
    counts.update({f"alerts_by_{name}": by(key, alerts) for name, key in keys.items() if name != "label"})
    return counts


def freeze(out_dir: str) -> Dict[str, Any]:
    """Write ``FROZEN.json``: sha256 of the manifest and of every frame file, one hash of the whole
    set, and its make-up. Refused if a frame is missing."""
    manifest_path = os.path.join(out_dir, MANIFEST)
    rows = read_jsonl(manifest_path)
    frames: Dict[str, str] = {}
    missing = []
    for row in rows:
        for rel in row.get("frames") or []:
            path = os.path.join(out_dir, rel)
            if os.path.isfile(path):
                frames[rel] = _sha256(path)
            else:
                missing.append(rel)
    if missing:
        raise ValueError(f"{len(missing)} frame files are missing, e.g. {', '.join(missing[:5])}; nothing frozen")
    manifest_sha = _sha256(manifest_path)
    record = {"frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"), "clips": len(rows),
              "manifest_sha256": manifest_sha, "set_sha256": set_sha256(manifest_sha, frames),
              "counts": set_counts(rows), "frames": frames}
    _atomic_write(os.path.join(out_dir, FROZEN), lambda f: f.write(json.dumps(record, indent=1)))
    return record


def frozen_changes(out_dir: str) -> Optional[List[str]]:
    """None when the folder is not frozen; else what changed since (``[]``: nothing)."""
    path = os.path.join(out_dir, FROZEN)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        frozen = json.load(f)
    changes = []
    manifest_path = os.path.join(out_dir, MANIFEST)
    if not os.path.isfile(manifest_path):
        changes.append(f"{MANIFEST} is missing")
    elif _sha256(manifest_path) != frozen.get("manifest_sha256"):
        changes.append(f"{MANIFEST} changed")
    for rel, sha in sorted((frozen.get("frames") or {}).items()):
        full = os.path.join(out_dir, rel)
        if not os.path.isfile(full):
            changes.append(f"{rel} is missing")
        elif _sha256(full) != sha:
            changes.append(f"{rel} changed")
    return changes


def frozen_status(out_dir: str) -> Optional[str]:
    """For the summary: ``frozen <set sha12>``, ``CHANGED since frozen ...``, or None (never frozen)."""
    changes = frozen_changes(out_dir)
    if changes is None:
        return None
    if changes:
        return f"CHANGED since frozen ({len(changes)} differences, e.g. {changes[0]})"
    with open(os.path.join(out_dir, FROZEN), encoding="utf-8") as f:
        return f"frozen {str(json.load(f).get('set_sha256'))[:12]}"


def check_frozen(out_dir: str, strict: bool = False) -> None:
    """Before a run: warn loudly if the frozen set changed; with *strict*, raise ``FrozenChanged``
    when it changed or was never frozen."""
    changes = frozen_changes(out_dir)
    if changes is None:
        if strict:
            raise FrozenChanged(f"{out_dir} is not frozen (no {FROZEN}); run freeze first, or drop --strict-frozen.")
        return
    if not changes:
        return
    shown = "; ".join(changes[:10]) + (f"; and {len(changes) - 10} more" if len(changes) > 10 else "")
    banner = "!" * 78
    log.warning("%s\nTHE FROZEN EVAL SET CHANGED since %s was written: %s\nScores from this run do NOT "
                "compare with scores made on the frozen set.\n%s", banner, FROZEN, shown, banner)
    if strict:
        raise FrozenChanged(f"{out_dir} changed since it was frozen ({shown}). Nothing was asked.")


def format_freeze(record: Dict[str, Any]) -> str:
    c = record["counts"]
    lines = [f"frozen            {record['clips']} clips, set sha256 {record['set_sha256']}",
             f"manifest sha256   {record['manifest_sha256']}"]
    for name in ("source", "label", "category", "day_night", "subset"):
        lines.append(f"{name:<18}{', '.join(f'{k} {v}' for k, v in c[f'by_{name}'].items())}")
        if f"alerts_by_{name}" in c:
            lines.append(f"{'  alerts':<18}{', '.join(f'{k} {v}' for k, v in c[f'alerts_by_{name}'].items())}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# run (box, calls the model)
# ----------------------------------------------------------------------------
class FakeBackend:
    """No network. Answers from our own label so a run can be checked end to end:
    alert -> escalation (even clip hash) or suspicious (odd); normal -> a short sentence;
    empty -> "No special activity." with no people. Clips in *fail_ids* raise."""

    model_name = "fake"

    def __init__(self, fail_ids: Iterable[str] = ()) -> None:
        self.fail_ids = set(fail_ids)
        self.calls = 0
        self.last_usage = {"prompt_tokens": 1000, "completion_tokens": 50}
        self.prompts: List[str] = []

    def ask(self, row: Dict[str, Any], frames: List[Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
        self.calls += 1
        self.prompts.append(inference.build_prompt(row.get("camera", ""), int(time.time()),
                                                   datetime.now().strftime("%H:%M:%S"), 0, 0))
        clip_id = str(row.get("clip_id", ""))
        if clip_id in self.fail_ids:
            raise RuntimeError(f"forced failure for {clip_id}")
        ours = row.get("ours_label")
        if ours == "alert":
            label = "escalation" if clip_hash(clip_id) % 2 == 0 else "suspicious"
            parsed = {"summary": "Two men force the door open.", "label": label, "people": 2,
                      "animals": 0, "vehicle_moving": False}
        elif ours == "empty":
            parsed = {"summary": NO_ACTIVITY, "label": "normal", "people": 0, "animals": 0, "vehicle_moving": False}
        else:
            parsed = {"summary": "A person walks to the door.", "label": "normal", "people": 1,
                      "animals": 0, "vehicle_moving": False}
        return json.dumps(parsed), parsed


class GptAsker:
    """Asks inference.GptBackend as the box does (same prompt function and request, 5 frames; which frames differ)."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.model_name = getattr(backend, "model_name", "gpt")

    @property
    def last_usage(self) -> Dict[str, int]:
        return dict(getattr(self.backend, "last_usage", None) or {})

    def ask(self, row: Dict[str, Any], frames: List[Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
        # Hours 0,0 = always inside the alert window; the prompt does not use them.
        return self.backend.analyze(frames, row.get("camera", ""), int(time.time()), 0, 0)


class _PromptState:
    local_time: Optional[str] = None


@contextlib.contextmanager
def prompt_override(template: Optional[str]) -> Iterator[_PromptState]:
    """Swap ``inference.build_prompt`` while the block runs, and always put it back.

    With a *template*, the prompt is its text with ``{camera_name}``,
    ``{local_time_str}`` and ``{owner_language}`` (default ``en``) filled in
    (nothing else is touched, so JSON braces are safe). Without one, the box's own
    prompt is used, and any extra keyword arguments are passed through to it.
    Either way, when ``state.local_time`` is set it replaces the time of day the
    backend passes in.
    """
    original = inference.build_prompt
    state = _PromptState()

    def build(camera_name: str, t_sec: int, local_time_str: str, start_hour: int, end_hour: int,
              **kwargs: Any) -> str:
        when = state.local_time or local_time_str
        if template is None:
            return original(camera_name, t_sec, when, start_hour, end_hour, **kwargs)
        return (template.replace("{camera_name}", camera_name).replace("{local_time_str}", when)
                .replace("{owner_language}", str(kwargs.get("owner_language") or "en")))

    inference.build_prompt = build
    try:
        yield state
    finally:
        inference.build_prompt = original


def _load_frames(out_dir: str, rels: Sequence[str]) -> List[Any]:
    import cv2  # noqa: PLC0415

    frames = []
    for rel in rels:
        frame = cv2.imread(os.path.join(out_dir, rel))
        if frame is None:
            raise FileNotFoundError(f"frame {rel} is missing or unreadable")
        frames.append(frame)
    return frames


def _clip_time(row: Dict[str, Any]) -> Optional[str]:
    """The time of day the prompt is told for a clip (the manifest's, else from its name)."""
    return row.get("local_time") or clip_local_time(str(row.get("clip_id", "")))


def input_fingerprint(out_dir: str, row: Dict[str, Any]) -> str:
    """sha256 (first 12 hex) of what the model is shown for a manifest clip: the camera name, the
    clip's time of day and the frame files' bytes. A saved answer is reused only while it matches."""
    h = hashlib.sha256()
    for part in (str(row.get("camera") or ""), _clip_time(row) or ""):
        h.update(part.encode("utf-8") + b"\0")
    for rel in row.get("frames") or []:
        try:
            with open(os.path.join(out_dir, rel), "rb") as f:
                data = f.read()
        except OSError:
            h.update(b"M")                           # missing: asking it fails, and is retried
            continue
        h.update(b"F" + len(data).to_bytes(8, "big") + data)
    return h.hexdigest()[:12]


def _answer_row(clip_id: str, raw: Any, parsed: Any, error: str, prompt_id: str, model: str,
                prompt_sha12: str, input_sha12: str, usage: Optional[Dict[str, int]] = None,
                latency_s: Optional[float] = None) -> Dict[str, Any]:
    """One results line: the model's answer to one clip. An answer that is not JSON, or not a JSON
    object, is an error with the raw text kept; an object is labelled exactly as the box labels it."""
    if not error and parsed is None:
        error = "the answer was not JSON"
    elif not error and not isinstance(parsed, dict):
        error = "answer is not a JSON object"
    p: Dict[str, Any] = parsed if isinstance(parsed, dict) and not error else {}
    summary = p.get("summary")
    pin = (usage or {}).get("prompt_tokens")
    pout = (usage or {}).get("completion_tokens")
    return {
        "clip_id": clip_id,
        "ai_label": "" if error else inference.label_of(p),
        "ai_summary": "" if summary is None else str(summary).strip(),
        "ai_people": p.get("people"), "ai_animals": p.get("animals"), "ai_vehicle_moving": p.get("vehicle_moving"),
        "raw": raw if isinstance(raw, str) else ("" if raw is None else str(raw)),
        "error": error, "prompt_id": prompt_id, "prompt_sha12": prompt_sha12, "model": model,
        "input_sha12": input_sha12,
        "prompt_tokens": pin, "completion_tokens": pout,
        "cost_usd": providers.cost_usd(model, pin or 0, pout or 0) if usage else None,
        "latency_s": None if latency_s is None else round(latency_s, 3),
    }


def _results_paths(out_dir: str, tag: str) -> Tuple[str, str, str]:
    base = os.path.join(out_dir, RESULTS_DIR, tag)
    return base + ".jsonl", base + ".csv", base + ".summary.json"


def _lock_path(out_dir: str, tag: str) -> str:
    return os.path.join(out_dir, RESULTS_DIR, tag + ".lock")


class ResultsLocked(Exception):
    """Another run holds the results file's lock."""


def _pid_alive(pid: int) -> Optional[bool]:
    """Whether process *pid* runs; None when this machine cannot tell (Windows without psutil,
    where ``os.kill(pid, 0)`` would end the process instead of probing it)."""
    try:
        import psutil  # noqa: PLC0415

        return bool(psutil.pid_exists(pid))
    except ImportError:
        pass
    if os.name == "nt":
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _lock_message(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            pid: Optional[int] = int(f.read().strip())
    except (OSError, ValueError):
        pid = None
    alive = _pid_alive(pid) if pid and pid > 0 else None
    if pid and alive is False:
        return (f"{path} was left by process {pid}, which is no longer running (a run that crashed). "
                f"Nothing was asked. If no other run uses these results, delete {path} and run again.")
    who = f"process {pid}" if pid else "another process"
    hint = "" if alive else f" If you are sure no run is going, delete {path}."
    return (f"another run ({who}) is writing these results ({path} exists). Nothing was asked. "
            f"Wait for it to finish, or use another --tag.{hint}")


@contextlib.contextmanager
def _results_lock(path: str) -> Iterator[None]:
    """Hold ``<tag>.lock`` (holding this process's id) while the block runs; refuse if it exists."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ResultsLocked(_lock_message(path)) from None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        yield
    finally:
        with contextlib.suppress(OSError):
            os.remove(path)


def current_answers(rows: Iterable[Dict[str, Any]], fingerprints: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Per clip, the latest answer without an error that was given for the CURRENT inputs; when there
    is none, the latest line. So a later error (a frame briefly missing) never hides a valid answer."""
    rows = list(rows)
    latest = latest_answers(rows)
    for r in rows:
        clip_id = str(r.get("clip_id", ""))
        if not r.get("error") and clip_id in fingerprints and r.get("input_sha12") == fingerprints[clip_id]:
            latest[clip_id] = r
    return latest


def latest_answers(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """The last line per clip. Answers are only ever appended, so a later line (a retry, or a
    re-ask after the inputs changed) replaces an earlier one; the earlier one stays in the file."""
    latest: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        latest[str(r.get("clip_id", ""))] = r
    return latest


def score_rows(manifest: Sequence[Dict[str, Any]], latest: Dict[str, Dict[str, Any]],
               fingerprints: Dict[str, str]) -> Tuple[List[Dict[str, Any]], int]:
    """The rows to score: each manifest clip's latest answer next to the manifest's CURRENT camera
    and truth. Answers for clips no longer in the manifest are kept in the file but not scored; an
    answer given for other inputs (fingerprint differs) is left out and counted as outdated."""
    scored, outdated = [], 0
    for m in manifest:
        r = latest.get(m["clip_id"])
        if r is None:
            continue
        if r.get("input_sha12") != fingerprints[m["clip_id"]]:
            outdated += 1
            continue
        scored.append({**{k: r.get(k) for k in ANSWER_COLUMNS}, "clip_id": m["clip_id"],
                       "camera": m.get("camera", ""), "ours_label": m.get("ours_label", ""),
                       "ours_text": m.get("ours_text", ""), "local_time": _clip_time(m),
                       "batch": m.get("batch", ""), "source": source_of(m), "category": m.get("category"),
                       "subset": m.get("subset"), "day_night": day_night_of(m)})
    return scored, outdated


def _utf8_safe(value: Any) -> Any:
    """A string with any lone surrogate replaced, so it can be written as UTF-8."""
    return value.encode("utf-8", "replace").decode("utf-8") if isinstance(value, str) else value


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    def write(f: Any) -> None:
        writer = csv.DictWriter(f, fieldnames=list(RESULT_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _utf8_safe(row.get(k)) for k in RESULT_COLUMNS})

    _atomic_write(path, write)


def _write_summary(path: str, summary: Dict[str, Any]) -> None:
    _atomic_write(path, lambda f: f.write(json.dumps(summary, indent=2)))


def _score(out_dir: str, tag: str, manifest: Sequence[Dict[str, Any]], latest: Dict[str, Dict[str, Any]],
           fingerprints: Dict[str, str], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Score, then write the derived ``.csv`` and ``.summary.json`` (both rebuilt every time)."""
    _, csv_path, summary_path = _results_paths(out_dir, tag)
    scored, outdated = score_rows(manifest, latest, fingerprints)
    _write_csv(csv_path, scored)
    summary = {**summarize(scored), "outdated": outdated, "tag": tag, "frozen": frozen_status(out_dir), **meta}
    _write_summary(summary_path, summary)
    return summary


def _fingerprints(out_dir: str, manifest: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    return {m["clip_id"]: input_fingerprint(out_dir, m) for m in manifest}


def run_eval(out_dir: str, backend: Any, prompt_text: Optional[str] = None, model: Optional[str] = None,
             tag: Optional[str] = None, limit: Optional[int] = None,
             progress: Callable[[str], None] = log.info, overwrite: bool = False,
             wait: bool = False, strict_frozen: bool = False) -> Dict[str, Any]:
    """Ask *backend* about each clip in the manifest; write results and the summary; return it.

    A frozen set (``FROZEN.json``) is checked first: if it changed since it was frozen, a loud
    warning is logged; with *strict_frozen*, ``FrozenChanged`` is raised (nothing asked) when the
    set changed or was never frozen.

    Only one run per results file: ``<tag>.lock`` is taken first, and ``ResultsLocked`` is raised
    (nothing read or asked) if another run holds it. Answers are appended and never removed:
    a clip is asked again only when its latest answer is an error, or was given for other inputs
    (camera, clip time or frame bytes changed); answers for clips that left the manifest stay.
    The score always uses the latest answer per clip and the manifest's current truth.
    If the results file holds answers from another prompt, wording or model, ``ResultsConflict``
    is raised and nothing is touched, unless *overwrite* is true (those answers are then removed).
    With *wait*, a run of more than CONFIRM_OVER clips names the model and pauses
    CONFIRM_SECONDS first (Ctrl+C aborts).
    """
    check_frozen(out_dir, strict_frozen)
    manifest = read_jsonl(os.path.join(out_dir, MANIFEST))
    prompt_id = prompt_id_of(prompt_text)
    model = model or getattr(backend, "model_name", "unknown")
    tag = tag or default_tag(prompt_text, model, fake=isinstance(backend, FakeBackend))
    sha12 = prompt_sha12_of(prompt_text)
    jsonl_path, _, _ = _results_paths(out_dir, tag)
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

    with _results_lock(_lock_path(out_dir, tag)):
        history = read_jsonl(jsonl_path) if os.path.isfile(jsonl_path) else []
        mine = (prompt_id, sha12, model)
        stale = [r for r in history if (r.get("prompt_id"), r.get("prompt_sha12"), r.get("model")) != mine]
        if stale and not overwrite:
            theirs = sorted({f"prompt {r.get('prompt_id')} (wording {r.get('prompt_sha12') or 'unrecorded'}), "
                             f"model {r.get('model')}" for r in stale})
            raise ResultsConflict(
                f"{jsonl_path} already holds {len({r.get('clip_id') for r in stale})} answers from "
                f"{'; '.join(theirs)}, but this run is prompt {prompt_id} (wording {sha12}), model {model}. "
                "Nothing was changed. Use another --tag, or pass --overwrite to replace them.")
        if stale:
            log.warning("%d rows in %s were made with another prompt or model; replacing them", len(stale), tag)
            history = [r for r in history
                       if (r.get("prompt_id"), r.get("prompt_sha12"), r.get("model")) == mine]

        fingerprints = _fingerprints(out_dir, manifest)
        latest = current_answers(history, fingerprints)

        def reusable(m: Dict[str, Any]) -> bool:
            r = latest.get(m["clip_id"])
            return r is not None and not r.get("error") and r.get("input_sha12") == fingerprints[m["clip_id"]]

        todo = manifest if limit is None else manifest[:limit]
        answered = [m for m in todo if reusable(m)]
        retry = [m for m in todo if m["clip_id"] in latest and latest[m["clip_id"]].get("error")]
        changed = [m for m in todo if m["clip_id"] in latest and not latest[m["clip_id"]].get("error")
                   and not reusable(m)]
        to_ask = len(todo) - len(answered)
        progress(f"asking {to_ask} clips ({len(answered)} already answered, {len(retry)} with errors to retry"
                 + (f", {len(changed)} whose frames, camera or time changed" if changed else "") + ")")
        if wait and to_ask > CONFIRM_OVER:
            progress(f"model {model}: this is a paid run; starting in {CONFIRM_SECONDS} seconds, Ctrl+C to abort "
                     "(--yes skips the wait)")
            time.sleep(CONFIRM_SECONDS)
        # Rewritten whole (every line kept) so a line cut off by a killed run is dropped before appending.
        _write_jsonl(jsonl_path, history)
        asked = 0
        with prompt_override(prompt_text) as state, \
                open(jsonl_path, "a", encoding="utf-8", newline="") as out:
            for n, m in enumerate(todo, 1):
                if reusable(m):
                    continue
                clip_id = m["clip_id"]
                state.local_time = _clip_time(m)
                raw: Any = ""
                try:
                    frames = _load_frames(out_dir, m["frames"])
                    started = time.monotonic()
                    raw, parsed = backend.ask(m, frames)
                    took = time.monotonic() - started
                    result = _answer_row(clip_id, raw, parsed, "", prompt_id, model, sha12, fingerprints[clip_id],
                                         usage=getattr(backend, "last_usage", None), latency_s=took)
                except Exception as exc:  # noqa: BLE001 - recorded, the run goes on
                    result = _answer_row(clip_id, raw, None, f"{type(exc).__name__}: {exc}", prompt_id, model,
                                         sha12, fingerprints[clip_id])
                latest[clip_id] = result
                out.write(json.dumps(result, ensure_ascii=True) + "\n")
                out.flush()
                asked += 1
                error = result["error"]
                progress(f"[{n}/{len(todo)}] {clip_id}: ours {m.get('ours_label')}, "
                         f"AI {result['ai_label'] or 'error'}" + (f" ({error})" if error else ""))

        return _score(out_dir, tag, manifest, latest, fingerprints,
                      {"prompt_id": prompt_id, "model": model, "prompt_sha12": sha12, "asked": asked})


def prompt_sha12_of(prompt_text: Optional[str]) -> str:
    return hashlib.sha256(_prompt_template(prompt_text).encode("utf-8")).hexdigest()[:12]


def _prompt_template(prompt_text: Optional[str]) -> str:
    """The prompt with its placeholders left in, so its hash names the wording only."""
    if prompt_text is not None:
        return prompt_text
    return inference.build_prompt("{camera_name}", 0, "{local_time_str}", 0, 0)


def _one_value(rows: Sequence[Dict[str, Any]], key: str) -> Optional[str]:
    values = sorted({str(r.get(key)) for r in rows if r.get(key) is not None})
    return ", ".join(values) if values else None


def load_summary(out_dir: str, tag: str) -> Dict[str, Any]:
    """The score of results file *tag*, always recomputed from its answers (latest per clip) and the
    current manifest; the ``.summary.json`` and ``.csv`` are rewritten from it, never trusted."""
    jsonl_path, _, _ = _results_paths(out_dir, tag)
    with _results_lock(_lock_path(out_dir, tag)):        # the derived files have one writer too
        manifest = read_jsonl(os.path.join(out_dir, MANIFEST))
        fingerprints = _fingerprints(out_dir, manifest)
        latest = current_answers(read_jsonl(jsonl_path), fingerprints)
        scored, _ = score_rows(manifest, latest, fingerprints)
        meta = {k: _one_value(scored, k) for k in ("prompt_id", "model", "prompt_sha12")}
        return _score(out_dir, tag, manifest, latest, fingerprints, meta)


# ----------------------------------------------------------------------------
# Choosing the box's vision model (spec 2026-10-06 section 6)
# ----------------------------------------------------------------------------
# Our alert text for a break-in: the one kind of alert a new primary may never newly miss.
SERIOUS_RE = re.compile(r"\b(forc\w*|break\w*|broke|climb\w*|pry\w*|jimm\w*|smash\w*)\b", re.IGNORECASE)


def serious_caught(rows: Sequence[Dict[str, Any]]) -> set:
    return {str(r.get("clip_id")) for r in rows
            if not r.get("error") and r.get("ours_label") == "alert" and r.get("ai_label") in CAUGHT_LABELS
            and SERIOUS_RE.search(str(r.get("ours_text") or ""))}


def beats(cand_rows: Sequence[Dict[str, Any]], base_rows: Sequence[Dict[str, Any]]) -> Tuple[bool, str]:
    """Whether *cand* beats *base*: more alerts caught, or as many with fewer false alarms; never
    by newly missing a forced-entry/climbing alert, never with more errors."""
    c, b = summarize(cand_rows), summarize(base_rows)
    if c["errors"] > b["errors"]:
        return False, f"more errors ({c['errors']} vs {b['errors']})"
    lost = sorted(serious_caught(base_rows) - serious_caught(cand_rows))
    if lost:
        return False, f"misses forced-entry/climbing clip(s) the other caught: {', '.join(lost)}"
    if c["alerts_caught"] > b["alerts_caught"]:
        return True, f"catches more alerts ({c['alerts_caught']} vs {b['alerts_caught']})"
    if c["alerts_caught"] == b["alerts_caught"] and c["normal_flagged"] < b["normal_flagged"]:
        return True, f"same alerts, fewer false alarms ({c['normal_flagged']} vs {b['normal_flagged']})"
    return False, (f"not better (alerts {c['alerts_caught']} vs {b['alerts_caught']}, "
                   f"false alarms {c['normal_flagged']} vs {b['normal_flagged']})")


def choose_pair(default: Tuple[str, List[Dict[str, Any]]],
                challenger: Tuple[str, List[Dict[str, Any]]]) -> Tuple[str, str, str]:
    """``(primary, fallback, reason)``: the default stays primary unless the challenger beats it;
    the other one is the fallback."""
    ok, why = beats(challenger[1], default[1])
    if ok:
        return challenger[0], default[0], f"{challenger[0]} {why}"
    return default[0], challenger[0], f"{challenger[0]} {why}"


def _money(v: Optional[float], fmt: str) -> str:
    return "-" if v is None else fmt.format(v)


def format_compare(default: Tuple[str, List[Dict[str, Any]]], challenger: Tuple[str, List[Dict[str, Any]]],
                   others: Dict[str, List[Dict[str, Any]]],
                   reference: Optional[List[Dict[str, Any]]] = None) -> str:
    primary, fallback, why = choose_pair(default, challenger)
    rows_of = {default[0]: default[1], challenger[0]: challenger[1], **others}
    c = summarize(rows_of[primary])
    lines: List[str] = []
    if reference is not None:
        r = summarize(reference)
        if c["alerts_caught"] < r["alerts_caught"] or c["normal_flagged"] > r["normal_flagged"]:
            lines.append(f"NOTE: {primary} is WORSE than gpt-4o on this set (alerts {c['alerts_caught']} vs "
                         f"{r['alerts_caught']}, false alarms {c['normal_flagged']} vs {r['normal_flagged']}); "
                         f"misses: {', '.join(c['missed_alerts']) or '-'}; false alarms: "
                         f"{', '.join(c['false_alarms']) or '-'}")
        else:
            lines.append(f"{primary} is no worse than gpt-4o on this set.")
    lines.append(f"{'model':44} {'alerts':>7} {'false':>7} {'err':>4} {'home al.':>9} {'ext al.':>8} "
                 f"{'night al.':>9} {'tok in':>7} {'sec':>5} {'$/call':>9} {'$/mo@150':>9}")
    table = dict(rows_of)
    if reference is not None:
        table["gpt-4o (reference)"] = reference
    for m, rows in table.items():
        s = summarize(rows)
        night = f"{s['night']['alerts_caught']}/{s['night']['alerts_total']}"
        home = f"{s['home']['alerts_caught']}/{s['home']['alerts_total']}"
        ext = f"{s['external']['alerts_caught']}/{s['external']['alerts_total']}"
        lines.append(f"{m:44} {s['alerts_caught']:>3}/{s['alerts_total']:<3} "
                     f"{s['normal_flagged']:>3}/{s['normal_total']:<3} {s['errors']:>4} {home:>9} {ext:>8} "
                     f"{night:>9} {_num(s['tokens_in_mean']):>7} {_num(s['latency_mean']):>5} "
                     f"{_money(s['cost_per_call'], '${:.5f}'):>9} {_money(s['per_month_150'], '${:.2f}'):>9}")
    lines += [f"primary: {primary}", f"fallback: {fallback}", f"  {why}"]
    for m, rows in others.items():
        if beats(rows, default[1])[0] and beats(rows, challenger[1])[0]:
            lines.append(f"  recommendation: {m} beats both 4B models (owner's call; not chosen automatically)")
    d = summarize(default[1])
    lines.append(f"  ({d['alerts_total']} alerts, {d['home']['alerts_total']} of them from our home cameras: "
                 "a one-clip difference is noise)")
    return "\n".join(lines)


def load_scored(out_dir: str, tag: str) -> List[Dict[str, Any]]:
    """Scored rows of results file *tag*: the latest answer per clip next to the current manifest."""
    jsonl_path, _, _ = _results_paths(out_dir, tag)
    with _results_lock(_lock_path(out_dir, tag)):
        manifest = read_jsonl(os.path.join(out_dir, MANIFEST))
        fingerprints = _fingerprints(out_dir, manifest)
        latest = current_answers(read_jsonl(jsonl_path), fingerprints)
        scored, _ = score_rows(manifest, latest, fingerprints)
    return scored


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def _make_gpt(provider: str, model: str) -> Any:
    try:
        import truststore  # noqa: PLC0415

        truststore.inject_into_ssl()
    except Exception:  # noqa: BLE001
        pass
    try:
        from dotenv import load_dotenv  # noqa: PLC0415
        from .boxconfig import PROJECT_ROOT  # noqa: PLC0415

        load_dotenv(os.path.join(PROJECT_ROOT, "api_key.env"))
    except Exception:  # noqa: BLE001
        pass
    try:
        backend = inference.build_gpt(provider, model, os.environ, timeout=120.0)
    except providers.ProviderError as exc:
        raise SystemExit(f"{exc}, or use --fake.") from None
    backend.model_name = providers.model_key(provider, model)
    return GptAsker(backend)


def _s3_client() -> Any:
    import boto3  # noqa: PLC0415

    return boto3.client("s3")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score the AI's prompt against our human tags.")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="Laptop: 5 frames, evenly across each reviewed clip of home_guard_dataset.",
                          description="Reads home_guard_dataset (annotations/clips.jsonl and clips/), from a local "
                                      "folder or s3://bucket/prefix. For S3: " + S3_HELP)
    prep.add_argument("--out", required=True, help="Folder to write frames/ and manifest.jsonl into.")
    prep.add_argument("--dataset", default=os.environ.get(DATASET_ENV),
                      help=f"home_guard_dataset: a folder or s3://bucket/prefix (default: ${DATASET_ENV}).")
    prep.add_argument("--sources", nargs="+", default=None,
                      help="Only these sources (house, external, uca, smarthome; default: all).")
    prep.add_argument("--batches", nargs="+", default=None, help="Only these batches (default: all).")
    prep.add_argument("--limit", type=int, default=None, help="Read at most N new clips this time.")

    addp = sub.add_parser("add", help="Laptop: add clips from outside the dataset, 5 frames inside each one's "
                                      "annotated segment.",
                          description="Each line of --picks: clip_id, source, batch, camera, ours_text, ours_label, "
                                      "segment [start, end] seconds, and s3_key (in --bucket) or clip (in --dataset); "
                                      "optional category, subset, day_night, local_time, note. " + S3_HELP)
    addp.add_argument("--dir", required=True)
    addp.add_argument("--picks", required=True, help="A jsonl file of picks.")
    addp.add_argument("--bucket", default=BUCKET)
    addp.add_argument("--dataset", default=os.environ.get(DATASET_ENV),
                      help="For picks that name a dataset clip instead of an s3_key.")

    frz = sub.add_parser("freeze", help="Write FROZEN.json: sha256 of the manifest and every frame, and the "
                                        "set's make-up. run then warns if the set changes.")
    frz.add_argument("--dir", required=True)

    runp = sub.add_parser("run", help="Ask the model about every prepared clip and print the score (laptop or "
                                      "box, wherever the provider is reachable).",
                          description="Caveat: the eval sees the whole clip (5 frames evenly across it, unmasked), "
                                      "the box only the last 5 buffered frames before the trigger, zone-masked, "
                                      "so alert recall reads somewhat optimistic compared with the box.")
    runp.add_argument("--dir", required=True, help="The folder prepare wrote.")
    runp.add_argument("--prompt-file", default=None,
                      help="Use this text as the prompt; {camera_name} and {local_time_str} are filled in.")
    runp.add_argument("--model", default="gpt-4o")
    runp.add_argument("--provider", default="openai", choices=sorted(providers.PROVIDERS),
                      help="Where the model is asked: openai, openrouter (OPENROUTER_API_KEY), ollama (the laptop "
                           "GPU, no key), vllm (VLLM_BASE_URL), dashscope-intl (DASHSCOPE_API_KEY).")
    runp.add_argument("--fake", action="store_true", help="No network: a stand-in model, for checking the setup.")
    runp.add_argument("--limit", type=int, default=None, help="Only the first N clips of the manifest.")
    runp.add_argument("--tag", default=None,
                      help="Results name (default: <prompt version, or file-<sha12>>__<model>). A file that holds "
                           "answers from another prompt, wording or model is refused unless --overwrite.")
    runp.add_argument("--overwrite", action="store_true",
                      help="Replace answers in the results file that came from another prompt, wording or model.")
    runp.add_argument("--yes", action="store_true",
                      help="Skip the 5-second pause before a paid run of more than 20 clips.")
    runp.add_argument("--strict-frozen", action="store_true",
                      help=f"Refuse (exit {EXIT_FROZEN}) unless the set is frozen and unchanged since.")

    summ = sub.add_parser("summary", help="Print the score of an earlier run again, recomputed from its "
                          "answers and the current manifest.")
    summ.add_argument("--dir", required=True)
    summ.add_argument("--tag", required=True)

    comp = sub.add_parser("compare", help="One table across results files, and the box's primary and fallback model.")
    comp.add_argument("--dir", required=True)
    comp.add_argument("--default", required=True, help="Results file of Qwen3-VL-4B-Instruct (the default primary).")
    comp.add_argument("--challenger", required=True, help="Results file of Qwen3.5-4B.")
    comp.add_argument("--others", nargs="*", default=[], help="Results files of the other models, reported only.")
    comp.add_argument("--reference", default=None, help="Results file of gpt-4o on the same prompt.")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr,
        format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S",
    )

    if args.command in ("prepare", "add"):
        try:
            if args.command == "prepare" and not args.dataset:
                raise ValueError(f"name the dataset with --dataset or ${DATASET_ENV}")
            dataset = Dataset(args.dataset, client=_s3_client() if args.dataset.startswith("s3://") else None) \
                if args.dataset else None
            if args.command == "prepare":
                print(format_prepare_counts(prepare(args.out, dataset, sources=args.sources, batches=args.batches,
                                                    limit=args.limit)))
            else:
                picks = read_jsonl(args.picks)
                client = _s3_client() if any(p.get("s3_key") for p in picks) else None
                print(format_add_counts(add_picks(args.dir, picks, client=client, bucket=args.bucket,
                                                  dataset=dataset)))
        except FrozenSet as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return EXIT_FROZEN
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}\n{S3_HELP}", file=sys.stderr)
            return 1
        return 0

    if args.command == "freeze":
        try:
            record = freeze(args.dir)
        except (OSError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(format_freeze(record))
        return 0

    if args.command == "run":
        if not os.path.isfile(os.path.join(args.dir, MANIFEST)):
            print(f"Error: {os.path.join(args.dir, MANIFEST)} not found; run prepare first.", file=sys.stderr)
            return 1
        if args.strict_frozen:
            try:
                check_frozen(args.dir, strict=True)      # before a backend is built or a key is needed
            except FrozenChanged as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return EXIT_FROZEN
        prompt_text = None
        if args.prompt_file:
            with open(args.prompt_file, encoding="utf-8") as f:
                prompt_text = f.read()
        model_id = providers.model_key(args.provider, args.model)
        backend = FakeBackend() if args.fake else _make_gpt(args.provider, args.model)
        model = None if args.fake else model_id
        tag = args.tag or default_tag(prompt_text, "fake" if args.fake else model_id, fake=args.fake)
        try:
            summary = run_eval(args.dir, backend, prompt_text=prompt_text, model=model, tag=tag,
                               limit=args.limit, overwrite=args.overwrite, wait=not (args.fake or args.yes),
                               strict_frozen=args.strict_frozen)
        except FrozenChanged as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return EXIT_FROZEN
        except ResultsConflict as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        except ResultsLocked as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 3
        except KeyboardInterrupt:
            print("Aborted.", file=sys.stderr)
            return 130
        print(format_summary(summary))
        return 0

    if args.command == "compare":
        tags = [args.default, args.challenger, *args.others] + ([args.reference] if args.reference else [])
        for tag in tags:
            if not os.path.isfile(_results_paths(args.dir, tag)[0]):
                print(f"Error: no results named {tag} in {os.path.join(args.dir, RESULTS_DIR)}", file=sys.stderr)
                return 1

        def named(tag: str) -> Tuple[str, List[Dict[str, Any]]]:
            rows = load_scored(args.dir, tag)
            return (_one_value(rows, "model") or tag), rows

        others = dict(named(t) for t in args.others)
        reference = load_scored(args.dir, args.reference) if args.reference else None
        print(format_compare(named(args.default), named(args.challenger), others, reference))
        return 0

    if not os.path.isfile(_results_paths(args.dir, args.tag)[0]):
        print(f"Error: no results named {args.tag} in {os.path.join(args.dir, RESULTS_DIR)}", file=sys.stderr)
        return 1
    if not os.path.isfile(os.path.join(args.dir, MANIFEST)):
        print(f"Error: {os.path.join(args.dir, MANIFEST)} not found; the score needs our current tags.",
              file=sys.stderr)
        return 1
    try:
        summary = load_summary(args.dir, args.tag)
    except ResultsLocked as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 3
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
