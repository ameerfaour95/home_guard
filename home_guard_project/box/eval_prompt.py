"""Score the cloud VLM's prompt against our human tags.

Two steps, on two machines, because the box cannot read tagging/ on S3:

    # On the laptop (reads S3): pick 5 frames from every tagged home clip.
    python -m home_guard_project.box.eval_prompt prepare --out eval_set
    # Copy eval_set/ to the box, then (calls the OpenAI API, about 220 clips):
    python -m home_guard_project.box.eval_prompt run --dir eval_set
    python -m home_guard_project.box.eval_prompt run --dir eval_set --prompt-file my_prompt.txt
    python -m home_guard_project.box.eval_prompt summary --dir eval_set --tag <tag>

``prepare`` writes ``frames/<clip_id>_<i>.jpg`` and ``manifest.jsonl`` (one row per
clip with our label and text). ``run`` asks the model about every clip with 5 frames,
writes ``results/<tag>.jsonl`` / ``.csv`` / ``.summary.json`` and prints the score.
Both steps resume where they stopped.

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

Our label comes from the tagger's text: ``[alert]`` (or the typo ``[alet]``) is
``alert``, "No special activity" is ``empty``, anything else ``normal``. Rows
marked ``[delete]`` or with no text are left out, as in the training set.

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
TAGGING_PREFIX = "tagging/"
HOME_BATCH_PREFIX = "ameer_house"
# Where a tagged clip's mp4 may live today, best first (video_s3_path is often stale).
CLIP_PREFIXES = ("dataset_multi/clips/", "dataset_ameer_house/", "tagging/")
JSONL_RE = re.compile(r"^tagging/([^/]+)/analysis_output/vlm_training\.jsonl$")

CONFIRM_OVER = 20          # a real run of more clips than this names the model and waits
CONFIRM_SECONDS = 5
FRAME_COUNT = 5            # what the box sends (inference.AlertSettings.clip_frames)
MAX_SIDE = 1280
JPEG_QUALITY = 90

MANIFEST = "manifest.jsonl"
FRAMES_DIR = "frames"
RESULTS_DIR = "results"
# What a results .jsonl line holds: the model's answer and what it answered (prompt, model,
# input fingerprint). Our truth is not stored: it is read from the current manifest when scoring.
USAGE_COLUMNS = ("prompt_tokens", "completion_tokens", "cost_usd", "latency_s")
ANSWER_COLUMNS = ("clip_id", "ai_label", "ai_summary", "ai_people", "ai_animals", "ai_vehicle_moving",
                  "raw", "error", "prompt_id", "prompt_sha12", "model", "input_sha12") + USAGE_COLUMNS
# The scored table (.csv): the answer next to the manifest's current camera and truth.
RESULT_COLUMNS = ("clip_id", "camera", "ours_label", "ours_text", "ai_label", "ai_summary", "ai_people", "ai_animals",
                  "ai_vehicle_moving", "raw", "error", "prompt_id", "prompt_sha12", "model", "input_sha12"
                  ) + USAGE_COLUMNS + ("local_time", "batch")
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


def is_dropped(description: str) -> bool:
    """A row the taggers threw out (``[delete]``) or never described."""
    return not (description or "").strip() or "[delete]" in description.lower()


def clip_stem(clip_id_or_key: str) -> str:
    return os.path.splitext(os.path.basename(str(clip_id_or_key)))[0]


def _key_rank(key: str) -> Tuple[int, int, str]:
    prefix = next((i for i, p in enumerate(CLIP_PREFIXES) if key.startswith(p)), len(CLIP_PREFIXES))
    crop = 1 if "/vlm_crops/" in f"/{key}" else 0      # a person crop shares the clip's name
    return crop, prefix, key                            # any full clip beats any crop


def build_index(keys: Iterable[str]) -> Dict[str, str]:
    """``{stem: best mp4 key}``: a full clip before any vlm_crops copy, then dataset_multi/clips/,
    dataset_ameer_house/, tagging/ in that order; ties broken by sorting."""
    best: Dict[str, str] = {}
    for key in keys:
        if not key.lower().endswith(".mp4"):
            continue
        stem = clip_stem(key)
        if stem not in best or _key_rank(key) < _key_rank(best[stem]):
            best[stem] = key
    return best


def match_rows(rows: Sequence[Dict[str, Any]], index: Dict[str, str]
               ) -> Tuple[List[Tuple[Dict[str, Any], str]], List[str]]:
    """Pair each row with its mp4 key by ``clip_id``; the ids with no mp4 come back separately."""
    matched, unmatched = [], []
    for row in rows:
        key = index.get(clip_stem(row.get("clip_id", "")))
        if key:
            matched.append((row, key))
        else:
            unmatched.append(str(row.get("clip_id", "")))
    return matched, unmatched


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
    return str(row.get("batch") or "").startswith(HOME_BATCH_PREFIX)


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
        "day": _counts([r for r in rows if is_night(r.get("local_time")) is False]),
        "night": _counts([r for r in rows if is_night(r.get("local_time")) is True]),
        "unknown_time": sum(is_night(r.get("local_time")) is None for r in rows),
        "home": _counts([r for r in rows if _is_home(r)]),
        "external": _counts([r for r in rows if not _is_home(r)]),
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


def sample_frames(path: str, k: int = FRAME_COUNT) -> List[Any]:
    """``k`` evenly spaced frames of the clip, long side at most MAX_SIDE; ``[]`` if it will not decode."""
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(path)
    claimed = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if cap.isOpened() else 0
    cap.release()
    wanted = sample_indices(claimed, k)
    got = _read_at(path, wanted) if wanted else {}
    if not wanted or any(i not in got for i in wanted):
        # The header's count was missing or wrong: count what really decodes, then pick again.
        _, decoded = _count_frames(path)
        wanted = sample_indices(decoded, k)
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
# prepare (laptop, reads S3)
# ----------------------------------------------------------------------------
def _list_keys(client: Any, bucket: str, prefix: str) -> List[str]:
    keys: List[str] = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []) or [])
    return keys


def _read_jsonl_from_s3(client: Any, bucket: str, key: str, tmp_dir: str) -> List[Dict[str, Any]]:
    local = os.path.join(tmp_dir, "rows.jsonl")
    client.download_file(Bucket=bucket, Key=key, Filename=local)
    try:
        return read_jsonl(local)
    finally:
        os.remove(local)


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


def prepare(out_dir: str, client: Any, bucket: str = BUCKET, batches: Optional[Sequence[str]] = None,
            limit: Optional[int] = None) -> Dict[str, Any]:
    """Fetch frames for every tagged clip of *batches* (default: the home batches) into *out_dir*.

    *limit* caps how many clips are newly downloaded this time; clips already in
    ``frames/`` are kept and never downloaded again.
    """
    os.makedirs(os.path.join(out_dir, FRAMES_DIR), exist_ok=True)

    tagging_keys = _list_keys(client, bucket, TAGGING_PREFIX)
    jsonl_keys = {m.group(1): k for k in tagging_keys if (m := JSONL_RE.match(k))}
    if batches:
        chosen = list(dict.fromkeys(batches))
        for b in chosen:
            if b not in jsonl_keys:
                log.warning("batch %s has no analysis_output/vlm_training.jsonl on S3", b)
        chosen = [b for b in chosen if b in jsonl_keys]
    else:
        chosen = sorted(b for b in jsonl_keys if b.startswith(HOME_BATCH_PREFIX))
    log.info("batches: %s", ", ".join(chosen) or "none")

    keys = list(tagging_keys)
    for prefix in CLIP_PREFIXES:
        if prefix != TAGGING_PREFIX:
            keys += _list_keys(client, bucket, prefix)
    index = build_index(keys)
    log.info("indexed %d clip names from %d keys", len(index), len(keys))

    rows: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp_dir:
        for batch in sorted(chosen):
            for row in _read_jsonl_from_s3(client, bucket, jsonl_keys[batch], tmp_dir):
                rows.append({**row, "_batch": batch})

    kept = [r for r in rows if not is_dropped(str(r.get("description") or ""))]
    matched, unmatched = match_rows(kept, index)
    matched.sort(key=lambda pair: (pair[0]["_batch"], clip_stem(pair[0]["clip_id"])))

    counts: Dict[str, Any] = {"rows": len(rows), "dropped": len(rows) - len(kept), "matched": len(matched),
                              "unmatched": unmatched, "new": 0, "cached": 0, "failed": [],
                              "skipped_by_limit": 0, "duplicates": []}
    manifest: List[Dict[str, Any]] = []
    seen: set = set()
    for row, key in matched:
        clip_id = clip_stem(row["clip_id"])
        if clip_id in seen:
            counts["duplicates"].append(clip_id)     # tagged in two batches: the first batch wins
            continue
        seen.add(clip_id)
        if _frames_exist(out_dir, clip_id):
            counts["cached"] += 1
        elif limit is not None and counts["new"] + len(counts["failed"]) >= limit:
            counts["skipped_by_limit"] += 1
            continue
        elif _fetch_frames(client, bucket, key, out_dir, clip_id):
            counts["new"] += 1
        else:
            counts["failed"].append(clip_id)
            continue
        label, text = parse_truth(str(row.get("description") or ""))
        manifest.append({
            "clip_id": clip_id, "batch": row["_batch"], "camera": row.get("camera_name") or "",
            "ours_text": text, "ours_label": label, "frames": frame_paths(clip_id), "s3_key": key,
            "local_time": clip_local_time(clip_id),
        })
    _write_jsonl(os.path.join(out_dir, MANIFEST), manifest)
    return counts


def _fetch_frames(client: Any, bucket: str, key: str, out_dir: str, clip_id: str) -> bool:
    fd, tmp = tempfile.mkstemp(suffix=".mp4", dir=tempfile.gettempdir())
    os.close(fd)
    try:
        client.download_file(Bucket=bucket, Key=key, Filename=tmp)
        frames = sample_frames(tmp)
        if len(frames) != FRAME_COUNT:
            log.warning("%s: could not read %d frames from %s", clip_id, FRAME_COUNT, key)
            return False
        for rel, frame in zip(frame_paths(clip_id), frames):
            _write_jpeg(os.path.join(out_dir, rel), frame)
        log.info("prepared %s", clip_id)
        return True
    except Exception as exc:  # noqa: BLE001 - one bad clip must not stop the rest
        log.warning("%s: %s", clip_id, exc)
        return False
    finally:
        with contextlib.suppress(OSError):
            os.remove(tmp)


def format_prepare_counts(c: Dict[str, Any]) -> str:
    lines = [
        f"rows read         {c['rows']}  ({c['dropped']} marked [delete] or empty, left out)",
        f"matched           {c['matched']}",
        f"unmatched         {len(c['unmatched'])}" + (f": {', '.join(c['unmatched'])}" if c["unmatched"] else ""),
        f"newly prepared    {c['new']}",
        f"cached            {c['cached']}",
    ]
    if c["failed"]:
        lines.append(f"failed            {len(c['failed'])}: {', '.join(c['failed'])}")
    if c["skipped_by_limit"]:
        lines.append(f"left for later    {c['skipped_by_limit']} (--limit)")
    if c["duplicates"]:
        lines.append(f"tagged twice      {len(c['duplicates'])} (first batch kept): {', '.join(c['duplicates'])}")
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
                       "batch": m.get("batch", "")})
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
    summary = {**summarize(scored), "outdated": outdated, "tag": tag, **meta}
    _write_summary(summary_path, summary)
    return summary


def _fingerprints(out_dir: str, manifest: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    return {m["clip_id"]: input_fingerprint(out_dir, m) for m in manifest}


def run_eval(out_dir: str, backend: Any, prompt_text: Optional[str] = None, model: Optional[str] = None,
             tag: Optional[str] = None, limit: Optional[int] = None,
             progress: Callable[[str], None] = log.info, overwrite: bool = False,
             wait: bool = False) -> Dict[str, Any]:
    """Ask *backend* about each clip in the manifest; write results and the summary; return it.

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
    lines.append("  (54 alerts, 18 of them from our home cameras: a one-clip difference is noise)")
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score the AI's prompt against our human tags.")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="Laptop: fetch 5 frames, evenly across each tagged home clip, from S3.",
                          description="Laptop only (reads tagging/ on S3). " + S3_HELP)
    prep.add_argument("--out", required=True, help="Folder to write frames/ and manifest.jsonl into.")
    prep.add_argument("--batches", nargs="+", default=None,
                      help=f"Tagging batches to use (default: every batch starting with {HOME_BATCH_PREFIX}).")
    prep.add_argument("--limit", type=int, default=None, help="Download at most N new clips this time.")
    prep.add_argument("--bucket", default=BUCKET)

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

    if args.command == "prepare":
        try:
            import boto3  # noqa: PLC0415

            counts = prepare(args.out, boto3.client("s3"), bucket=args.bucket, batches=args.batches,
                             limit=args.limit)
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}\n{S3_HELP}", file=sys.stderr)
            return 1
        print(format_prepare_counts(counts))
        return 0

    if args.command == "run":
        if not os.path.isfile(os.path.join(args.dir, MANIFEST)):
            print(f"Error: {os.path.join(args.dir, MANIFEST)} not found; run prepare first.", file=sys.stderr)
            return 1
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
                               limit=args.limit, overwrite=args.overwrite, wait=not (args.fake or args.yes))
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
