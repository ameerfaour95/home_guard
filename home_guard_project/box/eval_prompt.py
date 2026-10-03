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
(exit 2) unless ``--overwrite`` is given.

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

from . import inference

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
RESULT_COLUMNS = ("clip_id", "camera", "ours_label", "ours_text", "ai_label", "ai_summary", "ai_people",
                  "ai_vehicle_moving", "raw", "error", "prompt_id", "prompt_sha12", "model")

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
    return {
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
    }


def _pct(value: Optional[float]) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


def _num(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.1f}"


def format_summary(s: Dict[str, Any]) -> str:
    head = []
    if s.get("tag"):
        head.append(f"tag {s['tag']}  prompt {s.get('prompt_id', '?')}  model {s.get('model', '?')}")
    return "\n".join(head + [
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
    ])


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
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(buf.tobytes())
    os.replace(tmp, path)


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
    """The rows of a jsonl file. A cut-off last line (a run killed mid-write) is skipped with a
    warning; an unparseable line anywhere else still raises."""
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            log.warning("%s: the last line is cut off; skipping it", path)
    return rows


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


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
                      "vehicle_moving": False}
        elif ours == "empty":
            parsed = {"summary": NO_ACTIVITY, "label": "normal", "people": 0, "vehicle_moving": False}
        else:
            parsed = {"summary": "A person walks to the door.", "label": "normal", "people": 1,
                      "vehicle_moving": False}
        return json.dumps(parsed), parsed


class GptAsker:
    """Asks inference.GptBackend as the box does (same prompt function and request, 5 frames; which frames differ)."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.model_name = getattr(backend, "model_name", "gpt")

    def ask(self, row: Dict[str, Any], frames: List[Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
        # Hours 0,0 = always inside the alert window; the prompt does not use them.
        return self.backend.analyze(frames, row.get("camera", ""), int(time.time()), 0, 0)


class _PromptState:
    local_time: Optional[str] = None


@contextlib.contextmanager
def prompt_override(template: Optional[str]) -> Iterator[_PromptState]:
    """Swap ``inference.build_prompt`` while the block runs, and always put it back.

    With a *template*, the prompt is its text with ``{camera_name}`` and
    ``{local_time_str}`` filled in (nothing else is touched, so JSON braces are
    safe). Without one, the box's own prompt is used. Either way, when
    ``state.local_time`` is set it replaces the time of day the backend passes in.
    """
    original = inference.build_prompt
    state = _PromptState()

    def build(camera_name: str, t_sec: int, local_time_str: str, start_hour: int, end_hour: int) -> str:
        when = state.local_time or local_time_str
        if template is None:
            return original(camera_name, t_sec, when, start_hour, end_hour)
        return template.replace("{camera_name}", camera_name).replace("{local_time_str}", when)

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


def _result_row(row: Dict[str, Any], raw: str, parsed: Optional[Dict[str, Any]], error: str,
                prompt_id: str, model: str, prompt_sha12: str = "") -> Dict[str, Any]:
    p = parsed or {}
    return {
        "clip_id": row["clip_id"], "camera": row.get("camera", ""), "ours_label": row.get("ours_label", ""),
        "ours_text": row.get("ours_text", ""),
        "ai_label": inference.label_of(parsed) if parsed else "",
        "ai_summary": str(p.get("summary", "")).strip(),
        "ai_people": p.get("people"), "ai_vehicle_moving": p.get("vehicle_moving"),
        "raw": raw or "", "error": error, "prompt_id": prompt_id, "prompt_sha12": prompt_sha12, "model": model,
    }


def _results_paths(out_dir: str, tag: str) -> Tuple[str, str, str]:
    base = os.path.join(out_dir, RESULTS_DIR, tag)
    return base + ".jsonl", base + ".csv", base + ".summary.json"


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(RESULT_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in RESULT_COLUMNS})
    os.replace(tmp, path)


def run_eval(out_dir: str, backend: Any, prompt_text: Optional[str] = None, model: Optional[str] = None,
             tag: Optional[str] = None, limit: Optional[int] = None,
             progress: Callable[[str], None] = log.info, overwrite: bool = False,
             wait: bool = False) -> Dict[str, Any]:
    """Ask *backend* about each clip in the manifest; write results and the summary; return it.

    Resumes: a clip already answered for the same prompt id, prompt wording (hash) and model
    without an error is not asked again. A model error on one clip is recorded and the run
    goes on. If the results file holds rows from another prompt, wording or model,
    ``ResultsConflict`` is raised and nothing is touched, unless *overwrite* is true.
    With *wait*, a run of more than CONFIRM_OVER clips names the model and pauses
    CONFIRM_SECONDS first (Ctrl+C aborts).
    """
    manifest = read_jsonl(os.path.join(out_dir, MANIFEST))
    prompt_id = prompt_id_of(prompt_text)
    model = model or getattr(backend, "model_name", "unknown")
    tag = tag or default_tag(prompt_text, model, fake=isinstance(backend, FakeBackend))
    sha12 = prompt_sha12_of(prompt_text)
    jsonl_path, csv_path, summary_path = _results_paths(out_dir, tag)
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

    done: Dict[str, Dict[str, Any]] = {}
    if os.path.isfile(jsonl_path):
        for r in read_jsonl(jsonl_path):           # appended as it went: the last line per clip wins
            done[r["clip_id"]] = r
        stale = [cid for cid, r in done.items()
                 if (r.get("prompt_id"), r.get("prompt_sha12"), r.get("model")) != (prompt_id, sha12, model)]
        if stale and not overwrite:
            stale_set = set(stale)
            theirs = sorted({f"prompt {r.get('prompt_id')} (wording {r.get('prompt_sha12') or 'unrecorded'}), "
                             f"model {r.get('model')}" for cid, r in done.items() if cid in stale_set})
            raise ResultsConflict(
                f"{jsonl_path} already holds {len(stale)} answers from {'; '.join(theirs)}, but this run is "
                f"prompt {prompt_id} (wording {sha12}), model {model}. Nothing was changed. "
                "Use another --tag, or pass --overwrite to replace them.")
        if stale:
            log.warning("%d rows in %s were made with another prompt or model; replacing them", len(stale), tag)
            for cid in stale:
                del done[cid]

    todo = manifest if limit is None else manifest[:limit]
    answered = [r for r in todo if r["clip_id"] in done and not done[r["clip_id"]].get("error")]
    retry = [r for r in todo if r["clip_id"] in done and done[r["clip_id"]].get("error")]
    to_ask = len(todo) - len(answered)
    progress(f"asking {to_ask} clips ({len(answered)} already answered, {len(retry)} with errors to retry)")
    if wait and to_ask > CONFIRM_OVER:
        progress(f"model {model}: this is a paid run; starting in {CONFIRM_SECONDS} seconds, Ctrl+C to abort "
                 "(--yes skips the wait)")
        time.sleep(CONFIRM_SECONDS)
    _write_jsonl(jsonl_path, done.values())
    asked = 0
    with prompt_override(prompt_text) as state, open(jsonl_path, "a", encoding="utf-8") as out:
        for n, row in enumerate(todo, 1):
            prev = done.get(row["clip_id"])
            if prev is not None and not prev.get("error"):
                continue
            state.local_time = row.get("local_time") or clip_local_time(row["clip_id"])
            raw, parsed, error = "", None, ""
            try:
                frames = _load_frames(out_dir, row["frames"])
                raw, parsed = backend.ask(row, frames)
            except Exception as exc:  # noqa: BLE001 - recorded, the run goes on
                error = f"{type(exc).__name__}: {exc}"
            if not error and parsed is None:
                error = "the answer was not JSON"
            result = _result_row(row, raw, parsed, error, prompt_id, model, sha12)
            done[row["clip_id"]] = result
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            asked += 1
            progress(f"[{n}/{len(todo)}] {row['clip_id']}: ours {row.get('ours_label')}, "
                     f"AI {result['ai_label'] or 'error'}" + (f" ({error})" if error else ""))

    rows = [done[r["clip_id"]] for r in manifest if r["clip_id"] in done]
    _write_jsonl(jsonl_path, rows)
    _write_csv(csv_path, rows)
    summary = {**summarize(rows), "tag": tag, "prompt_id": prompt_id, "model": model, "asked": asked,
               "prompt_sha12": sha12}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def prompt_sha12_of(prompt_text: Optional[str]) -> str:
    return hashlib.sha256(_prompt_template(prompt_text).encode("utf-8")).hexdigest()[:12]


def _prompt_template(prompt_text: Optional[str]) -> str:
    """The prompt with its placeholders left in, so its hash names the wording only."""
    if prompt_text is not None:
        return prompt_text
    return inference.build_prompt("{camera_name}", 0, "{local_time_str}", 0, 0)


def load_summary(out_dir: str, tag: str) -> Dict[str, Any]:
    jsonl_path, _, summary_path = _results_paths(out_dir, tag)
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            return json.load(f)
    return {**summarize(read_jsonl(jsonl_path)), "tag": tag}


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def _make_gpt(model: str) -> Any:
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
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set (put it in api_key.env at the repo root), or use --fake.")
    return GptAsker(inference.GptBackend(key, model=model))


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

    runp = sub.add_parser("run", help="Box: ask the model about every prepared clip and print the score.",
                          description="Caveat: the eval sees the whole clip (5 frames evenly across it, unmasked), "
                                      "the box only the last 5 buffered frames before the trigger, zone-masked, "
                                      "so alert recall reads somewhat optimistic compared with the box.")
    runp.add_argument("--dir", required=True, help="The folder prepare wrote.")
    runp.add_argument("--prompt-file", default=None,
                      help="Use this text as the prompt; {camera_name} and {local_time_str} are filled in.")
    runp.add_argument("--model", default="gpt-4o")
    runp.add_argument("--fake", action="store_true", help="No network: a stand-in model, for checking the setup.")
    runp.add_argument("--limit", type=int, default=None, help="Only the first N clips of the manifest.")
    runp.add_argument("--tag", default=None,
                      help="Results name (default: <prompt version, or file-<sha12>>__<model>). A file that holds "
                           "answers from another prompt, wording or model is refused unless --overwrite.")
    runp.add_argument("--overwrite", action="store_true",
                      help="Replace answers in the results file that came from another prompt, wording or model.")
    runp.add_argument("--yes", action="store_true",
                      help="Skip the 5-second pause before a paid run of more than 20 clips.")

    summ = sub.add_parser("summary", help="Print the score of an earlier run again.")
    summ.add_argument("--dir", required=True)
    summ.add_argument("--tag", required=True)

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
        backend = FakeBackend() if args.fake else _make_gpt(args.model)
        model = None if args.fake else args.model
        tag = args.tag or default_tag(prompt_text, "fake" if args.fake else args.model, fake=args.fake)
        try:
            summary = run_eval(args.dir, backend, prompt_text=prompt_text, model=model, tag=tag,
                               limit=args.limit, overwrite=args.overwrite, wait=not (args.fake or args.yes))
        except ResultsConflict as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            print("Aborted.", file=sys.stderr)
            return 130
        print(format_summary(summary))
        return 0

    try:
        summary = load_summary(args.dir, args.tag)
    except FileNotFoundError:
        print(f"Error: no results named {args.tag} in {os.path.join(args.dir, RESULTS_DIR)}", file=sys.stderr)
        return 1
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
