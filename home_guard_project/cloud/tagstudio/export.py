"""From our tags to the training set and the eval. Local files only; nothing is written to S3.

1. **VLM training JSONL** (``vlm_training.jsonl``): the contract ``analysis/analyze.py`` ``write_vlm_jsonl``
   writes (``video_s3_path``, ``vlm_crop_s3_path``, ``description`` and the clip's metadata) plus the taxonomy:
   ``category``, ``category_name``, ``raw_label``, ``other_text`` and ``observation`` (zone, movement, flags,
   visibility, evidence frame). A clip marked delete, or with no description, is left out, as before.
2. **Eval manifest rows** (``eval_manifest.jsonl``): rows in ``box/eval_prompt.py``'s manifest shape (clip_id,
   batch, camera, ours_text, ours_label, frames, local_time) plus category / raw_label. ``frames_dir`` also writes
   the 5 frames per clip there, so ``eval_prompt run --dir`` can score a model against the rows. Not run here.

3. **LLaMA-Factory sharegpt** (``vlm_sharegpt.json``): one conversation per studio tag, the user turn a ``<video>``
   tag and the question, the assistant turn the tag as the box's own JSON answer in the exact field order of the
   clip's prompt version (fleet_contract/prompt_schemas.py), ``videos`` the crop the model saw (else the clip).
   Every row carries as many ``<video>`` tags as videos (the loader refuses anything else).

Decided means: our tag (unless marked delete or, by default, needs-check), or a migrated old tag that nothing
contradicts. A contradicted old tag waits for us.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ...fleet_contract import taxonomy
from . import evalfmt
from . import queue as work_queue
from .fields import Tag, default_raw_label
from .items import ALERT, EMPTY, OLD, ClipItem

TRAINING_FILE = "vlm_training.jsonl"
EVAL_FILE = "eval_manifest.jsonl"
SHAREGPT_FILE = "vlm_sharegpt.json"
VIDEO_TAG = "<video>"
QUESTION = "Describe what happens in this home security camera clip and answer in JSON."
_ANSWER_DEFAULTS = {"string": "", "integer": 0, "boolean": False, "array": []}
OBSERVATION_FIELDS = ("zone", "movement", "flags", "visibility", "appearance", "evidence_frame", "evidence_sec")


def decision(item: ClipItem, tag: Optional[Tag], include_needs_check: bool = False
             ) -> Tuple[Optional[Dict[str, Any]], str]:
    """``(decided tag, "")`` or ``(None, why it is left out)``."""
    old = item.opinions.get(OLD)
    if tag is not None:
        f = tag.fields
        if f.get("delete"):
            return None, "delete"
        if f.get("needs_check") and not include_needs_check:
            return None, "needs_check"
        category = f.get("category", "")
        return {
            "description": f.get("description") or (old.text if old else ""),
            "category": category, "other_text": f.get("other_text", ""),
            "raw_label": f.get("raw_label") or default_raw_label(category),
            "observation": {k: f[k] for k in OBSERVATION_FIELDS if f.get(k) not in (None, "", [])},
            "notes": f.get("notes", ""), "tag_source": "studio", "tagged_by": tag.by, "tagged_at": tag.at,
            "suggested_by": f.get("suggested_by", ""), "suggestion_use": f.get("suggestion_use", ""),
            "empty": category == "N10",
            "prompt_version": f.get("prompt_version", ""), "fields": dict(f),
            "tagger_words": f.get("tagger_words", ""), "tagger_language": f.get("tagger_language", ""),
            "converted_by": f.get("converted_by", ""),
        }, ""
    if old is not None and old.detail.get("delete"):
        return None, "delete"
    if old is None or not old.effective_label():
        return None, "untagged"
    if work_queue.assess(item).tier != work_queue.DONE:
        return None, "contradicted"
    label = old.effective_label()
    # An old tag is a level only (normal / nothing there / [alert]): no category is invented for it, and an [alert]
    # did not say suspicious or escalation, so its raw label stays unknown too.
    return {
        "description": old.text, "category": None, "other_text": "",
        "raw_label": "normal" if label in ("normal", EMPTY) else None,
        "observation": {}, "notes": "", "tag_source": "migrated", "tagged_by": old.detail.get("by", ""),
        "tagged_at": "", "empty": label == EMPTY, "old_alert": label == ALERT, "old_label": label,
    }, ""


def training_record(item: ClipItem, d: Dict[str, Any]) -> Dict[str, Any]:
    cat = taxonomy.get(d["category"])
    return {
        "video_s3_path": item.video_s3,
        "vlm_crop_s3_path": item.crop_s3,
        "description": d["description"],
        "camera_name": item.camera,                              # the raw id, for training traceability
        "camera_display": item.info.get("camera_display") or "",  # what staff read: the owner's name, else Camera N
        "duration_sec": round(float(item.duration_sec), 2) if item.duration_sec is not None else None,
        "num_persons": item.info.get("num_persons"),
        "num_cars": item.info.get("num_cars"),
        "kind": item.info.get("kind", ""),
        "date": item.date,
        "clip_id": item.clip_id,
        "source": item.source,
        "batch": item.batch,
        "category": d["category"],
        "category_name": cat.name if cat else ("other" if d["category"] == taxonomy.OTHER else None),
        "other_text": d["other_text"],
        "raw_label": d["raw_label"],
        "alert": d["raw_label"] in ("suspicious", "escalation") or bool(d.get("old_alert")),
        "observation": d["observation"],
        "tag_source": d["tag_source"],
        "tagged_by": d["tagged_by"],
        "tagged_at": d["tagged_at"],
        "old_label": d.get("old_label"),
        "suggested_by": d.get("suggested_by", ""),
        "suggestion_use": d.get("suggestion_use", ""),
        "taxonomy_version": taxonomy.TAXONOMY_VERSION,
        "video_expires": bool(item.info.get("production_only")),
        "prompt_version": d.get("prompt_version", ""),
        "answer": answer(item, d),
        "tagger_words": d.get("tagger_words", ""), "tagger_language": d.get("tagger_language", ""),
        "converted_by": d.get("converted_by", ""),
    }


def answer(item: ClipItem, d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A studio tag as the box's own JSON answer: the fields of the clip's prompt-version schema, in its order (summary
    is the description; a field the tag does not hold takes the schema's empty value, label defaults to the raw label,
    people to the clip's count). None for a migrated old tag (it holds no structured answer) or without a raw label."""
    from ...fleet_contract import prompt_schemas as ps  # noqa: PLC0415

    if d.get("tag_source") != "studio" or not d.get("raw_label"):
        return None
    f = d.get("fields") or {}
    _, schema = ps.answer_schema(d.get("prompt_version"))
    known = {"summary": d["description"], "category": d.get("category") or "", "raw_label": d["raw_label"],
             "label": f.get("label") or d["raw_label"], "people": f.get("people", item.info.get("num_persons")),
             "vehicles": f.get("vehicles", item.info.get("num_cars"))}
    out: Dict[str, Any] = {}
    for name, spec in schema["properties"].items():
        value = known[name] if name in known else f.get(name)
        if value is None or (spec.get("type") == "integer" and not isinstance(value, int)):
            value = _ANSWER_DEFAULTS.get(spec.get("type"), "")
        if "enum" in spec and value not in spec["enum"]:
            return None                      # a value outside the schema never becomes a training answer
        out[name] = value
    return out


def sharegpt_rows(training: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """LLaMA-Factory sharegpt rows (``messages`` + ``videos``) of the training rows that hold a structured answer."""
    rows = []
    for record in training:
        video = record.get("vlm_crop_s3_path") or record.get("video_s3_path")
        if not record.get("answer") or not video:
            continue
        rows.append({"messages": [{"role": "user", "content": VIDEO_TAG + QUESTION},
                                  {"role": "assistant", "content": json.dumps(record["answer"], ensure_ascii=False)}],
                     "videos": [video], "clip_id": record["clip_id"], "prompt_version": record.get("prompt_version", "")})
    return rows


def check_sharegpt(rows: Iterable[Dict[str, Any]]) -> List[str]:
    """What LLaMA-Factory's sharegpt loader would refuse (empty list = every row loads): alternating user/assistant
    turns, as many ``<video>`` tags as videos, an assistant answer that is one JSON object."""
    problems = []
    for n, row in enumerate(rows):
        msgs = row.get("messages") or []
        roles = [m.get("role") for m in msgs]
        if not msgs or roles != ["user", "assistant"] * (len(msgs) // 2) or len(msgs) % 2:
            problems.append(f"row {n}: turns must alternate user / assistant")
        tags = sum(str(m.get("content", "")).count(VIDEO_TAG) for m in msgs)
        if tags != len(row.get("videos") or []):
            problems.append(f"row {n}: {tags} {VIDEO_TAG} tag(s) for {len(row.get('videos') or [])} video(s)")
        try:
            if not isinstance(json.loads(msgs[-1]["content"]), dict):
                raise ValueError
        except (ValueError, KeyError, IndexError, TypeError):
            problems.append(f"row {n}: the assistant answer is not one JSON object")
    return problems


def eval_label(d: Dict[str, Any]) -> str:
    """``eval_prompt``'s truth words: ``alert`` / ``empty`` / ``normal``."""
    if d["raw_label"] in ("suspicious", "escalation") or d.get("old_alert") or \
            taxonomy.group_of(d["category"]) in ("S", "E"):
        return "alert"
    return "empty" if d.get("empty") else "normal"


def eval_row(item: ClipItem, d: Dict[str, Any]) -> Dict[str, Any]:
    parts = item.video_s3.split("/", 3) if item.video_s3.startswith("s3://") else []
    return {
        "clip_id": item.clip_id, "batch": item.batch, "camera": item.camera,
        "ours_text": evalfmt.parse_truth(d["description"])[1], "ours_label": eval_label(d),
        "frames": evalfmt.frame_paths(item.clip_id), "s3_key": parts[3] if len(parts) == 4 else "",
        "local_time": item.local_time or evalfmt.clip_local_time(item.clip_id),
        "category": d["category"], "raw_label": d["raw_label"], "source": item.source, "tag_source": d["tag_source"],
        "old_label": d.get("old_label"),
    }


def build(items: Iterable[ClipItem], tags: Dict[str, Tag], include_needs_check: bool = False
          ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """``(training rows, eval rows, counts)`` in clip-id order."""
    training, evals = [], []
    counts: Dict[str, int] = {"clips": 0, "training": 0, "eval": 0, "studio": 0, "migrated": 0, "untagged": 0,
                              "delete": 0, "needs_check": 0, "contradicted": 0, "no_description": 0,
                              "no_video_path": 0, "video_expires": 0}
    for item in sorted(items, key=lambda i: (i.clip_id, i.key)):
        counts["clips"] += 1
        d, why = decision(item, tags.get(item.key), include_needs_check)
        if d is None:
            counts[why] += 1
            continue
        counts[d["tag_source"]] += 1
        evals.append(eval_row(item, d))
        if not d["description"].strip():
            counts["no_description"] += 1
            continue
        if not item.video_s3:
            counts["no_video_path"] += 1
        record = training_record(item, d)
        counts["video_expires"] += int(record["video_expires"])
        training.append(record)
    counts["training"], counts["eval"] = len(training), len(evals)
    return training, evals, counts


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return path


def write_frames(rows: Iterable[Dict[str, Any]], frames_dir: str, video_for: Callable[[str], Optional[str]],
                 sample: Callable[[str], List[Any]] = evalfmt.sample_frames) -> Dict[str, int]:
    """``frames/<clip_id>_<i>.jpg`` under *frames_dir* for rows that lack them (eval_prompt's layout)."""
    import cv2  # noqa: PLC0415

    counts = {"written": 0, "cached": 0, "failed": 0}
    for row in rows:
        paths = [os.path.join(frames_dir, *rel.split("/")) for rel in row["frames"]]
        if all(os.path.isfile(p) for p in paths):
            counts["cached"] += 1
            continue
        video = video_for(row["clip_id"])
        frames = sample(video) if video else []
        if len(frames) != len(paths):
            counts["failed"] += 1
            continue
        for path, frame in zip(paths, frames, strict=True):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), evalfmt.JPEG_QUALITY])
            if not ok:
                raise RuntimeError(f"could not encode {path}")
            with open(path, "wb") as f:
                f.write(buf.tobytes())
        counts["written"] += 1
    return counts
