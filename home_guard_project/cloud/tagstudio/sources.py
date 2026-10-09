"""Where the studio's clips come from. Each source is an adapter that turns its records into ``ClipItem``s.

- :class:`DatasetSource`: the unified dataset (``annotations/clips.jsonl`` + ``vlm/clips.jsonl``), the old tags.
  Only this class knows that layout; unknown fields are ignored and missing ones tolerated, so a newer dataset
  still loads. An old tag is the starting point for a clip.
- :class:`OwnerFeedbackSource`: ``<dataset>/owner_feedback/``, the customers' Telegram answers and the clips they
  answered, in the box's own layout (``production_<site>/{meta,clips,vlm_crops}/<camera>/<day>/``, and
  ``feedback/production_<site>/feedback/<camera>/<day>/<alert_id>_<ms>.feedback.json``).
- :func:`event_items`: the customers' alerts the indexer keeps in the database (live from S3): the AI's answer
  (``ai_runs``), the owner's answers (``feedback`` and their raw revisions) and the clip's artifacts.

The answer and AI mappings mirror ``box/feedback.py`` (verdicts, owner tags, Undo) and the box's meta records.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from ...fleet_contract import taxonomy
from . import evalfmt
from .items import AI, EMPTY, OLD, OWNER, ClipItem, Opinion, alertish, iso_utc

# box/feedback.py: answers that judge an alert, the owner's tag words, and the Undo note.
LABELLING_VERDICTS = ("true_alert", "false_alarm", "real_but_wrong", "expected")
OWNER_LABELS = ("normal", "suspicious", "escalation", "empty", "other")
TAG_UNDONE_NOTE = "tag undone"

ANNOTATIONS = os.path.join("annotations", "clips.jsonl")
VLM = os.path.join("vlm", "clips.jsonl")
OWNER_FEEDBACK = "owner_feedback"


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _rel(root: str, rel: Any) -> str:
    if not rel:
        return ""
    return os.path.join(root, *str(rel).replace("\\", "/").split("/"))


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------- the unified dataset
def old_opinion(row: Dict[str, Any]) -> Optional[Opinion]:
    """The old tag: ``alert`` (S or E, unknown which), ``empty`` or ``normal``; a ``[delete]`` tag has no label and
    ``detail["delete"]``. No text and no alert flag: no old tag."""
    description = str(row.get("description") or "")
    alert = row.get("alert")
    if not description.strip() and alert is None:
        return None
    delete = "[delete]" in description.lower()
    truth, text = evalfmt.parse_truth(description)
    if delete and alert is None:
        label = ""
    elif alert is True or truth == "alert":
        label = "alert"
    elif truth == "empty":
        label = EMPTY
    elif alert is False:
        label = "normal"
    else:
        label = ""
    return Opinion(OLD, label=label, text=text,
                   detail={"by": row.get("description_by") or "", "batch": row.get("batch") or "", "delete": delete})


class DatasetSource:
    name = "dataset"

    def __init__(self, root: str, bucket: str = "security-camera-project-v1", s3_prefix: str = "home_guard_dataset"):
        self.root = str(root)
        self.bucket, self.s3_prefix = bucket, s3_prefix
        self._stamp: Optional[Tuple[float, float]] = None
        self._items: List[ClipItem] = []
        self._status: Dict[str, Any] = {"name": self.name, "root": self.root, "clips": 0, "error": ""}

    def _stamps(self) -> Tuple[float, float]:
        return _mtime(os.path.join(self.root, ANNOTATIONS)), _mtime(os.path.join(self.root, VLM))

    def load(self) -> List[ClipItem]:
        """Every clip, re-read only when the files changed."""
        stamp = self._stamps()
        if stamp == self._stamp:
            return self._items
        rows, bad = evalfmt.read_jsonl(os.path.join(self.root, ANNOTATIONS))
        vlm_rows, bad_vlm = evalfmt.read_jsonl(os.path.join(self.root, VLM))
        vlm = {str(r.get("clip_id")): r for r in vlm_rows if r.get("clip_id")}
        items: List[ClipItem] = []
        seen = set()
        for row in rows:
            clip_id = str(row.get("clip_id") or "").strip()
            if not clip_id or clip_id in seen:
                continue
            seen.add(clip_id)
            v = vlm.get(clip_id, {})
            clip_rel = row.get("clip") or v.get("video") or ""
            crop_rel = row.get("crop") or v.get("crop") or ""
            epoch = evalfmt.clip_epoch(clip_id)
            item = ClipItem(
                key=f"ds:{clip_id}", clip_id=clip_id, origin=self.name, source=str(row.get("source") or ""),
                batch=str(row.get("batch") or ""), camera=str(row.get("camera") or v.get("camera") or ""),
                date=str(row.get("date") or ""), duration_sec=row.get("duration_sec", v.get("duration_sec")),
                video=_rel(self.root, clip_rel), crop=_rel(self.root, crop_rel),
                video_s3=str(v.get("video_s3_path") or self._s3(clip_rel)),
                crop_s3=str(v.get("vlm_crop_s3_path") or self._s3(crop_rel)),
                meta_path=_rel(self.root, row.get("meta")), local_time=evalfmt.clip_local_time(clip_id),
                sort_ts=float(epoch) if epoch else 0.0,
                info={"num_persons": row.get("num_persons"), "num_cars": row.get("num_cars"),
                      "needs_check": row.get("needs_check"), "split": row.get("split"),
                      "boxes_by": row.get("boxes_by"), "kind": _kind(clip_id)},
            )
            op = old_opinion(row)
            if op is not None:
                item.opinions[OLD] = op      # no time: an old tag predates every studio tag
            items.append(item)
        self._items, self._stamp = items, stamp
        self._status.update(clips=len(items), bad_lines=bad + bad_vlm,
                            error="" if rows else f"no clips in {os.path.join(self.root, ANNOTATIONS)}")
        return items

    def _s3(self, rel: Any) -> str:
        return f"s3://{self.bucket}/{self.s3_prefix}/{str(rel).replace(os.sep, '/')}" if rel else ""

    def fps(self, item: ClipItem) -> Optional[float]:
        meta = _load_json(item.meta_path) if item.meta_path else None
        value = (meta or {}).get("fps_estimated")
        return float(value) if isinstance(value, (int, float)) and value > 0 else None

    def status(self) -> Dict[str, Any]:
        return dict(self._status)


def _kind(clip_id: str) -> str:
    stem = clip_id.rsplit("_", 1)
    return stem[1] if len(stem) == 2 and stem[1].isalpha() else ""


# ---------------------------------------------------------------- what an alert meta and an answer say
def ai_opinion(alert: Dict[str, Any], response: Dict[str, Any], observation: Dict[str, Any], kind: str,
               at: str, detail: Dict[str, Any]) -> Optional[Opinion]:
    """The AI's own, context-free answer (``raw_label``), with the Eye's observation fields when present."""
    raw = str(alert.get("raw_label") or response.get("raw_label") or alert.get("label") or response.get("label")
              or "").strip().lower()
    category = str(observation.get("category") or response.get("category") or alert.get("category") or "").strip()
    if not raw and kind == "false_positive":
        raw = EMPTY                 # the AI dismissed it: nobody and nothing moving
    if not raw and not category:
        return None
    for name in ("zone", "movement", "flags", "visibility", "evidence_frame", "other_text"):
        value = observation.get(name, response.get(name))
        if value not in (None, "", []):
            detail[name] = value
    return Opinion(AI, label=raw if raw in taxonomy.LABELS + (EMPTY,) else "",
                   category=taxonomy.normalize_id(category) if category else "",
                   text=str(alert.get("summary") or response.get("summary") or ""), at=at, detail=detail,
                   knows_empty=bool(category) or raw == EMPTY)


def _judges(answer: Dict[str, Any]) -> bool:
    if answer.get("verdict") == "none" and answer.get("note") == TAG_UNDONE_NOTE:
        return True
    return answer.get("verdict") in LABELLING_VERDICTS or answer.get("owner_label") in OWNER_LABELS


def answer_key(a: Dict[str, Any]) -> Tuple[str, str, str]:
    return str(a.get("time_utc") or ""), str(a.get("verdict") or ""), str(a.get("raw_text") or "")


def owner_opinion(answers: Iterable[Dict[str, Any]], ai_label: str = "") -> Optional[Opinion]:
    """The owner's latest answer that judges the alert; an Undo takes the answer back."""
    unique = {answer_key(a): a for a in answers if isinstance(a, dict)}
    judged = sorted((a for a in unique.values() if _judges(a)), key=lambda a: str(a.get("time_utc") or ""))
    current: Optional[Dict[str, Any]] = None
    for answer in judged:
        undo = answer.get("verdict") == "none" and answer.get("note") == TAG_UNDONE_NOTE
        current = None if undo else answer
    if current is None:
        return None
    owner_label = str(current.get("owner_label") or "")
    verdict = str(current.get("verdict") or "")
    label, disputes = "", False
    if owner_label in ("normal", "suspicious", "escalation", EMPTY):
        label = owner_label
    elif owner_label == "other" or verdict == "real_but_wrong":
        disputes = True
    elif verdict == "false_alarm":
        label = EMPTY
    elif verdict == "expected":
        label = "normal"
    elif verdict == "true_alert":
        label = ai_label if alertish(ai_label) and ai_label != "alert" else "alert"
    sender = current.get("from")
    return Opinion(OWNER, label=label, disputes_ai=disputes,
                   text=str(current.get("owner_text") or current.get("note") or current.get("raw_text") or ""),
                   at=str(current.get("time_utc") or ""),
                   detail={"verdict": verdict, "owner_label": owner_label, "raw_text": current.get("raw_text") or "",
                           "from": (sender.get("name") if isinstance(sender, dict) else sender) or "",
                           "source": current.get("source") or "", "answers": len(judged)})


# ---------------------------------------------------------------- owner_feedback/ (local copies)
class OwnerFeedbackSource:
    """The Telegram answers and their clips, as copied into the dataset's ``owner_feedback/`` folder."""

    name = "owner_feedback"

    def __init__(self, root: str):
        self.root = str(root)
        self._stamp: Optional[float] = None
        self._items: List[ClipItem] = []
        self._status: Dict[str, Any] = {"name": self.name, "root": self.root, "clips": 0, "answers": 0, "error": ""}

    def _walk(self, top: str, suffix: str) -> Iterator[str]:
        for dirpath, _, names in os.walk(top):
            for name in sorted(names):
                if name.endswith(suffix):
                    yield os.path.join(dirpath, name)

    def load(self) -> List[ClipItem]:
        stamp = _mtime(os.path.join(self.root, "feedback_index.jsonl"))
        if stamp == self._stamp and self._items:
            return self._items
        items: Dict[str, ClipItem] = {}
        answers: Dict[str, List[Dict[str, Any]]] = {}
        if os.path.isdir(self.root):
            for prefix in sorted(os.listdir(self.root)):
                prefix_dir = os.path.join(self.root, prefix)
                if not prefix.startswith(("production_", "dataset_")) or not os.path.isdir(prefix_dir):
                    continue
                for path in self._walk(os.path.join(prefix_dir, "meta"), ".meta.json"):
                    meta = _load_json(path)
                    if meta is None:
                        continue
                    stem = os.path.basename(path)[:-len(".meta.json")]
                    item = self._item(prefix, stem, path, meta)
                    items.setdefault(item.key, item)
                    for a in meta.get("owner_feedback") or []:
                        if isinstance(a, dict):
                            answers.setdefault(stem, []).append(a)
            for path in self._walk(os.path.join(self.root, "feedback"), ".feedback.json"):
                record = _load_json(path)
                alert = record.get("alert") if record and isinstance(record.get("alert"), dict) else {}
                if alert.get("alert_id"):
                    answers.setdefault(str(alert["alert_id"]), []).append(record)
        n = 0
        for item in items.values():
            ai = item.opinions.get(AI)
            op = owner_opinion(answers.get(item.clip_id, []), ai.effective_label() if ai else "")
            if op is not None:
                item.opinions[OWNER] = op
                n += 1
        self._items, self._stamp = list(items.values()), stamp
        self._status.update(clips=len(self._items), answers=n)
        return self._items

    def _item(self, prefix: str, stem: str, meta_path: str, meta: Dict[str, Any]) -> ClipItem:
        prefix_dir = os.path.join(self.root, prefix)
        clip_rel = str(meta.get("clip_path") or "").replace("\\", "/")
        crop = meta.get("vlm_crop") if isinstance(meta.get("vlm_crop"), dict) else {}
        crop_rel = str(crop.get("vlm_crop_path") or "").replace("\\", "/")
        start_local = str(meta.get("clip_start_local") or "")
        alert = meta.get("alert") if isinstance(meta.get("alert"), dict) else {}
        response = meta.get("model_response") if isinstance(meta.get("model_response"), dict) else {}
        observation = meta.get("observation") if isinstance(meta.get("observation"), dict) else {}
        teacher = meta.get("teacher") if isinstance(meta.get("teacher"), dict) else {}
        try:
            sort_ts = float(meta.get("clip_end_ts") or meta.get("trigger_ts") or 0.0)
        except (TypeError, ValueError):
            sort_ts = 0.0
        site = prefix.split("_", 1)[1] if "_" in prefix else prefix
        item = ClipItem(
            key=f"of:{prefix}/{stem}", clip_id=stem, origin=self.name, source=site, batch=prefix,
            camera=str(meta.get("camera_name") or ""), date=start_local[:10], duration_sec=meta.get("duration_sec"),
            video=_rel(prefix_dir, clip_rel) if evalfmt.exists(_rel(prefix_dir, clip_rel)) else "",
            crop=_rel(prefix_dir, crop_rel) if evalfmt.exists(_rel(prefix_dir, crop_rel)) else "",
            meta_path=meta_path, local_time=start_local[11:19] or None, sort_ts=sort_ts,
            fps=meta.get("fps_estimated") if isinstance(meta.get("fps_estimated"), (int, float)) else None,
            info={"kind": meta.get("kind") or "", "num_persons": alert.get("people"), "site": site},
        )
        detail = {"model": teacher.get("model") or alert.get("model") or response.get("model") or "",
                  "prompt_version": teacher.get("prompt_version") or alert.get("prompt_version") or "",
                  "final_label": alert.get("final_label") or alert.get("label") or "", "why": alert.get("why") or "",
                  "kind": meta.get("kind") or ""}
        op = ai_opinion(alert, response, observation, str(meta.get("kind") or ""),
                        iso_utc(meta.get("clip_end_ts") or meta.get("trigger_ts")), detail)
        if op is not None:
            item.opinions[AI] = op
        return item

    def status(self) -> Dict[str, Any]:
        return dict(self._status)


# ---------------------------------------------------------------- customer events (database, live from S3)
def _iso(value: Optional[datetime]) -> str:
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def event_items(session, bucket: str) -> List[ClipItem]:
    """Every indexed customer event as a clip, with its AI answer and the owner's answers."""
    from sqlalchemy import select  # noqa: PLC0415

    from ..models import AiRun, Artifact, Customer, Device, Event, Feedback, RawRevision  # noqa: PLC0415

    rows = session.execute(select(Event, Device, Customer).join(Device, Device.id == Event.device_pk)
                           .join(Customer, Customer.id == Device.customer_id)).all()
    if not rows:
        return []
    ids = [ev.id for ev, _, _ in rows]
    runs: Dict[int, Any] = {}
    for run in session.scalars(select(AiRun).where(AiRun.event_id.in_(ids)).order_by(AiRun.id)):
        if run.status == "real" or run.event_id not in runs:
            runs[run.event_id] = run
    arts: Dict[int, Dict[str, Any]] = {}
    gone: Dict[int, set] = {}            # videos the indexer saw once and S3 no longer has
    for art in session.scalars(select(Artifact).where(Artifact.event_id.in_(ids),
                                                      Artifact.role.in_(("original_video", "crop_video")))):
        kind = "clip" if art.role == "original_video" else "crop"
        if not art.available:
            gone.setdefault(art.event_id, set()).add(kind)
            continue
        current = arts.setdefault(art.event_id, {}).get(kind)
        # The training copy (dataset_<site>/) outlives the production copy, which S3 empties after 14 days.
        if current is None or (art.s3_key.startswith("dataset_") and not current.s3_key.startswith("dataset_")):
            arts[art.event_id][kind] = art
    answers: Dict[int, List[Dict[str, Any]]] = {}
    fb_rows = session.execute(select(Feedback, RawRevision.body)
                              .outerjoin(RawRevision, RawRevision.s3_key == Feedback.s3_key)
                              .where(Feedback.event_id.in_(ids)).order_by(RawRevision.id)).all()
    for fb, body in fb_rows:
        body = body if isinstance(body, dict) else {}
        answers.setdefault(fb.event_id, []).append({
            "time_utc": body.get("time_utc") or _iso(fb.received_at), "verdict": fb.verdict or body.get("verdict"),
            "note": fb.note, "raw_text": fb.raw_text, "source": fb.source, "owner_label": body.get("owner_label") or "",
            "owner_text": body.get("owner_text") or "", "from": body.get("from")})
    items = []
    for ev, device, customer in rows:
        run = runs.get(ev.id)
        parsed = run.parsed if run is not None and isinstance(run.parsed, dict) else {}
        alert = {"label": ev.label or "", "summary": ev.summary or ""}
        local = str(ev.clip_start_local or "")
        art = arts.get(ev.id, {})
        item = ClipItem(
            key=f"ev:{ev.id}", clip_id=ev.stem, origin="customer", source=ev.site, batch=f"{customer.name}",
            camera=ev.camera, date=ev.day or local[:10], duration_sec=ev.duration_sec,
            video_s3=f"s3://{bucket}/{art['clip'].s3_key}" if "clip" in art else "",
            crop_s3=f"s3://{bucket}/{art['crop'].s3_key}" if "crop" in art else "",
            local_time=local[11:19] or None, sort_ts=float(ev.start_ts or 0.0), fps=ev.fps, event_id=ev.id,
            artifacts={k: a.id for k, a in art.items()},
            info={"kind": ev.kind, "site": ev.site, "customer": customer.name, "customer_id": customer.id,
                  "consent_training": bool(customer.consent_training),
                  "consent_proposed": customer.consent_proposed is not None,
                  "production_only": bool(art) and all(a.s3_key.startswith("production_") for a in art.values()),
                  "media_missing": {kind: ("removed from S3 (the indexer saw it, the file is gone)"
                                           if kind in gone.get(ev.id, ()) else "never uploaded to S3")
                                    for kind in ("clip", "crop") if kind not in art},
                  "num_persons": parsed.get("people"), "detected": ev.detected or []},
        )
        detail = {"model": (run.model if run else "") or "", "prompt_version": (run.prompt_version if run else "") or "",
                  "final_label": ev.label or "", "kind": ev.kind, "alert_command": ev.alert_command or "",
                  "why": parsed.get("why") or "", "ai_status": run.status if run else "none"}
        observation = parsed.get("observation") if isinstance(parsed.get("observation"), dict) else {}
        op = ai_opinion({"summary": ev.summary or ""} if parsed else alert, parsed, observation, ev.kind,
                        iso_utc(ev.end_ts or ev.start_ts), detail)
        if op is not None:
            item.opinions[AI] = op
        owner = owner_opinion(answers.get(ev.id, []), op.effective_label() if op else "")
        if owner is not None:
            item.opinions[OWNER] = owner
        items.append(item)
    return items
