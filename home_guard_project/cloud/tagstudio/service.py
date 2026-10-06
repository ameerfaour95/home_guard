"""The tagging studio itself: every clip from every source, the teacher's suggestion, our tags, the work queue,
media access and the exports. The routes (``routes/tagging.py``) are thin wrappers around :class:`TagStudio`.

Clips found by more than one source are one clip: a local owner-feedback copy and the dataset's old tag join the
indexed event with the same site and stem (the event keeps its key, so the Label view and the tag agree).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ...fleet_contract import taxonomy
from . import export as exporter
from . import queue as work_queue
from .config import StudioPaths
from .fields import FIELDS, Tag, TagError, clean_fields, empty_form, fold
from .items import EMPTY, OLD, TEACHER, WHO, ClipItem
from .sources import DatasetSource, OwnerFeedbackSource, event_items
from .teacher import EvalResultsTeacher, OpenAICompatibleTeacher, TeacherRefused, first_suggestion

log = logging.getLogger(__name__)

MEDIA_TTL_SECONDS = 600
_MEDIA_DOMAIN = b"home-guard-admin/tagging-media/v1"
GROUP_NAMES = {"N": ("Normal", "רגיל"), "S": ("Suspicious", "חשוד"), "E": ("Escalation", "הסלמה")}


class StudioError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def taxonomy_table() -> Dict[str, Any]:
    table = taxonomy.as_table()
    table["groups"] = [{"id": g, "label": taxonomy.GROUP_LABEL[g], "name": GROUP_NAMES[g][0], "he": GROUP_NAMES[g][1],
                        "categories": [c for c in table["categories"] if c["group"] == g]} for g in ("N", "S", "E")]
    return table


class TagStudio:
    def __init__(self, paths: StudioPaths, bucket: str = "security-camera-project-v1"):
        self.paths, self.bucket = paths, bucket
        self.dataset = DatasetSource(str(paths.dataset), bucket=bucket)
        self.owner_feedback = OwnerFeedbackSource(str(paths.dataset / "owner_feedback"))
        self.eval_teacher = EvalResultsTeacher([str(p) for p in paths.eval_results])
        self.ask_teacher: Optional[OpenAICompatibleTeacher] = None
        self.teacher_error = ""
        if paths.teacher_base_url and paths.teacher_model:
            try:
                self.ask_teacher = OpenAICompatibleTeacher(paths.teacher_base_url, paths.teacher_model,
                                                           str(paths.exports / "teacher_cache.jsonl"),
                                                           api_key=paths.teacher_api_key)
            except TeacherRefused as e:
                self.teacher_error = str(e)
        self.teachers = ([self.ask_teacher] if self.ask_teacher else []) + [self.eval_teacher]

    # ------------------------------------------------------------ clips
    def items(self, session) -> Dict[str, ClipItem]:
        events = event_items(session, self.bucket) if session is not None else []
        by_site_stem = {(e.source, e.clip_id): e for e in events}
        by_stem: Dict[str, ClipItem] = {}
        for e in events:
            by_stem.setdefault(e.clip_id, e)
        out: Dict[str, ClipItem] = {e.key: e for e in events}
        for item in self.owner_feedback.load():
            match = by_site_stem.get((item.source, item.clip_id))
            if match is not None:
                match.merge(item)
            else:
                out[item.key] = item
        for item in self.dataset.load():
            match = by_stem.get(item.clip_id)
            if match is not None:
                match.merge(item)
            else:
                out[item.key] = item
        for item in out.values():
            item.info["has_media"] = any(ok for ok, _ in self.media_status(item).values())
            item.opinions.pop(TEACHER, None)
            op = first_suggestion(self.teachers, item)
            if op is not None:
                item.opinions[TEACHER] = op
        return out

    def tags(self, session) -> Dict[str, Tag]:
        from sqlalchemy import select  # noqa: PLC0415

        from ..models import TagEvent  # noqa: PLC0415

        rows = session.scalars(select(TagEvent).order_by(TagEvent.id))
        return fold({"key": r.clip_key, "at": _iso(r.created_at), "by": r.staff_name, "fields": r.fields}
                    for r in rows)

    def history(self, session, key: str) -> List[Dict[str, Any]]:
        from sqlalchemy import select  # noqa: PLC0415

        from ..models import TagEvent  # noqa: PLC0415

        rows = session.scalars(select(TagEvent).where(TagEvent.clip_key == key).order_by(TagEvent.id.desc()).limit(20))
        return [{"at": _iso(r.created_at), "by": r.staff_name, "fields": r.fields} for r in rows]

    # ------------------------------------------------------------ screens
    def state(self, session) -> Dict[str, Any]:
        rows = work_queue.build(self.items(session).values(), self.tags(session))
        counts = {name: 0 for name in work_queue.TIER_NAMES.values()}
        for _, a in rows:
            counts[a.tier_name] += 1
        sources = [self.dataset.status(), self.owner_feedback.status(),
                   {"name": "customers", "clips": sum(1 for i, _ in rows if i.origin == "customer"), "error": ""}]
        teachers = [t.status() for t in self.teachers]
        if self.teacher_error:
            teachers.append({"name": "openai", "error": self.teacher_error})
        return {"taxonomy": taxonomy_table(), "fields": FIELDS, "counts": counts, "total": len(rows),
                "sources": sources, "teachers": teachers, "can_ask_teacher": self.ask_teacher is not None,
                "paths": {"dataset": str(self.paths.dataset), "eval_results": "; ".join(str(p) for p in self.paths.eval_results),
                          "exports": str(self.paths.exports)}}

    def queue(self, session, tier: str = "open", origin: str = "", text: str = "", limit: int = 2000
              ) -> Dict[str, Any]:
        tags = self.tags(session)
        text = text.strip().lower()
        out = []
        for item, a in work_queue.build(self.items(session).values(), tags):
            if tier == "open" and a.tier == work_queue.DONE:
                continue
            if tier not in ("open", "all", "") and a.tier_name != tier:
                continue
            if origin and item.origin != origin:
                continue
            if text and not any(text in s.lower() for s in (item.clip_id, item.camera, item.source, item.batch)):
                continue
            labels = {who: {"label": op.effective_label(), "category": op.category, "disputes_ai": op.disputes_ai}
                      for who, op in item.opinions.items() if who in WHO}
            tag = tags.get(item.key)
            if tag is not None:
                labels["studio"] = {"label": tag.raw_label, "category": tag.fields.get("category", ""),
                                    "disputes_ai": False}
            out.append({**item.summary(), **a.as_dict(), "labels": labels, "has_media": bool(item.info.get("has_media"))})
        return {"items": out[:limit], "count": len(out)}

    def _item(self, session, key: str) -> Tuple[ClipItem, Dict[str, ClipItem]]:
        items = self.items(session)
        item = items.get(key)
        if item is None:
            raise StudioError("Clip not found", 404)
        return item, items

    def prefill(self, item: ClipItem) -> Dict[str, Any]:
        """The old tag as the starting point: its text, and what its label says about the category/raw label."""
        form = empty_form()
        old = item.opinions.get(OLD)
        if old is not None:
            form["description"] = old.text
            label = old.effective_label()
            if label == EMPTY:
                form["category"], form["raw_label"] = "N10", "normal"
            elif label == "normal":
                form["raw_label"] = "normal"
            form["delete"] = bool(old.detail.get("delete"))
        return form

    def detail(self, session, key: str) -> Dict[str, Any]:
        item, _ = self._item(session, key)
        tag = self.tags(session).get(key)
        if tag is not None:
            form, prefilled = {**self.prefill(item), **tag.fields}, "studio"
        else:
            form, prefilled = self.prefill(item), (OLD if OLD in item.opinions else "")
        fps = item.fps or (self.dataset.fps(item) if item.meta_path else None)
        status = self.media_status(item)
        return {
            "item": {**item.summary(), "info": item.info, "video_s3": item.video_s3, "crop_s3": item.crop_s3},
            "media": {kind: ok for kind, (ok, _) in status.items()},
            "media_reasons": {kind: why for kind, (_, why) in status.items()}, "fps": fps,
            "opinions": {who: op.as_dict() for who, op in item.opinions.items()},
            "tag": tag.as_dict() if tag else None, "form": form, "prefilled_from": prefilled,
            "assessment": work_queue.assess(item, tag).as_dict(), "history": self.history(session, key),
        }

    def save(self, session, staff, key: str, fields: Dict[str, Any], now: datetime) -> Dict[str, Any]:
        from ..models import TagEvent  # noqa: PLC0415

        item, items = self._item(session, key)
        try:
            clean = clean_fields(fields)
        except TagError as e:
            raise StudioError(str(e), 422) from None
        session.add(TagEvent(clip_key=key, clip_id=item.clip_id, fields=clean, staff_id=staff.id,
                             staff_name=staff.name, created_at=now))
        session.flush()
        tags = self.tags(session)
        rows = work_queue.build(items.values(), tags)
        next_key = next((i.key for i, a in rows if a.tier != work_queue.DONE and i.key != key), "")
        tag = tags[key]
        return {"tag": tag.as_dict(), "assessment": work_queue.assess(item, tag).as_dict(), "next_key": next_key}

    # ------------------------------------------------------------ media
    def media_status(self, item: ClipItem) -> Dict[str, Tuple[bool, str]]:
        """Per video kind: can it be opened, and if not, why (shown to the tagger, who can still tag from the cards)."""
        out: Dict[str, Tuple[bool, str]] = {}
        no_consent = item.origin == "customer" and item.info.get("consent_training") is False
        for kind, path in (("clip", item.video), ("crop", item.crop)):
            name = "full-frame clip" if kind == "clip" else "crop"
            local = bool(path) and os.path.isfile(path)
            if local or kind in item.artifacts:
                out[kind] = (False, "no training consent from this customer") if no_consent else (True, "")
            elif path:
                out[kind] = (False, f"{name} not on this computer ({path})")
            elif item.origin == "customer":
                out[kind] = (False, (item.info.get("media_missing") or {}).get(kind) or f"no {name} recorded")
            elif item.origin == "dataset":
                out[kind] = (False, f"the dataset has no {name} for this clip")
            else:
                out[kind] = (False, f"no {name} in the local copy")
        return out

    def local_media(self, item: ClipItem, kind: str) -> Optional[str]:
        path = item.video if kind == "clip" else item.crop if kind == "crop" else ""
        return path if path and os.path.isfile(path) else None

    def sign(self, secret: str, path: str, now: float) -> str:
        payload = base64.urlsafe_b64encode(json.dumps({"p": path, "e": int(now) + MEDIA_TTL_SECONDS}).encode()
                                           ).decode().rstrip("=")
        return f"{payload}.{_mac(secret, payload)}"

    @staticmethod
    def verify(secret: str, token: str, now: float) -> Optional[str]:
        payload, _, mac = token.partition(".")
        if not payload or not hmac.compare_digest(mac, _mac(secret, payload)):
            return None
        try:
            data = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        except (ValueError, TypeError):
            return None
        if not isinstance(data, dict) or float(data.get("e", 0)) < now:
            return None
        path = data.get("p")
        return path if isinstance(path, str) and os.path.isfile(path) else None

    # ------------------------------------------------------------ teacher and export
    def teach(self, session, key: str) -> Dict[str, Any]:
        if self.ask_teacher is None:
            raise StudioError(self.teacher_error or "No teacher model is configured (HG_TEACHER_BASE_URL, "
                              "HG_TEACHER_MODEL)", 409)
        item, _ = self._item(session, key)
        path = self.local_media(item, "clip") or self.local_media(item, "crop")
        if not path:
            raise StudioError("The teacher needs a local copy of the clip", 409)
        op = self.ask_teacher.ask(item, path)
        return {"suggestion": op.as_dict() if op else None}

    def export(self, session, include_needs_check: bool = False, out_dir: Optional[str] = None,
               frames_dir: Optional[str] = None) -> Dict[str, Any]:
        items = self.items(session)
        training, evals, counts = exporter.build(items.values(), self.tags(session), include_needs_check)
        out = str(out_dir or self.paths.exports)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        result = {"training_path": exporter.write_jsonl(os.path.join(out, stamp, exporter.TRAINING_FILE), training),
                  "eval_path": exporter.write_jsonl(os.path.join(out, stamp, exporter.EVAL_FILE), evals),
                  "counts": counts}
        if frames_dir:
            by_clip = {i.clip_id: i for i in items.values()}
            result["frames"] = exporter.write_frames(
                evals, frames_dir, lambda cid: self.local_media(by_clip[cid], "clip") if cid in by_clip else None)
        return result


def _mac(secret: str, payload: str) -> str:
    key = hmac.new(secret.encode("utf-8"), _MEDIA_DOMAIN, hashlib.sha256).digest()
    return hmac.new(key, payload.encode("ascii"), hashlib.sha256).hexdigest()[:40]


def _iso(value: Optional[datetime]) -> str:
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
