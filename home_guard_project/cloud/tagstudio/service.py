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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...fleet_contract import health, taxonomy
from ...fleet_contract.prompt_schemas import answer_schema, schema_kind
from . import export as exporter
from . import queue as work_queue
from .config import StudioPaths
from .fields import FIELDS, Tag, TagError, clean_fields, empty_form, fold
from .items import AI, EMPTY, OLD, OWNER, TEACHER, WHO, ClipItem
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
        names = self.camera_names(session, {i.info.get("site") or (i.source if i.origin != "dataset" else "")
                                            for i in out.values()})
        for item in out.values():
            site = item.info.get("site") or (item.source if item.origin != "dataset" else "")
            item.info["camera_display"] = health.camera_label(item.camera, names.get(site)) if item.camera else ""
            item.info["has_media"] = any(ok for ok, _ in self.media_status(item).values())
            item.opinions.pop(TEACHER, None)
            op = first_suggestion(self.teachers, item)
            if op is not None:
                item.opinions[TEACHER] = op
        return out

    def camera_names(self, session, sites) -> Dict[str, Dict[str, str]]:
        """{site: {camera id: the owner's name}}: the box's heartbeat camera_list, else a Camera.display_name (as the
        Chat tab reads them); health.camera_label maps an old id to the current camera on its channel."""
        if session is None:
            return {}
        from sqlalchemy import select  # noqa: PLC0415

        from ..models import Camera, Device  # noqa: PLC0415

        out: Dict[str, Dict[str, str]] = {}
        for dev in session.scalars(select(Device).where(Device.site.in_(sorted(s for s in sites if s)))):
            names = {cam: shown.strip() for cam, shown in session.execute(
                select(Camera.name, Camera.display_name).where(Camera.device_pk == dev.id)).all()
                if shown and shown.strip()}
            names.update({c["id"]: c["name"] for c in health.camera_list(dev.last_heartbeat) or () if c["name"]})
            out[dev.site] = names
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
            if text and not any(text in s.lower() for s in (item.clip_id, item.camera, item.source, item.batch,
                                                             str(item.info.get("camera_display") or ""))):
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
        """The old tag as the starting point: its text and its level as the raw label. An old tag never had a category,
        so none is filled in (it would bias the tagger), except "nothing there", which can only be N10. [alert] starts
        as suspicious; escalation is the tagger's call. A teacher suggestion is never filled in: it is applied with A."""
        form = empty_form()
        old = item.opinions.get(OLD)
        if old is not None:
            form["description"] = old.text
            label = old.effective_label()
            if label == EMPTY:
                form["category"], form["raw_label"] = "N10", "normal"
            elif label == "normal":
                form["raw_label"] = "normal"
            elif label == "alert":
                form["raw_label"] = "suspicious"
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
        prompt_version = form.get("prompt_version") or self.prompt_version_of(session, item)
        form["prompt_version"] = prompt_version
        name, answer = answer_schema(prompt_version)
        return {
            "prompt_version": prompt_version,
            "answer_schema": {"kind": schema_kind(prompt_version), "name": name, "fields": list(answer["properties"])},
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
        clean = self.with_prompt_version(session, item, self.tags(session).get(key), clean)
        require_category(self.tags(session).get(key), clean)
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

    def with_prompt_version(self, session, item: ClipItem, current: Optional[Tag],
                            clean: Dict[str, Any]) -> Dict[str, Any]:
        """Every saved tag carries the prompt version its schema follows: the one the form sent, else the tag's own,
        else the clip's (prompt_version_of); nothing is added for a clip no prompt answered."""
        if clean.get("prompt_version") or (current is not None and current.fields.get("prompt_version")):
            return clean
        version = self.prompt_version_of(session, item)
        return {**clean, "prompt_version": str(version)[:500]} if version else clean

    def clip_meta(self, session, item: ClipItem) -> Dict[str, Any]:
        """The clip's meta: an indexed event's newest revision, else the local copy's file; {} when none."""
        if item.event_id and session is not None:
            from .. import labeling  # noqa: PLC0415

            meta = labeling.meta_body(session, item.event_id)
            if meta:
                return meta
        if item.meta_path:
            try:
                with open(item.meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                return meta if isinstance(meta, dict) else {}
            except (OSError, ValueError):
                return {}
        return {}

    def model_input(self, session, key: str, s3=None) -> Dict[str, Any]:
        """What the AI saw of the clip (model_view.py: the frames the box sent, else its recipe, else rendered like
        the box), with the clip's prompt version; 404 when there is nothing to show."""
        import tempfile  # noqa: PLC0415

        from . import model_view  # noqa: PLC0415

        item, _ = self._item(session, key)
        meta = self.clip_meta(session, item)
        # a local copy first (no download), else the event's files in storage
        sent = model_view.local_sent_frames(meta, item.meta_path) if item.meta_path else []
        view = model_view.build(meta, sent, item.crop or None, item.video or None)
        if view is None and item.event_id and session is not None and s3 is not None:
            with tempfile.TemporaryDirectory(prefix="hg_model_input_") as work:
                view = model_view.build(meta, *self._event_model_files(session, item, s3, work))
        if view is None:
            raise StudioError("Nothing the AI saw is saved for this clip: no frames, no crop and no clip", 404)
        prompt_version = (meta.get("teacher") or {}).get("prompt_version") or ""
        return {**view.as_dict(), "key": key, "prompt_version": prompt_version}

    def _event_model_files(self, session, item: ClipItem, s3, work: str):
        """(sent frames, crop path, clip path) of an indexed event: the guard run's input frames (all or none) and
        the crop / clip videos downloaded into `work` (only what the view will need)."""
        from sqlalchemy import select  # noqa: PLC0415

        from ..models import AiRun, Artifact  # noqa: PLC0415

        run = session.scalar(select(AiRun).where(AiRun.event_id == item.event_id, AiRun.purpose == "guard")
                             .order_by(AiRun.id.desc()).limit(1))
        ids = [i for i in (run.input_artifact_ids or []) if isinstance(i, int)] if run is not None else []
        arts = {a.id: a for a in session.scalars(select(Artifact).where(Artifact.id.in_(ids)))} if ids else {}
        sent: List[bytes] = []
        if ids and all(i in arts and arts[i].available for i in ids):
            try:
                sent = [s3.get_bytes(arts[i].s3_key) for i in ids]
            except Exception:  # noqa: BLE001 - unreadable frames: the view is rendered instead
                sent = []
        if sent:
            return sent, None, None
        paths = {}
        for kind in ("crop", "clip"):
            art = session.get(Artifact, item.artifacts[kind]) if kind in item.artifacts else None
            if art is not None and art.available:
                path = os.path.join(work, f"{kind}.mp4")
                try:
                    s3.download_to(art.s3_key, path)
                    paths[kind] = path
                except Exception:  # noqa: BLE001
                    log.warning("could not fetch the %s of %s", kind, item.key)
        return [], paths.get("crop"), paths.get("clip")

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
    # ------------------------------------------------------------ "Suggest tag"
    def suggester(self):
        """The suggestion model (built once; ``suggest_client`` is injected by tests, never a real call there)."""
        from .suggest import SuggestConfig, Suggester  # noqa: PLC0415

        if getattr(self, "_suggester", None) is None:
            repo = Path(__file__).resolve().parents[3]
            keys = [Path(os.environ["HG_SUGGEST_KEY_FILE"])] if os.environ.get("HG_SUGGEST_KEY_FILE") else []
            keys += [repo / "api_key.env", repo.parent / "home_guard" / "api_key.env",
                     Path.home() / "Ameer" / "home_guard" / "api_key.env"]   # the founder laptop's key file
            config, client = SuggestConfig.resolve(key_files=keys), getattr(self, "suggest_client", None)
            if getattr(client, "model", None):  # a fake (tests, demo) says which model it stands for
                config = SuggestConfig(model=client.model, base_url="offline", api_key="")
            self._suggester = Suggester(config, str(self.paths.exports / "suggestions.jsonl"), client=client,
                                        frames_for=getattr(self, "suggest_frames", None))
        return self._suggester

    def prompt_version_of(self, session, item: ClipItem) -> str:
        """The prompt version a clip's tag follows: the saved tag's, else the clip's meta (teacher.prompt_version),
        else its AI run's; a clip the box's AI answered with no version recorded was answered by the legacy prompt
        (LEGACY_ASSUMED: before the Eye it was the only one); "" only for a clip no AI answered."""
        from ...fleet_contract.prompt_schemas import LEGACY_ASSUMED  # noqa: PLC0415

        tag = self.tags(session).get(item.key)
        if tag is not None and tag.fields.get("prompt_version"):
            return str(tag.fields["prompt_version"])
        meta = self.clip_meta(session, item)
        ai = item.opinions.get(AI)
        version = (meta.get("teacher") or {}).get("prompt_version") or (ai.detail.get("prompt_version") if ai else "")
        if version:
            return str(version)
        answered = ai is not None or bool(meta.get("model_response")) or bool((meta.get("alert") or {}).get("label"))
        return LEGACY_ASSUMED if answered else ""

    def converter(self):
        """The "In my words" converter (built once; ``convert_client`` is injected by tests and the demo)."""
        from .convert import ConvertConfig, Converter  # noqa: PLC0415

        if getattr(self, "_converter", None) is None:
            repo = Path(__file__).resolve().parents[3]
            keys = [repo / "api_key.env", repo.parent / "home_guard" / "api_key.env",
                    Path.home() / "Ameer" / "home_guard" / "api_key.env"]
            config, client = ConvertConfig.resolve(key_files=keys), getattr(self, "convert_client", None)
            if getattr(client, "model", None):
                config = ConvertConfig(model=client.model, base_url="offline", api_key="")
            self._converter = Converter(config, client=client)
        return self._converter

    def convert(self, session, key: str, words: str, words_source: str = "staff",
                feedback_id: Optional[int] = None) -> Dict[str, Any]:
        """The tagger's words as the clip's answer schema (a suggestion for the form; nothing is saved).

        Customer words (an owner's answer, a transcript) are sent only with that customer's training consent (403
        otherwise). The server decides: words claimed as the staff's own that contain the owner's answer to this
        clip are the owner's. Camera names, the house and the customer's name never leave (convert.redact)."""
        from .convert import ConvertError  # noqa: PLC0415

        item, _ = self._item(session, key)
        source = self.words_source_of(session, item, words, words_source, feedback_id)
        customer = self.household_of(session, item)
        if source != "staff":
            if customer is None:
                raise StudioError("This clip's customer is not known here, so their consent to training use cannot be "
                                  "checked: their words can't be sent to the converter", 403)
            if not customer["consent_training"]:
                raise StudioError("This customer hasn't agreed to training use, so their words can't be sent to the "
                                  "converter", 403)
        cameras, places = self.private_terms(session, item, customer)
        try:
            result = self.converter().convert(words, self.prompt_version_of(session, item), cameras, places)
        except ConvertError as e:
            raise StudioError(str(e), 409) from None
        return {**result, "key": key, "words_source": source}

    def owner_texts(self, session, item: ClipItem) -> List[str]:
        """What the clip's owner said about it: the owner card's words and every answer row's words."""
        texts = []
        owner = item.opinions.get(OWNER)
        if owner is not None:
            texts += [owner.text, str(owner.detail.get("owner_text") or ""), str(owner.detail.get("raw_text") or "")]
        if session is not None and item.event_id:
            from sqlalchemy import select  # noqa: PLC0415

            from ..models import Feedback  # noqa: PLC0415

            for fb in session.scalars(select(Feedback).where(Feedback.event_id == item.event_id)):
                texts += [fb.owner_text, fb.transcript, fb.raw_text, fb.note]
        return [t.strip() for t in texts if isinstance(t, str) and len(t.strip()) >= 4]

    def words_source_of(self, session, item: ClipItem, words: str, claimed: str, feedback_id: Optional[int]) -> str:
        """Where the words came from, decided here: the claimed source, checked against the answer it names, and
        "owner_answer" whenever the words carry the owner's own answer to this clip, whatever the client said."""
        claimed = claimed if claimed in ("staff", "owner_answer", "transcript") else "staff"
        if claimed != "staff" and feedback_id is not None and session is not None:
            from ..models import Feedback  # noqa: PLC0415

            fb = session.get(Feedback, feedback_id)
            if fb is None or fb.event_id != item.event_id:
                raise StudioError("That owner answer is not about this clip", 422)
        folded = " ".join(str(words or "").split()).casefold()
        if claimed == "staff" and any(" ".join(t.split()).casefold() in folded for t in self.owner_texts(session, item)):
            return "owner_answer"
        return claimed

    def household_of(self, session, item: ClipItem) -> Optional[Dict[str, Any]]:
        """{name, site, consent_training, device_pk} of a customer clip's household; None for our own dataset clips
        or a household this database does not know."""
        site = item.info.get("site") or (item.source if item.origin != "dataset" else "")
        if item.origin == "dataset" and not item.event_id:
            return None
        if session is not None and site:
            from sqlalchemy import select  # noqa: PLC0415

            from ..models import Customer, Device  # noqa: PLC0415

            device = session.scalar(select(Device).where(Device.site == site))
            customer = session.get(Customer, device.customer_id) if device is not None else None
            if customer is not None:
                return {"name": customer.name, "site": site, "consent_training": bool(customer.consent_training),
                        "device_pk": device.id}
        if "consent_training" in item.info:
            return {"name": item.info.get("customer") or "", "site": site,
                    "consent_training": bool(item.info.get("consent_training")), "device_pk": None}
        return None

    def private_terms(self, session, item: ClipItem, customer: Optional[Dict[str, Any]]):
        """(camera ids and names, house / site / customer names) the converter must never see."""
        cameras = {item.camera}
        places = {item.source, item.batch, item.info.get("site") or "", item.info.get("customer") or ""}
        if customer is not None:
            places |= {customer["name"], customer["site"]}
            places |= {part for part in str(customer["name"]).split() if len(part) >= 3}
            if session is not None and customer.get("device_pk"):
                from sqlalchemy import select  # noqa: PLC0415

                from ..models import Camera  # noqa: PLC0415

                for cam in session.scalars(select(Camera).where(Camera.device_pk == customer["device_pk"])):
                    cameras |= {cam.name, cam.display_name or ""}
        return sorted(t for t in cameras if t), sorted(t for t in places if t)

    def video_for(self, item: ClipItem, s3=None) -> Optional[str]:
        """A local copy of the clip's full-frame video (else its crop): this machine's file, or one downloaded once
        from S3 (read only) into the exports folder's media cache."""
        for kind in ("clip", "crop"):
            path = self.local_media(item, kind)
            if path:
                return path
        if s3 is None or not item.video_s3.startswith("s3://"):
            return None
        key = item.video_s3.split("/", 3)[3]
        dest = self.paths.exports / "media_cache" / (item.key.replace(":", "_").replace("/", "_") + ".mp4")
        if not dest.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_to(key, dest)
        return str(dest)

    def suggest(self, session, key: str, refresh: bool = False, s3=None) -> Dict[str, Any]:
        from .suggest import SuggestError  # noqa: PLC0415

        item, _ = self._item(session, key)
        try:
            suggester = self.suggester()
            if not refresh:
                cached = suggester.cached(key)
                if cached is not None:
                    return {**cached, "cached": True}
            version = self.prompt_version_of(session, item)
            try:   # the frames the box sent the AI (or rendered like the box), not frames picked from the clip
                view = self.model_input(session, key, s3)
            except StudioError:
                view = None
            if view and view.get("frames"):
                record = view.get("record") or {}
                index = [i for i in record.get("frame_indices") or [] if isinstance(i, int)]
                return suggester.suggest(key, None, camera=item.camera, refresh=refresh,
                                         jpegs=[base64.b64decode(f) for f in view["frames"]],
                                         frame_index=index if len(index) == len(view["frames"]) else [],
                                         fps=item.fps or record.get("fps"), prompt_version=version)
            video = self.video_for(item, s3)
            if not video:
                raise SuggestError("This clip has no video on this computer or in S3 to show the model")
            return suggester.suggest(key, video, camera=item.camera, refresh=refresh, prompt_version=version)
        except SuggestError as e:
            raise StudioError(str(e), 409) from None

    # ------------------------------------------------------------ boxes of dataset clips (not indexed events)
    def clip_boxes(self, session, key: str) -> Dict[str, Any]:
        """The Label view's annotation of a dataset clip: the newest saved version, else its YOLO boxes."""
        from sqlalchemy import select  # noqa: PLC0415

        from ..models import ClipAnnotation  # noqa: PLC0415
        from .boxes import preload_tracks  # noqa: PLC0415

        item, _ = self._item(session, key)
        if item.origin != "dataset" or item.event_id:
            raise StudioError("This clip is an indexed event: open its own annotation", 400)
        fps = item.fps or self.dataset.fps(item) or 7.0
        duration = float(item.duration_sec or 0) or None
        row = session.scalar(select(ClipAnnotation).where(ClipAnnotation.clip_key == key)
                             .order_by(ClipAnnotation.version.desc()).limit(1))
        old = item.opinions.get(OLD)
        preload, source = ([], None) if row is not None else preload_tracks(
            str(self.paths.dataset), item.clip_id, fps, (item.meta_path, item.video))
        tracks = [_track_dict(t) for t in preload] if row is None else row.tracks or []
        # the label files number the clip's own frames; fps is an estimate, so the clip is at least that long
        last = max((k["frame"] for t in tracks for k in t.get("keyframes", [])), default=-1)
        frames = max(round(duration * fps) if duration else 0, last + 1) or None
        common = dict(event_id=0, ai_status="none", ai_model=None, ai_prompt_version=None, fps=fps,
                      frame_count=frames, frame_size=None, ai_description=old.text if old else "")
        if row is None:
            return dict(common, version=0, status="new", tracks=tracks,
                        description=old.text if old else "", drop_clip=False, needs_review=False, author=None,
                        updated_utc=None, suggestions_used=bool(tracks), preload_source=source)
        return dict(common, version=row.version, status=row.status, tracks=row.tracks or [],
                    description=row.description, drop_clip=row.drop_clip, needs_review=row.needs_review,
                    author=row.author_name, updated_utc=row.created_at,
                    suggestions_used=any(t.get("source") in ("yolo", "suggestion") for t in row.tracks or []))

    def save_clip_boxes(self, session, staff, body, now: datetime) -> Dict[str, Any]:
        from sqlalchemy import select, text  # noqa: PLC0415

        from .. import labeling  # noqa: PLC0415
        from ..models import ClipAnnotation  # noqa: PLC0415

        item, _ = self._item(session, body.key)
        current = self.clip_boxes(session, body.key)
        session.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:k, 0))"), {"k": f"clipbox:{body.key}"})
        latest = session.scalar(select(ClipAnnotation.version).where(ClipAnnotation.clip_key == body.key)
                                .order_by(ClipAnnotation.version.desc()).limit(1)) or 0
        if body.base_version != latest:
            raise StudioError(f"Someone saved this clip since you opened it (now version {latest}). Reload it, "
                              "then save again.", 409)
        tracks = labeling.to_tracks([t.model_dump() for t in body.tracks])
        earlier = list(session.scalars(select(ClipAnnotation).where(ClipAnnotation.clip_key == body.key)))
        labeling.assign_track_ids(tracks, earlier, latest)
        frames = current["frame_count"]
        problems = labeling.problems(tracks, frames / current["fps"] if frames else None, frames)
        if problems:
            raise StudioError("; ".join(problems), 422)
        session.add(ClipAnnotation(clip_key=body.key, version=latest + 1, status=body.status,
                                   tracks=labeling.track_dicts(tracks), description=body.description,
                                   drop_clip=body.drop_clip, needs_review=body.needs_review, author_id=staff.id,
                                   author_name=staff.name, created_at=now))
        session.flush()
        return self.clip_boxes(session, body.key)

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
        sharegpt = exporter.sharegpt_rows(training)
        problems = exporter.check_sharegpt(sharegpt)
        if problems:
            raise StudioError("The training set would not load: " + "; ".join(problems[:3]), 500)
        sharegpt_path = os.path.join(out, stamp, exporter.SHAREGPT_FILE)
        os.makedirs(os.path.dirname(sharegpt_path), exist_ok=True)
        with open(sharegpt_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(sharegpt, f, ensure_ascii=False, indent=1)
        counts["sharegpt"] = len(sharegpt)
        result = {"training_path": exporter.write_jsonl(os.path.join(out, stamp, exporter.TRAINING_FILE), training),
                  "eval_path": exporter.write_jsonl(os.path.join(out, stamp, exporter.EVAL_FILE), evals),
                  "sharegpt_path": sharegpt_path, "counts": counts}
        if frames_dir:
            by_clip = {i.clip_id: i for i in items.values()}
            result["frames"] = exporter.write_frames(
                evals, frames_dir, lambda cid: self.local_media(by_clip[cid], "clip") if cid in by_clip else None)
        return result


NEEDS_CATEGORY = "Choose a category before saving (or mark the clip Delete)"
NEEDS_LABEL = "Choose the raw label before saving (or mark the clip Delete)"


def require_category(current: Optional[Tag], fields: Dict[str, Any]) -> None:
    """A saved tag always names a category, unless the clip is deleted: nothing is filled in for the tagger. A clip
    answered with the legacy prompt (no category in its schema) needs its raw label instead."""
    from ...fleet_contract.prompt_schemas import EYE, schema_kind  # noqa: PLC0415

    merged = {**(current.fields if current else {}), **fields}
    if merged.get("delete"):
        return
    if schema_kind(merged.get("prompt_version")) == EYE:
        if not merged.get("category"):
            raise StudioError(NEEDS_CATEGORY, 422)
    elif not merged.get("raw_label") and not merged.get("category"):
        raise StudioError(NEEDS_LABEL, 422)


def _track_dict(t) -> Dict[str, Any]:
    return {"track_id": t.track_id, "label": t.label, "source": t.source,
            **({"entity": t.entity} if getattr(t, "entity", None) else {}),
            "keyframes": [{"frame": k.frame, "t_sec": k.t_sec, "xyxy": list(k.xyxy), "enabled": k.enabled}
                          for k in t.keyframes]}


def _mac(secret: str, payload: str) -> str:
    key = hmac.new(secret.encode("utf-8"), _MEDIA_DOMAIN, hashlib.sha256).digest()
    return hmac.new(key, payload.encode("ascii"), hashlib.sha256).hexdigest()[:40]


def _iso(value: Optional[datetime]) -> str:
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
