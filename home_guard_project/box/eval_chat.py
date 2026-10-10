"""The golden conversation suite for the owner's Telegram assistant (the brain).

Every case is one real owner turn (tests/golden/conversations/*.yaml): the turns before it, the alerts in play, the
memory as it should be by then, the owner's words, and what a sharp, warm, human security operator would do. The
runner puts each case through the brain's real entry point (``OwnerAgentV2.handle``, or the inbox's tag path for an
answer after "🏷️ תיוג אחר") on a stubbed box - fake cameras and photos from the case, a temporary event book, activity
book, camera facts, mute and house state seeded from the case - and records the reply, the tool calls, the media
sent and every memory write. It scores:

(a) deterministic checks: the writes (type, scope, expiry), forbidden writes, the number of questions back, banned
    phrases (memory dumps, "מה לתקן?", empty "הבנתי אותך.", English, internal handles, camera ids), must-do actions
    (look at the cameras, send the video, look at the clip);
(b) an LLM judge (OpenRouter, openai/gpt-4o by default) on 1-5 scales with the ideal reply as a reference, plus four
    "stupid message" flags (nonsensical / irrelevant / repetitive / robotic) - any flag is a hard fail.

Usage::

    python -m home_guard_project.box.eval_chat run --suite tests/golden --runs 2 --report out.md
    python -m home_guard_project.box.eval_chat run --suite tests/golden --no-judge --only g10_
    python -m home_guard_project.box.eval_chat check --suite tests/golden        # schema + ideal replies only

The key comes from OPENROUTER_API_KEY (environment) or ``--key-file``. Nothing reaches Telegram, nothing touches the
box's own files (usage ledger off, every store in a temporary folder).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import glob
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("HOMEGUARD_USAGE_LEDGER", "off")

CHAT = "-5326761586"
OWNER = {"user_id": 1, "name": "Hello_24"}
JPEG = (b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00" + bytes(range(64))
        + b"\xff\xd9")
# $ per million tokens (input, output), OpenRouter list prices.
PRICES = {"openai/gpt-4o": (2.5, 10.0), "openai/gpt-4o-mini": (0.15, 0.6), "openai/gpt-4.1": (2.0, 8.0),
          "openai/gpt-4.1-mini": (0.4, 1.6), "anthropic/claude-sonnet-4.5": (3.0, 15.0)}
DEFAULT_BIG, DEFAULT_FAST, DEFAULT_JUDGE = "openrouter:openai/gpt-4o", "openrouter:openai/gpt-4o-mini", "openai/gpt-4o"

INTENTS = ("place_fact", "person_mark", "activity_explain", "tag_only", "question_live", "question_history",
           "question_meta", "complaint", "ack", "preference", "command")
WRITE_TYPES = ("place_fact", "person_mark", "activity", "tag", "pause", "camera_off", "alias", "preference",
               "house_state", "expect", "mark_removed", "activity_removed", "fact_removed")
DIMENSIONS = ("understood", "routing", "questions", "human_tone", "evidence", "concise", "no_recitation",
              "top_company")
PASS_AT = {"understood": 4, "routing": 4, "questions": 4, "human_tone": 4, "evidence": 3, "concise": 4,
           "no_recitation": 4, "top_company": 4}
STUPID_FLAGS = ("nonsensical", "irrelevant", "repetitive", "robotic")
MUST_DO = ("look", "send_video", "ask_clip", "send_photo")

# Phrases no reply may contain (the owner's complaints, 2026-10-07 .. 2026-10-10). A case lifts one with ``allow``.
GLOBAL_MUST_NOT = {
    "closing_offer": r"אם (?:יש|תרצה|צריך) (?:עוד )?(?:משהו|לדעת)[^.!?\n]*(?:אני כאן|אשמח)|אני כאן[!.]?\s*$",
    "what_to_fix": r"מה לתקן\??|מה תרצה לשנות|מה (?:אתה )?רוצה שאעשה",
    "memory_dump": r"שמור אצלי עכשיו|כבר מסומנים ב|מה שבזיכרון:|הנה מה שאני זוכר",
    "expected_alert_jargon": r"התרעה צפויה|כהתרעה אמיתית|כהתרעה שגויה",
    "internal_handles": r"\[handles:|receipts:|\bE\d+=|\bR\d+ [a-z_]+|tool_call|alert_id",
    "camera_id": r"ameer_[a-z0-9_]+_ch\d+|\b[a-z]+_ch\d+\b",
    "how_can_i_help": r"איך (?:אני )?(?:יכול|אוכל) לעזור",
}
EMPTY_EMPATHY = re.compile(r"^\s*(?:אני )?(?:מבין|הבנתי)(?: אותך| את התסכול שלך)?[.!]?\s*$")

_local = threading.local()


# ---------------------------------------------------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------------------------------------------------
def _yaml():
    import yaml  # noqa: PLC0415

    return yaml


def load_suite(path: str) -> List[Dict[str, Any]]:
    """Every case under *path* (a folder searched for conversations/*.yaml, a folder of .yaml, or one .yaml file),
    with the file's ``defaults`` merged in, in file order."""
    if os.path.isfile(path):
        files = [path]
    else:
        files = sorted(glob.glob(os.path.join(path, "conversations", "*.yaml"))) or \
            sorted(glob.glob(os.path.join(path, "*.yaml")))
    cases: List[Dict[str, Any]] = []
    for name in files:
        with open(name, encoding="utf-8") as f:
            doc = _yaml().safe_load(f) or {}
        defaults = doc.get("defaults") or {}
        for case in doc.get("cases") or []:
            merged = dict(defaults)
            merged.update(case)
            merged["_file"] = os.path.basename(name)
            cases.append(merged)
    return cases


def validate_case(case: Dict[str, Any]) -> List[str]:
    """Problems with one case's shape (empty when it is well formed)."""
    errs: List[str] = []
    cid = case.get("id") or "?"
    for key in ("id", "source", "date", "time", "message", "expect"):
        if not case.get(key):
            errs.append(f"{cid}: missing {key}")
    expect = case.get("expect") or {}
    if expect.get("intent") not in INTENTS:
        errs.append(f"{cid}: intent {expect.get('intent')!r} not one of {INTENTS}")
    if not str(expect.get("ideal") or "").strip():
        errs.append(f"{cid}: no ideal reply")
    if expect.get("max_questions") not in (0, 1):
        errs.append(f"{cid}: max_questions must be 0 or 1")
    for w in list(expect.get("writes") or []) + [{"type": t} for t in expect.get("forbid_writes") or []]:
        types = (w.get("type") if isinstance(w.get("type"), list) else [w.get("type")]) if isinstance(w, dict) else [None]
        if any(t not in WRITE_TYPES for t in types):
            errs.append(f"{cid}: unknown write {w!r}")
    for m in expect.get("must_do") or []:
        if str(m).split(":", 1)[0] not in MUST_DO:
            errs.append(f"{cid}: unknown must_do {m!r}")
    for a in expect.get("allow") or []:
        if a not in GLOBAL_MUST_NOT and a != "english":
            errs.append(f"{cid}: unknown allow {a!r}")
    msg = case.get("message") or {}
    if not str(msg.get("text") or "").strip():
        errs.append(f"{cid}: empty message")
    alerts = {a.get("id") for a in case.get("alerts") or []}
    if msg.get("reply_to") and msg["reply_to"] not in alerts:
        errs.append(f"{cid}: reply_to {msg['reply_to']} is not one of the case's alerts")
    elif msg.get("reply_to") and case.get("time") and case.get("date"):
        try:
            if msg["reply_to"] not in {a.get("id") for a in visible(case)["alerts"]}:
                errs.append(f"{cid}: reply_to {msg['reply_to']} is not before the message")
        except (ValueError, KeyError, TypeError) as exc:
            errs.append(f"{cid}: bad time ({exc})")
    if msg.get("via", "message") not in ("message", "tag"):
        errs.append(f"{cid}: via must be message or tag")
    if msg.get("via") == "tag" and not msg.get("reply_to"):
        errs.append(f"{cid}: a tag answer needs the alert it tags (reply_to)")
    return errs


def ts_of(case: Dict[str, Any], clock: Any, date: Optional[str] = None) -> float:
    """``"13:48"`` / ``"13:48:05"`` on the case's date, or a full ``"2026-10-15 18:00"``."""
    text = str(clock).strip()
    if re.match(r"\d{4}-\d{2}-\d{2}", text):
        day, _, hms = text.partition(" ")
    else:
        day, hms = date or str(case["date"]), text
    parts = [int(x) for x in (hms or "00:00").split(":")]
    while len(parts) < 3:
        parts.append(0)
    y, mo, d = (int(x) for x in day.split("-"))
    return dt.datetime(y, mo, d, *parts).timestamp()


def visible(case: Dict[str, Any], keep_turns: int = 14) -> Dict[str, Any]:
    """What the box knew just before the owner's message: the alerts and turns before the case's time (a day file
    keeps the whole day in ``defaults``), the last *keep_turns* turns, and the memory live at that moment (an entry
    with ``at`` later than the message, or ``gone`` at or before it, is left out)."""
    now = ts_of(case, case["time"])
    alerts = [a for a in case.get("alerts") or [] if ts_of(case, a["time"]) < now]
    history = [h for h in case.get("history") or [] if ts_of(case, h["time"]) < now][-keep_turns:]
    memory: Dict[str, Any] = {}
    for key, rows in (case.get("memory") or {}).items():
        memory[key] = [r for r in rows or [] if (not r.get("at") or ts_of(case, r["at"]) <= now)
                       and (not r.get("gone") or ts_of(case, r["gone"]) > now)]
    return {"alerts": alerts, "history": history, "memory": memory}


# ---------------------------------------------------------------------------------------------------------------------
# Deterministic checks (also used on the ideal replies by the pytest subset)
# ---------------------------------------------------------------------------------------------------------------------
def count_questions(text: str, buttons: Sequence[str] = ()) -> int:
    """Questions back to the owner: sentences ending in "?" with 5+ words (a short echo like "אה, הם של הפרגולה?"
    before the real question is not a second question - the owner's own example, F.4); a lone short question
    still counts once, and two or more choice buttons are a question too (one button, "📹 שלח את הסרטון", is an
    offer)."""
    parts = re.findall(r"[^?.!\n]*\?+", text or "")
    n = sum(1 for p in parts if len(p.replace("?", " ").split()) >= 5)
    if n == 0 and parts:
        n = 1
    return max(n, 1) if len(buttons or ()) >= 2 else n


def english_words(text: str) -> List[str]:
    """Latin words of 3+ letters that are not a model/brand token the owner uses himself."""
    allowed = {"ai", "p1", "p2", "p3", "car1", "car2", "led", "ok", "gpt", "qwen"}
    words = re.findall(r"[A-Za-z][A-Za-z']{2,}", text or "")
    return [w for w in words if w.lower() not in allowed]


def _clock(ts: Optional[float]) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%H:%M") if ts else ""


def write_matches(want: Dict[str, Any], got: Dict[str, Any], now: float) -> bool:
    types = want.get("type") if isinstance(want.get("type"), list) else [want.get("type")]
    if got.get("type") not in types:
        return False
    if want.get("camera") is not None:
        allowed = want["camera"] if isinstance(want["camera"], list) else [want["camera"]]
        have = list(got.get("cameras") or []) + [got.get("camera") or ""]
        if not any(c in have for c in allowed):
            return False
    scope = want.get("scope")
    if scope == "house" and got.get("camera") not in ("", None) and not got.get("whole_house"):
        return False
    if scope == "camera" and got.get("camera") in ("", None) and not got.get("cameras"):
        return False
    expiry = want.get("expiry")
    until = got.get("until")
    if expiry == "none" and until:
        return False
    if expiry == "today" and (not until or dt.datetime.fromtimestamp(until).date() != dt.datetime.fromtimestamp(now).date()):
        return False
    if isinstance(expiry, str) and expiry.startswith("until:") and _clock(until) != expiry.split(":", 1)[1]:
        return False
    if want.get("daily_to") and got.get("daily_to") != want["daily_to"]:
        return False
    return True


def deterministic(case: Dict[str, Any], out: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Each check: ``{"name", "ok", "detail"}``. *out* is a run result (or a fake one built from the ideal)."""
    expect = case.get("expect") or {}
    text = str(out.get("text") or "")
    writes = out.get("writes") or []
    tools = out.get("tools") or []
    now = ts_of(case, case["time"])
    checks: List[Dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    if out.get("error"):
        add("no_error", False, str(out["error"])[:200])
    add("replied", bool(text.strip()), "" if text.strip() else "no reply text")
    allow = set(expect.get("allow") or [])
    for key, pattern in GLOBAL_MUST_NOT.items():
        if key in allow:
            continue
        hit = re.search(pattern, text)
        if hit:
            add(f"no_{key}", False, hit.group(0))
    if EMPTY_EMPATHY.match(text):
        add("no_empty_empathy", False, text.strip())
    for phrase in expect.get("must_not") or []:
        if re.search(phrase, text):
            add("must_not", False, phrase)
    if expect.get("must_include_any"):
        ok = any(re.search(p, text) for p in expect["must_include_any"])
        add("must_include_any", ok, "" if ok else " | ".join(expect["must_include_any"]))
    if "english" not in allow:
        eng = english_words(text)
        hebrew = bool(re.search(r"[\u0590-\u05FF]", text))
        add("hebrew", len(eng) <= 1 and (hebrew or not eng), " ".join(eng[:6]))
    q = count_questions(text, out.get("buttons") or ())
    limit = int(expect.get("max_questions", 0))
    add("max_questions", q <= limit, f"{q} > {limit}" if q > limit else "")
    # writes
    asked = q >= 1
    wanted = expect.get("writes") or []
    missing = [w for w in wanted if not any(write_matches(w, g, now) for g in writes)
               and not ((w.get("or_ask") or expect.get("write_or_ask")) and asked)]
    if wanted:
        ok = not missing
        add("writes", ok, "" if ok else "missing " + json.dumps(missing, ensure_ascii=False))
    forbidden = set(expect.get("forbid_writes") or [])
    bad = [g for g in writes if g.get("type") in forbidden]
    if forbidden:
        add("forbid_writes", not bad, "" if not bad else json.dumps(bad, ensure_ascii=False)[:200])
    if expect.get("no_live_mark_until"):
        clock = expect["no_live_mark_until"]
        live = [m for m in out.get("marks_after") or [] if _clock(m.get("until")) == clock]
        add("no_live_mark_until", not live, json.dumps(live, ensure_ascii=False)[:160] if live else "")
    for item in expect.get("must_do") or []:
        kind, _, cam = str(item).partition(":")
        if kind == "look":
            looked = [t for t in tools if t in ("check_camera", "look_around")]
            ok = bool(looked) and (not cam or cam in (out.get("photo_cameras") or []) or "look_around" in looked)
            add("must_do_look", ok, "" if ok else f"tools={tools}")
        elif kind == "send_video":
            add("must_do_send_video", bool(out.get("videos")), "" if out.get("videos") else f"tools={tools}")
        elif kind == "send_photo":
            add("must_do_send_photo", bool(out.get("photos")), "" if out.get("photos") else f"tools={tools}")
        elif kind == "ask_clip":
            ok = any(t in ("ask_vision", "describe_event", "assess_event") for t in tools)
            add("must_do_ask_clip", ok, "" if ok else f"tools={tools}")
    return checks


def ideal_result(case: Dict[str, Any]) -> Dict[str, Any]:
    """The ideal reply as a run result: the writes it implies are the expected ones (for the self-check)."""
    expect = case.get("expect") or {}
    now = ts_of(case, case["time"])
    writes = []
    for w in expect.get("writes") or []:
        g = dict(w)
        if isinstance(g.get("type"), list):
            g["type"] = g["type"][0]
        exp = w.get("expiry")
        if exp == "today":
            g["until"] = now + 60
        elif isinstance(exp, str) and exp.startswith("until:"):
            g["until"] = ts_of(case, exp.split(":", 1)[1])
        cam = w.get("camera")
        cam = cam[0] if isinstance(cam, list) else cam
        g["camera"] = cam if cam is not None else ("" if w.get("scope") == "house" else "cam")
        writes.append(g)
    tools: List[str] = []
    out: Dict[str, Any] = {"text": expect.get("ideal", ""), "writes": writes, "tools": tools, "buttons": (),
                           "photos": [], "videos": [], "photo_cameras": [], "marks_after": []}
    for item in expect.get("must_do") or []:
        kind, _, cam = str(item).partition(":")
        if kind == "look":
            tools.append("check_camera")
            out["photo_cameras"].append(cam)
            out["photos"].append("x.jpg")
        elif kind == "send_video":
            out["videos"].append("x.mp4")
        elif kind == "send_photo":
            out["photos"].append("x.jpg")
        elif kind == "ask_clip":
            tools.append("ask_vision")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# The stubbed box
# ---------------------------------------------------------------------------------------------------------------------
class CountingModel:
    """Wraps a chat model: counts tokens and calls, and stops the run on a 402 (no credit)."""

    def __init__(self, inner: Any, name: str, ledger: "Spend") -> None:
        self._inner, self._name, self._ledger = inner, name, ledger

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        msg = self._inner.chat(*args, **kwargs)
        usage = getattr(msg, "usage", (0, 0)) or (0, 0)
        self._ledger.add(self._name, int(usage[0] or 0), int(usage[1] or 0))
        err = str(getattr(msg, "error", "") or "")
        if "402" in err or "insufficient" in err.lower() or "credit" in err.lower():
            self._ledger.stop = f"OpenRouter refused for credit: {err[:160]}"
        return msg

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class Spend:
    def __init__(self, cap_usd: float) -> None:
        self.cap, self.by_model, self.stop = cap_usd, {}, ""
        self._lock = threading.Lock()

    def add(self, model: str, tin: int, tout: int) -> None:
        with self._lock:
            row = self.by_model.setdefault(model, [0, 0, 0])
            row[0] += tin
            row[1] += tout
            row[2] += 1

    def usd(self) -> float:
        total = 0.0
        for model, (tin, tout, _) in self.by_model.items():
            a, b = PRICES.get(model.split(":", 1)[-1], (2.5, 10.0))
            total += tin / 1e6 * a + tout / 1e6 * b
        return total

    def over(self) -> bool:
        return bool(self.stop) or self.usd() >= self.cap


class FakeVision:
    """The live look answers the case's ``live`` text for the camera; a clip question answers its ``answer``."""

    model_name = "golden-fake-vision"

    def __init__(self, world: "World") -> None:
        self.world = world

    def look(self, camera: str, images: List[bytes], guard: bool, question: str = "",
             what: str = "a live photo") -> Dict[str, Any]:
        self.world.vision_calls.append(("look", camera, question))
        if what != "a live photo":
            alert = self.world.alert_by_camera_recent(camera)
            text = (alert or {}).get("answer") or (alert or {}).get("summary_en") or (alert or {}).get("summary") \
                or "Nothing clear is visible."
            return {"ok": True, "description": text, "quality": "clear", "people": int((alert or {}).get("people") or 0),
                    "label": "normal", "why": ""}
        live = self.world.live(camera)
        out = {"ok": True, "description": live["description"], "quality": "clear", "people": live["people"]}
        if guard:
            out.update(label=live.get("label", "normal"), why=live.get("why", ""))
        return out

    def ask(self, camera: str, images: List[bytes], question: str, language: str = "English") -> Dict[str, Any]:
        self.world.vision_calls.append(("ask", camera, question))
        alert = self.world.alert_by_camera_recent(camera)
        answer = (alert or {}).get("answer") or (alert or {}).get("summary") or "Nothing clear is visible."
        return {"ok": True, "answer": answer, "frame": 2, "seen": True}


class FakeDeliverer:
    def __init__(self, world: "World") -> None:
        self.world = world
        self._n = 1000

    def _id(self) -> int:
        self._n += 1
        return self._n

    def photo(self, chat_id: str, path: str, caption: str = "") -> Dict[str, Any]:
        self.world.photos.append(path)
        return {"ok": True, "message_id": self._id()}

    def video(self, chat_id: str, path: str, caption: str = "") -> Dict[str, Any]:
        self.world.videos.append(path)
        return {"ok": True, "message_id": self._id()}

    def text(self, chat_id: str, text: str, reply_to: Optional[int] = None, **kwargs: Any) -> Dict[str, Any]:
        self.world.texts.append(text)
        return {"ok": True, "message_id": self._id()}

    def typing(self, chat_id: str) -> None:
        return None


DEFAULT_LIVE = "A quiet yard and driveway; no people, no moving vehicles."


class World:
    """One case's box: a temporary folder with every store the brain touches, and a clock at the case's time."""

    def __init__(self, case: Dict[str, Any], big: Any, fast: Any) -> None:
        from .brain.agent import OwnerAgentV2  # noqa: PLC0415
        from .brain.memory import ChatMemory  # noqa: PLC0415
        from .brain.receipts import ReceiptBook  # noqa: PLC0415
        from .brain.tools import Services  # noqa: PLC0415
        from .activity_memory import ActivityBook  # noqa: PLC0415
        from .events import EventBook  # noqa: PLC0415
        from .feedback import MuteState  # noqa: PLC0415
        from .house_state import HouseStateStore  # noqa: PLC0415

        self.case = case
        self.root = tempfile.mkdtemp(prefix="golden_")
        self.clock = {"now": ts_of(case, case["time"])}
        house = case.get("house") or {}
        self.cameras: List[str] = list(house.get("cameras") or [])
        self.aliases: Dict[str, List[str]] = {k: list(v) for k, v in (house.get("aliases") or {}).items()}
        self.alerts: Dict[str, Dict[str, Any]] = {}
        self.photos: List[str] = []
        self.photo_cameras: List[str] = []
        self.videos: List[str] = []
        self.texts: List[str] = []
        self.vision_calls: List[Tuple[str, str, str]] = []
        self.option_writes: List[Dict[str, Any]] = []
        self.alias_writes: List[Dict[str, Any]] = []
        self.camera_writes: List[Dict[str, Any]] = []
        self.events = EventBook(os.path.join(self.root, "events"))
        self.activities = ActivityBook(os.path.join(self.root, "activity.json"))
        self.mute = MuteState(os.path.join(self.root, "alert_mute.json"))
        self.house = HouseStateStore(os.path.join(self.root, "house_state.jsonl"),
                                     mute_path=self.mute.path, now=lambda: self.clock["now"])
        os.makedirs(os.path.join(self.root, "meta"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "clips"), exist_ok=True)
        os.makedirs(os.path.join(self.root, ".live"), exist_ok=True)
        world = self

        class Registry:
            def snapshot(self) -> Any:
                from .brain.registry import CameraState, HouseSnapshot  # noqa: PLC0415

                now = world.clock["now"]
                cams = []
                for c in world.cameras:
                    muted = None
                    try:
                        muted = world.mute.muted_until(now, c) or None
                    except Exception:  # noqa: BLE001
                        muted = None
                    cams.append(CameraState(c, c not in world.disabled, tuple(world.aliases.get(c, ())), live=True,
                                            last_seen=now - 2, muted_until=muted))
                return HouseSnapshot(now=now, mode="guard", mode_ends=now + 3600, mode_started=now - 3600,
                                     start_hour=0, end_hour=0, cameras=tuple(cams))

        self.disabled: set = set()
        self.registry = Registry()
        self.services = Services(
            roots=lambda: [self.root], desc_dir=os.path.join(self.root, ".desc"), feedback_dir=self.root,
            work_dir=os.path.join(self.root, ".live"), mute=self.mute, deliver=FakeDeliverer(self),
            vision=FakeVision(self), grab_photo=self._grab_photo, record_live=self._record_live,
            cut_segment=self._cut_segment, clip_frames=lambda path, count=4: [JPEG] * 4,
            clip_frames_at=lambda path, count=8, start=None, end=None: [(float(i), JPEG) for i in range(4)],
            set_camera=self._set_camera, add_alias=self._add_alias, remove_alias=self._remove_alias,
            request_restart=lambda: None, embedder=None, now=lambda: self.clock["now"],
            set_option=self._set_option, read_settings=lambda: {"owner_language": "he"}, alert_settings=None,
            house=self.house, events=self.events, activities=self.activities)
        self.agent = OwnerAgentV2(big, self.registry, ChatMemory(os.path.join(self.root, ".conversations")),
                                  ReceiptBook(os.path.join(self.root, ".receipts"), now=lambda: self.clock["now"]),
                                  self.services, fast_model=fast, now=lambda: self.clock["now"])

    # -- services ------------------------------------------------------------------------------------------------
    def live(self, camera: str) -> Dict[str, Any]:
        entry = (self.case.get("live") or {}).get(camera)
        if isinstance(entry, str):
            entry = {"description": entry}
        entry = dict(entry or {})
        entry.setdefault("description", DEFAULT_LIVE)
        entry.setdefault("people", 0)
        return entry

    def _grab_photo(self, camera: str) -> Dict[str, Any]:
        if camera not in self.cameras:
            return {"error": "camera_unknown"}
        path = os.path.join(self.root, ".live", f"{camera}_{int(self.clock['now'])}_{len(self.photos)}.jpg")
        with open(path, "wb") as f:
            f.write(JPEG)
        self.photo_cameras.append(camera)
        return {"camera": camera, "image": path}

    def _record_live(self, camera: str, seconds: float) -> Dict[str, Any]:
        path = os.path.join(self.root, ".live", f"{camera}_{int(self.clock['now'])}_live.mp4")
        with open(path, "wb") as f:
            f.write(b"\x00\x00\x00\x18ftypmp42")
        return {"ok": True, "path": path, "start": self.clock["now"], "end": self.clock["now"] + seconds}

    def _cut_segment(self, clip_path: str, clip_start_ts: float, clip_end_ts: float, start_ts: float, seconds: float,
                     out_path: str, *args: Any, **kwargs: Any) -> Optional[Tuple[float, float]]:
        shutil.copyfile(clip_path, out_path)
        return clip_start_ts, clip_end_ts

    def _set_camera(self, camera: str, active: bool) -> Dict[str, Any]:
        (self.disabled.discard if active else self.disabled.add)(camera)
        self.camera_writes.append({"camera": camera, "active": active})
        return {"ok": True}

    def _add_alias(self, camera: str, alias: str, cameras: Sequence[str]) -> List[str]:
        names = self.aliases.setdefault(camera, [])
        if alias not in names:
            names.append(alias)
        self.alias_writes.append({"camera": camera, "alias": alias})
        return list(names)

    def _remove_alias(self, camera: str, alias: str) -> List[str]:
        names = self.aliases.setdefault(camera, [])
        if alias in names:
            names.remove(alias)
        self.alias_writes.append({"camera": camera, "alias": alias, "removed": True})
        return list(names)

    def _set_option(self, key: str, value: Any) -> Dict[str, Any]:
        self.option_writes.append({"key": key, "value": value})
        return {"ok": True}

    def alert_by_camera_recent(self, camera: str) -> Optional[Dict[str, Any]]:
        rows = [a for a in self.alerts.values() if a["camera"] == camera and a["ts"] <= self.clock["now"] + 1]
        reply_to = (self.case.get("message") or {}).get("reply_to")
        if reply_to in self.alerts and self.alerts[reply_to]["camera"] == camera:
            return self.alerts[reply_to]
        return max(rows, key=lambda a: a["ts"]) if rows else None

    # -- seeding -------------------------------------------------------------------------------------------------
    def seed(self) -> None:
        from .brain.memory import ChatMemory  # noqa: PLC0415,F401

        case = self.case
        view = visible(case)
        memory = view["memory"]
        start = min([self.clock["now"]] + [ts_of(case, a["time"]) for a in view["alerts"]]
                    + [ts_of(case, h["time"]) for h in view["history"]]) - 60
        for m in memory.get("marks") or []:
            at = ts_of(case, m.get("at") or start)
            daily = m.get("daily") or []
            self.events.mark_known(m.get("camera") or "", m["text"], "Hello_24", ts_of(case, m["until"]), now=at,
                                   daily_from=daily[0] if daily else "", daily_to=daily[1] if daily else "")
        for a in memory.get("activities") or []:
            at = ts_of(case, a.get("at") or start)
            daily = a.get("daily") or []
            self.activities.add(a.get("cameras") or [], a.get("actions") or ["working_ground"], a["cause"],
                                ts_of(case, a["until"]), at, place=a.get("place", ""),
                                owner_words=a.get("owner_words", a["cause"]), daily_from=daily[0] if daily else "",
                                daily_to=daily[1] if daily else "", by="Hello_24")
        from .brain.tools import profiles_for  # noqa: PLC0415

        store = profiles_for(self.services)
        for fct in memory.get("facts") or []:
            store.add_fact(fct.get("camera") or "", fct["text"], by="Hello_24", now=start)
        for pause in memory.get("pauses") or []:
            from .feedback import Feedback  # noqa: PLC0415

            self.mute.apply(Feedback(action="mute", mute_until=ts_of(case, pause["until"]),
                                     camera=pause.get("camera")), start)
        # alerts and the turns before the owner's message, in time order
        timeline: List[Tuple[float, int, str, Dict[str, Any]]] = []
        for a in view["alerts"]:
            timeline.append((ts_of(case, a["time"]), 0, "alert", a))
        for i, h in enumerate(view["history"]):
            timeline.append((ts_of(case, h["time"]), 1 + i, "turn", h))
        for ts, _, kind, item in sorted(timeline, key=lambda x: (x[0], x[1])):
            self.clock["now"] = ts
            if kind == "alert":
                self._seed_alert(item, ts)
            else:
                self._seed_turn(item, ts)
        self.clock["now"] = ts_of(case, case["time"])

    def _seed_alert(self, a: Dict[str, Any], ts: float) -> None:
        alert_id = a["id"]
        camera = a["camera"]
        clip_rel = f"clips/{alert_id}.mp4"
        with open(os.path.join(self.root, clip_rel), "wb") as f:
            f.write(b"\x00\x00\x00\x18ftypmp42")
        label = a.get("label", "suspicious")
        command = {"normal": "[none]", "suspicious": "[send_message]", "escalation": "[call_owner]"}.get(label, "")
        meta = {"camera_name": camera, "clip_path": clip_rel, "clip_end_ts": ts, "clip_start_ts": ts - 12,
                "trigger_ts": ts - 10, "kind": "alert", "mode": "inference",
                "alert": {"summary": a.get("summary", ""), "label": label, "alert_command": command,
                          "people": a.get("people", 1)},
                "yolo": {"trigger_classes": a.get("classes") or ["person"]}}
        with open(os.path.join(self.root, "meta", f"{alert_id}.meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        record = {"alert_id": alert_id, "camera": camera, "ts": ts, "label": label,
                  "summary": a.get("summary", ""), "people": a.get("people", 1), "answer": a.get("answer", "")}
        self.alerts[alert_id] = record
        try:
            self.events.decide(camera, ts, label, a.get("people", 1), a.get("summary", ""), alert_id)
        except Exception:  # noqa: BLE001 - the session is context only
            pass
        if a.get("in_chat", True):
            self.agent.note_alert(CHAT, {k: record[k] for k in ("alert_id", "camera", "ts", "label", "summary")})

    def _seed_turn(self, h: Dict[str, Any], ts: float) -> None:
        if h.get("tag_for"):
            alert = self.alerts.get(h["tag_for"])
            if alert:
                self.agent.note_tag(CHAT, {k: alert[k] for k in ("alert_id", "camera", "ts", "label", "summary")},
                                    h.get("owner", ""), OWNER)
            return
        state = self.agent.memory.load(CHAT)
        handles = []
        if h.get("about") and h["about"] in self.alerts:
            alert = self.alerts[h["about"]]
            handles.append(state.add_handle("event", alert["alert_id"], alert["camera"], alert["ts"],
                                            alert.get("summary", "")))
        for cam in h.get("photos") or []:
            handle = state.add_handle("photo", f"{cam}_{int(ts)}.jpg", cam, ts)
            state.note_observation(handle, self.live(cam)["description"])
            handles.append(handle)
        state.add_turn(str(OWNER["user_id"]), str(h.get("owner") or ""), str(h.get("bot") or ""), handles, [], ts)
        self.agent.memory.save(CHAT, state)

    # -- memory state ---------------------------------------------------------------------------------------------
    def state(self) -> Dict[str, Any]:
        from .brain.tools import profiles_for  # noqa: PLC0415

        now = self.clock["now"]
        store = profiles_for(self.services)
        facts = []
        try:
            for f in store.house_facts():
                facts.append({"id": f["id"], "camera": "", "text": f["text"]})
            for cam, rows in store.all_facts().items():
                for f in rows:
                    facts.append({"id": f["id"], "camera": cam, "text": f["text"]})
        except Exception:  # noqa: BLE001
            pass
        tags = sorted(glob.glob(os.path.join(self.root, "feedback", "**", "*.json"), recursive=True))
        try:
            with open(self.house.path, encoding="utf-8") as f:
                house_lines = [json.loads(x) for x in f if x.strip()]
        except OSError:
            house_lines = []
        return {"marks": self.events.list_known(now), "activities": [vars(a).copy() for a in self.activities.live(now)],
                "facts": facts, "tags": tags, "house": house_lines, "mute": self.mute.snapshot(),
                "aliases": json.loads(json.dumps(self.aliases)), "options": list(self.option_writes),
                "cameras_off": sorted(self.disabled)}


def diff_writes(before: Dict[str, Any], after: Dict[str, Any], world: World) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    old = {m["id"]: m for m in before["marks"]}
    new = {m["id"]: m for m in after["marks"]}
    for k, m in new.items():
        if k not in old:
            out.append({"type": "person_mark", "camera": m.get("camera") or "", "text": m.get("text"),
                        "until": m.get("until"), "daily_from": m.get("daily_from"), "daily_to": m.get("daily_to"),
                        "whole_house": not m.get("camera")})
    for k, m in old.items():
        if k not in new:
            out.append({"type": "mark_removed", "camera": m.get("camera") or "", "text": m.get("text")})
    old_a = {a["id"]: a for a in before["activities"]}
    new_a = {a["id"]: a for a in after["activities"]}
    for k, a in new_a.items():
        if k not in old_a:
            out.append({"type": "activity", "cameras": a.get("cameras"), "camera": (a.get("cameras") or [""])[0],
                        "text": a.get("cause"), "until": a.get("until"), "actions": a.get("actions"),
                        "place": a.get("place"), "daily_to": a.get("daily_to")})
        elif a.get("until") != old_a[k].get("until") or a.get("cameras") != old_a[k].get("cameras"):
            out.append({"type": "activity", "cameras": a.get("cameras"), "camera": (a.get("cameras") or [""])[0],
                        "text": a.get("cause"), "until": a.get("until"), "changed": True})
    for k in old_a:
        if k not in new_a:
            out.append({"type": "activity_removed", "text": old_a[k].get("cause")})
    old_f = {f["id"] for f in before["facts"]}
    new_f = {f["id"]: f for f in after["facts"]}
    for k, f in new_f.items():
        if k not in old_f:
            out.append({"type": "place_fact", "camera": f["camera"], "text": f["text"], "until": None,
                        "whole_house": not f["camera"]})
    for k in old_f - set(new_f):
        out.append({"type": "fact_removed"})
    for path in after["tags"][len(before["tags"]):] if len(after["tags"]) > len(before["tags"]) else []:
        try:
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
        except (OSError, ValueError):
            rec = {}
        out.append({"type": "tag", "camera": rec.get("camera") or "", "text": rec.get("owner_text") or
                    rec.get("raw_text") or "", "label": rec.get("owner_label") or rec.get("verdict") or ""})
    new_tags = set(after["tags"]) - set(before["tags"])
    if len(new_tags) and len(after["tags"]) <= len(before["tags"]):
        out.append({"type": "tag"})
    for line in after["house"][len(before["house"]):]:
        kind = str(line.get("kind") or line.get("op") or line.get("type") or "")
        out.append({"type": "expect" if "expect" in kind else "house_state", "detail": kind})
    if json.dumps(after["mute"], sort_keys=True) != json.dumps(before["mute"], sort_keys=True):
        until = after["mute"].get("all") or max([0.0] + list((after["mute"].get("cameras") or {}).values()))
        out.append({"type": "pause", "until": until or None, "camera": "" if after["mute"].get("all") else
                    ",".join(sorted(after["mute"].get("cameras") or {}))})
    for w in world.alias_writes:
        out.append({"type": "alias", "camera": w["camera"], "text": w["alias"]})
    for w in world.option_writes[len(before["options"]):]:
        out.append({"type": "preference", "text": f"{w['key']}={w['value']}"})
    for w in world.camera_writes:
        if not w["active"]:
            out.append({"type": "camera_off", "camera": w["camera"]})
    return out


# ---------------------------------------------------------------------------------------------------------------------
# One case through the brain
# ---------------------------------------------------------------------------------------------------------------------
def _patch_aliases() -> None:
    """Camera names come from the running case (thread-local), never from the laptop's or the box's files."""
    from . import camera_names  # noqa: PLC0415
    from .brain import aliases as brain_aliases  # noqa: PLC0415

    if getattr(camera_names, "_golden_patched", False):
        return
    original = camera_names._load

    def load(aliases: Optional[Dict[str, List[str]]]) -> Dict[str, List[str]]:
        if aliases is not None:
            return aliases
        world = getattr(_local, "world", None)
        return world.aliases if world is not None else original(None)

    def load_file(path: str = "") -> Dict[str, List[str]]:
        world = getattr(_local, "world", None)
        return {k: list(v) for k, v in world.aliases.items()} if world is not None else {}

    camera_names._load = load
    brain_aliases.load_aliases = load_file
    camera_names._golden_patched = True


def run_case(case: Dict[str, Any], big: Any, fast: Any) -> Dict[str, Any]:
    """The reply, tools, media and memory writes of one case. Never raises (an error is in ``error``)."""
    from .brain.known_memory import tag_line  # noqa: PLC0415
    from .brain.mode import hhmm  # noqa: PLC0415
    from .brain.registry import display  # noqa: PLC0415
    from .brain.style import clean_outgoing  # noqa: PLC0415
    from .feedback import Feedback, save_feedback, verdict_for  # noqa: PLC0415
    from .telegram_agent import is_tag_answer  # noqa: PLC0415

    _patch_aliases()
    started = time.time()
    out: Dict[str, Any] = {"id": case["id"], "text": "", "buttons": [], "tools": [], "writes": [], "photos": [],
                           "videos": [], "photo_cameras": [], "error": "", "path": ""}
    world: Optional[World] = None
    try:
        world = World(case, big, fast)
        _local.world = world
        world.seed()
        before = world.state()
        msg = case["message"]
        text = str(msg["text"])
        alert = world.alerts.get(msg.get("reply_to") or "")
        alert_arg = {k: alert[k] for k in ("alert_id", "camera", "ts", "label", "summary")} if alert else None
        if msg.get("via") == "tag" and alert and is_tag_answer(text, None, {}):
            out["path"] = "tag"
            label = world.agent.tag_label_for(alert_arg, text)
            save_feedback(world.root, alert_arg, Feedback(verdict=verdict_for(label, alert["label"]), owner_label=label,
                                                          owner_text=text, source="text"),
                          text, OWNER, CHAT, world.clock["now"])
            ask = world.agent.note_tag(CHAT, alert_arg, text, OWNER, label=label)
            snap = world.registry.snapshot()
            reply = tag_line(hhmm(alert["ts"]), display(snap, alert["camera"], "he"), label, text, "he")
            if ask is not None and getattr(ask, "text", ""):
                reply += "\n" + ask.text
                out["buttons"] = list(getattr(ask, "buttons", ()) or ())
            out["text"] = reply
        else:
            out["path"] = "brain"
            reply = world.agent.handle(text, CHAT, OWNER, alert_arg, bool(alert_arg))
            if reply is None:
                out["text"] = ""
            else:
                out["text"] = clean_outgoing(str(reply.text or ""), world.cameras, "he")
                out["buttons"] = [str(b) for b in reply.buttons or ()]
                rows = [label for row in (reply.rows or ()) for label, _ in row]
                out["row_buttons"] = rows
                out["tools"] = list(reply.tools_called or ())
                out["receipts"] = [f"{r.tool}:{r.status}" for r in reply.receipts or ()]
                for fn in reply.after or ():
                    try:
                        fn()
                    except Exception:  # noqa: BLE001
                        pass
        after = world.state()
        out["writes"] = diff_writes(before, after, world)
        out["marks_after"] = after["marks"]
        out["photos"] = [os.path.basename(p) for p in world.photos]
        out["videos"] = [os.path.basename(p) for p in world.videos]
        out["photo_cameras"] = list(world.photo_cameras)
        out["vision_calls"] = len(world.vision_calls)
    except Exception as exc:  # noqa: BLE001 - one broken case must not stop the suite
        import traceback  # noqa: PLC0415

        out["error"] = f"{type(exc).__name__}: {exc}"
        out["trace"] = traceback.format_exc()[-1500:]
    finally:
        _local.world = None
        if world is not None:
            shutil.rmtree(world.root, ignore_errors=True)
    out["seconds"] = round(time.time() - started, 1)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------------------------------------------------
JUDGE_SYSTEM = """You grade replies of a home-security assistant that talks with a homeowner in Hebrew on Telegram.
The bar: a top-company human agent (think the best remote-monitoring operator at a premium security firm) - sharp,
warm, short, specific, answers first, no filler, never recites its internal state, never asks what it can work out
itself, never pretends. The owner hates: questions he already answered, "מה לתקן?", empty "הבנתי אותך.", the bot
dumping its memory ("שמור אצלי עכשיו: ..."), mentioning unrelated things (e.g. the workers when he talks about the
neighbour's house), English, invented times (e.g. "until 23:59"), internal ids/handles, closing offers ("אם יש עוד
משהו אני כאן"), and asking "until when?" about something permanent (a place is permanent).
Routing rules the owner agreed on: what he writes after pressing the 🏷️ tag button is a TAG (training label for the
detection model only, no memory, no question). A plain message explaining people/places/activities is MEMORY (and
changes what the box does). Complaints, questions and acks are neither: no tag, no memory write.

You get the context, the owner's message, the expected behaviour, an IDEAL reply written by a human expert (a
reference for quality, NOT an exact target: a different wording that is as good scores as high), and the actual
reply with the tools it called and the memory writes it made. Score the ACTUAL reply, 1-5 each (5 = as good as the
ideal or better, 3 = acceptable but clearly weaker, 1 = wrong/harmful):
- understood: got what the owner meant without needing more explanation.
- routing: the right memory / tag / nothing routing, with the right scope and expiry (judge the writes made).
- questions: asks nothing it could work out; at most one short question, only if truly needed.
- human_tone: sounds like a human security operator, not a bot; warm but not sugary; no canned phrases.
- evidence: claims are backed by what it looked at (cameras, the clip, the event book); no unbacked certainty. If
  the turn needs no evidence, score 5 unless the reply invents facts.
- concise: short, answers first, no filler or repetition.
- no_recitation: does not recite memory/internal state that the owner did not ask for; no ids or handles.
- top_company: would a top-company human agent send exactly this message?
Flags (true/false), each a "stupid message": nonsensical (doesn't make sense), irrelevant (talks about something
else), repetitive (repeats a line already sent in the context or within itself), robotic (template/system voice).
Return JSON only: {"scores": {"understood": n, "routing": n, "questions": n, "human_tone": n, "evidence": n,
"concise": n, "no_recitation": n, "top_company": n}, "flags": {"nonsensical": b, "irrelevant": b, "repetitive": b,
"robotic": b}, "reason": "<one or two sentences>"}"""


def _context_text(case: Dict[str, Any]) -> str:
    lines = []
    view = visible(case)
    names = (case.get("house") or {}).get("aliases") or {}
    for a in view["alerts"][-6:]:
        lines.append(f"[{a['time']}] BOX ALERT ({a.get('label', 'suspicious')}) camera "
                     f"{names.get(a['camera'], [a['camera']])[0]}: {a.get('summary', '')}")
    for h in view["history"][-8:]:
        if h.get("tag_for"):
            lines.append(f"[{h['time']}] OWNER (after the 🏷️ tag button, a tag): {h.get('owner', '')}")
            continue
        if h.get("owner"):
            lines.append(f"[{h['time']}] OWNER: {h['owner']}")
        if h.get("bot") or h.get("photos"):
            shot = f" [+ photos: {', '.join(h['photos'])}]" if h.get("photos") else ""
            lines.append(f"[{h['time']}] BOT: {h.get('bot') or ''}{shot}")
    mem = view["memory"]
    if any(mem.get(k) for k in ("marks", "activities", "facts", "pauses")):
        lines.append("MEMORY BEFORE: " + json.dumps(mem, ensure_ascii=False))
    live = case.get("live") or {}
    if live:
        lines.append("WHAT THE CAMERAS SHOW NOW (only if the bot looks): " + json.dumps(live, ensure_ascii=False))
    return "\n".join(lines) or "(no earlier context)"


def judge_case(case: Dict[str, Any], result: Dict[str, Any], judge: Any) -> Dict[str, Any]:
    expect = case.get("expect") or {}
    msg = case["message"]
    user = (f"CONTEXT (oldest first; times are local):\n{_context_text(case)}\n\n"
            f"OWNER'S MESSAGE at {case['time']}"
            f"{' (a Telegram reply to the alert ' + msg['reply_to'] + ')' if msg.get('reply_to') else ''}"
            f"{' (typed after pressing the 🏷️ tag button)' if msg.get('via') == 'tag' else ''}: {msg['text']}\n\n"
            f"EXPECTED: intent={expect.get('intent')}; memory writes={json.dumps(expect.get('writes') or [], ensure_ascii=False)}"
            f"{' (or one short question first)' if expect.get('write_or_ask') else ''}; forbidden writes="
            f"{expect.get('forbid_writes') or []}; at most {expect.get('max_questions', 0)} question(s); must do="
            f"{expect.get('must_do') or []}. Notes: {case.get('notes') or expect.get('notes') or '-'}\n"
            f"IDEAL REPLY (reference): {expect.get('ideal')}\n\n"
            f"ACTUAL REPLY: {result.get('text') or '(nothing sent)'}\n"
            f"ACTUAL BUTTONS: {result.get('buttons') or []}\n"
            f"ACTUAL TOOLS CALLED: {result.get('tools') or []}; photos sent: {len(result.get('photos') or [])}; "
            f"videos sent: {len(result.get('videos') or [])}\n"
            f"ACTUAL MEMORY WRITES: {json.dumps(_writes_for_judge(result.get('writes') or []), ensure_ascii=False)}")
    messages = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]
    for attempt in range(2):
        msg_out = judge.chat(messages, None)
        raw = str(getattr(msg_out, "content", "") or "")
        parsed = _json_obj(raw)
        if parsed and isinstance(parsed.get("scores"), dict):
            scores = {d: _score(parsed["scores"].get(d)) for d in DIMENSIONS}
            flags = {f: bool((parsed.get("flags") or {}).get(f)) for f in STUPID_FLAGS}
            return {"scores": scores, "flags": flags, "reason": str(parsed.get("reason") or "")[:400]}
        if getattr(msg_out, "error", ""):
            return {"error": str(msg_out.error)[:200]}
    return {"error": "judge returned no JSON"}


def _writes_for_judge(writes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for w in writes:
        w = dict(w)
        if w.get("until"):
            w["until"] = dt.datetime.fromtimestamp(w["until"]).strftime("%a %d.%m %H:%M")
        out.append(w)
    return out


def _score(v: Any) -> int:
    try:
        return max(1, min(5, int(round(float(v)))))
    except (TypeError, ValueError):
        return 1


def _json_obj(raw: str) -> Optional[Dict[str, Any]]:
    start, end = raw.find("{"), raw.rfind("}")
    if not 0 <= start < end:
        return None
    try:
        parsed = json.loads(raw[start:end + 1])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def verdict(case: Dict[str, Any], result: Dict[str, Any], judged: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    checks = deterministic(case, result)
    failed = [c for c in checks if not c["ok"]]
    stupid = bool(judged and any((judged.get("flags") or {}).values()))
    low = [d for d, s in ((judged or {}).get("scores") or {}).items() if s < PASS_AT[d]]
    ok = not failed and (judged is None or ("error" not in judged and not stupid and not low))
    return {"pass": ok, "det_pass": not failed, "failed_checks": failed, "stupid": stupid, "low_dims": low}


# ---------------------------------------------------------------------------------------------------------------------
# Running the suite
# ---------------------------------------------------------------------------------------------------------------------
def read_key(key_file: str = "") -> Dict[str, str]:
    env = {"OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY", "")}
    path = key_file or os.path.expanduser("~/.homeguard/api_key.env")
    if not env["OPENROUTER_API_KEY"] and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    if k.strip() == "OPENROUTER_API_KEY":
                        env["OPENROUTER_API_KEY"] = v.strip().strip('"').strip("'")
    return env


def make_models(env: Dict[str, str], spend: Spend, big_spec: str = DEFAULT_BIG, fast_spec: str = DEFAULT_FAST,
                judge_name: str = DEFAULT_JUDGE, need_judge: bool = True) -> Tuple[Any, Any, Any]:
    from .brain.models import make_model  # noqa: PLC0415

    big = make_model(big_spec, env)
    fast = make_model(fast_spec, env) if fast_spec else None
    judge = make_model(f"openrouter:{judge_name}", env) if need_judge else None
    if big is None or (need_judge and judge is None):
        raise SystemExit("no OpenRouter key: set OPENROUTER_API_KEY or pass --key-file")
    return (CountingModel(big, big_spec, spend), CountingModel(fast, fast_spec, spend) if fast else None,
            CountingModel(judge, judge_name, spend) if judge else None)


def run_suite(cases: List[Dict[str, Any]], runs: int, big: Any, fast: Any, judge: Any, spend: Spend,
              workers: int = 4, log: Callable[[str], None] = print) -> List[List[Dict[str, Any]]]:
    """``results[run][i]`` for every case: the run result with ``judge`` and ``verdict``."""
    all_runs: List[List[Dict[str, Any]]] = []
    for r in range(runs):
        results: List[Optional[Dict[str, Any]]] = [None] * len(cases)

        def one(i: int) -> None:
            case = cases[i]
            if spend.over():
                results[i] = {"id": case["id"], "error": "skipped: " + (spend.stop or "spend cap reached"),
                              "text": "", "writes": [], "tools": []}
            else:
                results[i] = run_case(case, big, fast)
            res = results[i]
            judged = None
            if judge is not None and not spend.over() and not str(res.get("error", "")).startswith("skipped"):
                try:
                    judged = judge_case(case, res, judge)
                except Exception as exc:  # noqa: BLE001
                    judged = {"error": f"{type(exc).__name__}: {exc}"}
            res["judge"] = judged
            res["verdict"] = verdict(case, res, judged)
            log(f"run {r + 1} {case['id']:<10} {'PASS' if res['verdict']['pass'] else 'fail'} "
                f"${spend.usd():.3f} {res.get('seconds', 0)}s  {(res.get('text') or res.get('error') or '')[:70]!r}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(one, range(len(cases))))
        all_runs.append([x for x in results if x is not None])
        if spend.stop:
            log("STOPPED: " + spend.stop)
            break
    return all_runs


def summarize(cases: List[Dict[str, Any]], all_runs: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    per_run = []
    for results in all_runs:
        dims = {d: [] for d in DIMENSIONS}
        for res in results:
            for d, s in (((res.get("judge") or {}).get("scores")) or {}).items():
                dims[d].append(s)
        per_run.append({
            "passed": sum(1 for x in results if x["verdict"]["pass"]),
            "det_passed": sum(1 for x in results if x["verdict"]["det_pass"]),
            "stupid": sum(1 for x in results if x["verdict"]["stupid"]),
            "dims": {d: (statistics.mean(v) if v else None) for d, v in dims.items()},
            "n": len(results)})
    flips = 0
    if len(all_runs) > 1:
        for i in range(len(cases)):
            vals = {run[i]["verdict"]["pass"] for run in all_runs if i < len(run)}
            flips += len(vals) > 1
    return {"per_run": per_run, "flips": flips}


def _mean_score(res: Dict[str, Any]) -> float:
    scores = ((res.get("judge") or {}).get("scores")) or {}
    return statistics.mean(scores.values()) if scores else 0.0


def write_report(path: str, cases: List[Dict[str, Any]], all_runs: List[List[Dict[str, Any]]], spend: Spend,
                 title: str, meta: Dict[str, Any]) -> str:
    s = summarize(cases, all_runs)
    by_id = {c["id"]: c for c in cases}
    L: List[str] = [f"# {title}", ""]
    for k, v in meta.items():
        L.append(f"- {k}: {v}")
    L.append(f"- cost: ${spend.usd():.3f} total, ${spend.usd() / max(1, len(all_runs)):.3f} per full run "
             f"(tokens by model: {json.dumps(spend.by_model)})")
    if spend.stop:
        L.append(f"- STOPPED EARLY: {spend.stop}")
    L += ["", "## Summary", "", "| run | pass (all) | pass (deterministic) | stupid messages | "
          + " | ".join(DIMENSIONS) + " |", "|" + "---|" * (4 + len(DIMENSIONS))]
    for i, pr in enumerate(s["per_run"]):
        dims = " | ".join(f"{pr['dims'][d]:.2f}" if pr["dims"][d] is not None else "-" for d in DIMENSIONS)
        L.append(f"| {i + 1} | {pr['passed']}/{pr['n']} | {pr['det_passed']}/{pr['n']} | {pr['stupid']} | {dims} |")
    if len(s["per_run"]) > 1:
        L.append("")
        L.append(f"Variance between runs: {s['flips']} case(s) flipped pass/fail; per-dimension mean difference: "
                 + ", ".join(f"{d} {abs((s['per_run'][0]['dims'][d] or 0) - (s['per_run'][1]['dims'][d] or 0)):.2f}"
                             for d in DIMENSIONS))
    L.append(f"\nPass thresholds per dimension: {json.dumps(PASS_AT)}; any stupid-message flag is a hard fail.")
    # per intent
    L += ["", "## By intent (run 1)", "", "| intent | cases | pass | det pass | mean judge |", "|---|---|---|---|---|"]
    first = all_runs[0] if all_runs else []
    for intent in INTENTS:
        rows = [r for r in first if by_id[r["id"]]["expect"]["intent"] == intent]
        if rows:
            L.append(f"| {intent} | {len(rows)} | {sum(r['verdict']['pass'] for r in rows)} | "
                     f"{sum(r['verdict']['det_pass'] for r in rows)} | "
                     f"{statistics.mean([_mean_score(r) for r in rows]):.2f} |")
    # worst
    scored = sorted(((min(_mean_score(run[i]) for run in all_runs), i) for i in range(len(first))), key=lambda x: x[0])
    L += ["", "## The 10 worst replies", ""]
    for score, i in scored[:10]:
        case = by_id[first[i]["id"]]
        res = min((run[i] for run in all_runs), key=_mean_score)
        L.append(f"### {case['id']} ({case['source']}, {case['expect']['intent']}) - mean judge {score:.2f}")
        L.append(f"- owner: {case['message']['text']}")
        L.append(f"- bot: {(res.get('text') or '(nothing)').strip()}")
        L.append(f"- ideal: {case['expect']['ideal']}")
        j = res.get("judge") or {}
        if j.get("reason"):
            L.append(f"- judge: {j['reason']}")
        fails = [f"{c['name']} ({c['detail']})" for c in res["verdict"]["failed_checks"]]
        if fails:
            L.append(f"- failed checks: {'; '.join(fails)}")
        L.append("")
    # per case
    L += ["## Every case", "", "| case | source | intent | " + " | ".join(f"run {i + 1}" for i in range(len(all_runs)))
          + " | failed checks (run 1) | judge (run 1) |", "|" + "---|" * (5 + len(all_runs))]
    for i, res in enumerate(first):
        case = by_id[res["id"]]
        marks = []
        for run in all_runs:
            v = run[i]["verdict"]
            marks.append(("PASS" if v["pass"] else "fail") + (" STUPID" if v["stupid"] else ""))
        fails = "; ".join(f"{c['name']}" + (f": {c['detail'][:60]}" if c["detail"] else "")
                          for c in res["verdict"]["failed_checks"]) or "-"
        j = res.get("judge") or {}
        sc = " ".join(f"{d[:4]}{v}" for d, v in (j.get("scores") or {}).items()) or j.get("error", "-")
        L.append(f"| {case['id']} | {case['source']} | {case['expect']['intent']} | " + " | ".join(marks)
                 + f" | {fails.replace('|', '/')} | {sc} |")
    L += ["", "## Replies (run 1)", ""]
    for res in first:
        case = by_id[res["id"]]
        L.append(f"- **{case['id']}** owner: {case['message']['text']}")
        L.append(f"  - bot: {(res.get('text') or res.get('error') or '(nothing)').strip()}".replace("\n", " / "))
        if res.get("writes"):
            L.append(f"  - writes: {json.dumps(_writes_for_judge(res['writes']), ensure_ascii=False)[:300]}")
        if res.get("tools"):
            L.append(f"  - tools: {', '.join(res['tools'])}")
    text = "\n".join(L) + "\n"
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump({"runs": all_runs, "spend": spend.by_model, "usd": spend.usd(), "meta": meta}, f,
                      ensure_ascii=False, indent=1, default=str)
    return text


def check_suite(cases: List[Dict[str, Any]]) -> List[str]:
    """Schema problems, duplicate ids, and ideal replies that fail their own deterministic checks."""
    problems: List[str] = []
    seen = set()
    for case in cases:
        problems += validate_case(case)
        if case.get("id") in seen:
            problems.append(f"{case.get('id')}: duplicate id")
        seen.add(case.get("id"))
        if not validate_case(case):
            for c in deterministic(case, ideal_result(case)):
                if not c["ok"]:
                    problems.append(f"{case['id']}: the ideal reply fails {c['name']} ({c['detail']})")
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="eval_chat", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the suite through the brain")
    r.add_argument("--suite", default="tests/golden")
    r.add_argument("--runs", type=int, default=2)
    r.add_argument("--report", default="")
    r.add_argument("--only", default="", help="comma-separated case ids or id prefixes")
    r.add_argument("--no-judge", action="store_true")
    r.add_argument("--judge", default=DEFAULT_JUDGE, help="OpenRouter model id of the judge")
    r.add_argument("--big", default=DEFAULT_BIG)
    r.add_argument("--fast", default=DEFAULT_FAST)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--cap", type=float, default=5.0, help="stop starting new cases above this spend ($)")
    r.add_argument("--key-file", default="")
    r.add_argument("--title", default="Golden conversation suite")
    c = sub.add_parser("check", help="schema + ideal replies, no model")
    c.add_argument("--suite", default="tests/golden")
    args = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if args.cmd == "check":
        cases = load_suite(args.suite)
        problems = check_suite(cases)
        print(f"{len(cases)} cases; {len(problems)} problem(s)")
        for x in problems:
            print("  " + x)
        return 1 if problems else 0
    cases = load_suite(args.suite)
    problems = [x for case in cases for x in validate_case(case)]
    if problems:
        print("\n".join(problems))
        return 2
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        cases = [c for c in cases if any(c["id"] == k or c["id"].startswith(k) for k in keys)]
    spend = Spend(args.cap)
    env = read_key(args.key_file)
    big, fast, judge = make_models(env, spend, args.big, args.fast, args.judge, need_judge=not args.no_judge)
    import logging  # noqa: PLC0415

    logging.basicConfig(level=logging.ERROR)
    t0 = time.time()
    all_runs = run_suite(cases, args.runs, big, fast, judge, spend, workers=args.workers)
    meta = {"date": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "cases": len(cases), "runs": len(all_runs),
            "assistant models": f"{args.big} + {args.fast}", "judge": "none" if args.no_judge else args.judge,
            "wall time": f"{time.time() - t0:.0f}s"}
    try:
        import subprocess  # noqa: PLC0415

        meta["code"] = subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True,
                                      cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    text = write_report(args.report, cases, all_runs, spend, args.title, meta)
    if not args.report:
        print(text)
    s = summarize(cases, all_runs)
    print(json.dumps({"per_run": s["per_run"], "flips": s["flips"], "usd": round(spend.usd(), 3)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
