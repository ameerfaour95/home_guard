# home_guard_project/box/brain/vision.py
"""One look at camera pictures by the vision model, for the assistant's tools.

Used for a live photo (check_camera) and for frames of a saved clip
(describe_event in Assistant mode, assess_event in Guard mode), and to answer
one follow-up question about a saved clip with the frame that shows it
(ask_vision: "what was in his hand?"). The answer
keeps picture quality apart from activity - "clear" means the picture is
usable, not that nothing is happening (version 1 told the owner a blurry view
was "clear"). Guard mode adds the tagging label and a short "why". A refusal
or a failure comes back explicitly; neither ever means "normal".
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import logging
import os
import threading
import time
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional

from ..inference import LABEL_RULES, LABELS, label_of, parse_vlm_json

log = logging.getLogger("box.brain.vision")

QUALITIES = ("clear", "blurry", "dark", "no_signal")
VISION_VERSION = "2026-10-03.v1"


class VisionRefused(Exception):
    """The model declined to look."""


def look_schema(guard: bool) -> Dict[str, Any]:
    props: Dict[str, Any] = {
        "description": {"type": "string"},
        "quality": {"type": "string", "enum": list(QUALITIES)},
        "people": {"type": "integer"},
    }
    if guard:
        props["label"] = {"type": "string", "enum": list(LABELS)}
        props["why"] = {"type": "string"}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def look_prompt(camera: str, guard: bool, question: str = "", what: str = "a live photo") -> str:
    lines = [
        f'You are the eyes of a home security system, looking at {what} from the homeowner\'s own camera "{camera}".',
        "",
        '"description": what is visible and what any people, vehicles or animals are doing, in one to three short',
        "sentences. Describe only what is there; where unsure, say \"appears to\". Never guess names, age or ethnicity.",
        '"quality": how usable the picture is - "clear", "blurry", "dark", or "no_signal" (black, grey, frozen or',
        "garbled). This is about the picture, NOT about whether anything is happening: a sharp, empty yard is \"clear\".",
        '"people": how many people are visible (0 if none).',
    ]
    if guard:
        lines += [
            "",
            'Give the scene ONE "label":',
            LABEL_RULES,
            "Dark clothing alone never makes a scene suspicious; judge what people do.",
            '"why": one short clause naming the behaviour behind a suspicious or escalation label; "" for normal.',
        ]
    if question:
        lines += ["", f'Also answer the homeowner\'s question inside "description": "{question}". '
                      "If the pictures cannot show it, say so."]
    lines += ["", "Reply with exactly one JSON object with these fields and nothing else."]
    return "\n".join(lines)


def ask_schema() -> Dict[str, Any]:
    props = {"answer": {"type": "string"}, "frame": {"type": "integer"}, "seen": {"type": "boolean"}}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def ask_prompt(camera: str, question: str, frames: int, language: str = "English") -> str:
    return "\n".join([
        f"You are the eyes of a home security system, looking at {frames} numbered frames (1 to {frames}, in time "
        f'order) from a saved video of the homeowner\'s own camera "{camera}".',
        f'The homeowner asks: "{question}"',
        "",
        f'"answer": the answer in {language}, in one or two short sentences, only from what the frames show. If the',
        "frames cannot show it (too dark, too far, hidden, out of the picture), say so - never guess. Never guess",
        "names, age or ethnicity.",
        '"frame": the number of the frame that shows the answer best; 0 if none does.',
        '"seen": true only when the frames clearly show the answer.',
        "",
        "Reply with exactly one JSON object with these fields and nothing else.",
    ])


class Vision:
    """*complete(prompt, jpeg_list, schema) -> raw text* does the model call (injected; raises VisionRefused)."""

    def __init__(self, complete: Callable[[str, List[bytes], Dict[str, Any]], str], model_name: str = "") -> None:
        self._complete = complete
        self.model_name = model_name

    def look(self, camera: str, images: List[bytes], guard: bool, question: str = "",
             what: str = "a live photo") -> Dict[str, Any]:
        if not images:
            return {"ok": False, "refused": False, "error": "no_pictures"}
        try:
            raw = self._complete(look_prompt(camera, guard, question, what), list(images), look_schema(guard))
        except VisionRefused:
            return {"ok": False, "refused": True, "error": "refused"}
        except Exception as exc:  # noqa: BLE001 - offline, TLS, a model error
            log.warning("Vision call failed: %s", exc)
            return {"ok": False, "refused": False, "error": "vision_failed"}
        parsed = parse_vlm_json(raw) if isinstance(raw, str) else None
        if (not isinstance(parsed, dict) or not isinstance(parsed.get("description"), str)
                or not parsed["description"].strip()):
            log.warning("Vision returned no usable answer")
            return {"ok": False, "refused": False, "error": "no_answer"}
        quality = str(parsed.get("quality") or "").strip().lower()
        try:
            people = max(0, int(parsed.get("people") or 0))
        except (TypeError, ValueError, OverflowError):
            log.warning("Vision returned an invalid people count; using zero")
            people = 0
        out: Dict[str, Any] = {
            "ok": True,
            "description": str(parsed["description"]).strip(),
            "quality": quality if quality in QUALITIES else "clear",
            "people": people,
        }
        if guard:
            if str(parsed.get("label") or "").strip().lower() not in LABELS:
                log.warning("Guard look returned no valid label")
                return {"ok": False, "refused": False, "error": "no_answer"}
            out["label"] = label_of(parsed)
            out["why"] = str(parsed.get("why") or "").strip() if out["label"] != "normal" else ""
        return out

    def ask(self, camera: str, images: List[bytes], question: str, language: str = "English") -> Dict[str, Any]:
        """One question about numbered frames: ``{"ok", "answer", "frame" (1-based, 0 for none), "seen"}``."""
        if not images:
            return {"ok": False, "refused": False, "error": "no_pictures"}
        try:
            raw = self._complete(ask_prompt(camera, question, len(images), language), list(images), ask_schema())
        except VisionRefused:
            return {"ok": False, "refused": True, "error": "refused"}
        except Exception as exc:  # noqa: BLE001 - offline, TLS, a model error
            log.warning("Vision question failed: %s", exc)
            return {"ok": False, "refused": False, "error": "vision_failed"}
        parsed = parse_vlm_json(raw) if isinstance(raw, str) else None
        if (not isinstance(parsed, dict) or not isinstance(parsed.get("answer"), str)
                or not parsed["answer"].strip()):
            log.warning("Vision returned no usable answer to a question")
            return {"ok": False, "refused": False, "error": "no_answer"}
        try:
            frame = int(parsed.get("frame") or 0)
        except (TypeError, ValueError, OverflowError):
            frame = 0
        return {"ok": True, "answer": parsed["answer"].strip(), "frame": frame if 1 <= frame <= len(images) else 0,
                "seen": parsed.get("seen") is True}


class BudgetedVision:
    """A cap on vision calls per local day, so a chatty family cannot run up the bill. Never raises."""

    def __init__(self, vision: Any, limit_per_day: int, path: str, now: Callable[[], float] = time.time) -> None:
        self._vision, self.path, self._now = vision, path, now
        try:
            self.limit = max(0, int(limit_per_day))
        except (TypeError, ValueError, OverflowError):
            log.warning("Invalid vision budget limit; disabling calls")
            self.limit = 0
        self.model_name = getattr(vision, "model_name", "")
        self._memory: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._warned: set = set()

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning(message)

    def _count(self) -> Dict[str, Any]:
        today = dt.datetime.fromtimestamp(self._now()).strftime("%Y-%m-%d")
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {}
        except (OSError, ValueError, TypeError):
            self._warn_once("read", "Vision budget could not be read; using the in-memory count")
            data = {}
        count = 0
        if isinstance(data, dict) and data.get("day") == today:
            try:
                count = max(0, int(data.get("count", 0)))
            except (TypeError, ValueError, OverflowError):
                self._warn_once("count", "Invalid vision budget count; using zero")
        elif not isinstance(data, dict):
            self._warn_once("shape", "Invalid vision budget file; using zero")
        if self._memory.get("day") == today:
            count = max(count, self._memory["count"])
        return {"day": today, "count": count}

    def look(self, camera: str, images: List[bytes], guard: bool, question: str = "",
             what: str = "a live photo") -> Dict[str, Any]:
        return self._spend(lambda: self._vision.look(camera, images, guard, question, what))

    def ask(self, camera: str, images: List[bytes], question: str, language: str = "English") -> Dict[str, Any]:
        return self._spend(lambda: self._vision.ask(camera, images, question, language))

    def _spend(self, call: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        """One vision call from today's budget."""
        try:
            with self._lock:
                data = self._count()
                if data["count"] >= self.limit:
                    self._warn_once("budget:" + data["day"], "Vision daily budget used up")
                    return {"ok": False, "refused": False, "error": "daily_budget"}
                data["count"] += 1
                self._memory = data
                temp = None
                try:
                    os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                    temp = os.fspath(self.path) + ".tmp"
                    with open(temp, "w", encoding="utf-8") as f:
                        json.dump(data, f, allow_nan=False)
                    os.replace(temp, self.path)
                except (OSError, ValueError, TypeError):
                    self._warn_once("write", "Vision budget not saved; keeping the in-memory count")
                finally:
                    if temp is not None:
                        try:
                            os.remove(temp)
                        except OSError:
                            pass
            return call()
        except Exception as exc:  # noqa: BLE001 - includes bad clocks and injected backends
            log.warning("Budgeted vision failed: %s", exc)
            return {"ok": False, "refused": False, "error": "vision_failed"}


def make_vision(env: Dict[str, str], model: str = "gpt-4o") -> Optional[Vision]:
    """The OpenAI-backed Vision, or None without a key. Uses the OS trust store (TLS interception)."""
    if not isinstance(env, Mapping):  # os.environ is a Mapping, not a dict
        log.warning("Vision environment must be a mapping")
        return None
    key = env.get("OPENAI_API_KEY", "")
    if not isinstance(key, str):
        log.warning("Vision API key must be a string")
        return None
    if not key:
        return None
    http_client = None
    try:
        import ssl  # noqa: PLC0415

        import httpx  # noqa: PLC0415
        from openai import OpenAI  # noqa: PLC0415

        http_client = httpx.Client(verify=ssl.create_default_context())
        client = OpenAI(api_key=key, http_client=http_client, timeout=30.0, max_retries=1)
    except Exception as exc:  # noqa: BLE001 - unavailable client/TLS configuration
        log.warning("Vision client could not be created: %s", exc)
        if http_client is not None:
            try:
                http_client.close()
            except Exception:  # noqa: BLE001 - cleanup must not escape either
                pass
        return None

    def complete(prompt: str, images: List[bytes], schema: Dict[str, Any]) -> str:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for data in images:
            b64 = base64.b64encode(data).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        resp = client.chat.completions.create(
            model=model, temperature=0, messages=[{"role": "user", "content": content}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "camera_look", "strict": True, "schema": schema}},
        )
        msg = resp.choices[0].message
        if getattr(msg, "refusal", None):
            raise VisionRefused(str(msg.refusal))
        return msg.content or ""

    return Vision(complete, model)
