""""Suggest tag": a teacher model fills in the whole tag, in the training format we export.

The question and the answer format follow the box's Eye (``box/eye_prompt.py``, intent ``alert_triage``): the same
description rules ("who, what, in order"; "No special activity." when nothing happens), the same taxonomy, and its
strict JSON schema, reduced to the fields a tag holds. The scene is judged WITHOUT the situation (no hour, no house
state): a tag is context-free, the box adds the situation itself. Keep this in step with eye_prompt.py.

The model is OpenAI-compatible (default: OpenRouter ``qwen/qwen3-vl-32b-instruct``; ``HG_SUGGEST_MODEL`` /
``HG_SUGGEST_BASE_URL`` change it). Gemini is refused: its terms forbid training a competing model on its outputs.
One answer per clip and model is cached (``suggestions.jsonl`` in the exports folder), so a clip is paid for once.
"""
from __future__ import annotations

import base64
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ...fleet_contract import taxonomy as tx
from . import evalfmt
from .teacher import FORBIDDEN_TEACHERS, parse_raw

PROMPT_VERSION = "2026-10-06.studio-suggest-v1 (eye-v3 alert_triage, context-free)"
DEFAULT_MODEL = "qwen/qwen3-vl-32b-instruct"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
CACHE_FILE = "suggestions.jsonl"
FRAMES = 5   # what the box sends the Eye


def _obj(props: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _enum(values: Sequence[str]) -> Dict[str, Any]:
    return {"type": "string", "enum": list(values)}


_STR, _INT, _BOOL = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}

# eye_prompt's alert_triage schema, minus what depends on the situation (label, house notes, why).
SCHEMA = _obj({
    "summary": _STR, "category": _enum(tx.CATEGORY_IDS), "other_text": _STR, "zone": _enum(tx.ZONES),
    "movement": _enum(tx.MOVEMENTS), "flags": {"type": "array", "items": _enum(tx.FLAGS)},
    "people": _INT, "vehicles": _INT, "vehicle_moving": _BOOL, "animals": _INT,
    "visibility": _enum(tx.VISIBILITY), "appearance": {"type": "array", "items": _STR},
    "evidence_frame": _INT, "raw_label": _enum(tx.LABELS),
})
RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "studio_tag", "strict": True, "schema": SCHEMA}}


def prompt(camera: str = "") -> str:
    where = f' "{camera}"' if camera else ""
    return f"""
You are tagging a short clip from a home security camera{where} for a training set. These are sequential frames
(frame 1 first). Judge only what you see: there is no time of day and no house situation here.

How to describe:
- Say who is there and what they do, in the order it happens. Where something is uncertain, say "appears to"
  or "seems to".
- Say "a man", "a woman", "a person", "two men", "a group of people"; never guess names, age, ethnicity or
  who the person is.
- Describe only what is there and what happens. Do not mention what is absent or the background (parked
  cars, walls, plants) unless someone acts on it.
- Clothing alone is never a reason: dark clothes, a hood, a cap or a courier's helmet mean nothing by
  themselves. Judge what people do. Hiding the face on purpose while coming toward a door, window, gate or
  car IS something they do: S5.
- Write in English only.

What a scene can be (one category id). The serious ones come first:
{tx.prompt_list(order=("E", "S", "N"))}

Look first, judge last. Fill the fields in this order:
- "summary": what happens, in one to three short sentences (usually 10 to 25 words). If nobody is there and
  nothing moves (parked cars, plants, light changes), write exactly "No special activity."
- "category": the one id above that fits what you SEE. "other" when none fits, with a few words in
  "other_text"; otherwise "other_text" is an empty string.
  Before you choose a Normal category, check every frame for: a hand on a door handle, window, gate latch or
  car door; reaching into or over something; climbing; crouching or hiding; a face hidden on purpose while
  approaching; picking something up and leaving with it; looking into windows or cars; running away. If any of
  these happens, the category is S or E, not N. N1 is only someone who passes without stopping at the
  property; N7 is only someone using their own car the normal way.
- "zone": where it happens: {' | '.join(tx.ZONES)}.
- "movement": {' | '.join(tx.MOVEMENTS)}.
- "flags": each one you clearly see: {', '.join(tx.FLAGS)}; [] when none.
- "people", "vehicles" (parked ones too), "vehicle_moving", "animals" (not birds): what is visible.
- "visibility": "clear", or "partial" when darkness, distance or cover hides what the person does.
- "appearance": up to 4 short phrases that would recognise the same person or vehicle again: clothing colour
  and type, what they carry, a vehicle's colour and type. Never the face, hair, body, age or sex. [] when
  nobody is there. Appearance never decides the category.
- "evidence_frame": the frame number (1 = first) that shows the category best; 0 when nothing happens.
- "raw_label": "normal" for an N category, "suspicious" for S, "escalation" for E; for "other", your own judgement.

Reply with EXACTLY ONE strict JSON object and nothing else.
""".strip()


LEGACY_PROMPT_VERSION = "2026-10-09.studio-suggest-legacy-v1 (box legacy answer, context-free)"


def legacy_prompt(camera: str = "") -> str:
    """The question for a clip answered with the box's legacy prompt: its answer fields (fleet_contract
    prompt_schemas.VLM_SCHEMA), judged without the house notes or the hour, like the Eye variant above."""
    head = prompt(camera).split("What a scene can be", 1)[0].strip()
    return f"""
{head}

Fill the fields in this order:
- "summary": what happens, in one to three short sentences (usually 10 to 25 words). If nobody is there and
  nothing moves, write exactly "No special activity."
- "label" and "raw_label" (the same: there are no house notes here): "normal" for ordinary activity, "suspicious"
  when it is worth a look (trying a handle, looking into windows or cars, a face hidden on purpose while
  approaching, hiding, climbing), "escalation" for danger or a crime (forced entry, theft, a weapon, violence).
- "applied_fact_id": an empty string.
- "serious_behaviour": true if anyone shows a hidden or covered face, tries doors, gates or car doors, or looks
  into windows or cars; otherwise false.
- "people", "vehicle_moving", "animals" (not birds): what is visible.
- "why": the reason for the label, in English, one short sentence.
- "summary_owner": an empty string.

Reply with EXACTLY ONE strict JSON object and nothing else.
""".strip()


class SuggestError(RuntimeError):
    """The suggestion could not be made (no key, no network, refused model, unusable answer); the message says why."""


@dataclass(frozen=True)
class SuggestConfig:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""

    @classmethod
    def resolve(cls, env: Optional[Mapping[str, str]] = None, key_files: Sequence[Path] = ()) -> "SuggestConfig":
        env = os.environ if env is None else env
        key = str(env.get("HG_SUGGEST_API_KEY") or env.get("OPENROUTER_API_KEY") or "").strip()
        if not key:
            for path in key_files:
                key = _key_from_file(Path(path), "OPENROUTER_API_KEY")
                if key:
                    break
        return cls(model=str(env.get("HG_SUGGEST_MODEL") or DEFAULT_MODEL).strip(),
                   base_url=str(env.get("HG_SUGGEST_BASE_URL") or DEFAULT_BASE_URL).strip(), api_key=key)


def _key_from_file(path: Path, name: str) -> str:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            k, sep, v = line.strip().partition("=")
            if sep and k.strip() == name:
                return v.strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


def sample(video: str, k: int = FRAMES) -> Tuple[List[Any], List[int], float]:
    """``(frames, their indices in the clip, fps)``; frames evenly spaced, long side at most 1280."""
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(video)
    frames, fps = [], float(cap.get(cv2.CAP_PROP_FPS) or 0) or 7.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    idx = sorted(set(evalfmt.sample_indices(len(frames), k)))
    out = []
    for i in idx:
        frame = frames[i]
        h, w = frame.shape[:2]
        scale = min(1.0, evalfmt.MAX_SIDE / max(h, w))
        out.append(cv2.resize(frame, (round(w * scale), round(h * scale))) if scale < 1 else frame)
    return out, idx, fps


def to_fields(parsed: Dict[str, Any], frame_index: Sequence[int], fps: float) -> Dict[str, Any]:
    """The answer as tag fields (what the form and the export hold). Anything outside the taxonomy is dropped."""
    category = str(parsed.get("category") or "").strip()
    known = tx.get(category)
    category = known.id if known else tx.OTHER if category.lower() == tx.OTHER else ""
    raw = str(parsed.get("raw_label") or "").strip().lower()
    fields: Dict[str, Any] = {
        "category": category if category in tx.CATEGORY_IDS else "",
        "other_text": str(parsed.get("other_text") or "")[:500] if category == tx.OTHER else "",
        "raw_label": raw if raw in tx.LABELS else (tx.BY_ID[category].label if category in tx.BY_ID else ""),
        "zone": parsed.get("zone") if parsed.get("zone") in tx.ZONES else "",
        "movement": parsed.get("movement") if parsed.get("movement") in tx.MOVEMENTS else "",
        "flags": [f for f in tx.FLAGS if f in (parsed.get("flags") or [])],
        "visibility": parsed.get("visibility") if parsed.get("visibility") in tx.VISIBILITY else "",
        "appearance": [str(a).strip()[:60] for a in (parsed.get("appearance") or []) if str(a).strip()][:4],
        "description": str(parsed.get("summary") or "").strip()[:2000],
        "evidence_frame": None, "evidence_sec": None,
    }
    try:
        n = int(parsed.get("evidence_frame") or 0)
    except (TypeError, ValueError):
        n = 0
    if 1 <= n <= len(frame_index):
        frame = frame_index[n - 1]
        fields["evidence_frame"], fields["evidence_sec"] = frame, round(frame / (fps or 7.0), 3)
    return fields


def _http_client():
    """Verify TLS against the Windows certificate store when truststore is installed: an antivirus that inspects
    HTTPS signs with its own root, which certifi's bundle does not know (CERTIFICATE_VERIFY_FAILED)."""
    import ssl  # noqa: PLC0415

    import httpx  # noqa: PLC0415

    try:
        import truststore  # noqa: PLC0415

        return httpx.Client(verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT), timeout=90)
    except ImportError:
        return None


class Suggester:
    def __init__(self, config: SuggestConfig, cache_path: str, client: Any = None,
                 frames_for: Optional[Callable[[str], Tuple[List[Any], List[int], float]]] = None):
        if any(word in f"{config.base_url} {config.model}".lower() for word in FORBIDDEN_TEACHERS):
            raise SuggestError("Gemini cannot suggest tags: its terms forbid training a competing model on its output")
        self.config, self.cache_path = config, str(cache_path)
        self._client, self._frames_for = client, frames_for or sample
        self._lock = threading.Lock()

    def cached(self, key: str) -> Optional[Dict[str, Any]]:
        rows, _ = evalfmt.read_jsonl(self.cache_path)
        found = None
        for row in rows:
            if row.get("key") == key and row.get("model") == self.config.model and isinstance(row.get("fields"), dict):
                found = row
        return found

    def suggest(self, key: str, video: Optional[str], camera: str = "", refresh: bool = False,
                jpegs: Sequence[bytes] = (), frame_index: Sequence[int] = (), fps: Optional[float] = None,
                prompt_version: Optional[str] = None) -> Dict[str, Any]:
        """The suggestion for clip *key* (cached unless *refresh*): ``{key, model, at, fields, cached}``.

        *jpegs*: the model-input frames (model_view.py: what the box sent the AI), with each one's clip frame index
        (*frame_index*) and the clip's *fps*; without them, frames are sampled from *video*. The answer follows the
        schema of the clip's *prompt_version*: the box's legacy answer, else the Eye's category form."""
        from ...fleet_contract import prompt_schemas as ps  # noqa: PLC0415

        if not refresh:
            row = self.cached(key)
            if row is not None:
                return {**row, "cached": True}
        if not self.config.api_key and self._client is None:
            raise SuggestError("No API key for the suggestion model: set OPENROUTER_API_KEY (api_key.env or the "
                               "Admin service's environment)")
        legacy = ps.schema_kind(prompt_version) == ps.LEGACY
        if jpegs:
            data, index, fps = [bytes(j) for j in jpegs], list(frame_index), fps or 7.0
        else:
            frames, index, fps = self._frames_for(video) if video else ([], [], 7.0)
            import cv2  # noqa: PLC0415

            data = []
            for frame in frames:
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok:
                    data.append(buf.tobytes())
        if not data:
            raise SuggestError("The clip's video could not be read")
        content: List[Dict[str, Any]] = [{"type": "text", "text": legacy_prompt(camera) if legacy else prompt(camera)}]
        for jpeg in data:
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")}})
        try:
            if self._client is None:
                from openai import OpenAI  # noqa: PLC0415

                self._client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url, timeout=90,
                                      max_retries=1, http_client=_http_client())
            extra = {"reasoning": {"enabled": False}} if "openrouter.ai" in self.config.base_url else None
            response_format = ({"type": "json_schema", "json_schema": {"name": "legacy_alert", "strict": True,
                                                                       "schema": ps.VLM_SCHEMA}}
                               if legacy else RESPONSE_FORMAT)
            response = self._client.chat.completions.create(
                model=self.config.model, messages=[{"role": "user", "content": content}], temperature=0,
                response_format=response_format, extra_body=extra)
            raw = response.choices[0].message.content or ""
        except SuggestError:
            raise
        except Exception as exc:  # noqa: BLE001 - network, auth, quota: say what happened
            raise SuggestError(f"The suggestion model could not be reached ({type(exc).__name__}: "
                               f"{str(exc)[:200]})") from None
        parsed = parse_raw(raw)
        if not parsed:
            raise SuggestError("The suggestion model's answer was not JSON")
        if legacy:
            from .convert import to_form  # noqa: PLC0415

            fields = to_form(parsed, prompt_version)
        else:
            fields = to_fields(parsed, index, fps)
        row = {"key": key, "model": self.config.model,
               "prompt_version": LEGACY_PROMPT_VERSION if legacy else PROMPT_VERSION,
               "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "raw": raw, "fields": fields,
               "frames": len(data), "from_model_input": bool(jpegs)}
        with self._lock:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {**row, "cached": False}
