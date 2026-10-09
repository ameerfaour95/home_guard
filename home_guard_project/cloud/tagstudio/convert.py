""""In my words" -> the tag: a text model restructures what the tagger wrote (any language) into the clip's answer
schema, in exactly the field order of the clip's prompt version (fleet_contract/prompt_schemas.py), enforced with a
strict json_schema. Only the words, the taxonomy and the schema are sent: never a picture, never the clip.

The result fills the form as a suggestion; the tagger confirms it. What is saved is the tagger's own words
(``tagger_words``, with ``tagger_language`` and ``words_source``) as the ground truth, the confirmed structured tag,
``converted_by`` (the model setting) and ``converted_model`` (the exact model id that answered).

Privacy (the reviewer, 2026-10-09): words that came from a customer (an owner's answer, a transcript) are sent only
when that customer agreed to training use; the server decides where the words came from (service.convert). Before
any call, camera ids and names, the house / site and the customer's name are replaced (:func:`redact`): only the
cleaned words, the taxonomy and the schema leave the machine.

The model is a setting: ``HG_CONVERT_MODEL`` (default ``google/gemini-3.1-flash-lite`` through OpenRouter, as the
owner asked), ``HG_CONVERT_BASE_URL``, key ``OPENROUTER_API_KEY`` (api_key.env). Gemini's terms restrict using its
output to train competing models; the output here is the human's own words restructured and confirmed by the human
(docs/admin/TAG_AI_CONVERT.md). TLS verifies against the Windows certificate store, as Suggest does.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from ...fleet_contract import prompt_schemas as ps
from ...fleet_contract import taxonomy as tx
from .suggest import _http_client, _key_from_file
from .teacher import parse_raw

DEFAULT_MODEL = "google/gemini-3.1-flash-lite"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
MAX_WORDS = 2000
# answer field -> tag form field (the rest keep their names)
FORM_NAMES = {"summary": "description"}


class ConvertError(RuntimeError):
    """The words could not be converted (no key, no network, an unusable answer); the message says why."""


@dataclass(frozen=True)
class ConvertConfig:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""

    @classmethod
    def resolve(cls, env: Optional[Mapping[str, str]] = None, key_files: Sequence[Path] = ()) -> "ConvertConfig":
        env = os.environ if env is None else env
        key = str(env.get("HG_CONVERT_API_KEY") or env.get("OPENROUTER_API_KEY") or "").strip()
        for path in key_files if not key else ():
            key = _key_from_file(Path(path), "OPENROUTER_API_KEY")
            if key:
                break
        return cls(model=str(env.get("HG_CONVERT_MODEL") or DEFAULT_MODEL).strip(),
                   base_url=str(env.get("HG_CONVERT_BASE_URL") or DEFAULT_BASE_URL).strip(), api_key=key)


def language(text: str) -> str:
    """The words' language by script: "he" (Hebrew letters), "ar" (Arabic), "ru" (Cyrillic), else "en"."""
    for code, pattern in (("he", r"[֐-׿]"), ("ar", r"[؀-ۿ]"), ("ru", r"[Ѐ-ӿ]")):
        if re.search(pattern, text or ""):
            return code
    return "en"


# Hebrew house words and what they are in English (the same idea as box/messenger.py's GLOSSARY, the other way):
# the converter writes the scene in English, and these words have one right translation in a yard.
GLOSSARY = {
    "משטחים": "pallets",
    "טנדר": "pickup truck",
    "פרגולה": "pergola",
    "מעקה": "railing",
    "שער": "gate",
    "חניה": "parking",
    "מחסן": "shed",
    "גדר": "fence",
    "עובדים / פועלים": "workers",
}

_CAMERA_ID = re.compile(r"\b[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*_ch\d+\b|\b(?:production|dataset)_[A-Za-z0-9_]+\b")


def redact(words: str, cameras: Sequence[str] = (), places: Sequence[str] = ()) -> str:
    """*words* with every camera id or name replaced by "camera" and every house, site or customer name by "the house"
    (longest first, any case), and anything shaped like a camera id (``<site>_ch6``) or a storage prefix as well."""
    out = str(words or "")
    terms = [(t, "camera") for t in cameras] + [(t, "the house") for t in places]
    for term, repl in sorted({(str(t).strip(), r) for t, r in terms if t and len(str(t).strip()) >= 2},
                             key=lambda tr: -len(tr[0])):
        out = re.sub(re.escape(term), repl, out, flags=re.IGNORECASE)
    return _CAMERA_ID.sub("camera", out)


def prompt(words: str, prompt_version: Optional[str]) -> str:
    kind = ps.schema_kind(prompt_version)
    fields = ", ".join(ps.field_order(prompt_version))
    rules = ['- "summary": the scene in English, one to three short sentences, who and what in the order it happens.',
             '- "raw_label": how serious the scene itself is: normal, suspicious or escalation.',
             '- "label": the same as raw_label unless the words say a house note changes it.',
             '- Numbers you are not told: 0. Text you are not told: an empty string. Yes/no you are not told: false.',
             '- Never add anything the words do not say.']
    if kind == ps.EYE:
        rules.insert(1, '- "category": the one id below that fits; "other" with a few words in "other_text" when '
                        'none fits.\n' + tx.prompt_list(order=("E", "S", "N")))
    else:
        rules += ['- "why": the reason for the label in the tagger\'s words, in English, short.',
                  '- "summary_owner": the summary in the language the tagger wrote in (empty when that is English).',
                  '- "serious_behaviour": true only for a hidden or covered face, trying doors, gates or car doors, '
                  'or looking into windows or cars.']
    glossary = "\n".join(f"  {he} = {en}" for he, en in GLOSSARY.items())
    return (f"A person who watched a home security clip described it in their own words (any language). Restructure "
            f"those words into the JSON answer below, with the fields in this order: {fields}.\n"
            + "\n".join(rules) + f"\n- Use these English words for Hebrew house words:\n{glossary}"
            + f"\n\nTheir words:\n\"\"\"\n{words.strip()[:MAX_WORDS]}\n\"\"\"\n\n"
            "Reply with exactly one JSON object and nothing else.")


def to_form(parsed: Dict[str, Any], prompt_version: Optional[str]) -> Dict[str, Any]:
    """The answer as tag form fields, in the schema's order; values outside the schema's choices are dropped."""
    _, schema = ps.answer_schema(prompt_version)
    out: Dict[str, Any] = {}
    for name, spec in schema["properties"].items():
        value = parsed.get(name)
        kind = spec.get("type")
        if "enum" in spec:
            value = value if value in spec["enum"] else ""
        elif kind == "integer":
            value = int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 else 0
        elif kind == "boolean":
            value = bool(value) if isinstance(value, bool) else False
        elif kind == "array":
            allowed = (spec.get("items") or {}).get("enum")
            value = [v for v in (value or []) if isinstance(v, str) and (allowed is None or v in allowed)]
        else:
            value = str(value or "").strip()
        out[FORM_NAMES.get(name, name)] = value
    if out.get("category") and out["category"] != tx.OTHER:
        out["other_text"] = ""
    if "evidence_frame" in out:
        out["evidence_frame"] = out["evidence_frame"] or None   # words never point at a frame
    return out


class Converter:
    def __init__(self, config: ConvertConfig, client: Any = None):
        self.config, self._client = config, client

    def convert(self, words: str, prompt_version: Optional[str], cameras: Sequence[str] = (),
                places: Sequence[str] = ()) -> Dict[str, Any]:
        """``{model, model_id, language, prompt_version, schema, fields, words_sent}`` for *words*, sent with the
        camera names in *cameras* and the house / customer names in *places* replaced (:func:`redact`)."""
        words = str(words or "").strip()
        if not words:
            raise ConvertError("Write what you saw first")
        sent = redact(words, cameras, places)
        if not self.config.api_key and self._client is None:
            raise ConvertError("No API key for the converter: set OPENROUTER_API_KEY (api_key.env or the Admin "
                               "service's environment)")
        name, schema = ps.answer_schema(prompt_version)
        try:
            if self._client is None:
                from openai import OpenAI  # noqa: PLC0415

                self._client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url, timeout=60,
                                      max_retries=1, http_client=_http_client())
            response = self._client.chat.completions.create(
                model=self.config.model, messages=[{"role": "user", "content": prompt(sent, prompt_version)}],
                temperature=0,
                response_format={"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}})
            raw = response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - network, auth, quota: say what happened
            raise ConvertError(f"The converter could not be reached ({type(exc).__name__}: {str(exc)[:200]})") from None
        parsed = parse_raw(raw)
        if not parsed:
            raise ConvertError("The converter's answer was not JSON")
        model_id = getattr(response, "model", None)
        return {"model": self.config.model, "model_id": model_id if isinstance(model_id, str) and model_id
                else self.config.model, "language": language(words), "prompt_version": prompt_version or "",
                "schema": name, "fields": to_form(parsed, prompt_version), "raw": json.dumps(parsed, ensure_ascii=False),
                "words_sent": sent}
