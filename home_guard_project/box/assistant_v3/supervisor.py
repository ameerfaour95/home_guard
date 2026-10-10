"""The pre-send supervisor (report §8.5b): every v3 reply passes it before Telegram.

Stage 1 is code (named checks, about 5 ms, $0): language, internal ids and Markdown, closing offers, generic
questions, empty empathy, repetition of the last five replies, memory recital, times and claims with no source,
the number of questions, length. Stage 2 is a small-model critic on the reply's meaning (does it answer HIS
message? is it about something else? the tone for an angry owner?). On a failure the writer gets ONE rewrite with
the named failure; if that fails too, a Hebrew template built from what was done is sent. Never English, never
empty. Every intervention is logged (``box.assistant_v3.supervisor``) and returned in the turn's trace.
"""
from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..brain.claims import unbacked_claims
from ..brain.style import replace_camera_ids

log = logging.getLogger("box.assistant_v3.supervisor")

ALLOWED_LATIN = {"ai", "p1", "p2", "p3", "car1", "car2", "led", "ok", "gpt", "qwen", "wifi", "lte"}
_IDS = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)*_ch\d+\b|\bE\d{1,3}\b|\bcam\d+x?\b|\bR\d+\b|\[handles?:|receipts?:|"
                  r"tool_call|alert_id|known_id|\bM\d\b|\bA\d\b|\bF\d\b", re.IGNORECASE)
_MD = re.compile(r"\*\*|__|^#+\s|^\s*[-*]\s", re.MULTILINE)
CLOSING = re.compile(r"אם (?:יש|תרצה|צריך|תצטרך) (?:עוד )?(?:משהו|לדעת|מידע)|אני כאן[!.]?\s*$|אשמח לעזור|"
                     r"תודה על ההבהרה|תודה שעדכנת|איך (?:אני )?(?:יכול|אוכל) לעזור|במה עוד")
GENERIC_Q = re.compile(r"מה לתקן|מה תרצה|מה (?:אתה )?רוצה שאעשה|מה תרצה לשנות|תרצה ש|האם תרצה|אם תרצה,? אוכל|"
                       r"רוצה שאשלח|לשלוח לך\?")
JARGON = re.compile(r"התרעה צפויה|כהתרעה (?:אמיתית|שגויה)|כהתרעת שווא|מתייג את זה|תיוג כשגרה")
EMPATHY = re.compile(r"אני מבין את התסכול|אני מבין אותך|מבין את הכעס|אני מצטער לשמוע|מתנצל על אי הנוחות")
ONLY_ACK = re.compile(r"^\s*(?:הבנתי|אני מבין|מבין)(?: אותך)?[.!]?\s*$")
_TIME = re.compile(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)")
_SENT = re.compile(r"[^.!?\n]+[.!?]?")


@dataclass
class Check:
    name: str
    detail: str = ""
    hard: bool = True


@dataclass
class Verdict:
    ok: bool
    failed: List[Check] = field(default_factory=list)

    def reason(self) -> str:
        return "; ".join(f"{c.name}: {c.detail}" if c.detail else c.name for c in self.failed)


def english(text: str) -> List[str]:
    return [w for w in re.findall(r"[A-Za-z][A-Za-z']{2,}", text or "") if w.lower() not in ALLOWED_LATIN]


def questions(text: str) -> int:
    return len([p for p in re.findall(r"[^?.!\n]*\?", text or "") if len(p.split()) >= 2]) or (1 if "?" in (text or "") else 0)


def sentences(text: str) -> int:
    return len([s for s in _SENT.findall(text or "") if len(s.split()) >= 2])


def similar(a: str, b: str) -> float:
    a, b = " ".join(a.split()), " ".join(b.split())
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def code_checks(reply: str, *, lang: str, last_replies: Sequence[str], allowed_times: str, receipts: Sequence[Any],
                memory_subjects: Sequence[str], owner_text: str, may_ask: bool, act_kinds: Sequence[str],
                evidence_text: str = "", long_ok: bool = False, strict_memory: str = "") -> Verdict:
    failed: List[Check] = []
    text = reply.strip()
    if not text:
        return Verdict(False, [Check("empty")])
    if text == "👍":
        return Verdict(True)
    if lang.startswith("he"):
        eng = english(text)
        if len(eng) > 1 or (eng and not re.search(r"[֐-׿]", text)):
            failed.append(Check("english", " ".join(eng[:5])))
    m = _IDS.search(text)
    if m:
        failed.append(Check("internal_id", m.group(0)))
    if CLOSING.search(text):
        failed.append(Check("closing_offer", CLOSING.search(text).group(0)))
    if JARGON.search(text):
        failed.append(Check("jargon", JARGON.search(text).group(0)))
    g = GENERIC_Q.search(text)
    if g:
        failed.append(Check("generic_question", g.group(0)))
    if EMPATHY.search(text) or ONLY_ACK.match(text):
        failed.append(Check("empty_empathy", text[:40]))
    for prev in last_replies[-5:]:
        if similar(text, prev) >= 0.85 or (len(prev) > 25 and prev.strip() in text):
            failed.append(Check("repetition", prev[:60]))
            break
    if "question_memory" not in act_kinds:
        for subject in memory_subjects:
            if recites(text, subject, owner_text, strict_memory):
                failed.append(Check("memory_recital", subject[:40]))
                break
    for m in _TIME.finditer(text):
        hhmm = f"{int(m.group(1)):02d}:{m.group(2)}"
        if hhmm not in allowed_times and m.group(0) not in allowed_times:
            failed.append(Check("invented_time", hhmm))
            break
    try:
        bad = unbacked_claims(text, list(receipts))
    except Exception:  # noqa: BLE001
        bad = []
    if set(act_kinds) & {"question_memory", "question_meta"}:
        bad = [b for b in bad if b not in ("save", "known")]     # talking ABOUT memory is not claiming a save
    if bad:
        failed.append(Check("unbacked_claim", ",".join(bad)))
    q = questions(text)
    if q > 1 or (q == 1 and not may_ask and not _rhetorical(text)):
        failed.append(Check("question", f"{q} question(s)"))
    if not long_ok and (sentences(text) > 3 or len(text.split()) > 55):
        failed.append(Check("too_long", f"{len(text.split())} words", hard=False))
    return Verdict(not [c for c in failed if c.hard], failed)


def _rhetorical(text: str) -> bool:
    """A short echo question inside the answer ("אה, הם של הפרגולה?") is not a question back."""
    parts = [p for p in re.findall(r"[^?.!\n]*\?", text) if p.strip()]
    return all(len(p.split()) <= 4 for p in parts)


_MARKED = re.compile(r"מסומנ|סימנתי|שמור אצלי|שמורים|בזיכרון|זוכר ש|רשום אצלי|כבר יודע|עד יום|כל יום \d")


def _content(text: str) -> List[str]:
    out = []
    for w in re.findall(r"[֐-׿A-Za-z]{3,}", text or ""):
        w = re.sub(r"^[ושהבלמכ](?=[א-ת]{3,})", "", w)
        if w not in ("של", "את", "עם", "זה", "הם", "עד"):
            out.append(w[:5])
    return out


CAMERA_WORDS = {"פרגול", "כניסה", "כניס", "ראשית", "מצלמ"}


def recites(reply: str, subject: str, owner_text: str, strict: str = "") -> bool:
    """The reply brings up a memory record the owner did not raise: two of the record's words in a row (its name,
    "העובדים של הפרגולה") together with memory talk ("מסומנים", "עד יום ה׳", "שמור אצלי"). Seeing workers at work
    and saying so is not a recital; "the workers are marked until 18:00" in a talk about the neighbour is."""
    words = _content(subject)
    if not words:
        return False
    said = set(_content(owner_text))
    have = _content(reply)
    pairs = {(a, b) for a, b in zip(words, words[1:])}
    if strict == "pair" and any((a, b) in pairs for a, b in zip(have, have[1:])) and not (set(words) & said):
        return True                    # its name ("העובדים של הפרגולה") in a turn he did not raise it
    if strict == "irrelevant" and (any((a, b) in pairs for a, b in zip(have, have[1:])) or
                                   (words[0] in have and _MARKED.search(reply))):
        return True                    # he just said it is unrelated: not one more word about it
    if not strict and not _MARKED.search(reply):
        return False
    if strict == "single":
        # A turn that is not a question about what is seen (a preference, a complaint, an ack): naming a memory's
        # people the owner did not name is bringing them up.
        said = set(_content(owner_text))
        key = words[0]
        if key in _content(reply) and key not in said and key not in ("פרגו", "כניס", "מצלמ"):
            return True
    if len(words) < 2 or not _MARKED.search(reply):
        return False
    have = _content(reply)
    pairs = {(a, b) for a, b in zip(words, words[1:])}
    hit = any((a, b) in pairs for a, b in zip(have, have[1:]))
    if not hit:
        return False
    said = _content(owner_text)
    return not any((a, b) in pairs for a, b in zip(said, said[1:])) and not (set(words) & set(said)) - CAMERA_WORDS


def sanitize(text: str, cameras: Sequence[str], lang: str) -> str:
    """The safe, mechanical repairs: camera ids to names, handles out (with a Hebrew prefix: "ב-E1"), Markdown out."""
    out = replace_camera_ids(text, list(cameras), lang) if cameras else text
    out = re.sub(r"\s*\((?:E\d+|cam\d+x?)(?:,\s*(?:E\d+|cam\d+x?))*\)", "", out)
    out = re.sub(r"(?:(?<=\s)|^)[בלמה]?-?(?:E\d{1,3}|cam\d+x?)\b", "", out)
    out = re.sub(r"\b(?:E\d{1,3}|cam\d+x?)\b", "", out)
    out = out.replace("**", "").replace("__", "")
    out = re.sub(r"^#+\s*", "", out, flags=re.MULTILINE)
    out = re.sub(r"\s+([.,!?])", r"\1", out)
    return re.sub(r"[ \t]{2,}", " ", out).strip()


def critic_messages(system: str, owner_text: str, acts: List[Dict[str, Any]], last: Sequence[str], evidence: str,
                    draft: str) -> List[Dict[str, Any]]:
    user = ("OWNER'S MESSAGE: " + owner_text + "\nWHAT IT MEANS (acts): " + str(acts)[:600]
            + "\nASSISTANT'S LAST REPLIES (oldest first):\n" + "\n".join(f"- {x[:200]}" for x in last[-3:])
            + "\nEVIDENCE / DONE THIS TURN:\n" + (evidence[:1500] or "(none)")
            + "\n\nDRAFT REPLY: " + draft)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


CRITIC_SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["pass", "rewrite"]},
                                                  "problem": {"type": "string"}, "fix": {"type": "string"}},
                 "required": ["verdict", "problem", "fix"], "additionalProperties": False}


def log_intervention(trace: List[str], what: str) -> None:
    trace.append(what)
    log.warning("v3 supervisor: %s", what)        # every intervention is in the box log (and the eval's report)


def times_in(*texts: str) -> str:
    """Every HH:MM in the sources, as one string the time check searches."""
    found = []
    for t in texts:
        for m in _TIME.finditer(t or ""):
            found.append(f"{int(m.group(1)):02d}:{m.group(2)}")
    return " ".join(found)


def first(value: Optional[str]) -> str:
    return str(value or "").strip()
