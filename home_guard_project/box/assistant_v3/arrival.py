"""When did the vehicle standing at a camera come? Read from the house recorder's recording (box/recorder.py).

2026-10-12 13:37-13:39, about the pickup in his live picture of the pergola: "תן לי את הוידיו מתי הבנאדם יצא ממנו",
"אני צריך מתי הבנאדם יצא מהטנדר", "הטנדר לא היה פה בבוקר, הוא הגיע מתישהו, אני צריך בדיוק את הזמן שהוא הגיע". He got
10 live seconds, then "I'll send a photo when the man gets out" and "I'll send a photo from camera 2 when the truck
arrives": two watches nobody asked for, and no answer. The pergola alerts only on people, so the box kept nothing of
the truck; the recorder had it coming in at 12:56.

A question with "when" about a vehicle (in his words, his last message, or the box's last reply: "ממנו" after "נראה
טנדר"), and no "let me know" in it, is answered here, before the acts run: the vehicles at the camera (named, else the
one of his last picture) are looked up in the recording, the time each came is the evidence, and the recording of the
newest arrival is sent. The watch and the live video the understanding read into it are dropped.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from ...prompts import load
from ..brain.receipts import DONE
from ..brain.tools import TOOLS
from .skills import internal
from .timeparse import clock

log = logging.getLogger("box.assistant_v3.arrival")

_WHEN = re.compile(r"(?<![א-ת])(?:מתי|ממתי|באיזו שעה|באיזה שעה|כמה זמן|הזמן ש|השעה ש)|\b(?:when|since when|what time)\b",
                   re.IGNORECASE)
# "tell me when it leaves" is a watch; "tell me when it ARRIVED" is not
_NOTIFY = re.compile(r"(?<![א-ת])(?:תודיע|תעדכן|תגיד לי|תכתוב לי|דיווח|ברגע ש|יגיע|תגיע|יגיעו|ייכנס|יחזור|תחזור)"
                     r"|\b(?:let me know|notify|tell me when|alert me)\b", re.IGNORECASE)
_PAST = re.compile(r"(?<![א-ת])ש?(?:הגיע|הגיעה|הגיעו|נכנס|נכנסה|חנה|חנתה|היה|הייתה|היתה)(?![א-ת])"
                   r"|\b(?:arrived|came|got here)\b", re.IGNORECASE)
CLIP_BEFORE_SEC = 5.0
CLIP_SEC = 20.0


def asks_arrival(text: str, vehicle: str) -> bool:
    """A "when did it come" question about *vehicle* ("" when no vehicle is in the talk)."""
    if not vehicle or not _WHEN.search(text or ""):
        return False
    return bool(_PAST.search(text)) or not _NOTIFY.search(text)


def answer(t: Any) -> bool:
    """Answer the turn's "when did the vehicle come" from the recording. True when it did (acts adjusted)."""
    from . import handlers as hd  # noqa: PLC0415
    from .context import last_replies  # noqa: PLC0415

    replies = last_replies(t.state, 1)
    owner = t.mem_owner_words()[:1]
    vehicle = hd.vehicle_words(t.text, *owner, *(replies[-1:] if replies else []))
    if not asks_arrival(t.text, vehicle):
        return False
    recorder = getattr(t.ctx.services, "recorder", None)
    detect = getattr(t.ctx.services, "detect", None)
    camera = hd._named_camera(t, t.text) or t.last_photo_camera(30 * 60) or hd._named_camera(t, *t.mem_owner_words()[:2]) \
        or hd._named_camera(t, replies[-1] if replies else "") or next(
            (a.camera for a in t.und.acts if a.camera and a.camera != "house"), "")
    if not camera:
        return False
    # a past question: no watch, no 10 live seconds
    dropped = [a.act if a.act != "command" else a.command for a in t.und.acts
               if a.act == "watch_request" or (a.act == "command" and a.command in ("send_video", "send_photo"))]
    t.und.acts = [a for a in t.und.acts if not (a.act == "watch_request" or
                                               (a.act == "command" and a.command in ("send_video", "send_photo")))]
    t.plan.trace.append(f"arrival question about {vehicle} at {camera}" + (f"; dropped {dropped}" if dropped else ""))
    if not t.und.has("question_history"):
        from .acts import Act  # noqa: PLC0415

        t.und.acts.append(Act(act="question_history", quote=t.text[:120], camera=camera))   # it is answered
    name = t.name(camera)
    if recorder is None or detect is None or not recorder.available(camera):
        t.did(f"אין לי גישה להקלטה של המקליט של {name}, אז לא בדקתי מתי {vehicle} הגיע.")
        return True
    found = recorder.arrival(camera, detect, t.now)
    log.info("v3 arrival %s at %s: %s", vehicle, camera, found)
    if not found.get("ok"):
        t.did(f"לא הצלחתי לקרוא את ההקלטה של המקליט של {name} כרגע, אז לא בדקתי מתי {vehicle} הגיע.")
        return True
    rows: List[Dict[str, Any]] = list(found.get("vehicles") or [])
    if not rows:
        t.plan.evidence.append(f"בהקלטה של המקליט אין רכב ב{name} ב-{clock(found.get('latest') or t.now)} "
                               f"(ההקלטה זמינה רק כדקה וחצי אחרי הזמן).")
        return True
    lone = len(rows) == 1
    for row in sorted(rows, key=lambda r: -float(r.get("arrived") or 0)):
        what = vehicle if lone else "רכב"
        if row.get("arrived"):
            t.plan.evidence.append(f"מההקלטה של המקליט (לא מהיומן של הקופסה): {what} הגיע ל{name} ב-{clock(row['arrived'])} "
                                   f"(עוד לא היה שם ב-{clock(row['before'])}; נבדק בגלאי של הקופסה, תמונה אחרי תמונה).")
        else:
            t.plan.evidence.append(f"מההקלטה של המקליט: {what} עומד ב{name} כבר מלפני {clock(row['since_before'])} "
                                   f"(בדקתי עד שם אחורה).")
    newest = max((r for r in rows if r.get("arrived")), key=lambda r: float(r["arrived"]), default=None)
    if newest is not None:
        _send_clip(t, recorder, camera, float(newest["before"]) - CLIP_BEFORE_SEC, name)
    t.plan.notes.append(load("assistant_v3_note_vehicle_arrival.prompt"))
    return True


def _send_clip(t: Any, recorder: Any, camera: str, start: float, name: str) -> Optional[str]:
    rec = recorder.clip(camera, start, CLIP_SEC)
    if not rec.get("ok"):
        t.plan.trace.append(f"recorder clip failed: {rec}")
        return None
    handle = t.state.add_handle("clip", rec["path"], camera, rec["start"])
    entry = t.state.resolve(handle)
    entry["bounds"] = f"מהמקליט {clock(rec['start'])}–{clock(rec['end'])}"
    t.called.append("send_media")
    with internal(t.ctx):
        out = TOOLS["send_media"](t.ctx, {"handle": handle})
    if out.get("ok") and out.get("status", DONE) == DONE:
        t.did(f"נשלח סרטון מההקלטה של המקליט (לא חי) מ{name}, {clock(rec['start'])}–{clock(rec['end'])}, "
              f"הרגע שהרכב הגיע.")
        return handle
    return None
