"""The event layer's real outcome per clip (box/events.py + box/inference.py's alert record), never lumped as
'No alert'. Metas are shaped like what the box writes (beelink-collector-box 79e9785)."""
from home_guard_project.fleet_contract.event_outcome import decision_of, outcome, session_id, would_raise


def _meta(**alert):
    return {"camera_name": "ameer_week_0_1_ch3", "alert": {"summary": "two workers on the pergola",
                                                           "alert_command": "[send_message]", **alert}}


def _event(notify, reason, **kw):
    return {"notify": notify, "session_id": "a1b2c3d4", "reason": reason, "reply_to": None, "new_people": 0,
            "known_text": "", "entities": [], "fresh": [], "unmarked": False, "counted_by": "head-count", **kw}


def test_sent_and_held_and_known():
    sent = decision_of(_meta(sent=True, event=_event(True, "suspicious, first in this event"),
                             dispatch={"sent": True}))
    assert outcome(sent) == ("sent", "Sent") and session_id(sent) == "a1b2c3d4"
    held = decision_of(_meta(sent=False, event=_event(False, "normal: kept in the event, not sent"),
                             not_sent_reason="normal: kept in the event, not sent",
                             dispatch={"sent": False, "reason": "normal: kept in the event, not sent"}))
    assert outcome(held) == ("held", "Kept in the event, not sent (normal)")
    known = decision_of(_meta(sent=False, event=_event(False, "suspicious, but the owner said who is here",
                                                       known_text="עובדים בפרגולה"),
                              not_sent_reason="suspicious, but the owner said who is here"))
    assert outcome(known) == ("known", "Not sent: owner said known (עובדים בפרגולה)")
    assert outcome(known, private=False) == ("known", "Not sent: owner said known")
    again = decision_of(_meta(sent=False, event=_event(False, "suspicious already reported in this event, nobody new")))
    assert outcome(again) == ("held", "Kept in the event, not sent (already reported, nobody new)")
    theirs = decision_of(_meta(sent=False, event=_event(False, "normal on the neighbour ground, nothing done there: not ours")))
    assert outcome(theirs) == ("not_ours", "Not sent: nothing done on our ground")


def test_lowered_second_look_and_baseline_shadow():
    lowered = decision_of(_meta(sent=False, downgraded="appearance only", label="normal",
                                event=_event(False, "normal: kept in the event, not sent")))
    assert outcome(lowered)[1] == "Lowered: appearance only  ·  Kept in the event, not sent (normal)"
    look = decision_of(_meta(sent=True, event=_event(True, "suspicious, first in this event"),
                             second_look={"answered": True, "confirmed": False, "class": "weapon",
                                          "what_it_is": "a garden hose", "evidence_frame": 3, "verified": False}))
    assert outcome(look) == ("sent", "Second look: not a weapon (a garden hose)  ·  Sent")
    shadow = decision_of(_meta(sent=False, event=_event(False, "normal: kept in the event, not sent"),
                               baseline={"mode": "shadow", "rarity": "rare", "would_raise": True, "raise": False,
                                         "text_en": "Not usual for this camera at this hour"}))
    assert would_raise(shadow) and outcome(shadow)[1].endswith("Would raise: rare for this camera")
    raised = decision_of(_meta(sent=True, event=_event(True, "normal, but rare here: a quiet message, once in this event (x)"),
                               baseline={"mode": "on", "would_raise": True, "raise": True}, raised="rare here"))
    assert not would_raise(raised) and outcome(raised) == ("sent", "Raised: rare here  ·  Sent: rare for this camera")
    investigated = decision_of(_meta(sent=False, investigation={"verdict": "short visit", "lowered": True},
                                     event=_event(False, "normal: kept in the event, not sent")))
    assert outcome(investigated)[1].startswith("Lowered: short visit (investigator)")


def test_old_metas_and_the_top_level_copies():
    assert decision_of({"alert": {"alert_command": "[none]"}}) == {
        "v": 1, "sent": None, "not_sent_reason": "", "muted": False, "false_positive": False, "command": "[none]"}
    assert outcome(decision_of({"alert": {"alert_command": "[none]"}})) == ("", "")
    assert outcome(decision_of(_meta(dispatch={"sent": True}))) == ("sent", "Sent")
    assert outcome(decision_of(_meta(dispatch={"sent": False, "reason": "Forbidden"}))) == ("undelivered", "Not delivered (Forbidden)")
    assert outcome(decision_of(_meta(false_positive=True))) == ("dismissed", "Dismissed: the AI saw nothing to alert on")
    assert outcome(decision_of(_meta(muted=True, sent=False))) == ("muted", "Not sent: camera muted by the owner")
    top = decision_of({"baseline": {"mode": "shadow", "would_raise": True}, "alert": {}})
    assert would_raise(top)
    assert decision_of(None) == {"v": 1, "sent": None, "not_sent_reason": "", "muted": False, "false_positive": False,
                                 "command": ""}
