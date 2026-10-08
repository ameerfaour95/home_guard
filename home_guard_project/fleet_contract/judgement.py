"""Is an owner's message a judgement of an alert at all? The box's own rule, vendored verbatim.

The block between the markers is a verbatim copy of the section "Is a message a judgement of an alert at all?" of
home_guard_project/box/feedback.py (branch beelink-collector-box): a question, a complaint or insult, or a command is
never a verdict. The Admin Center's Inbox uses the same rule to pre-flag an answer as "probably not a label", so the
box and the admin agree. tests/fleet_contract/test_judgement.py fails when the box's copy changes: copy it again.
"""
from __future__ import annotations

import re

# >>> vendored from box/feedback.py
# Is a message a judgement of an alert at all?
# ----------------------------------------------------------------------------
# The chat of 2026-10-07: "אני לא יודע על איזה סרטון אתה מדבר בכלל", "איזה תזכורת, אתה מטומטם ומגזים" and
# "די עם ההודעה המטופשת הזאת" were each filed as a verdict on the newest alert (or as the explanation of a
# clip). A question, a complaint or a command is never a judgement: it never reaches a verdict path by itself.
_QUESTION_WORDS = re.compile(
    r"\?|(?<!\w)ו?(?:איזה|איזו|אילו|מה|למה|מדוע|איפה|היכן|מתי|מי|כמה|האם|איך|כיצד)(?!\w)|"
    r"\b(?:which|what|why|where|when|who|how)\b", re.IGNORECASE)
_INSULT_WORDS = re.compile(
    r"(?<!\w)[ושה]{0,2}(?:מטומט\w*|טיפש\w*|מטופש\w*|אידיוט\w*|דביל\w*|פאקינג|חרא\w*)(?!\w)|(?<!\w)יא(?!\w)|"
    r"\b(?:stupid|idiot\w*|dumb|wtf|useless|ridiculous)\b", re.IGNORECASE)
_COMPLAINT_WORDS = re.compile(
    r"(?<!\w)[ושה]{0,2}(?:מעצבנ?\w*|מגזימ?\w*|נמאס|תפסיק\w*|תפסיקי|הפסק\w*|עצור|תעצור)(?!\w)|(?<!\w)די(?!\w)|"
    r"\b(?:annoying|enough|stop|shut up)\b", re.IGNORECASE)
_COMMAND_WORDS = re.compile(
    r"^\s*/|(?<!\w)ו?(?:תראה|תראי|הראה|תשלח|תשלחי|שלח|תביא|תביאי|תן|תני|תשתיק|השתק|תכבה|כבה|תדליק|הדלק|"
    r"תמשיך|תזכור|תזכרי|תקרא|תקראי|תרשום|תבדוק|בדוק)(?!\w)|"
    r"^\s*(?:show|send|give|pause|mute|turn|call|remember|check|play)\b", re.IGNORECASE)


def is_question(text: str) -> bool:
    return bool(isinstance(text, str) and _QUESTION_WORDS.search(text))


def is_insult(text: str) -> bool:
    return bool(isinstance(text, str) and _INSULT_WORDS.search(text))


def is_complaint(text: str) -> bool:
    """A complaint or an insult ("די עם ההודעה", "תפסיק", "יא מטומטם", "stop", "annoying")."""
    return bool(isinstance(text, str) and (_COMPLAINT_WORDS.search(text) or _INSULT_WORDS.search(text)))


def is_command(text: str) -> bool:
    return bool(isinstance(text, str) and _COMMAND_WORDS.search(text))


def not_a_judgement(text: str) -> bool:
    """True for a question, a complaint or insult, or a command: such a message never becomes a verdict (or the
    explanation of a clip) without the owner pointing at the alert themselves."""
    return not isinstance(text, str) or not text.strip() or is_question(text) or is_complaint(text) \
        or is_command(text)


# <<< vendored
