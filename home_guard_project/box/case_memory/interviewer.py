"""The Interviewer: after the owner marks an alert "normal", at most three short questions, then a summary card.

A pure state machine (no model, no I/O): it pre-fills what the event already says (camera, hour, day, path,
clothing words), asks only for what is missing, offers quick-reply buttons (always with "skip" and "don't know"),
and ends with a card the owner saves, edits or cancels. The scope question ("always at this time, or only
today?") matters most: too wide silences real alerts, too narrow saves nothing. "Only today" is not a case but an
expecting note that ends at midnight.

Slots: who (always asked), when (always asked), how to recognise (asked when the Eye saw clothing words),
what to do next time (asked when there is room under the three-question cap; default: daily digest, still behind
the trust ladder). Where comes from the tracker's path and is shown on the card, not asked.

The state round-trips through ``to_dict``/``from_dict`` so the Telegram side can keep it between messages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from . import texts
from .gates import default_max_dwell, hours_around
from .models import ALERT, ALL_DAYS, DIGEST, QUIET, WHO, WORKWEEK, Case, Example, Scope, Signature
from .signature import clean_appearance

MAX_QUESTIONS = 3
SKIP, DONT_KNOW = "skip", "dont_know"
WHO_SLOT, WHEN_SLOT, RECOGNISE_SLOT, NEXT_SLOT = "who", "when", "recognise", "next"
SLOTS = (WHO_SLOT, WHEN_SLOT, RECOGNISE_SLOT, NEXT_SLOT)
WIDE_MARGIN_MIN = 90       # "also at other hours": a wider window the owner sees on the card and can edit


@dataclass(frozen=True)
class Option:
    code: str
    he: str
    en: str

    def label(self, lang: str) -> str:
        return self.he if lang == "he" else self.en


@dataclass(frozen=True)
class Question:
    slot: str
    text_he: str
    text_en: str
    options: Tuple[Option, ...]
    free_text: bool = False        # the owner may type (or speak) an answer instead

    def text(self, lang: str) -> str:
        return self.text_he if lang == "he" else self.text_en


@dataclass(frozen=True)
class SummaryCard:
    text_he: str
    text_en: str
    options: Tuple[Option, ...]

    def text(self, lang: str) -> str:
        return self.text_he if lang == "he" else self.text_en


@dataclass(frozen=True)
class Outcome:
    """The end of an interview. ``case`` is a draft to save with ``CaseStore.add`` (keeper.save_interview);
    ``expecting`` is the text of an only-today note."""
    kind: str                       # case / expecting / cancelled
    case: Optional[Case] = None
    expecting: str = ""
    camera: str = ""


Step = Union[Question, SummaryCard, Outcome]

_SKIP = (Option(SKIP, "דלג", "Skip"), Option(DONT_KNOW, "לא יודע", "Don't know"))


@dataclass
class Interview:
    event_id: str
    signature: Signature
    answers: Dict[str, Any] = field(default_factory=dict)
    asked: List[str] = field(default_factory=list)
    stage: str = "asking"           # asking / card / edit / done
    editing: str = ""
    outcome_kind: str = ""

    # -- persistence ------------------------------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {"event_id": self.event_id, "signature": self.signature.to_dict(), "answers": dict(self.answers),
                "asked": list(self.asked), "stage": self.stage, "editing": self.editing,
                "outcome_kind": self.outcome_kind}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Interview":
        return cls(event_id=data["event_id"], signature=Signature.from_dict(data["signature"]),
                   answers=dict(data.get("answers") or {}), asked=list(data.get("asked") or []),
                   stage=data.get("stage", "asking"), editing=data.get("editing", ""),
                   outcome_kind=data.get("outcome_kind", ""))

    # -- what to ask ------------------------------------------------------------------------------------------------
    def _plan(self) -> List[str]:
        """The questions this event needs, at most three."""
        plan = [WHO_SLOT, WHEN_SLOT]
        plan.append(RECOGNISE_SLOT if self.signature.appearance else NEXT_SLOT)
        return plan[:MAX_QUESTIONS]

    def question(self, slot: str) -> Question:
        sig = self.signature
        clock = f"{sig.minute // 60:02d}:{sig.minute % 60:02d}"
        if slot == WHO_SLOT:
            opts = tuple(Option(f"who:{w}", texts.WHO_TEXT[w][0] + ("..." if w == "other" else ""),
                                texts.WHO_TEXT[w][1].capitalize() + ("..." if w == "other" else "")) for w in WHO)
            return Question(slot, 'סימנת "רגיל". מי זה?', 'You marked this "normal". Who is it?', opts + _SKIP,
                            free_text=True)
        if slot == WHEN_SLOT:
            where_he = texts.path_sentence(sig.path, "he")
            where_en = texts.path_sentence(sig.path, "en")
            he = f"זה היה ב{texts.day_name(sig.weekday, 'he')} בסביבות {clock}" + (f", {where_he}" if where_he else "")
            en = f"This was on {texts.day_name(sig.weekday, 'en')} around {clock}" + (f", {where_en}" if where_en else "")
            if sig.weekday in WORKWEEK:
                regular = Option("when:workweek", "כן, כל יום חול", "Yes, every weekday")
            else:
                regular = Option("when:weekday", f"כן, כל {texts.day_name(sig.weekday, 'he')}",
                                 f"Yes, every {texts.day_name(sig.weekday, 'en')}")
            opts = (regular, Option("when:every_day", "כל יום", "Every day"),
                    Option("when:today", "רק היום", "Only today"),
                    Option("when:other_hours", "גם בשעות אחרות", "At other hours too"))
            return Question(slot, he + ". ככה זה תמיד בשעה הזו, או רק היום?",
                            en + ". Always at this time, or only today?", opts + _SKIP)
        if slot == RECOGNISE_SLOT:
            words = ", ".join(sig.appearance)
            return Question(slot, f"ראיתי {words}. ככה מזהים אותו?", f"I saw {words}. Is that how to recognise them?",
                            (Option("rec:yes", "כן", "Yes"), Option("rec:varies", "הבגדים משתנים", "Clothes change"))
                            + _SKIP)
        if slot == NEXT_SLOT:
            return Question(slot, "מה לעשות בפעם הבאה?", "What should I do next time?",
                            (Option("next:quiet", "הודעה שקטה", "A quiet message"),
                             Option("next:digest", "רק בסיכום היומי", "Only in the daily digest"),
                             Option("next:alert", "להודיע כרגיל", "Alert as usual")) + _SKIP)
        raise ValueError(f"unknown slot {slot!r}")

    def step(self) -> Step:
        """What to show now."""
        if self.stage == "done":
            return self.outcome()
        if self.stage == "edit":
            if self.editing:
                return self.question(self.editing)
            return Question("edit", "מה לשנות?", "What should I change?",
                            (Option("edit:who", "מי", "Who"), Option("edit:when", "מתי", "When"),
                             Option("edit:recognise", "איך מזהים", "How to recognise"),
                             Option("edit:next", "מה לעשות", "What to do")))
        if self.stage == "asking":
            if self.answers.get("typing"):
                return Question(WHO_SLOT, "מי זה? כתוב בכמה מילים.", "Who is it? A few words.", _SKIP, free_text=True)
            for slot in self._plan():
                if slot not in self.asked:
                    return self.question(slot)
            self.stage = "card"
        return self.card()

    def answer(self, code: str, text: str = "") -> Step:
        """Record the owner's button (*code*) or typed/spoken *text*, and return the next step."""
        code = str(code or "").strip()
        if self.stage == "card":
            if code == "card:save":
                self.stage, self.outcome_kind = "done", self._final_kind()
            elif code == "card:cancel":
                self.stage, self.outcome_kind = "done", "cancelled"
            elif code == "card:edit":
                self.stage, self.editing = "edit", ""
            return self.step()
        if self.stage == "edit" and not self.editing:
            slot = code.partition(":")[2]
            if slot in SLOTS:
                self.editing = slot
            return self.step()
        current = self.step()
        if not isinstance(current, Question):
            return current
        slot = current.slot
        self._record(slot, code, text)
        if slot == WHO_SLOT and code == "who:other" and not self.answers.get("title") and not self.answers.get("typing"):
            self.answers["typing"] = True            # "Other..." waits once for the typed (or spoken) answer
            return self.step()
        self.answers.pop("typing", None)
        if self.stage == "edit":
            self.stage, self.editing = "card", ""
            return self.card()
        if slot not in self.asked:
            self.asked.append(slot)
        if slot == WHEN_SLOT and self.answers.get(WHEN_SLOT) == "today":
            self.stage = "card"                      # only today: no more questions, an expecting note
        return self.step()

    def _record(self, slot: str, code: str, text: str) -> None:
        if code in (SKIP, DONT_KNOW):
            self.answers[slot] = code
            return
        prefix, _, value = code.partition(":")
        if slot == WHO_SLOT:
            typed = " ".join(str(text or "").split())[:80]
            if typed and not value:
                self.answers[WHO_SLOT], self.answers["title"] = "other", typed
            elif value in WHO:
                self.answers[WHO_SLOT] = value
                if typed:
                    self.answers["title"] = typed
            return
        if slot == WHEN_SLOT and value in ("workweek", "weekday", "every_day", "today", "other_hours"):
            self.answers[WHEN_SLOT] = value
        elif slot == RECOGNISE_SLOT and value in ("yes", "varies"):
            self.answers[RECOGNISE_SLOT] = value
        elif slot == NEXT_SLOT and value in (QUIET, DIGEST, ALERT):
            self.answers[NEXT_SLOT] = value
        elif text:
            self.answers[f"{slot}_text"] = " ".join(str(text).split())[:200]

    # -- what the answers mean --------------------------------------------------------------------------------------
    def _final_kind(self) -> str:
        return "expecting" if self.answers.get(WHEN_SLOT) == "today" else "case"

    def scope(self) -> Scope:
        sig = self.signature
        when = self.answers.get(WHEN_SLOT)
        if when == "workweek":
            days = WORKWEEK
        elif when == "every_day":
            days = ALL_DAYS
        else:                                         # this weekday, skipped or don't know: the narrow default
            days = (sig.weekday,)
        margin = WIDE_MARGIN_MIN if when == "other_hours" else 30
        dwell = default_max_dwell([sig.dwell_s]) if sig.dwell_s is not None else None
        house = sig.house_state if sig.house_state != "away" else "home_awake"   # away only when said explicitly
        category = sig.category if sig.category and sig.category != "other" else ""
        return Scope(camera=sig.camera, hours=hours_around(sig.minute, margin, margin), weekdays=tuple(sorted(days)),
                     house_states=(house,), night=sig.phase == "late_night", people=sig.people,
                     vehicles=sig.vehicles, path=sig.path, entry_edge=sig.entry_edge, exit_edge=sig.exit_edge,
                     categories=(category,) if category else (), max_dwell_s=dwell)

    def draft(self) -> Case:
        sig = self.signature
        who = self.answers.get(WHO_SLOT)
        who = who if who in WHO else "other"
        recognise = clean_appearance(list(sig.appearance)) if self.answers.get(RECOGNISE_SLOT) == "yes" else ()
        effect = self.answers.get(NEXT_SLOT)
        effect = effect if effect in (QUIET, DIGEST, ALERT) else DIGEST
        note = " ".join(str(self.answers.get(k, "")) for k in ("who_text", "when_text") if self.answers.get(k))
        return Case(id="", scope=self.scope(), who=who, title=str(self.answers.get("title") or ""), note=note,
                    recognise=recognise, effect=effect, examples=[Example(self.event_id, sig)],
                    created_at=sig.ts, source={"alert_id": self.event_id, "origin": "interview"})

    def card(self) -> SummaryCard:
        sig = self.signature
        opts = (Option("card:save", "שמור", "Save"), Option("card:edit", "ערוך", "Edit"),
                Option("card:cancel", "ביטול", "Cancel"))
        draft = self.draft()
        title_he, title_en = texts.case_title(draft, "he"), texts.case_title(draft, "en")
        if self.answers.get(WHEN_SLOT) == "today":
            return SummaryCard(f"אזכור להיום בלבד: {title_he}, מצלמת {sig.camera}. לא אשמור את זה כשגרה.",
                               f"I'll remember for today only: {title_en}, camera {sig.camera}. Not saved as a routine.",
                               (opts[0], opts[2]))
        s = draft.scope
        where_he, where_en = texts.path_sentence(s.path, "he"), texts.path_sentence(s.path, "en")
        line_he = f"מצלמת {s.camera}, {texts.days_text(s.weekdays, 'he')} {texts.hours_text(s.hours)}"
        line_en = f"camera {s.camera}, {texts.days_text(s.weekdays, 'en')} {texts.hours_text(s.hours)}"
        if where_he:
            line_he += f", {where_he}"
            line_en += f", {where_en}"
        if s.people:
            line_he += f", {s.people} {'אדם' if s.people == 1 else 'אנשים'}"
            line_en += f", {s.people} {'person' if s.people == 1 else 'people'}"
        if draft.recognise:
            line_he += f". מזהים לפי: {', '.join(draft.recognise)}"
            line_en += f". Recognised by: {', '.join(draft.recognise)}"
        effect_he = {DIGEST: "בפעמים הבאות לא אתריע, ואראה לך בסיכום היומי.",
                     QUIET: "בפעמים הבאות אשלח הודעה שקטה.",
                     ALERT: f"אמשיך להתריע, ואוסיף שזה כנראה {title_he}."}[draft.effect]
        effect_en = {DIGEST: "Next time I won't alert; you'll see it in the daily digest.",
                     QUIET: "Next time I'll send a quiet message.",
                     ALERT: f"I'll keep alerting and add that it's probably {title_en}."}[draft.effect]
        if draft.effect != ALERT:
            effect_he += " בהתחלה אמשיך להתריע ולשאול אותך, עד שאהיה בטוח."
            effect_en += " At first I'll keep alerting and ask you, until I'm sure."
        guard_he = "לא חל על לילה, על בית ריק, על יותר אנשים או על מי שנוגע בדלת או מתעכב."
        guard_en = "Never for night, an empty house, more people, or someone who touches the door or lingers."
        if s.night:
            guard_he = guard_he.replace("לילה, על ", "")
            guard_en = guard_en.replace("night, ", "")
        return SummaryCard(f"אזכור: {title_he}. {line_he}.\n{effect_he}\n{guard_he}",
                           f"I'll remember: {title_en}. {line_en}.\n{effect_en}\n{guard_en}", opts)

    def outcome(self) -> Outcome:
        kind = self.outcome_kind or "cancelled"
        if kind == "case":
            return Outcome("case", case=self.draft(), camera=self.signature.camera)
        if kind == "expecting":
            draft = self.draft()
            return Outcome("expecting", expecting=texts.case_title(draft, "he"), camera=self.signature.camera)
        return Outcome("cancelled", camera=self.signature.camera)

    @property
    def questions_asked(self) -> int:
        return len(self.asked)


def start(event_id: str, signature: Signature) -> Tuple[Interview, Step]:
    """Begin an interview for an alert the owner marked "normal"."""
    interview = Interview(event_id, signature)
    return interview, interview.step()
