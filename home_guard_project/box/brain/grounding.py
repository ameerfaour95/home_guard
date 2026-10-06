# home_guard_project/box/brain/grounding.py
"""The safety net for visual details: a colour, a piece of clothing, what is in a hand, a plate.

After an alert the chat model has only the vision agent's observation - a few
sentences, never the pictures. It is partial: a detail it does not mention was
not checked, and the model must ask the clip (ask_vision) instead of filling it
in. This finds the details in an answer that neither the observation nor a
vision answer backs. Details are compared as concepts, so a Hebrew answer
("לבנה") is backed by an English observation ("a white car").
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

# A Hebrew stem with its prefixes (בלבן) and endings (לבנה, לבנים). Stems that are also common words are left out
# ("שק" in "שקט", "לום" in "שלום", "כסף", "אלה").
_HE = r"(?<!\w)[ושהבלמכ]{0,2}(?:%s)\w*"


def _en(words: str) -> str:
    return rf"\b(?:{words})\b"


# (concept, English words, Hebrew stems)
DETAILS: Tuple[Tuple[str, str, str], ...] = (
    ("white", "white", "לבן|לבנ"), ("black", "black", "שחור"), ("red", "red", "אדום|אדומ"),
    ("blue", "blue|navy", "כחול"), ("green", "green", "ירוק"), ("yellow", "yellow", "צהוב"),
    ("grey", "gr[ae]y", "אפור"), ("brown", "brown", "חום|חומ"), ("orange", "orange", "כתום|כתומ"),
    ("purple", "purple", "סגול"), ("pink", "pink", "ורוד"), ("silver", "silver", "כסוף|כסופ"),
    ("gold", "gold(?:en)?", "זהב|זהוב"), ("beige", "beige", "בז'|בז׳"),
    ("hoodie", "hoodie|hood(?:ed)?", "קפוצ'ון|קפוצ׳ון|ברדס"), ("mask", "mask(?:ed)?|balaclava", "מסכה|מסיכה"),
    ("hat", "hat|cap|beanie", "כובע"), ("jacket", "jacket|coat", "מעיל|ז'קט|ז׳קט"),
    ("shirt", "shirt|t-shirt", "חולצ"), ("trousers", "trousers|pants|jeans", "מכנס|ג'ינס|ג׳ינס"),
    ("shorts", "shorts", "שורט"), ("dress", "dress|skirt", "שמלה|חצאית"), ("uniform", "uniform|vest", "מדים|אפוד"),
    ("helmet", "helmet", "קסדה"), ("glasses", "glasses|sunglasses", "משקפ"), ("gloves", "gloves", "כפפ"),
    ("backpack", "backpack", "תיק גב"), ("phone", "phone|mobile", "טלפון|פלאפון|נייד"),
    ("bag", "bag|handbag|sack", r"תיק(?:ים|ה|ו)?(?!\w)|שקית"),
    ("package", "package|parcel|carton|delivery box", "חבילה|ארגז|קופסה"),
    ("hammer", "hammer", "פטיש"), ("knife", "knife|blade", "סכין"),
    ("gun", "gun|pistol|rifle|weapon|firearm", "אקדח|נשק|רובה"), ("crowbar", "crowbar|pry bar", "ראש חץ"),
    ("umbrella", "umbrella", "מטריה|מטרייה"), ("bottle", "bottle", "בקבוק"), ("cigarette", "cigarette", "סיגרי"),
    ("flashlight", "flashlight|torch", "פנס"), ("ladder", "ladder", "סולם"), ("keys", "keys?", "מפתח"),
    ("stick", "stick|bat|baton|pipe", "מקל|צינור"), ("tool", "tool|drill|screwdriver", "מברג|מקדחה"),
)
_PATTERNS = [(name, re.compile(_en(en) + "|" + (_HE % he), re.IGNORECASE)) for name, en, he in DETAILS]
_PLATE = re.compile(r"(?<!\d)(?:\d{2,3}-\d{2,3}-\d{2,3}|\d{7,8})(?!\d)")


def details_in(text: str) -> List[str]:
    """The visual-detail concepts *text* mentions, in DETAILS order, then any plate numbers."""
    if not isinstance(text, str) or not text:
        return []
    found = [name for name, pattern in _PATTERNS if pattern.search(text)]
    return found + [f"plate {m.group()}" for m in _PLATE.finditer(text)]


def ungrounded_details(answer: str, evidence: str) -> List[str]:
    """The details of *answer* that *evidence* (observations, vision answers, tool results) does not mention."""
    have = set(details_in(evidence if isinstance(evidence, str) else ""))
    return [d for d in details_in(answer) if d not in have]


def evidence_text(parts: Sequence[str]) -> str:
    return "\n".join(p for p in parts if isinstance(p, str) and p)
