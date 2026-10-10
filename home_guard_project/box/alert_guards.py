"""Code guards on the model's label, before an alert reaches the owner (stage 1 of the alert fix, 2026-10-08).

Why: in two days of real alerts the model called workers "suspicious" for a mask, a hoodie or a covered face, and
gave a red "possible weapon" for a long work tool and a red "smashed the car window" for a man who parked. The
owner's rules:

- Appearance alone (mask, hood, covered face, dark clothes, hat, sunglasses, a blurred or pixelated face) is never
  suspicious; only actions are. :func:`appearance_only` finds a "suspicious" whose reason names nothing but looks,
  and the guard loop lowers it to normal. It never touches an escalation.
- A red for a weapon, a tool used as a weapon, a car break-in, violence or a person down (lying, kneeling) that
  comes from one model answer gets a second look before it goes out red (:func:`verify_class` picks the question). Clear serious things - a break-in
  through the house's door or window, climbing in, fire or smoke, a person lying motionless (:func:`clear_class`) -
  go out at once.

Matching is on words, in English and Hebrew. English words are matched at a word start (so "hat" does not match
"that"); Hebrew words as substrings, so a prefix letter (ב, ה, ו, ש, ל) still matches. Pure functions, no imports
beyond ``re`` and the prompt files: unit-tested on their own.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

from ..prompts import load, render

_FLAGS = re.IGNORECASE


def _any(patterns: Sequence[str]) -> "re.Pattern[str]":
    return re.compile("|".join(f"(?:{p})" for p in patterns), _FLAGS)


# ---------- appearance vs action ----------
APPEARANCE = _any([
    r"\bmask", r"\bhood", r"\bbalaclava", r"\bski[- ]mask", r"\bhats?\b", r"\bcaps?\b", r"\bbeanie",
    r"\bsunglasses", r"\bpixelat", r"\bblur", r"\bdark (?:cloth|outfit|attire|jacket|hoodie|garment)",
    r"\bcover(?:ed|ing|s)? (?:his |her |their |the )?faces?", r"\bfaces? (?:is |are |was |were )?(?:covered|hidden|obscured|concealed|not visible)",
    r"\b(?:hidden|obscured|concealed) faces?", r"\bhid(?:es|ing)? (?:his|her|their) faces?",
    r"מסכה", r"מסיכה", r"קפוצ['׳]?ון", r"ברדס", r"פנים מכוסות", r"פנים מוסתרות", r"פניו מכוסות", r"פניו מוסתרות",
    r"מכסה את פניו", r"כיסוי ראש", r"כיסוי פנים", r"בגדים כהים", r"לבוש כהה", r"כובע", r"משקפי שמש", r"פיקסל", r"מפוקסל", r"מטושטש", r"רעול",
])

# Anything here keeps the "suspicious": an action, or night (walking around the property at night stays suspicious).
ACTION = _any([
    r"\bhandles?\b", r"\bdoors?\b", r"\bwindows?\b", r"\bgates?\b", r"\bfence", r"\blocks?\b", r"\bclimb",
    r"\bhid(?:e|es|ing)\b(?! (?:his|her|their) faces?)",r"\bcrouch", r"\bsteal", r"\bstole",
    r"\btak(?:e|es|ing|en)\b", r"\btook\b", r"\btamper", r"\bpeek", r"\bpeer", r"\blook(?:s|ed|ing)? (?:in|into|inside)\b",
    r"\bloiter", r"\blurk", r"\bbreak", r"\bforc", r"\bpry", r"\bentrance", r"\bnight\b",
    r"\b(?:cover|block|turn|mov|spray|paint|point|push|hit)\w* (?:the |a |at the )?camera",
    r"ידית", r"דלת", r"חלון", r"שער", r"גדר", r"מנעול", r"טיפוס", r"מטפס", r"מסתתר", r"מתחבא", r"כורע", r"גונב",
    r"לוקח", r"לקח", r"מציץ", r"פוגע במצלמה", r"פורץ", r"כניסה", r"לילה",
])


# What may be left in a reason that is only about looks: who, the look itself, filler, and harmless presence
# (standing, walking past, a phone). Anything else - any other word - is taken as an action and keeps the label.
# 2026-10-08 (home-guard-32's replay on eval_set_v2): the first version kept a "suspicious" only when its reason had
# a word from ACTION, and lowered 41 REAL alerts whose action was in other words ("loading items into a parked
# vehicle", "carrying a large bag", "moving around the property"). Deny by default: an unknown word keeps the alert.
_BENIGN_EN = set("""
a an the this that one two three several some person people man men woman women someone somebody individual
individuals figure figures guy boy girl child adult with without wearing wears wear dressed in and or is are was
were be been being appears appear appeared seems seem possibly likely maybe partially partly fully mostly
completely its it his her their them they he she of to by from for at on as due because while also only just
face faces head heads features identity covered covering covers obscured obscuring hidden hiding concealed
concealing masked mask masks balaclava ski hood hooded hoodie hoodies sweatshirt hat hats cap caps beanie helmet
sunglasses glasses dark black darkly colored coloured clothing clothes clothed outfit attire jacket garment
garments pants shirt long sleeve sleeves pixelated pixelation pixels blurry blurred blur unclear not visible
cannot be seen unidentified unknown unrecognizable worker workers suspicious suspiciously behavior behaviour appearance presence
present noted observed seen visible stands standing stood walks walking walked passes passing passed past by near
nearby next wall frame scene view area outside here there phone looking looks look at talking talks
walk stand pass across through along away have has had watches watching watch watched
light lighter white grey gray blue red green brown beige yellow orange pink purple navy khaki colorful colourful
patterned striped plain bright coat coats vest shorts shoes sneakers trousers jeans top dress skirt
balcony porch patio yard backyard courtyard driveway path pathway walkway paved garden lawn grass house home
property sidewalk pavement stone tiles
""".split())
_BENIGN_HE = set("""
אדם אנשים גבר גברים אישה נשים מישהו דמות דמויות ילד ילדה אחד אחת שני שניים שלושה עם בלי לבוש לבושה לובש לובשת לבושים
ו או של על ידי עקב בגלל כנראה נראה נראית נראים אולי חלקית לגמרי הוא היא הם הן את
פנים פניו פניה פניהם מכוסות מכוסים מוסתרות מוסתרים מוסתר מכוסה מכסה מסתיר מסתירים כיסוי ראש
מסכה מסכות מסיכה רעול רעולת רעולי ברדס קפוצון כובע כובעים קסדה משקפי שמש
בגדים בגד כהים כהה שחורים שחור שחורה ארוכים פיקסלית פיקסלים מפוקסל מטושטש מטושטשות לא ברור ברורות
התנהגות חשודה חשוד הופעת הופעה נוכחות עומד עומדת עומדים הולך הולכת הולכים עובר עוברת עוברים ליד קיר
בתמונה בפריים באזור בחוץ כאן שם טלפון מסתכל מסתכלת מדבר מדברת
התנהלות התנהגותו חליפה חליפת מסתירה מסתירות עובד עובדים עובדת פועל פועלים מפוקסלות מפוקסלים מפוקסלת מטושטשים מטושטשת
בהיר בהירה בהירים בהירות לבן לבנה לבנים לבנות אפור אפורה אפורים כחול כחולה כחולים אדום אדומה אדומים ירוק ירוקה
ירוקים צבעוני צבעונית צבעוניים מודפסת מודפס מעיל מעילים קפל חולצה חולצות מכנסיים מכנסי נעליים
מרפסת חצר שביל שבילים חניה חנייה פטיו דשא בית גינה מסלול מרוצף מרוצפת אבן אבנים מדרכה ריצוף
לאורך דרך הולכת נראה נראית נראים נראו
""".split())
_TOKEN = re.compile(r"[A-Za-z]+|[א-ת]+")
_HE_PREFIXES = ("ו", "ה", "ב", "ל", "ש", "מ", "כ")


def _benign(token: str) -> bool:
    t = token.lower()
    if t in _BENIGN_EN or t in _BENIGN_HE:
        return True
    if t[:1] in _HE_PREFIXES and len(t) > 2:          # "והפנים", "במסכה": one or two prefix letters
        if t[1:] in _BENIGN_HE or (t[1:2] in _HE_PREFIXES and t[2:] in _BENIGN_HE):
            return True
    return False


def appearance_only(text: str) -> bool:
    """True when *text* (the model's why / alert_reason) is about looks and nothing else: it names an appearance
    (APPEARANCE) and every other word is filler or harmless presence (standing, walking past, a phone). Such a
    "suspicious" is lowered to normal. Any other word - an action in words we never listed - keeps the label, and
    so does any listed action (ACTION). Empty text, or text naming no appearance, is False."""
    text = str(text or "").replace("'", "").replace("׳", "").replace('"', " ")
    if not APPEARANCE.search(text) or ACTION.search(text):
        return False
    return all(_benign(tok) for tok in _TOKEN.findall(text))


# Presence: someone is there, walking or standing (2026-10-09 18:22 ch6: "הולך לאורך המסלול" went out 🟡).
PRESENCE = _any([
    r"\bwalk", r"\bstand", r"\bstood\b", r"\bpass(?:es|ed|ing)?\b", r"\bpresen", r"\bvisible\b", r"\bseen\b",
    r"הולך", r"הולכת", r"הולכים", r"עומד", r"עומדת", r"עומדים", r"עובר", r"עוברת", r"עוברים", r"נוכחות", r"נראה", r"נראית",
])
# What in the summary keeps the label: any ACTION, or a carried thing, a try, a way in, a run (the summary is too long
# for the every-token rule, so here it is only searched for these).
SUMMARY_ACTION = _any([
    r"\bcarr(?:y|ies|ied|ying)\b", r"\bbags?\b", r"\bsacks?\b", r"\bbackpack", r"\bpackage", r"\bparcel", r"\bboxe?s?\b",
    r"\bload", r"\bgrab", r"\bopen", r"\btr(?:y|ies|ied|ying)\b", r"\benter", r"\bran\b", r"\brun", r"\bflee", r"\bfled",
    r"\bjump", r"\bthrow", r"\bthrew", r"\bpick", r"\bhold", r"\bremov", r"\bsearch", r"\brummag", r"\bvehicle", r"\bcars?\b",
    r"\bapproach", r"\baround\b", r"\bflashlight", r"\btorch", r"\bphotograph", r"\bfilm", r"\bweapon", r"\bknife", r"\bgun",
    # ...or comes toward the house or the camera (eval_set_v2 Security_smartbench_0908: hooded people "walk toward
    # the camera" were window-peepers)
    r"\btowards?\b", r"\bcamera", r"\binto\b", r"\bup to\b", r"\bclose to\b", r"\bcloser\b",
    r"לכיוון", r"מתקרב", r"מצלמה", r"נושא", r"סוחב", r"שק", r"תיק", r"פותח", r"מנסה", r"רץ", r"בורח",
])


# A fence or a wall the person walks NEXT TO is where they are, not what they do ("walking along a paved path next to
# a stone wall and a black fence", 18:22). A gate, a door, a window - or behind / over a fence - still count.
_BOUNDARY = r"(?:the |a |an )?(?:[\w-]+ ){0,2}?(?:fences?|walls?)\b"
_BOUNDARY_PLACE = re.compile(rf"\b(?:next to|near|along(?:side)?|beside|by|past) {_BOUNDARY}(?:,? and {_BOUNDARY})*",
                             _FLAGS)


def presence_only(why: str, summary: str = "") -> bool:
    """True when the Eye's *why* says only that someone is there - walking or standing (PRESENCE), with places,
    colours, clothes and filler - and names nothing else: every token benign (the deny-by-default rule of
    :func:`appearance_only`), no ACTION in the why, and no ACTION or SUMMARY_ACTION in the *summary*. Such a
    "suspicious" is lowered to normal like an appearance-only one. Only the why is held to every token (the summary is
    too long for that). Empty why, or a why that names no presence, is False."""
    why = str(why or "").replace("'", "").replace("׳", "").replace('"', " ")
    summary = _BOUNDARY_PLACE.sub(" ", str(summary or ""))
    if not PRESENCE.search(why) or ACTION.search(why) or ACTION.search(summary) or SUMMARY_ACTION.search(summary):
        return False
    return all(_benign(tok) for tok in _TOKEN.findall(why))


# ---------- lingering: the investigator watches the tracker before it goes out (stage 2b, 2026-10-08) ----------
# A "suspicious" whose reason is only about time spent - loitering, standing there, looking around - is a question of
# how long the person really stayed, which the tracker measures (inference.investigate_lingering). Any other action
# (a handle, climbing, taking something, peeking in, night) keeps it suspicious whatever the time. Places (door,
# gate, entrance) are not actions here: "loitering by the gate" is exactly the case.
LINGER = _any([
    r"\bloiter", r"\bling(?:er|ers|ered|ering)\b", r"\bstand(?:s|ing)? (?:\w+ ){0,4}for\b", r"\bstood (?:\w+ ){0,4}for\b",
    r"\blook(?:s|ed|ing)? around\b", r"\bwander", r"\bhang(?:s|ing)? (?:a)?round\b", r"\bpac(?:es|ing)\b",
    r"מסתובב", r"שוהה", r"שוהים", r"עומד(?:ת|ים|ות)? (?:\S+ ){0,3}זמן", r"מסתכל(?:ת|ים|ות)? (?:\S+ )?(?:סביב|מסביב|לצדדים)",
])
OTHER_ACTION = _any([
    r"\bhandles?\b", r"\blocks?\b", r"\bclimb", r"\bhid(?:e|es|ing)\b(?! (?:his|her|their) faces?)", r"\bcrouch",
    r"\bsteal", r"\bstole", r"\btak(?:e|es|ing|en)\b", r"\btook\b", r"\btamper", r"\bpeek", r"\bpeer",
    r"\blook(?:s|ed|ing)? (?:in|into|inside|through)\b", r"\bbreak", r"\bforc", r"\bpry", r"\btr(?:y|ies|ied|ying)\b",
    r"\bopen", r"\bnight\b", r"\bphotograph", r"\bfilm",
    r"\b(?:cover|block|turn|mov|spray|paint|point|push|hit)\w* (?:the |a |at the )?camera",
    r"ידית", r"מנעול", r"טיפוס", r"מטפס", r"מסתתר", r"מתחבא", r"כורע", r"גונב", r"לוקח", r"לקח", r"מציץ", r"פוגע במצלמה",
    r"פורץ", r"מנסה", r"פותח", r"מצלם", r"לילה",
])


def about_lingering(text: str) -> bool:
    """True when *text* (the model's why / alert_reason) makes it suspicious only for lingering - loitering,
    standing there for a while, looking around, wandering - and names no other action."""
    text = str(text or "")
    return bool(LINGER.search(text)) and not OTHER_ACTION.search(text)


# ---------- the second look before a red ----------
_VEHICLE = r"(?:\bcars?\b|\bvehicles?\b|\btrucks?\b|\bvans?\b|רכב|מכונית)"
VERIFY_PATTERNS = {
    "weapon": _any([r"\bweapon", r"\bguns?\b", r"\bknife", r"\bknives", r"\brifle", r"\bpistol", r"\bfirearm",
                    r"\bmachete", r"נשק", r"אקדח", r"סכין", r"רובה"]),
    # A long or blunt thing in a red (2026-10-09 13:25 ch6: pavers workers "holding a long metal bar" went out red):
    # a bat or a pipe CAN be a weapon, so it is asked about, never just dropped. What is climbed or fixed to the house
    # (a drainpipe, a light pole, window bars) is not a thing in someone's hands. Hebrew words that hide inside common
    # words (מוט in מוטל "lying", לום in שלום / כלום, מקל in מקלט / מקלחת, אלה "these") only as whole words.
    "tool_weapon": _any([
        r"\bbars?\b(?! (?:on|of|over) (?:the |a )?windows?)", r"\brods?\b",
        r"(?<!light )(?<!lamp )(?<!utility )(?<!flag )(?<!fence )(?<!street )(?<!power )\bpoles?\b",
        r"(?<!drain )(?<!gutter )(?<!rain )(?<!water )\bpipes?\b", r"\bstick\b(?! (?:to|out|around|by|with)\b)",
        r"\b(?:a|the|wooden|long|metal|big|large|thick|his|her|their|with|holding|carrying|swinging|waving) sticks\b",
        r"\bbats?\b", r"\bclubs?\b", r"\bcrowbars?\b", r"\bhammers?\b", r"\bsledge", r"\baxes?\b", r"\bax\b",
        r"\bshovels?\b", r"\bplanks?\b", r"\bbatons?\b",
        r"(?<![א-ת])[בוהלמ]?ה?מוט(?:ות)?(?![א-ת])", r"צינור", r"(?<![א-ת])[בוהלמש]?ה?מקל(?:ות)?(?![א-ת])",
        r"(?<![א-ת])(?:ב|עם |מחזיק |מחזיקה |אוחז |אוחזת |מניף |מניפה )אלה(?![א-ת])", r"(?<![א-ת])[בוה]?ה?אלת ",
        r"מחבט", r"פטיש", r"גרזן", r"את חפירה", r"קרש", r"(?<![א-ת])[בוה]?ה?לום(?![א-ת])",
    ]),
    # A car must be named: "smashed a window" alone may be the house's, and that goes out red at once.
    "vehicle": _any([rf"(?:\bsmash|\bshatter|\bbreak|\bbroke|\bforc|\bpry|ניפץ|מנפץ|שובר|שבר|פורץ|פריצה).{{0,40}}{_VEHICLE}",
                     rf"{_VEHICLE}.{{0,30}}(?:\bsmash|\bshatter|\bbroken into|\bbreak-in|\bforced|נופץ|נפרץ|פריצה)"]),
    # Hitting a person, not a window: the person must be named.
    "violence": _any([r"\bfight", r"\battack", r"\bassault", r"\bpunch", r"\bbeat(?:s|ing)? (?:up )?(?:a |an |the |another )?(?:man|woman|person|someone|him|her|child)",
                      r"\bhit(?:s|ting)? (?:a |an |the |another )?(?:man|woman|person|someone|him|her|child|boy|girl)",
                      r"\bkick(?:s|ing)? (?:a |an |the |another )?(?:man|woman|person|someone|him|her)",
                      r"אלימות", r"תוקף", r"תוקפים", r"קטטה", r"מתקוטט", r"מכה (?:אדם|גבר|אישה|ילד|אותו|אותה|את ה(?:גבר|אישה|ילד|אדם))",
                      r"מכים (?:אדם|גבר|אישה|אותו|אותה)", r"הכה (?:אדם|גבר|אישה|אותו|אותה)"]),
}
# A person down - lying, on the ground, kneeling, crouching (2026-10-09 14:03 ch6: the owner's pavers workers went out
# red for "a person lies on the ground while two others stand nearby"). Lying motionless / unconscious / not moving
# is CLEAR and goes out at once; this is the rest, which may be work.
PERSON_DOWN = _any([
    r"\bl(?:ie|ies|ying|ay|ays|aying|ain)\b (?:\w+ ){0,3}?(?:on|in) (?:the |a )?(?:ground|floor|pavement|pavers?|sidewalk|"
    r"road|grass|street|concrete|tiles?|dirt|asphalt)\b", r"\blying\b", r"\blies\b",
    r"\b(?:person|man|woman|someone|somebody|worker|he|she|one|people|men|workers|they)\b (?:\w+ ){0,3}?on the "
    r"(?:ground|floor|pavement)\b",       # a PERSON on the ground ("throwing items on the floor" is not, eval_set_v2)
    r"\bkneel", r"\bknelt\b", r"\bcrouch", r"\bsquat", r"\bprone\b", r"\bon (?:his|her|their|all) (?:knees|back|stomach|fours)\b",
    r"שוכב", r"(?:אדם|גבר|אישה|מישהו|פועל|עובד|אנשים|פועלים|הוא|היא)(?: \S+){0,2} על (?:הקרקע|הרצפה|האדמה|המדרכה)",
    r"כורע", r"רכון", r"רכונה", r"רכונים", r"על הברכיים", r"על ברכיו",
])
# ...but not when the red names anything else: violence, a weapon, a break-in or theft, a way in (door, window, gate,
# fence), hiding, night, a fall or an injury, a child or an old person. Those reds go on as before (their own look,
# or out at once). These words count anywhere in the red; a car or a bag (DOWN_THING) only where it is said about the
# person on the ground.
DOWN_ACT = _any([
    r"\bfight", r"\battack", r"\bassault", r"\bhit\b", r"\bhits\b", r"\bhitting", r"\bbeat", r"\bpunch", r"\bkick",
    r"\bstrik", r"\bstruck", r"\bslap", r"\bspray", r"\bbaton", r"\bcharg", r"\brun(?:s|ning)? (?:forward|toward|at)\b",
    r"\bsurround", r"\bcorner(?:s|ed|ing)\b", r"\bhands? (?:raised|up)\b", r"\braises? (?:his |her |their )?hands",
    r"\bstomp", r"\bpush", r"\bshov", r"\bknock", r"\bstruggl", r"\bwrestl", r"\bgrab", r"\bdrag", r"\bpin(?:s|ned|ning)?\b",
    r"\bheld\b", r"\bhold(?:s|ing)? (?:\w+ ){0,2}down\b", r"\brestrain", r"\btie[sd]?\b", r"\btying", r"\bchok", r"\bstrangl",
    r"\bthreat", r"\bviolen", r"\baggress", r"\babus", r"\brob", r"\bmug", r"\bsteal", r"\bstole", r"\btheft", r"\bthie",
    r"\bburgl", r"\bbreak", r"\bbroke", r"\bsmash", r"\bforc", r"\bpr(?:y|ies|ied|ying)\b", r"\block", r"\btamper",
    r"\bweapon", r"\bgun", r"\bknife", r"\bknives", r"\bpistol", r"\brifle",
    r"\bdoors?\b", r"\bwindows?\b", r"\bgates?\b", r"\bfence", r"\bwall\b", r"\bclimb", r"\benter", r"\bpeek", r"\bpeer", r"\blook(?:s|ed|ing)? (?:in|into|inside|through)\b",
    r"\bhid(?:e|es|ing|den)\b", r"\bsneak", r"\bnight\b",
    r"\bfall", r"\bfell\b", r"\bcollaps", r"\bfaint", r"\bpass(?:es|ed)? out", r"\binjur", r"\bhurt", r"\bblood", r"\bbleed",
    r"\bwound", r"\bpain\b", r"\bhelp", r"\bdistress", r"\bseizure", r"\bvictim", r"\bbody\b", r"\bdead\b", r"\blimp\b",
    r"\bstill\b", r"\bmotionless", r"\bnot moving", r"\bunresponsive", r"\bimmobile", r"\bscream", r"\bcry", r"\bcries",
    r"\bchild", r"\bkid\b", r"\bbaby", r"\btoddler", r"\belderly", r"\bold (?:man|woman|person)",
    # ...or a theft, a robbery or a getaway around it (eval_set_v2 "as if the Eye had said 'lying on the ground'")
    r"\blung", r"\bthrow", r"\bthrew", r"\bcut", r"\bload", r"\btak(?:e|es|ing|en)\b", r"\btook\b", r"\bflee", r"\bfled",
    r"\bran\b", r"\brun(?:s|ning)? (?:away|off)", r"\bcarr\w* (?:\w+ ){0,5}(?:away|off|out)\b", r"\bstore\b", r"\bshop",
    r"\bcounter\b", r"\bregister\b", r"\bcash", r"\bshutter", r"\bpower tool", r"\btamper",
    r"\bdriv\w* (?:\w+ ){0,2}(?:away|off)\b", r"\bspeed(?:s|ing)? (?:away|off)\b", r"\bcamera", r"\blens\b",
    r"\bporch\b", r"\bentrance", r"\bsteps\b", r"\bstairs",
    r"מצלמה", r"כניסה", r"מדרגות", r"נמלט", r"נוסע משם", r"נסע משם",
    r"מכה", r"מכים", r"תוקף", r"תקיפה", r"אלימות", r"קטטה", r"דוחף", r"גורר", r"מחזיק אותו", r"כבול", r"שודד", r"גונב",
    r"פורץ", r"נשק", r"סכין", r"אקדח", r"דלת", r"חלון", r"שער", r"גדר", r"מטפס", r"מסתתר", r"מתחבא",
    r"לילה", r"נפל", r"נופל", r"התמוטט", r"מתעלף", r"התעלף", r"פצוע", r"פציעה", r"(?<![א-ת])ה?דם(?![א-ת])", r"עזרה",
    r"ללא תנועה", r"לא זז", r"ילד", r"תינוק", r"קשיש", r"צועק", r"בוכה",
    r"לוקח", r"לקח", r"בורח", r"ברח", r"זורק", r"זרק", r"מעמיס", r"חותך", r"חנות", r"קופה",
])
# A thing: crouching OVER it, lying under it, rummaging in it (eval_set_v2: a burglar crouching to search the floor,
# kneeling to rummage through a bag, crouching over a scooter) is about the thing, not a person down. Counted only in
# what is said about the person on the ground (:func:`down_scope`): 2026-10-09 14:41 / 15:44 ch6, the pavers workers
# went out red with no look because ANOTHER man "stands next to a white car" / "walks past him carrying a bag".
# Interacting with the person on the ground is what the look itself asks about, so it is not a thing.
DOWN_THING = _any([
    r"\bcars?\b", r"\bvehicles?\b", r"\btrucks?\b", r"\bvans?\b", r"\bmotorcycle", r"\bbikes?\b",
    r"\brummag", r"\bsearch", r"\bpick(?:s|ed|ing)? (?:\w+ ){0,3}up\b", r"\bpick(?:s|ed|ing)? up\b", r"\bhandl",
    r"\binspect", r"\binteract\w*(?! with (?:the |a |an |another |that |this )?(?:\w+ )?(?:person|man|woman|worker|"
    r"him|her|them|people|men)\b)", r"\badjust", r"\bbags?\b", r"\bbackpack", r"\bpackage", r"\bparcel",
    r"\bboxe?s?\b", r"\bbins?\b", r"\bcrate", r"\btrash", r"\bscooter", r"\bbicycle",
    r"מחטט", r"מחפש", r"מרים", r"תיק", r"חבילה", r"קורקינט", r"אופניים", r"רכב", r"מכונית",
])
# Where one person's words end and another's begin: a sentence end, a dash, or "while / and / as / with / ," before
# a new person ("while another man", "and two others", Hebrew "ואדם אחר", "בעוד", "בזמן ש"). A plain "and" goes on
# with the same person ("crouches and searches the floor"), and so does a sentence that starts with "he / she / they".
_NEW_PERSON = (r"(?:another|the other|the others|others|someone|somebody|a second|one of them|"
               r"(?:a|an|the|one|two|three|four|some|several|second|other)\s+(?:[\w-]+\s+){0,3}?"
               r"(?:man|men|woman|women|person|people|persons|worker|workers|guy|guys|individual|individuals|figure|"
               r"figures|boy|girl|others?)\b)")
_HE_NEW_PERSON = r"(?:אדם|גבר|אישה|מישהו|פועל|עובד|אנשים|גברים|פועלים|עובדים|שני|שניים|שלושה|אחר|אחד|אחת|אחרים)"
_CLAUSE_BREAK = re.compile(
    r"[.!?;:]+(?:\s+|$)|\s+[-–—]+\s+"
    rf"|,\s*(?={_NEW_PERSON}|{_HE_NEW_PERSON}(?![א-ת]))"
    rf"|\s+(?:while|whilst|whereas|and|as|but|with)\s+(?={_NEW_PERSON})"
    rf"|\s+(?=(?:בעוד|בזמן ש|כאשר|ו(?!אחר כך){_HE_NEW_PERSON}(?![א-ת])))", _FLAGS)
# Bent over something counts as the person down too ("bends down near the rear of the white car").
_BENT = _any([r"\bb(?:end|ends|ending|ent)\b", r"\bstoop", r"\blean(?:s|ed|ing)? (?:over|into|down|in)\b",
              r"\breach(?:es|ed|ing)? (?:into|under|inside)\b", r"מתכופף", r"התכופף", r"רוכן", r"גוהר"])
_SAME_PERSON = re.compile(r"\s*(?:he|she|they|his|her|their|him|then|הוא|היא|הם|הן|ואז|אז)(?![\w])", _FLAGS)


def down_scope(text: str) -> str:
    """What *text* says about the person on the ground: the clauses that name a person down (:data:`PERSON_DOWN`) or
    bent over something, each with the sentences after it that go on with "he / she / they". Another person's clause ("while another man
    walks past carrying a bag") is left out."""
    kept: List[str] = []
    inside = False
    for part in _CLAUSE_BREAK.split(str(text or "")):
        if not part or not part.strip():
            continue
        inside = bool(PERSON_DOWN.search(part) or _BENT.search(part)) or (inside and bool(_SAME_PERSON.match(part)))
        if inside:
            kept.append(part.strip())
    return " | ".join(kept)

VERIFY_ORDER = ("weapon", "tool_weapon", "vehicle", "violence", "person_down")

# The tool question is only for a tool that is THERE - held, carried, lying near someone - not for a red that already
# names what was done with it. eval_set_v2 (2026-10-09, qwen3.5-9b, 5 frames): asked the tool question, the second
# look said "no" on 11 of 14 real tool alerts it was shown (a baton beating, a window pried with a tool, a glass door
# smashed, a car window broken with a tool...). So when the red names an act - hitting, swinging, smashing, prying,
# forcing, cutting, throwing, threatening, attacking, stealing, using it on a door, lock or car - there is no tool
# question and the red goes on as before (its own vehicle / violence look, or out at once).
TOOL_IN_USE = _any([
    r"\bswing", r"\bswung", r"\bwield", r"\bbrandish", r"\bstrik", r"\bstruck", r"\bhit", r"\bbeat", r"\bbash",
    r"\bsmash", r"\bshatter", r"\bbreak", r"\bbroke", r"\bpr(?:y|ies|ied|ying)\b", r"\bforc", r"\bcut",
    r"\bthrow", r"\bthrew", r"\bthrown", r"\bthreat", r"\bmenac", r"\baggress", r"\bviolen", r"\battack", r"\bassault",
    r"\bfight", r"\bdamag", r"\bvandal", r"\bsteal", r"\bstole", r"\btheft", r"\brob", r"\bburgl", r"\btamper",
    r"\bpoint(?:s|ed|ing)? (?:\w+ ){0,3}at\b", r"\bjab", r"\bpok(?:e|es|ed|ing)\b", r"\bchas",
    # ...or a red about something else (a theft, a getaway, a way in) where a tool is only one of the things seen
    r"\bload", r"\bgrab", r"\bsnatch", r"\btak(?:e|es|ing|en)\b", r"\btook\b", r"\bremov", r"\bloot",
    r"\bcarr\w* (?:\w+ ){0,5}(?:away|off|out of)\b", r"\bdriv\w* (?:\w+ ){0,2}away", r"\bflee", r"\bfled",
    r"\bran away", r"\brun(?:s|ning)? (?:away|off)", r"\benter", r"\bclimb", r"\bsneak",
    r"\b(?:us(?:e|es|ed|ing)|work(?:s|ed|ing)?) (?:\w+ ){0,5}on (?:the |a |an |its |his |her )?(?:\w+ )?"
    r"(?:doors?|windows?|locks?|cars?|vehicles?|gates?|shutters?|handles?|motorcycles?|bikes?)",
    r"מניף", r"הניף", r"מכה", r"מכים", r"היכה", r"הכה", r"חובט", r"תוקף", r"תקיפה", r"מאיים", r"איום", r"אלימות",
    r"שובר", r"שבר", r"מנפץ", r"ניפץ", r"פורץ", r"פריצה", r"לפרוץ", r"חותך", r"זורק", r"זרק", r"גונב", r"גניבה",
    r"משחית", r"ונדליזם", r"רודף", r"מכוון",
    r"מעמיס", r"העמיס", r"חוטף", r"חטף", r"לוקח", r"לקח", r"בורח", r"ברח", r"נכנס", r"מטפס", r"מתגנב",
])

# Clear serious things go out red at once, even when a verify word is there too.
CLEAR = _any([
    r"\bfire\b", r"\bflames?\b", r"\bsmoke\b", r"\blying motionless", r"\bunconscious", r"\bnot moving on the ground",
    r"\bclimb(?:s|ed|ing)? (?:in|into|through)\b", r"\bclimb(?:s|ed|ing)? (?:over|up) (?:the |a )?(?:fence|wall)",
    r"\bforc(?:e|es|ed|ing) (?:open )?(?:the |a )?(?:front |back |house |main )?(?:door|window|entry)",
    r"\bforced entry", r"\b(?:break|breaks|breaking|broke) into (?:the )?(?:house|home|building)", r"\bburglar",
    r"\b(?:break|breaks|breaking|broke|smash(?:es|ed|ing)?) (?:the |a )?(?:front |back |house |main )(?:door|window)",
    r"שריפה", r"(?<![א-ת])[בוה]?ה?אש(?![א-ת])", r"עשן", r"להבות", r"שוכב ללא תנועה", r"מחוסר הכרה", r"מטפס פנימה", r"נכנס דרך החלון",
    r"פורץ לבית", r"פריצה לבית", r"פורץ את הדלת", r"שובר את הדלת", r"פורץ דלת", r"פורץ חלון",
])

VERIFY_QUESTIONS = {
    "weapon": load("verify_question_weapon.prompt"),
    "tool_weapon": load("verify_question_tool_weapon.prompt"),
    "vehicle": load("verify_question_vehicle.prompt"),
    "violence": load("verify_question_violence.prompt"),
    "person_down": load("verify_question_person_down.prompt"),
}


def clear_class(text: str) -> bool:
    """A clear serious thing (house break-in, climbing in, fire or smoke, a person lying motionless)."""
    return bool(CLEAR.search(str(text or "")))


def verify_classes(text: str, reason: Optional[str] = None) -> List[str]:
    """Every second-look class *text* points to, in ``VERIFY_ORDER``; [] when it needs none (no verify word, or a
    clear class that goes out at once).

    ``tool_weapon`` only when the tool is the red's reason and nothing was done with it: named in *reason* (the
    model's why + alert_reason; *text* when *reason* is None or empty) while *text* names no act (:data:`TOOL_IN_USE`).
    A tool only in the summary of a red that is about something else (a theft, a break-in) is left alone.

    ``person_down`` the same way: a person lying, kneeling or crouching (:data:`PERSON_DOWN`) in *reason*, no other
    class, and *text* names nothing else (:data:`DOWN_ACT`: violence, a weapon, a break-in, a way in, hiding, night, a
    fall or an injury, a child), nor a thing (:data:`DOWN_THING`: a car, a bag) in what it says about the person on
    the ground (:func:`down_scope`) - another man by a car or with a bag does not count."""
    text = str(text or "")
    if clear_class(text):
        return []
    found = [name for name in VERIFY_ORDER if name in VERIFY_PATTERNS and name != "tool_weapon"
             and VERIFY_PATTERNS[name].search(text)]
    why = str(reason or "").strip() or text
    if VERIFY_PATTERNS["tool_weapon"].search(why) and not TOOL_IN_USE.search(text):
        found.append("tool_weapon")
    if not found and PERSON_DOWN.search(why) and not DOWN_ACT.search(text):
        # The why and the summary apart (the why's last clause must not run into the summary's first). A summary that
        # names no person down cannot say which person the why meant: all of it counts, as before.
        head, rest = (why, text[len(why):]) if reason and text.startswith(why) else ("", text)
        about = [down_scope(head)] if head else []
        about.append(down_scope(rest) if PERSON_DOWN.search(rest) else rest)
        if not DOWN_THING.search(" | ".join(about)):
            found.append("person_down")
    return [name for name in VERIFY_ORDER if name in found]


def verify_class(text: str) -> Optional[str]:
    """The main second look an escalation needs (``weapon``, ``tool_weapon``, ``vehicle``, ``violence`` or
    ``person_down``), or None."""
    found = verify_classes(text)
    return found[0] if found else None


# A tool's "no" names the tool by design ("a worker with a metal bar"), so its answer contradicts itself only when it
# names what the tool did to a person or a way in: threatening, hitting, swinging at, breaking a door / window / car.
# Words after a "not / no / without" are the model saying what it is NOT, so they are left out first.
TOOL_ACT = _any([
    r"\bthreat", r"\bmenac", r"\battack", r"\bassault", r"\bswing(?:s|ing)? (?:\w+ ){0,3}at\b",
    r"\b(?:strik(?:e|es|ing)|struck|hit(?:s|ting)?|beat(?:s|ing)?) (?:a |an |the |another )?(?:man|woman|person|someone|him|her|child|people)",
    r"(?:\bbreak|\bbroke|\bsmash|\bshatter|\bforc|\bpr(?:y|ies|ying|ied))\w* (?:\w+ ){0,4}(?:doors?|windows?|cars?|vehicles?|locks?|gates?|shutters?)\b",
    r"\bbreak(?:ing)?[- ]in\b",
    r"מאיים", r"איום", r"תוקף", r"מכה (?:אדם|גבר|אישה|ילד|אותו|אותה|את ה)", r"מניף (?:\S+ ){0,3}(?:על|לעבר|כלפי)",
    r"(?:פורץ|פריצה|לפרוץ|שובר|לשבור|מנפץ|לנפץ|כופה).{0,30}(?:דלת|חלון|רכב|מכונית|מנעול|שער|תריס)",
])
# A person-down "no" names the ground by design ("a worker kneeling on the pavement laying pavers"), so it
# contradicts itself only when it names harm: hurt, collapsed, unconscious, a fall, being attacked or held down.
DOWN_HARM = _any([
    r"\bhurt", r"\binjur", r"\bcollaps", r"\bunconscious", r"\bfaint", r"\bpass(?:es|ed)? out", r"\bfell\b",
    r"\bfall(?:s|en)?\b", r"\bmotionless", r"\bunresponsive", r"\bnot moving", r"\bblood", r"\bbleed", r"\bin pain",
    r"\battack", r"\bassault", r"\bbeat", r"\bhit(?:s|ting)? (?:\w+ ){0,2}(?:man|woman|person|him|her)", r"\bkick",
    r"\bheld down", r"\bhold(?:s|ing)? (?:\w+ ){0,2}down", r"\bpinned", r"\brestrain", r"\bfight", r"\bstruggl",
    r"פצוע", r"נפצע", r"התמוטט", r"מחוסר הכרה", r"התעלף", r"נפל", r"ללא תנועה", r"לא זז", r"(?<![א-ת])ה?דם(?![א-ת])",
    r"מותקף", r"תוקף",
    r"מכה", r"מוחזק", r"מחזיקים אותו",
])
_NEGATED = _any([r"\b(?:not|no|without|never|nobody|none|isn'?t|aren'?t)\b[^,.;:]*",
                 r"(?<![א-ת])(?:לא|אין|ללא|בלי)(?![א-ת])[^,.;:]*"])


def answer_names(classes: Sequence[str], what_it_is: str) -> bool:
    """Does a second look's "no" name, in its own words, one of the *classes* it was asked about? Then it
    contradicts itself and the red stays ("not confirmed: a physical altercation", eval_set_v2 Abuse004). For
    ``tool_weapon`` naming the tool is not enough (:data:`TOOL_ACT`), for ``person_down`` naming the ground is not
    enough (:data:`DOWN_HARM`)."""
    what = str(what_it_is or "")
    own = {"tool_weapon": TOOL_ACT, "person_down": DOWN_HARM}
    for name in classes:
        if name not in own:
            if name in verify_classes(what):
                return True
            continue
        kept = _NEGATED.sub(" ", what)
        if (own[name].search(kept) or clear_class(kept)
                or any(c not in own for c in verify_classes(kept))):
            return True
    return False


def verify_question(classes: Sequence[str]) -> str:
    """One question for the one verification call; several classes are asked together ("is any of these so?")."""
    questions = [VERIFY_QUESTIONS[c] for c in classes if c in VERIFY_QUESTIONS]
    if len(questions) <= 1:
        return questions[0] if questions else ""
    return load("verify_any.prompt") + " " + " ".join(f"({i}) {q}" for i, q in enumerate(questions, 1))


def verify_prompt(question: str, frames: int) -> str:
    """The second look's question with its strict JSON answer."""
    return render("verify_second_look.prompt", frames=frames, question=question)
