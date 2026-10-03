"""Every sentence the box itself writes to the owner, in English, Hebrew and Arabic.

The model writes the facts of an answer in the owner's language; everything the
code writes - action confirmations, errors, announcements - comes from here, so
a confirmation is never invented by the model and never in the wrong language.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

LANGS = ("en", "he", "ar")
DEFAULT_LANG = "en"
SUPPORTED_LANGS = ("en", "he")   # what the assistant speaks today; the Arabic sentences wait for later
LANGUAGE_NAMES = {"en": "English", "he": "Hebrew", "ar": "Arabic"}

TEMPLATES: Dict[str, Dict[str, str]] = {
    "alert_types_changed": {"en": "✓ {camera} alerts on: {old} → {new}", "he": "✓ {camera} מתריעה על: {old} ← {new}",
                            "ar": "✓ {camera} تنبه على: {old} ← {new}"},
    "alert_types_changed_house": {"en": "✓ The house alerts on: {old} → {new}", "he": "✓ הבית מתריע על: {old} ← {new}",
                                  "ar": "✓ المنزل ينبه على: {old} ← {new}"},
    "sensitivity_changed": {"en": "✓ {camera}: how sure before alerting - {changes}",
                            "he": "✓ {camera}: כמה בטוח הזיהוי צריך להיות לפני התראה - {changes}",
                            "ar": "✓ {camera}: مدى التأكد المطلوب قبل التنبيه - {changes}"},
    "sensitivity_change": {"en": "{kind} {old}% → {new}%", "he": "{kind} {old}% ← {new}%",
                           "ar": "{kind} {old}% ← {new}%"},
    "reason_which_camera": {"en": "say which camera, or 'house'", "he": "ציין איזו מצלמה, או 'הבית'",
                            "ar": "حدد أي كاميرا، أو 'المنزل'"},
    "undo_what_alert_types": {"en": "the alert types of {where}", "he": "סוגי ההתראות של {where}",
                              "ar": "أنواع التنبيهات في {where}"},
    "undo_what_sensitivity": {"en": "the sensitivity of {where}", "he": "הרגישות של {where}",
                              "ar": "الحساسية في {where}"},
    "type_person": {"en": "people", "he": "אנשים", "ar": "أشخاص"},
    "type_vehicle": {"en": "vehicles", "he": "רכבים", "ar": "مركبات"},
    "type_animal": {"en": "animals", "he": "בעלי חיים", "ar": "حيوانات"},
    "the_house": {"en": "the house", "he": "הבית", "ar": "المنزل"},
    "what_set_alert_types": {"en": "Changing the alert types", "he": "שינוי סוגי ההתראות", "ar": "تغيير أنواع التنبيهات"},
    "what_set_sensitivity": {"en": "Changing the sensitivity", "he": "שינוי הרגישות", "ar": "تغيير الحساسية"},
    "undo_button": {"en": "↩ Undo", "he": "↩ ביטול", "ar": "↩ تراجع"},
    "nothing_to_undo": {"en": "There is nothing left to undo here.", "he": "אין כאן מה לבטל.",
                        "ar": "لا يوجد ما يمكن التراجع عنه هنا."},
    "undo_changed_since": {"en": "Changed since - nothing to undo for {what}.",
                           "he": "השתנה מאז - אין מה לבטל עבור {what}.",
                           "ar": "تغيّر منذ ذلك الحين - لا يوجد ما يمكن التراجع عنه في {what}."},
    "undo_failed": {"en": "✗ Could not undo {what}: {reason}",
                    "he": "✗ לא הצלחתי לבטל את {what}: {reason}",
                    "ar": "✗ تعذّر التراجع عن {what}: {reason}"},
    "undo_what_pause_all": {"en": "the pause of all alerts", "he": "השתקת כל ההתראות",
                            "ar": "إيقاف جميع التنبيهات"},
    "undo_what_pause_camera": {"en": "the pause of alerts from {camera}", "he": "השתקת ההתראות מ-{camera}",
                               "ar": "إيقاف تنبيهات {camera}"},
    "undo_what_camera": {"en": "the change to {camera}", "he": "השינוי במצלמה {camera}",
                         "ar": "تغيير الكاميرا {camera}"},
    "undo_camera_still_paused": {"en": "✓ Undone, but alerts from {camera} stay paused until {until} by another pause.",
                                 "he": "✓ בוטל, אבל ההתראות מ-{camera} נשארות מושתקות עד {until} בגלל השתקה אחרת.",
                                 "ar": "✓ تم التراجع، لكن تنبيهات {camera} تبقى متوقفة حتى {until} بسبب إيقاف آخر."},
    "undo_all_still_paused": {"en": "✓ Undone, but alerts stay paused until {until} by another pause.",
                              "he": "✓ בוטל, אבל ההתראות נשארות מושתקות עד {until} בגלל השתקה אחרת.",
                              "ar": "✓ تم التراجع، لكن التنبيهات تبقى متوقفة حتى {until} بسبب إيقاف آخر."},
    "undo_back_on_except": {"en": "✓ Alerts are back on, except {pauses} (paused separately).",
                            "he": "✓ ההתראות חזרו לפעול, חוץ מ-{pauses} (מושתקות בנפרד).",
                            "ar": "✓ عادت التنبيهات للعمل، باستثناء {pauses} (متوقفة بشكل منفصل)."},
    "pause_until_item": {"en": "{camera} until {until}", "he": "{camera} עד {until}", "ar": "{camera} حتى {until}"},
    "setting_changed": {"en": "✓ {setting}: {old} → {new}",
                        "he": "✓ {setting}: {old} ← {new}",
                        "ar": "✓ {setting}: {old} ← {new}"},
    "setting_alert_hours": {"en": "Alert hours", "he": "שעות ההתראות", "ar": "ساعات التنبيه"},
    "setting_cooldown_minutes": {"en": "Time between alerts", "he": "זמן בין התראות", "ar": "الوقت بين التنبيهات"},
    "setting_sensitivity": {"en": "Detector sensitivity", "he": "רגישות הזיהוי", "ar": "حساسية الكشف"},
    "setting_language": {"en": "Box language", "he": "שפת המערכת", "ar": "لغة النظام"},
    "what_change_setting": {"en": "Changing the setting", "he": "שינוי ההגדרה", "ar": "تغيير الإعداد"},
    # -- receipts that succeeded ------------------------------------------------
    "sent_video": {"en": "✓ Video sent ({bounds})",
                   "he": "✓ הסרטון נשלח ({bounds})",
                   "ar": "✓ تم إرسال الفيديو ({bounds})"},
    "sent_photo": {"en": "✓ Photo sent ({camera})",
                   "he": "✓ התמונה נשלחה ({camera})",
                   "ar": "✓ تم إرسال الصورة ({camera})"},
    "sent_live_clip": {"en": "✓ New {seconds}-second video from {camera} sent",
                       "he": "✓ סרטון חדש של {seconds} שניות מ-{camera} נשלח",
                       "ar": "✓ تم إرسال فيديو جديد مدته {seconds} ثانية من {camera}"},
    "paused_all": {"en": "✓ Alerts paused until {until}. The cameras keep watching.",
                   "he": "✓ ההתראות מושתקות עד {until}. המצלמות ממשיכות לצפות.",
                   "ar": "✓ تم إيقاف التنبيهات حتى {until}. الكاميرات تواصل المراقبة."},
    "paused_camera": {"en": "✓ Alerts from {camera} paused until {until}. The camera keeps watching.",
                      "he": "✓ ההתראות מ-{camera} מושתקות עד {until}. המצלמה ממשיכה לצפות.",
                      "ar": "✓ تم إيقاف تنبيهات {camera} حتى {until}. الكاميرا تواصل المراقبة."},
    "resumed_all": {"en": "✓ Alerts are back on.",
                    "he": "✓ ההתראות חזרו לפעול.",
                    "ar": "✓ عادت التنبيهات للعمل."},
    "resumed_camera": {"en": "✓ Alerts from {camera} are back on.",
                       "he": "✓ ההתראות מ-{camera} חזרו לפעול.",
                       "ar": "✓ عادت تنبيهات {camera} للعمل."},
    "camera_off_requested": {"en": "⏳ Turning {camera} off - the box restarts for a moment.",
                             "he": "⏳ מכבה את {camera} - הקופסה מופעלת מחדש לרגע.",
                             "ar": "⏳ جارٍ إيقاف {camera} - يُعاد تشغيل الجهاز للحظة."},
    "camera_on_requested": {"en": "⏳ Turning {camera} on - the box restarts for a moment.",
                            "he": "⏳ מדליק את {camera} - הקופסה מופעלת מחדש לרגע.",
                            "ar": "⏳ جارٍ تشغيل {camera} - يُعاد تشغيل الجهاز للحظة."},
    "camera_off_done": {"en": "✓ {camera} is off.",
                        "he": "✓ {camera} כבויה.",
                        "ar": "✓ {camera} متوقفة."},
    "camera_on_done": {"en": "✓ {camera} is on.",
                       "he": "✓ {camera} פועלת.",
                       "ar": "✓ {camera} تعمل."},
    "verdict_saved": {"en": "✓ Noted: {verdict}",
                      "he": "✓ נרשם: {verdict}",
                      "ar": "✓ تم التسجيل: {verdict}"},
    "alias_saved": {"en": "✓ \"{alias}\" now means {camera}",
                    "he": "✓ \"{alias}\" מעכשיו זה {camera}",
                    "ar": "✓ \"{alias}\" تعني الآن {camera}"},
    "failed": {"en": "✗ {what} could not be done: {reason}",
               "he": "✗ {what} לא הצליח: {reason}",
               "ar": "✗ تعذّر {what}: {reason}"},
    # -- what failed ------------------------------------------------------------
    "what_send_media": {"en": "Sending the video", "he": "שליחת הסרטון", "ar": "إرسال الفيديو"},
    "what_check_camera": {"en": "Taking a live photo", "he": "צילום תמונה חיה", "ar": "التقاط صورة مباشرة"},
    "what_record_clip": {"en": "Recording a new video", "he": "הקלטת סרטון חדש", "ar": "تسجيل فيديو جديد"},
    "what_pause_alerts": {"en": "Pausing alerts", "he": "השתקת ההתראות", "ar": "إيقاف التنبيهات"},
    "what_resume_alerts": {"en": "Turning alerts back on", "he": "החזרת ההתראות", "ar": "إعادة تشغيل التنبيهات"},
    "what_set_camera_active": {"en": "Changing the camera", "he": "שינוי מצב המצלמה", "ar": "تغيير حالة الكاميرا"},
    "what_record_verdict": {"en": "Saving your answer", "he": "שמירת התשובה", "ar": "حفظ إجابتك"},
    "what_set_alias": {"en": "Saving the camera name", "he": "שמירת שם המצלמה", "ar": "حفظ اسم الكاميرا"},
    # -- why it failed ----------------------------------------------------------
    "reason_not_on_box": {"en": "the video is no longer on the box (older than {days} days)",
                          "he": "הסרטון כבר לא שמור (ישן מ-{days} ימים)",
                          "ar": "الفيديو لم يعد محفوظًا (أقدم من {days} يومًا)"},
    "reason_telegram": {"en": "Telegram did not accept it", "he": "טלגרם לא קיבל אותו", "ar": "لم يقبله تيليجرام"},
    "reason_camera_unknown": {"en": "there is no camera by that name", "he": "אין מצלמה בשם הזה",
                              "ar": "لا توجد كاميرا بهذا الاسم"},
    "reason_camera_off": {"en": "that camera is turned off", "he": "המצלמה הזאת כבויה", "ar": "هذه الكاميرا متوقفة"},
    "reason_camera_offline": {"en": "the camera is not answering", "he": "המצלמה לא עונה", "ar": "الكاميرا لا تستجيب"},
    "reason_busy": {"en": "that camera is already recording", "he": "המצלמה כבר מקליטה", "ar": "الكاميرا تسجل بالفعل"},
    "reason_too_many": {"en": "only 3 videos can be sent at once", "he": "אפשר לשלוח עד 3 סרטונים בבת אחת",
                        "ar": "يمكن إرسال 3 مقاطع فيديو فقط في المرة الواحدة"},
    "reason_last_camera": {"en": "at least one camera must stay on", "he": "לפחות מצלמה אחת חייבת להישאר פעילה",
                           "ar": "يجب أن تبقى كاميرا واحدة على الأقل قيد التشغيل"},
    "reason_error": {"en": "something went wrong on the box", "he": "משהו השתבש בקופסה", "ar": "حدث خطأ في الجهاز"},
    # -- verdicts ---------------------------------------------------------------
    "verdict_true_alert": {"en": "a real alert", "he": "התראה אמיתית", "ar": "تنبيه حقيقي"},
    "verdict_false_alarm": {"en": "a false alarm", "he": "התראת שווא", "ar": "إنذار كاذب"},
    "verdict_real_but_wrong": {"en": "real, but described wrongly", "he": "אמיתי, אבל התיאור שגוי",
                               "ar": "حقيقي لكن الوصف خاطئ"},
    "verdict_expected": {"en": "expected activity", "he": "פעילות צפויה", "ar": "نشاط متوقع"},
    "verdict_missed_event": {"en": "an event the box missed", "he": "אירוע שהקופסה פספסה", "ar": "حدث فاته الجهاز"},
    # -- general ----------------------------------------------------------------
    "nothing_done": {"en": "I did not do anything yet - please tell me again what you need.",
                     "he": "עדיין לא עשיתי כלום - תכתוב לי שוב מה צריך.",
                     "ar": "لم أقم بأي إجراء بعد - أخبرني مرة أخرى بما تحتاجه."},
    "unavailable": {"en": "I couldn't work on that right now, but your message was saved.",
                    "he": "לא הצלחתי לטפל בזה כרגע, אבל ההודעה שלך נשמרה.",
                    "ar": "لم أتمكن من معالجة ذلك الآن، لكن رسالتك حُفظت."},
    "guard_started": {"en": "🛡️ Guarding started{until}. {live} of {total} cameras live.",
                      "he": "🛡️ השמירה התחילה{until}. {live} מתוך {total} מצלמות פעילות.",
                      "ar": "🛡️ بدأت الحراسة{until}. {live} من {total} كاميرات تعمل."},
    "guard_until": {"en": ", until {time}", "he": ", עד {time}", "ar": "، حتى {time}"},
    "guard_ended": {"en": "💬 Guarding ended. The box keeps a quiet log{next}.",
                    "he": "💬 השמירה הסתיימה. הקופסה ממשיכה לתעד בשקט{next}.",
                    "ar": "💬 انتهت الحراسة. يواصل الجهاز التسجيل بهدوء{next}."},
    "guard_ended_no_log": {"en": "💬 Guarding ended. Nothing is recorded{next}.",
                           "he": "💬 השמירה הסתיימה. שום דבר לא מתועד{next}.",
                           "ar": "💬 انتهت الحراسة. لا يتم تسجيل أي شيء{next}."},
    "guard_next": {"en": " until {time}", "he": " עד {time}", "ar": " حتى {time}"},
}


def t(key: str, lang: str, **values: object) -> str:
    """The sentence *key* in *lang* (English when that language has none), filled with *values*."""
    entry = TEMPLATES[key]
    return (entry.get(lang) or entry[DEFAULT_LANG]).format(**values)


_HEBREW = re.compile(r"[א-תװ-ײ]")
_ARABIC = re.compile(r"[ء-غف-يٱ-ۓۺ-ۼ]")
_LATIN = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> Optional[str]:
    """``he``/``ar``/``en`` from the letters of *text*; None when it has no letters ("8", "👍")."""
    he = len(_HEBREW.findall(text or ""))
    ar = len(_ARABIC.findall(text or ""))
    en = len(_LATIN.findall(text or ""))
    if he == ar == en == 0:
        return None
    # Hebrew or Arabic win over Latin letters of equal weight: camera names are Latin.
    if he and he >= ar and he * 2 >= en:
        return "he"
    if ar and ar > he and ar * 2 >= en:
        return "ar"
    return "en"


_OVERRIDES = (
    (re.compile(r"\b(?:in|answer in|reply in|speak)\s+english\b", re.IGNORECASE), "en"),
    (re.compile(r"\b(?:in|answer in|reply in|speak)\s+hebrew\b", re.IGNORECASE), "he"),
    (re.compile(r"\b(?:in|answer in|reply in|speak)\s+arabic\b", re.IGNORECASE), "ar"),
    (re.compile("באנגלית"), "en"),
    (re.compile("בעברית"), "he"),
    (re.compile("בערבית"), "ar"),
    (re.compile("بالإنجليزية|بالانجليزية"), "en"),
    (re.compile("بالعبرية"), "he"),
    (re.compile("بالعربية"), "ar"),
)


def language_override(text: str) -> Optional[str]:
    """The language the owner explicitly asked for ("answer in English", "בעברית"), or None."""
    for pattern, lang in _OVERRIDES:
        if pattern.search(text or ""):
            return lang
    return None
