"""House notes constrain labels, never admission. All providers and delivery are local fakes."""
import builtins
import json
import os
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest import mock

from home_guard_project.box import inference as inf
from home_guard_project.box import telegram_agent as agent
from home_guard_project.box.brain.i18n import t


CAM = "back_east"
TS = datetime(2026, 10, 3, 12).timestamp()
LABEL_CASES = (*inf.LABELS, "invalid", None)  # None means the field is missing.


def note(**changes):
    return dict(dict(id="F12", camera=CAM, area="pergola", hours=["07:00", "17:00"],
                     kind="people", text="people at the pergola are workers", effect="lower",
                     source="owner", added_by="1", added_at="2026-10-01T00:00:00",
                     expires_at=None), **changes)


def answer(**changes):
    return dict(dict(summary="People wait at the pergola.", raw_label="suspicious", label="normal",
                     applied_fact_id="F12", serious_behaviour=False, people=2, vehicle_moving=False,
                     animals=0, why="", summary_owner=""), **changes)


class DecisionTests(unittest.TestCase):
    def decide(self, raw="suspicious", label="normal", facts=None, fact_id="F12", serious=False,
               kinds=("people",), ts=TS):
        return inf.final_label(raw, label, fact_id, serious, [note()] if facts is None else facts,
                               kinds, ts, CAM)

    def test_valid_lower(self):
        self.assertEqual(self.decide(), ("normal", True, note()))

    def test_escalation_never_softens(self):
        for label in inf.LABELS:
            self.assertEqual(self.decide(raw="escalation", label=label), ("escalation", False, None))

    def test_serious_behaviour_never_softens(self):
        self.assertEqual(self.decide(serious=True), ("suspicious", False, None))

    def test_missing_or_non_boolean_serious_never_softens(self):
        for value in (None, "false", 0):
            self.assertEqual(self.decide(serious=value), ("suspicious", False, None))

    def test_both_effects_require_all_checks(self):
        for effect, raw, label in (("lower", "suspicious", "normal"), ("raise", "normal", "normal")):
            for changes in ({"camera": "front"}, {"hours": ["17:00", "19:00"]},
                            {"expires_at": "2026-10-02T00:00:00"}, {"kind": "vehicles"}):
                with self.subTest(effect=effect, changes=changes):
                    self.assertEqual(self.decide(raw, label, [note(effect=effect, **changes)]),
                                     (raw, False, None))
            for facts, fact_id in (([note(effect=effect)], "F99"), ([note(effect=effect)], "")):
                self.assertEqual(self.decide(raw, label, facts, fact_id), (raw, False, None))

    def test_raise_is_only_normal_to_suspicious(self):
        facts = [note(effect="raise")]
        self.assertEqual(self.decide("normal", "normal", facts), ("suspicious", False, facts[0]))
        for raw, label in (("normal", "escalation"), ("suspicious", "escalation"),
                           ("suspicious", "normal")):
            base = max((raw, label), key=inf.LABELS.index)
            self.assertEqual(self.decide(raw, label, facts), (base, False, None))

    def test_invalid_labels_are_empty_and_loud(self):
        for raw, label in (("bad", "bad"), ("", "bad"), ("", "")):
            self.assertEqual(self.decide(raw, label), ("", False, None))
            self.assertFalse(inf.is_silent(self.decide(raw, label)[0]))

    def test_hours_boundaries_and_midnight(self):
        for hours, hour, minute, valid in ((["07:00", "17:00"], 7, 0, True),
                                         (["07:00", "17:00"], 17, 0, False),
                                         (["22:30", "06:00"], 22, 29, False),
                                         (["22:30", "06:00"], 23, 0, True),
                                         (["22:30", "06:00"], 5, 59, True),
                                         (None, 3, 0, True)):
            result = self.decide(facts=[note(hours=hours)], ts=datetime(2026, 10, 3, hour, minute).timestamp())
            self.assertEqual(result[1], valid)

    def test_no_fact_cannot_change_label(self):
        for raw in LABEL_CASES:
            for label in LABEL_CASES:
                with self.subTest(raw=raw, label=label):
                    self.assertEqual(self.decide(raw, label, []),
                                     (label if label in inf.LABELS else "", False, None))

    def test_notes_label_matrix(self):
        for raw in LABEL_CASES:
            for label in LABEL_CASES:
                valid = [value for value in (raw, label) if value in inf.LABELS]
                base = max(valid, key=inf.LABELS.index) if valid else ""
                for effect, fact_id, serious in (("lower", "F99", False), ("lower", "F12", False),
                                                 ("lower", "F12", True), ("raise", "F12", False)):
                    with self.subTest(raw=raw, label=label, effect=effect, fact_id=fact_id, serious=serious):
                        fact = note(effect=effect)
                        lower = base == "suspicious" and effect == "lower" and fact_id == "F12" and not serious
                        raise_ = base == "normal" and effect == "raise"
                        expected = "normal" if lower else "suspicious" if raise_ else base
                        self.assertEqual(self.decide(raw, label, [fact], fact_id, serious),
                                         (expected, lower, fact if lower or raise_ else None))


class PromptTests(unittest.TestCase):
    def test_schema_requires_every_property(self):
        props = inf.VLM_SCHEMA["properties"]
        self.assertEqual(set(inf.VLM_SCHEMA["required"]), set(props))
        self.assertEqual(props["raw_label"]["enum"], list(inf.LABELS))
        self.assertEqual(props["applied_fact_id"], {"type": "string"})
        self.assertEqual(props["serious_behaviour"], {"type": "boolean"})
        self.assertTrue(inf.PROMPT_VERSION.endswith("-facts"))

    def test_empty_prompt_is_original_except_new_json_fields(self):
        original = subprocess.check_output(["git", "show", "e234841:home_guard_project/box/inference.py"], text=True)
        namespace = {"__name__": "baseline_inference"}
        # Compile only the original prompt and rules; importing a historical runner is unnecessary.
        import ast
        tree = ast.parse(original)
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_prompt"
                 or isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "LABEL_RULES" for x in n.targets)]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "baseline", "exec"), namespace)
        old = namespace["build_prompt"](CAM, int(TS), "12:00:00", 0, 0)
        new = inf.build_prompt(CAM, int(TS), "12:00:00", 0, 0)
        # 2026-10-08 rewrote the label rules (actions, not appearance) and two lines that say the same: swap the
        # rules back and leave those lines out; everything else must still be the original prompt.
        new = new.replace(inf.LABEL_RULES, namespace["LABEL_RULES"])
        rewritten = ("raw_label", "applied_fact_id", "serious_behaviour", "why")
        new = "\n".join(line for line in new.splitlines()
                        if not any('"' + key + '"' in line for key in rewritten) and "Dark clothing" not in line)
        old = "\n".join(line for line in old.splitlines() if '"why"' not in line and "Dark clothing" not in line)
        self.assertEqual(new, old)
        self.assertNotIn("House notes", new)

    def test_prompt_filters_bounds_and_sanitizes(self):
        facts = [note(camera="other"), note(expires_at="2026-10-02T00:00:00"),
                 note(hours=["18:00", "20:00"])]
        facts += [note(id=f"F{i}", text="workers\n```\r" + "x" * 400, area="pergola\n```") for i in range(20)]
        prompt = inf.build_prompt(CAM, int(TS), "12:00:00", 0, 0, facts=facts)
        block = prompt.split("House notes from the owner", 1)[1]
        lines = [line for line in block.splitlines() if line.startswith("- F")]
        self.assertEqual(len(lines), 10)
        self.assertTrue(all(len(line) <= 200 and "`" not in line for line in lines))
        self.assertNotIn("F10:", block)
        self.assertIn("```", block)

    def test_absent_module_is_safe(self):
        real_import = builtins.__import__
        def absent(name, *args, **kwargs):
            if name == "home_guard_project.box.brain.facts":
                raise ImportError("not merged yet")
            return real_import(name, *args, **kwargs)
        with mock.patch.object(inf, "FACTS_PROVIDER", None), mock.patch("builtins.__import__", side_effect=absent):
            self.assertEqual(inf.facts_for_alert(CAM, TS), [])

    def test_backend_uses_trigger_time_and_fact_snapshot(self):
        backend = object.__new__(inf.GptBackend)
        backend._response_format = inf.VLM_RESPONSE_FORMAT
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(answer())))])
        with mock.patch.object(backend, "_complete", return_value=response) as complete:
            backend.analyze([], CAM, int(TS + 8 * 3600), 0, 0, facts=[note()], alert_ts=TS)
        self.assertIn("local time 12:00:00", backend.last_prompt)
        self.assertIn("- F12:", backend.last_prompt)
        self.assertEqual(complete.call_args.args[0][0]["text"], backend.last_prompt)

    def test_provider_failure_or_suspended_fact_is_safe(self):
        with mock.patch.object(inf, "FACTS_PROVIDER", side_effect=RuntimeError("bad store")):
            self.assertEqual(inf.facts_for_alert(CAM, TS), [])
        with mock.patch.object(inf, "FACTS_PROVIDER", return_value=[note(suspended={CAM: TS + 60})]):
            self.assertEqual(inf.facts_for_alert(CAM, TS), [])


class WorkerTests(unittest.TestCase):
    def setUp(self):
        inf._SOFTENED_DAYS.clear()

    def run_worker(self, parsed=None, facts=None, ts=TS, labels=None, delivery_ok=True, module_absent=False):
        parsed = answer() if parsed is None else parsed
        backend = mock.Mock()
        backend.model_name = "fake"
        backend.last_prompt = "local test"
        backend.analyze.return_value = (json.dumps(parsed), parsed)
        assistant = mock.Mock()
        assistant.is_muted.return_value = False
        assistant.send_alert.return_value = {"sent": delivery_ok}
        job = inf.AlertJob(camera=CAM, stem=f"{CAM}_{int(ts)}_alert", ts=ts,
                           labels=["person"] if labels is None else labels)
        provider = mock.Mock(return_value=[note()] if facts is None else facts)
        real_import = builtins.__import__
        def local_import(name, *args, **kwargs):
            if module_absent and name == "home_guard_project.box.brain.facts":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)
        with mock.patch.object(inf, "FACTS_PROVIDER", None if module_absent else provider), \
                mock.patch("builtins.__import__", side_effect=local_import), \
                mock.patch.object(inf, "owner_language", return_value="en"), \
                mock.patch.object(inf, "_jpegs", return_value=[]):
            inf._worker(backend, {"alert_channel": "telegram"}, {}, inf.AlertSettings(), CAM, [], assistant, job)
        self.assertTrue(job.ready.is_set())
        return job, assistant, backend, provider

    def test_lower_records_text_and_sound_each_day(self):
        for offset, silent in ((0, False), (1, True), (86400, False), (86401, True)):
            job, assistant, backend, provider = self.run_worker(ts=TS + offset)
            self.assertEqual(job.alert["final_label"], "normal")
            self.assertEqual(job.alert["raw_label"], "suspicious")
            self.assertEqual(job.alert["label"], "normal")
            self.assertTrue(job.alert["softened"])
            self.assertEqual(job.alert["fact_effect"], "lower")
            self.assertEqual(job.alert["applied_fact_id"], "F12")
            self.assertIs(job.alert["serious_behaviour"], False)
            self.assertEqual(job.alert["silent"], silent)
            text = assistant.send_alert.call_args.args[1]
            # The owner reads the camera's name (no family name here: the id with spaces), never the id.
            self.assertEqual(text, f"🟢 {CAM.replace('_', ' ')}: People wait at the pergola. Normal: {note()['text']} (your note, 07:00-17:00)")
            self.assertTrue(assistant.send_alert.call_args.args[0]["softened"])
            provider.assert_called_once_with(CAM, TS + offset)
            self.assertEqual(backend.analyze.call_args.kwargs["facts"], [note()])

    def test_failed_delivery_does_not_consume_first_sound(self):
        self.run_worker(delivery_ok=False)
        job, _, _, _ = self.run_worker(ts=TS + 1)
        self.assertFalse(job.alert["silent"])

    def test_sound_cache_prunes_400_days_under_lock(self):
        for day in range(400):
            ts = (datetime.fromtimestamp(TS) + timedelta(days=day)).timestamp()
            job, _, _, _ = self.run_worker(ts=ts)
            self.assertFalse(job.alert["silent"])
            self.assertEqual(set(inf._SOFTENED_DAYS), {("F12", datetime.fromtimestamp(ts).date().isoformat())})
        tomorrow = TS + 400 * 86400
        original_dispatch = inf.dispatch_alert
        dispatch_states = []
        def check_pruned(*args, **kwargs):
            # Assert outside the worker: it catches delivery exceptions.
            dispatch_states.append((inf._SOFTENED_LOCK.locked(), set(inf._SOFTENED_DAYS)))
            return original_dispatch(*args, **kwargs)
        with mock.patch.object(inf, "dispatch_alert", side_effect=check_pruned):
            self.run_worker(ts=tomorrow, delivery_ok=False)
        self.assertEqual(dispatch_states, [(True, set())])
        self.assertFalse(inf._SOFTENED_DAYS)

    def test_sound_cache_caps_500_entries_and_keeps_recent_repeats_silent(self):
        for number in range(501):
            fact_id = f"F{number}"
            job, _, _, _ = self.run_worker(answer(applied_fact_id=fact_id), [note(id=fact_id)])
            self.assertFalse(job.alert["silent"])
            self.assertLessEqual(len(inf._SOFTENED_DAYS), 500)
        self.assertEqual(len(inf._SOFTENED_DAYS), 500)
        job, _, _, _ = self.run_worker(answer(applied_fact_id="F500"), [note(id="F500")])
        self.assertTrue(job.alert["silent"])

    def test_raise_is_loud_and_names_note(self):
        job, assistant, _, _ = self.run_worker(answer(raw_label="normal", label="normal"), [note(effect="raise")])
        self.assertEqual(job.alert["final_label"], "suspicious")
        self.assertFalse(job.alert["softened"])
        self.assertFalse(job.alert["silent"])
        self.assertEqual(job.alert["fact_effect"], "raise")
        self.assertIn(f"Suspicious: {note()['text']} (your note, 07:00-17:00)", assistant.send_alert.call_args.args[1])

    def test_kind_uses_detector_not_model_counts(self):
        for kind, detector in (("people", "person"), ("vehicles", "truck"), ("animals", "dog")):
            job, _, _, _ = self.run_worker(facts=[note(kind=kind)], labels=[detector])
            self.assertTrue(job.alert["softened"])
        job, _, _, _ = self.run_worker(labels=["car"])
        self.assertEqual(job.alert["final_label"], "suspicious")

    def test_model_cannot_apply_eleventh_fact(self):
        job, _, backend, _ = self.run_worker(answer(applied_fact_id="F10"), [note(id=f"F{i}") for i in range(11)])
        self.assertEqual(job.alert["final_label"], "suspicious")
        self.assertEqual(len(backend.analyze.call_args.kwargs["facts"]), 10)

    def test_missing_raw_with_facts_uses_valid_label(self):
        parsed = answer()
        del parsed["raw_label"]
        job, _, _, _ = self.run_worker(parsed)
        self.assertEqual(job.alert["final_label"], "normal")
        self.assertTrue(job.alert["silent"])

    def test_review_case_no_facts_escalation_is_loud(self):
        for absent in (False, True):
            with self.subTest(module_absent=absent):
                job, assistant, _, _ = self.run_worker(
                    {"label": "escalation", "raw_label": "normal", "people": 1}, [], module_absent=absent)
                self.assertEqual(job.alert["label"], "escalation")
                self.assertEqual(job.alert["raw_label"], "normal")
                self.assertFalse(job.alert["silent"])
                self.assertEqual(job.alert["alert_command"], "[call_owner]")
                assistant.send_alert.assert_called_once()

    def test_no_notes_matrix_matches_e234841(self):
        for absent in (False, True):
            for raw in (*LABEL_CASES, {}, [], 42, True):
                for label in LABEL_CASES:
                    for people in (0, 1):
                        with self.subTest(absent=absent, raw=raw, label=label, people=people):
                            parsed = {"summary": "Scene", "people": people}
                            if raw is not None:
                                parsed["raw_label"] = raw
                            if label is not None:
                                parsed["label"] = label
                            job, assistant, backend, _ = self.run_worker(parsed, [], module_absent=absent)
                            expected = label if label in inf.LABELS else ""
                            self.assertEqual(job.alert["label"], expected)
                            self.assertEqual(job.alert["final_label"], expected)
                            self.assertFalse(job.alert["softened"])
                            self.assertEqual(job.false_positive, people == 0)
                            self.assertEqual(assistant.send_alert.called, people > 0)
                            self.assertNotIn("facts", backend.analyze.call_args.kwargs)
                            if people:
                                self.assertEqual(job.alert["silent"], expected == "normal")
                                self.assertEqual(job.alert["alert_command"], inf.LABEL_COMMANDS.get(expected, "[send_message]"))

    def test_notes_matrix_through_worker(self):
        for raw in LABEL_CASES:
            for label in LABEL_CASES:
                valid = [value for value in (raw, label) if value in inf.LABELS]
                base = max(valid, key=inf.LABELS.index) if valid else ""
                for effect, fact_id, serious in (("lower", "F99", False), ("lower", "F12", False),
                                                 ("lower", "F12", True), ("raise", "F12", False)):
                    with self.subTest(raw=raw, label=label, effect=effect, fact_id=fact_id, serious=serious):
                        inf._SOFTENED_DAYS.clear()
                        parsed = answer(applied_fact_id=fact_id, serious_behaviour=serious)
                        for key, value in (("raw_label", raw), ("label", label)):
                            if value is None:
                                del parsed[key]
                            else:
                                parsed[key] = value
                        lower = base == "suspicious" and effect == "lower" and fact_id == "F12" and not serious
                        raise_ = base == "normal" and effect == "raise"
                        expected = "normal" if lower else "suspicious" if raise_ else base
                        job, assistant, _, _ = self.run_worker(parsed, [note(effect=effect)])
                        self.assertEqual(job.alert["final_label"], expected)
                        self.assertEqual(job.alert["softened"], lower)
                        self.assertEqual(job.alert["fact_effect"], effect if lower or raise_ else "")
                        self.assertEqual(job.alert["silent"], expected == "normal" and not lower)
                        self.assertEqual(job.alert["alert_command"], inf.LABEL_COMMANDS.get(expected, "[send_message]"))
                        assistant.send_alert.assert_called_once()

    def test_no_facts_matches_baseline_admission_label_sound(self):
        # e234841: counts admit first; known labels are stored, only normal is silent.
        for label in (*inf.LABELS, "unknown", "", None):
            for people in (0, 1):
                with self.subTest(label=label, people=people):
                    parsed = {"summary": "Scene", "label": label, "people": people}
                    job, assistant, _, _ = self.run_worker(parsed, [])
                    valid = label if label in inf.LABELS else ""
                    self.assertEqual(job.false_positive, people == 0)
                    self.assertEqual(job.alert["label"], valid)
                    self.assertEqual(assistant.send_alert.called, people > 0)
                    if people:
                        self.assertEqual(job.alert["silent"], valid == "normal")
                        self.assertEqual(job.alert["alert_command"], inf.LABEL_COMMANDS.get(valid, "[send_message]"))

    def test_fact_never_changes_admission(self):
        job, assistant, _, _ = self.run_worker(answer(people=0))
        self.assertTrue(job.false_positive)
        assistant.send_alert.assert_not_called()

    def test_training_and_teacher_use_raw_label(self):
        job, _, _, _ = self.run_worker()
        self.assertEqual(job.teacher["parsed"]["label"], "suspicious")
        self.assertEqual(json.loads(job.teacher["raw"])["label"], "normal")  # original evidence retained
        with mock.patch("home_guard_project.box.alert_clips.write_alert_clip", return_value=None) as write:
            inf._save_clip(job, [], "production", "training")
        self.assertEqual(write.call_args_list[0].args[4]["label"], "normal")
        self.assertEqual(write.call_args_list[1].args[4]["label"], "suspicious")

    def test_alert_meta_and_archive_retain_fact_fields(self):
        import numpy as np
        from home_guard_project.box.alert_clips import encode_frame
        from home_guard_project.box.archive import load_records
        job, _, _, _ = self.run_worker()
        frames = [(TS + i / 5, encode_frame(np.zeros((48, 64, 3), dtype=np.uint8))) for i in range(5)]
        with tempfile.TemporaryDirectory() as root, \
                mock.patch("home_guard_project.box.alert_clips._to_h264", return_value=False):
            production, training = os.path.join(root, "production"), os.path.join(root, "training")
            inf._save_clip(job, frames, production, training)
            paths = [os.path.join(d, n) for d, _, names in os.walk(production) for n in names if n.endswith(".meta.json")]
            self.assertEqual(len(paths), 1)
            with open(paths[0], encoding="utf-8") as saved:
                meta = json.load(saved)
            for key in ("raw_label", "label", "final_label", "applied_fact_id", "softened", "fact_effect", "serious_behaviour"):
                self.assertEqual(meta["alert"][key], job.alert[key])
            record, = load_records([production])
            self.assertTrue(record.softened)
            self.assertEqual(record.applied_fact_id, "F12")

    def test_paused_record_has_empty_fact_decision(self):
        job = inf.AlertJob(camera=CAM, stem="paused", ts=TS, labels=["person"])
        assistant = mock.Mock()
        assistant.is_muted.return_value = True
        backend = mock.Mock()
        inf._worker(backend, {}, {}, inf.AlertSettings(), CAM, [], assistant, job)
        backend.analyze.assert_not_called()
        for key, value in (("raw_label", ""), ("label", ""), ("final_label", ""), ("applied_fact_id", ""),
                           ("softened", False), ("fact_effect", ""), ("serious_behaviour", False)):
            self.assertEqual(job.alert[key], value)


class ButtonTests(unittest.TestCase):
    def test_fallback_delivery_preserves_note_sound_and_button(self):
        from home_guard_project.box import telegram_notify as notify
        cfg = SimpleNamespace(dry_run=False, enabled=True, chat_ids=["1"], bot_token="fake")
        for image in (None, b"jpg"):
            with mock.patch.object(notify, "load_telegram_config", return_value=cfg), \
                    mock.patch.object(notify, "_http_post", return_value={"ok": True}) as post, \
                    mock.patch.object(notify, "_http_post_multipart", return_value={"ok": True}) as photo:
                inf.dispatch_alert({"alert_channel": "telegram"}, {}, "[send_message]", "summary", "reason",
                                   image=image, alert={"alert_id": "a", "applied_fact_id": "F12", "softened": True},
                                   graded="Normal: workers (your note, all day)", silent=True)
            fields = (photo if image else post).call_args.args[2]
            self.assertEqual(fields["caption" if image else "text"], "Normal: workers (your note, all day)")
            self.assertEqual(fields["disable_notification"], "true")
            button = json.loads(fields["reply_markup"])["inline_keyboard"][0][0]
            self.assertEqual(button["callback_data"], "nt:F12:a")

    def keyboard(self, softened=True, fact_id="F12", alert_id="back_east_123_alert", lang="en"):
        cfg = SimpleNamespace(dry_run=False, enabled=True, chat_ids=["1"], bot_token="fake")
        post = mock.Mock(return_value={"ok": True, "result": {"message_id": 1}})
        agent.send_alert(cfg, mock.Mock(), {"alert_id": alert_id, "label": "normal", "softened": softened,
                                          "applied_fact_id": fact_id}, "local test", post=post, lang=lang)
        return json.loads(post.call_args.args[2]["reply_markup"])["inline_keyboard"]

    def test_not_them_only_on_softened_alert(self):
        buttons = [b for row in self.keyboard() for b in row]
        self.assertIn({"text": "Not them", "callback_data": "nt:F12:back_east_123_alert"}, buttons)
        self.assertFalse(any(b["callback_data"].startswith("nt:") for row in self.keyboard(False) for b in row))

    def test_callback_64_byte_limit(self):
        for alert_id, expected in (("a" * 57, True), ("a" * 58, False), ("א" * 29, False)):
            buttons = [b for row in self.keyboard(alert_id=alert_id) for b in row]
            self.assertEqual(any(b["callback_data"].startswith("nt:") for b in buttons), expected)

    def test_localized_note_and_button_keys(self):
        for lang in ("en", "he", "ar"):
            for key in ("house_fact_normal", "house_fact_suspicious", "house_fact_all_day", "house_fact_not_them"):
                self.assertNotEqual(t(key, lang, text="workers", hours="07:00-17:00"), key)
            self.assertTrue(any(b["text"] == t("house_fact_not_them", lang)
                                for row in self.keyboard(lang=lang) for b in row))
