"""The AI usage ledger (box/usage_ledger.py, 2026-10-09): one line per AI call, per agent and model, for the Admin.

Every call site is hooked; each test drives one with a stubbed response carrying usage and reads the line back from
the day's file. The contract the Admin reads (docs/contracts/usage_ledger.md) is pinned at the end.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import tempfile
import time
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import httpx  # noqa: F401  (imported before the TLS setup is patched)
import numpy as np
import openai  # noqa: F401

from home_guard_project.box import usage_ledger as ul

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONTRACT = os.path.join(ROOT, "docs", "contracts", "usage_ledger.md")
OPENROUTER = "https://openrouter.ai/api/v1/"
OPENAI = "https://api.openai.com/v1/"
EYE_JSON = json.dumps({"summary": "a person walks", "label": "normal", "people": 1, "vehicle_moving": False,
                       "animals": 0, "why": "", "summary_owner": ""})


def usage(prompt=1000, completion=50, cost=None, cached=0):
    u = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                        prompt_tokens_details=SimpleNamespace(cached_tokens=cached))
    if cost is not None:
        u.cost = cost
    return u


def answer(content, model="qwen/qwen3.5-9b", **kw):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None,
                                                                            refusal=None),
                                                    finish_reason="stop")],
                           usage=usage(**kw), model=model)


class FakeClient:
    """An OpenAI-shaped client: answers (or raises) each ``chat.completions.create`` in turn."""

    def __init__(self, base_url, *answers):
        self.base_url = base_url
        self.answers = list(answers)
        self.sent = []
        self.max_retries = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.sent.append(kwargs)
        got = self.answers.pop(0)
        if isinstance(got, BaseException):
            raise got
        return got


def patched_openai():
    stack = ExitStack()
    stack.enter_context(mock.patch("ssl.create_default_context"))
    stack.enter_context(mock.patch("httpx.Client"))
    stack.enter_context(mock.patch("openai.OpenAI"))
    return stack


class Status(Exception):
    def __init__(self, status):
        super().__init__(f"status {status}")
        self.status_code = status


class LedgerCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        ul.configure(self.dir, enabled=True, site="test_site", box_id="b" * 32)

    def tearDown(self):
        ul.configure(None, enabled=None)
        shutil.rmtree(self.dir, ignore_errors=True)

    def lines(self):
        ul.flush()
        out = []
        for name in sorted(os.listdir(self.dir)):
            if name.endswith(".jsonl"):
                with open(os.path.join(self.dir, name), encoding="utf-8") as f:
                    out += [json.loads(x) for x in f if x.strip()]
        return out

    def one(self):
        lines = self.lines()
        self.assertEqual(len(lines), 1, lines)
        return lines[0]

    def gpt(self, client, model="qwen/qwen3.5-9b"):
        from home_guard_project.box.inference import GptBackend

        with patched_openai():
            b = GptBackend("k", model, timeout=5.0)
        b._client = client
        return b


# ---------------------------------------------------------------------------------------------------------------------
class LineTests(LedgerCase):
    def test_a_line_has_exactly_the_contract_fields(self):
        ul.record("eye", answer("{}", cost=0.0012), time.monotonic(), client=SimpleNamespace(base_url=OPENROUTER),
                  images=3, camera="cam1", alert_id="cam1_1_alert")
        line = self.one()
        self.assertEqual(tuple(line), ul.FIELDS)
        self.assertEqual((line["box_id"], line["site"], line["agent"], line["provider"], line["model"]),
                         ("b" * 32, "test_site", "eye", "openrouter", "qwen/qwen3.5-9b"))
        self.assertEqual((line["usd"], line["usd_source"], line["route"], line["ok"], line["error_kind"]),
                         (0.0012, "provider", "direct", True, ""))
        self.assertEqual((line["images"], line["camera"], line["alert_id"]), (3, "cam1", "cam1_1_alert"))
        self.assertTrue(os.path.isfile(os.path.join(self.dir, ul.day_of(line["ts"]) + ".jsonl")))

    def test_no_provider_cost_uses_the_price_table_by_the_model_asked(self):
        resp = answer("{}", model="gpt-4o-mini-2024-07-18", prompt=1_000_000, completion=1_000_000)
        ul.record("brain_fast", resp, None, client=SimpleNamespace(base_url=OPENAI), model="gpt-4o-mini")
        line = self.one()
        self.assertEqual((line["provider"], line["usd_source"]), ("openai", "price_table"))
        self.assertAlmostEqual(line["usd"], 0.15 + 0.60)

    def test_an_unpriced_model_is_unknown_and_costs_zero(self):
        ul.record("other", answer("{}", model="local/thing"), None, provider="ollama", model="local/thing")
        line = self.one()
        self.assertEqual((line["usd"], line["usd_source"]), (0.0, "unknown"))

    def test_failures_are_recorded_with_their_kind(self):
        for exc, kind in ((TimeoutError("late"), "timeout"), (Status(429), "429"), (Status(503), "5xx"),
                          (ValueError("bad"), "other")):
            ul.record("eye", None, time.monotonic(), error=exc, provider="openrouter", model="qwen/qwen3.5-9b")
        lines = self.lines()
        self.assertEqual([x["error_kind"] for x in lines], ["timeout", "429", "5xx", "other"])
        self.assertTrue(all(x["ok"] is False and x["usd"] == 0.0 for x in lines))

    def test_the_gateway_names_its_upstream(self):
        resp = answer("{}", model="openrouter:qwen/qwen3-vl-8b-instruct", cost=0.0004)
        ul.record("eye_fallback", resp, None, client=SimpleNamespace(base_url=OPENROUTER), model="qwen/qwen3.5-9b")
        line = self.one()
        self.assertEqual((line["route"], line["provider"], line["model"], line["usd_source"]),
                         ("gateway", "openrouter", "qwen/qwen3-vl-8b-instruct", "provider"))

    def test_the_http_layer_can_say_how_the_call_went(self):
        ul.note_route("fallback-direct")
        ul.record("eye", answer("{}"), None, provider="openrouter")
        ul.record("eye", answer("{}"), None, provider="openrouter")
        self.assertEqual([x["route"] for x in self.lines()], ["fallback-direct", "direct"])

    def test_cached_tokens_and_anthropic_usage(self):
        ul.record("brain", answer("{}", cached=800), None, provider="openai", model="gpt-4o")
        anth = SimpleNamespace(usage=SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=900,
                                                     cache_creation_input_tokens=0), model="claude-x")
        ul.record("brain", anth, None, provider="anthropic", model="claude-x")
        a, b = self.lines()
        self.assertEqual(a["cached_tokens"], 800)
        self.assertEqual((b["prompt_tokens"], b["completion_tokens"], b["cached_tokens"]), (1000, 20, 900))

    def test_an_unknown_agent_is_other_and_a_scope_agent_overrides(self):
        ul.record("nobody", None, None)
        with ul.scope(agent="eye_rescue", camera="c9"):
            ul.record("eye", None, None)
        a, b = self.lines()
        self.assertEqual(a["agent"], "other")
        self.assertEqual((b["agent"], b["camera"]), ("eye_rescue", "c9"))

    def test_off_writes_nothing_and_record_never_raises(self):
        ul.configure(self.dir, enabled=False)
        self.assertIsNone(ul.record("eye", answer("{}"), None))
        ul.configure(self.dir, enabled=True)
        with mock.patch.object(ul, "build_line", side_effect=RuntimeError("boom")):
            self.assertIsNone(ul.record("eye", answer("{}"), None))
        self.assertEqual(self.lines(), [])

    def test_lines_go_to_their_local_day(self):
        day1 = dt.datetime(2026, 10, 8, 23, 59).timestamp()
        day2 = dt.datetime(2026, 10, 9, 0, 1).timestamp()
        ul.append_lines(self.dir, [ul.build_line("eye", now=day1), ul.build_line("eye", now=day2)])
        self.assertEqual(sorted(n for n in os.listdir(self.dir) if n.endswith(".jsonl")),
                         ["2026-10-08.jsonl", "2026-10-09.jsonl"])


# ---------------------------------------------------------------------------------------------------------------------
class EyeHookTests(LedgerCase):
    def test_eye(self):
        client = FakeClient(OPENROUTER, answer(EYE_JSON, prompt=9000, completion=120, cost=0.00093))
        b = self.gpt(client)
        frames = [np.zeros((40, 60, 3), np.uint8)] * 4
        with ul.scope(camera="cam6", alert_id="cam6_1_alert"):
            b.analyze(frames, "cam6", 0, 0, 0)
        line = self.one()
        self.assertEqual((line["agent"], line["provider"], line["images"], line["usd"], line["usd_source"]),
                         ("eye", "openrouter", 4, 0.00093, "provider"))
        self.assertEqual((line["camera"], line["alert_id"], line["prompt_tokens"]), ("cam6", "cam6_1_alert", 9000))

    def test_eye_fallback_and_the_failed_primary(self):
        from home_guard_project.box import inference as inf

        primary = self.gpt(FakeClient(OPENROUTER, Status(502)))
        fallback = self.gpt(FakeClient(OPENROUTER, answer(EYE_JSON, model="qwen/qwen3-vl-8b-instruct", cost=0.002)),
                            "qwen/qwen3-vl-8b-instruct")
        import dataclasses

        settings = dataclasses.replace(inf.AlertSettings(), vlm_backend="gpt", vlm_provider="openrouter",
                                       vlm_model="qwen/qwen3.5-9b", vlm_fallback_provider="openrouter",
                                       vlm_fallback_model="qwen/qwen3-vl-8b-instruct", dry_run=False)
        with mock.patch.object(inf, "build_gpt", side_effect=[primary, fallback]):
            backend = inf.make_backend(settings, {})
        backend.analyze([], "cam", 0, 0, 0)
        failed, good = self.lines()
        self.assertEqual((failed["agent"], failed["ok"], failed["error_kind"]), ("eye", False, "5xx"))
        self.assertEqual((good["agent"], good["model"], good["usd"]), ("eye_fallback", "qwen/qwen3-vl-8b-instruct", 0.002))

    def test_eye_rescue(self):
        from home_guard_project.box import inference as inf

        b = self.gpt(FakeClient(OPENROUTER, answer(EYE_JSON)))
        out = inf.vlm_rescue(b, [np.zeros((1000, 1500, 3), np.uint8)] * 2, "cam", 0, 0, 0, "both failed")
        self.assertTrue(out[2]["answered"])
        self.assertEqual(self.one()["agent"], "eye_rescue")

    def test_second_look_in_its_own_thread_keeps_the_alert(self):
        from home_guard_project.box import inference as inf

        b = self.gpt(FakeClient(OPENROUTER, answer(json.dumps({"confirmed": True, "what_it_is": "a knife",
                                                                "evidence_frame": 1}))))
        with ul.scope(camera="cam2", alert_id="cam2_9_alert"):
            look = inf.second_look(b, [np.zeros((40, 60, 3), np.uint8)] * 2, ["weapon"], "en")
        self.assertTrue(look["answered"])
        line = self.one()
        self.assertEqual((line["agent"], line["camera"], line["alert_id"], line["images"]),
                         ("second_look", "cam2", "cam2_9_alert", 2))

    def test_activity_look(self):
        from home_guard_project.box import activity_memory as am

        book = am.ActivityBook(os.path.join(self.dir, "activity.json"))
        at = lambda h, m=0, d=9: dt.datetime(2026, 10, d, h, m).timestamp()  # noqa: E731
        book.add(cameras=["ch6"], actions=["lying"], cause="electricians", until=at(18, 0, 15), now=at(13, 54),
                 cause_en="electricians installing LED lights", place="stairs", place_words="stairs",
                 daily_from="07:00", daily_to="18:00", by="owner")
        b = self.gpt(FakeClient(OPENROUTER, answer(json.dumps({"confirmed": True, "what_it_is": "a worker"}))))
        look = am.red_look(b, [np.zeros((40, 60, 3), np.uint8)], "ch6", at(14, 3), "A person lies on the ground",
                           "he", activities=book)
        self.assertTrue(look["answered"])
        self.assertEqual(self.one()["agent"], "activity_look")

    def test_a_worker_thread_carries_the_alert_scope(self):
        from home_guard_project.box import inference as inf

        seen = {}

        def worker():
            ul.record("describer", None, None)
            seen["ok"] = True

        run = ul.bound(worker, camera="cam3", alert_id="cam3_5_alert")
        import threading

        t = threading.Thread(target=run)
        t.start()
        t.join()
        line = self.one()
        self.assertEqual((line["camera"], line["alert_id"]), ("cam3", "cam3_5_alert"))
        self.assertIs(getattr(ul.bound(inf._worker), "__wrapped__", None), inf._worker)


class OtherHookTests(LedgerCase):
    def test_describer(self):
        from home_guard_project.box.describer import Describer

        client = FakeClient(OPENROUTER, answer('{"scene": "x"}', cost=0.0003))
        Describer(client, "qwen/qwen3.5-9b", timeout=5.0).ask("prompt", [b"jpg1", b"jpg2"])
        line = self.one()
        self.assertEqual((line["agent"], line["images"], line["usd"]), ("describer", 2, 0.0003))

    def test_translator_and_its_hedge(self):
        from home_guard_project.box.messenger import Messenger

        client = FakeClient(OPENROUTER, answer("{}", model="google/gemini-3.1-flash-lite", cost=0.0001),
                            answer("{}", model="google/gemini-2.5-flash-lite", cost=0.00005))
        m = Messenger(client, "google/gemini-3.1-flash-lite", timeout=5.0, hedge_model="google/gemini-2.5-flash-lite")
        m._ask({"summary": "hi"}, "he", (), fields=True)
        m._ask({"summary": "hi"}, "he", (), fields=True, model="google/gemini-2.5-flash-lite")
        a, b = self.lines()
        self.assertEqual((a["agent"], a["model"]), ("translator", "google/gemini-3.1-flash-lite"))
        self.assertEqual((b["agent"], b["model"]), ("translator_fast", "google/gemini-2.5-flash-lite"))

    def test_brain_and_brain_fast(self):
        from home_guard_project.box.brain.models import AnthropicChat, OpenAIChat

        big = OpenAIChat(FakeClient(OPENROUTER, answer("hello", model="openai/gpt-4o", cost=0.01)), "openai/gpt-4o")
        fast = OpenAIChat(FakeClient(OPENAI, answer("hi", model="gpt-4o-mini")), "gpt-4o-mini")
        fast.usage_agent = "brain_fast"
        big.chat([{"role": "user", "content": "x"}], [])
        fast.chat([{"role": "user", "content": "x"}], [])
        resp = SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")], stop_reason="end_turn",
                               usage=SimpleNamespace(input_tokens=10, output_tokens=5), model="claude-sonnet-5-5")
        anth = AnthropicChat(SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: resp)), "claude-sonnet-5-5")
        anth.chat([{"role": "user", "content": "x"}], [])
        a, b, c = self.lines()
        self.assertEqual((a["agent"], a["usd_source"], b["agent"], b["usd_source"]),
                         ("brain", "provider", "brain_fast", "price_table"))
        self.assertEqual((c["agent"], c["provider"], c["prompt_tokens"]), ("brain", "anthropic", 10))

    def test_brain_fast_is_named_where_the_assistant_is_built(self):
        src = open(os.path.join(ROOT, "home_guard_project", "box", "brain", "agent.py"), encoding="utf-8").read()
        self.assertIn('fast.usage_agent = "brain_fast"', src)

    def test_openrouter_brain_asks_for_the_cost(self):
        from home_guard_project.box.brain import models

        with mock.patch("openai.OpenAI") as ctor, mock.patch.object(models, "_http_client"):
            chat = models.make_model("openrouter:openai/gpt-4o", {"OPENROUTER_API_KEY": "k"})
        self.assertEqual(chat._extra_body, {"usage": {"include": True}})
        self.assertTrue(ctor.called)

    def test_brain_tool_vision(self):
        from home_guard_project.box.brain.vision import _completer

        client = FakeClient(OPENROUTER, answer('{"seen": true}', cost=0.0002))
        with mock.patch("openai.OpenAI", return_value=client), mock.patch("httpx.Client"), \
                mock.patch("ssl.create_default_context"):
            complete = _completer({"OPENROUTER_API_KEY": "k"}, "openrouter", "qwen/qwen3.5-9b")
            complete("look", [b"a", b"b", b"c"], {"type": "object"})
        line = self.one()
        self.assertEqual((line["agent"], line["images"], line["usd_source"]), ("brain_tool_vision", 3, "provider"))
        self.assertEqual(client.sent[0]["extra_body"]["usage"], {"include": True})

    def test_embeddings(self):
        from home_guard_project.box.embeddings import Embedder

        body = {"data": [{"embedding": [0.1, 0.2]}], "usage": {"prompt_tokens": 7, "total_tokens": 7},
                "model": "text-embedding-3-small"}
        fake = SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)
        with mock.patch("httpx.post", return_value=fake), mock.patch("ssl.create_default_context"):
            Embedder("sk", os.path.join(self.dir, "cache.json"))._fetch(["hello"])
        with mock.patch("httpx.post", side_effect=TimeoutError("slow")), mock.patch("ssl.create_default_context"):
            Embedder("sk", os.path.join(self.dir, "cache2.json"))._fetch(["again"])
        ok, failed = self.lines()
        self.assertEqual((ok["agent"], ok["provider"], ok["prompt_tokens"], ok["usd_source"]),
                         ("embeddings", "openai", 7, "price_table"))
        self.assertEqual((failed["ok"], failed["error_kind"]), (False, "timeout"))

    def test_case_judge(self):
        from home_guard_project.box.case_memory.judge import OpenAICompatibleJudge

        client = FakeClient(OPENAI, answer('{"verdict": "unsure"}', model="gpt-6-luna"))
        OpenAICompatibleJudge(client=client)({"messages": [], "response_format": {"type": "json_object"}})
        self.assertEqual(self.one()["agent"], "case_judge")

    def test_transcription(self):
        from home_guard_project.box import voice

        result = SimpleNamespace(text="hello", usage=SimpleNamespace(input_tokens=40, output_tokens=3))
        client = SimpleNamespace(base_url=OPENAI, audio=SimpleNamespace(
            transcriptions=SimpleNamespace(create=lambda **kw: result)))
        with mock.patch("openai.OpenAI", return_value=client), mock.patch("httpx.Client"), \
                mock.patch("ssl.create_default_context"):
            voice.make_transcriber({"OPENAI_API_KEY": "sk"})(b"ogg", "v.ogg", "he")
        line = self.one()
        self.assertEqual((line["agent"], line["prompt_tokens"], line["completion_tokens"]), ("transcription", 40, 3))

    def test_v1_assistant(self):
        from home_guard_project.box.agent import _OpenAIChat

        _OpenAIChat(FakeClient(OPENAI, answer("hi", model="gpt-4o-mini")), "gpt-4o-mini").chat([], [])
        self.assertEqual(self.one()["agent"], "brain")


# ---------------------------------------------------------------------------------------------------------------------
class ReadBackTests(LedgerCase):
    def write_day(self):
        now = dt.datetime(2026, 10, 9, 12).timestamp()
        lines = []
        for agent, model, usd in (("eye", "qwen/qwen3.5-9b", 0.001), ("eye", "qwen/qwen3.5-9b", 0.002),
                                  ("translator", "google/gemini-3.1-flash-lite", 0.0005)):
            line = ul.build_line(agent, answer("{}", model=model, cost=usd), now=now)
            lines.append(line)
        ul.append_lines(self.dir, lines)
        return now

    def test_usage_today_for_the_heartbeat(self):
        now = self.write_day()
        got = ul.usage_today(self.dir, now)
        self.assertEqual(got["eye"], {"calls": 2, "usd": 0.003})
        self.assertEqual(got["translator"], {"calls": 1, "usd": 0.0005})
        self.assertEqual(got["total"], {"calls": 3, "usd": 0.0035})
        self.assertEqual(ul.usage_today(self.dir, now + 86400), {"total": {"calls": 0, "usd": 0.0}})

    def test_summary_per_agent_per_model_and_the_trend(self):
        self.write_day()
        text = ul.summary("2026-10-09", 1, self.dir)
        self.assertIn("per agent:", text)
        self.assertIn("per model:", text)
        self.assertRegex(text, r"eye\s+2\s+0")
        self.assertIn("google/gemini-3.1-flash-lite", text)
        week = ul.summary("2026-10-10", 7, self.dir)
        self.assertIn("2026-10-09       3 calls  $0.003500", week)
        self.assertIn("2026-10-04", week)

    def test_the_cli(self):
        self.write_day()
        with mock.patch("sys.stdout") as out:
            self.assertEqual(ul.main(["summary", "--date", "2026-10-09", "--dir", self.dir]), 0)
        self.assertIn("per agent:", "".join(c.args[0] for c in out.write.call_args_list))

    def test_heartbeat_carries_usage_today(self):
        from home_guard_project.box import __main__ as box_main

        cfg = SimpleNamespace(site="test_site", mode="inference")
        with mock.patch.object(box_main, "build_heartbeat", return_value={}), \
                mock.patch.object(box_main, "load_registration", return_value=None), \
                mock.patch.object(box_main.control, "is_stopped", return_value=False), \
                mock.patch("home_guard_project.box.box_identity.box_id", return_value="b" * 32), \
                mock.patch.object(ul, "usage_today", return_value={"total": {"calls": 4, "usd": 0.01}}):
            status = box_main._status(cfg)
        self.assertEqual(status["usage_today"], {"total": {"calls": 4, "usd": 0.01}})


class UploadTests(LedgerCase):
    def test_day_files_are_copied_when_they_grow_and_not_again(self):
        from home_guard_project.box.outbox import USAGE_STATE_NAME, copy_usage_ledger

        outbox = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, outbox, True)
        state = os.path.join(self.dir, USAGE_STATE_NAME)
        now = dt.datetime(2026, 10, 9, 12).timestamp()
        ul.append_lines(self.dir, [ul.build_line("eye", now=now)])
        self.assertEqual(copy_usage_ledger(self.dir, outbox, state), 1)
        self.assertEqual(copy_usage_ledger(self.dir, outbox, state), 0)            # unchanged: nothing to do
        ul.append_lines(self.dir, [ul.build_line("describer", now=now)])
        self.assertEqual(copy_usage_ledger(self.dir, outbox, state), 1)            # grew: the whole day again
        with open(os.path.join(outbox, "usage", "2026-10-09.jsonl"), encoding="utf-8") as f:
            self.assertEqual([json.loads(x)["agent"] for x in f], ["eye", "describer"])
        os.remove(os.path.join(outbox, "usage", "2026-10-09.jsonl"))              # the archive's retention
        self.assertEqual(copy_usage_ledger(self.dir, outbox, state), 0)            # is not undone
        self.assertEqual(copy_usage_ledger(os.path.join(self.dir, "missing"), outbox, state), 0)

    def test_the_upload_takes_the_ledger_with_the_chat(self):
        src = open(os.path.join(ROOT, "home_guard_project", "box", "__main__.py"), encoding="utf-8").read()
        self.assertIn("usage_dir=usage_ledger.default_dir()", src)


# ---------------------------------------------------------------------------------------------------------------------
class ContractTests(unittest.TestCase):
    def setUp(self):
        with open(CONTRACT, encoding="utf-8") as f:
            self.doc = f.read()

    def test_every_field_and_agent_is_documented(self):
        for name in ul.FIELDS + ul.AGENTS + ul.ROUTES + ul.USD_SOURCES + tuple(k for k in ul.ERROR_KINDS if k):
            self.assertIn(f"`{name}`", self.doc, name)

    def test_the_example_line_is_a_real_line(self):
        block = re.search(r"```json\n(\{.*?\})\n```", self.doc, re.S)
        self.assertIsNotNone(block)
        example = json.loads(block.group(1))
        self.assertEqual(tuple(example), ul.FIELDS)
        self.assertIn(example["agent"], ul.AGENTS)

    def test_paths_and_heartbeat_are_named(self):
        for text in ("production_<site>/usage/<YYYY-MM-DD>.jsonl", "<state_dir>/usage/<YYYY-MM-DD>.jsonl",
                     "usage_today", "usd_source"):
            self.assertIn(text, self.doc)


if __name__ == "__main__":
    unittest.main()
