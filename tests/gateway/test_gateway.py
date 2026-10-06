"""Tests for the API gateway: auth, alias routing, fallback, caps, metering, admin report, token CLI.

No network: the providers are an httpx.MockTransport, and the end-to-end tests run the
gateway's HTTP server on 127.0.0.1 and reach it with the box's own OpenAI client.
"""
from __future__ import annotations

import io
import json
import logging
import os
import sqlite3
import tempfile
import threading
import unittest
from typing import Any, Callable, Dict, List, Optional

import httpx

from home_guard_project.box import providers
from home_guard_project.gateway import cli, router
from home_guard_project.gateway.settings import ConfigError, parse_config
from home_guard_project.gateway.server import Gateway, make_server
from home_guard_project.gateway.store import Store, hash_token

# A laptop antivirus sets SSLKEYLOGFILE to a pipe the OpenSSL in this venv cannot open, which kills the
# process ("no OPENSSL_Applink") once a client makes a TLS context. Nothing here needs it.
os.environ.pop("SSLKEYLOGFILE", None)

ENV = {"OPENAI_API_KEY": "sk-openai-secret-0001", "OPENROUTER_API_KEY": "sk-or-secret-0002"}

RAW_CONFIG: Dict[str, Any] = {
    "caps": {"box_daily_usd": 1.0, "fleet_daily_usd": None},
    "aliases": {
        "eye": {"kind": "chat", "upstreams": [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "openrouter", "model": "qwen/qwen3-vl-8b-instruct"},
        ]},
        "gpt-4o-mini": [{"provider": "openai", "model": "gpt-4o-mini"}],
        "text-embedding-3-small": {"kind": "embeddings", "upstreams": [
            {"provider": "openai", "model": "text-embedding-3-small"}]},
    },
}

IMAGE = "data:image/jpeg;base64," + "A" * 4000


def chat_payload(model: str = "eye") -> Dict[str, Any]:
    return {"model": model, "temperature": 0, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is at the door?"},
        {"type": "image_url", "image_url": {"url": IMAGE}}]}],
        "response_format": {"type": "json_object"}}


def completion(text: str = '{"summary": "a person at the door"}', p_in: int = 1000, p_out: int = 100,
               model: str = "gpt-4o-2024-08-06") -> Dict[str, Any]:
    return {"id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": p_in, "completion_tokens": p_out, "total_tokens": p_in + p_out}}


class FakeProviders:
    """Every provider, in process. *script* maps an upstream host to the replies it gives, in order
    (the last one repeats); a reply is a status code with a body, or an exception to raise."""

    def __init__(self, script: Optional[Dict[str, List[Any]]] = None) -> None:
        self.script = script or {}
        self.requests: List[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        replies = self.script.get(request.url.host) or [(200, completion())]
        sent = sum(1 for r in self.requests if r.url.host == request.url.host)
        reply = replies[min(sent, len(replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        status, body = reply
        return httpx.Response(status, json=body)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))

    def bodies(self) -> List[Dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]

    def hosts(self) -> List[str]:
        return [r.url.host for r in self.requests]


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_gateway(fake: FakeProviders, raw: Optional[Dict[str, Any]] = None,
                 env: Optional[Dict[str, str]] = None, **overrides: Any) -> Gateway:
    raw = json.loads(json.dumps(raw or RAW_CONFIG))
    raw.update(overrides)
    cfg = parse_config(raw, env=dict(ENV if env is None else env))
    return Gateway(cfg, Store(":memory:"), fake.client(), clock=Clock(), sleep=lambda s: None)


def post(gw: Gateway, token: str, payload: Any, path: str = "/v1/chat/completions"):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    status, out, headers = gw.post(path, {"Authorization": f"Bearer {token}"}, body)
    return status, json.loads(out), headers


class AuthTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeProviders()
        self.gw = make_gateway(self.fake)
        self.token = self.gw.store.add_box("house1")

    def test_a_valid_token_is_served(self) -> None:
        status, body, _ = post(self.gw, self.token, chat_payload())
        self.assertEqual(status, 200)
        self.assertEqual(body["choices"][0]["message"]["content"], '{"summary": "a person at the door"}')

    def test_missing_wrong_and_revoked_tokens_are_refused(self) -> None:
        for token in ("", "hgb_wrong"):
            status, body, headers = post(self.gw, token, chat_payload())
            self.assertEqual((status, body["error"]["code"]), (401, "invalid_token"))
            self.assertEqual(headers["x-should-retry"], "false")
        self.gw.store.revoke_box("house1")
        self.assertEqual(post(self.gw, self.token, chat_payload())[0], 401)
        self.assertEqual(self.fake.requests, [])          # no provider was asked

    def test_only_the_hash_is_stored(self) -> None:
        dump = "\n".join(self.gw.store._db.iterdump())
        self.assertNotIn(self.token, dump)
        self.assertIn(hash_token(self.token), dump)

    def test_a_rotated_token_replaces_the_old_one(self) -> None:
        new = self.gw.store.add_box("house1", rotate=True)
        self.assertIsNone(self.gw.store.box_for_token(self.token))
        self.assertEqual(self.gw.store.box_for_token(new).box_id, "house1")


class RoutingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeProviders()
        self.gw = make_gateway(self.fake)
        self.token = self.gw.store.add_box("house1")

    def test_the_alias_becomes_the_first_upstream_with_our_key(self) -> None:
        status, _, headers = post(self.gw, self.token, chat_payload())
        self.assertEqual(status, 200)
        req = self.fake.requests[0]
        self.assertEqual(str(req.url), "https://api.openai.com/v1/chat/completions")
        self.assertEqual(req.headers["Authorization"], "Bearer sk-openai-secret-0001")
        sent = self.fake.bodies()[0]
        self.assertEqual(sent["model"], "gpt-4o")
        self.assertEqual(sent["response_format"], {"type": "json_object"})     # the rest passes through
        self.assertEqual(headers["x-homeguard-upstream"], "openai:gpt-4o")

    def test_the_providers_extras_are_added(self) -> None:
        self.fake.script = {"api.openai.com": [(500, {"error": {"message": "down"}})]}
        post(self.gw, self.token, chat_payload())
        last = self.fake.bodies()[-1]
        self.assertEqual(self.fake.hosts()[-1], "openrouter.ai")
        self.assertEqual(last["model"], "qwen/qwen3-vl-8b-instruct")
        self.assertEqual(last["reasoning"], {"enabled": False})
        self.assertEqual(self.fake.requests[-1].headers["Authorization"], "Bearer sk-or-secret-0002")

    def test_embeddings(self) -> None:
        self.fake.script = {"api.openai.com": [(200, {"object": "list", "data": [{"embedding": [0.1, 0.2]}],
                                                      "usage": {"prompt_tokens": 50, "total_tokens": 50}})]}
        status, body, _ = post(self.gw, self.token, {"model": "text-embedding-3-small", "input": ["a"],
                                                     "dimensions": 256}, "/v1/embeddings")
        self.assertEqual((status, body["data"][0]["embedding"]), (200, [0.1, 0.2]))
        self.assertEqual(str(self.fake.requests[0].url), "https://api.openai.com/v1/embeddings")
        self.assertEqual(self.fake.bodies()[0]["dimensions"], 256)

    def test_unknown_models_wrong_kinds_and_streams_are_refused(self) -> None:
        for payload, path, code in (
                (chat_payload("gpt-6-unknown"), "/v1/chat/completions", "model_not_found"),
                ({"model": "text-embedding-3-small", "messages": []}, "/v1/chat/completions", "model_not_found"),
                ({"model": "eye", "input": ["a"]}, "/v1/embeddings", "model_not_found"),
                (dict(chat_payload(), stream=True), "/v1/chat/completions", "stream_unsupported"),
                (b"not json", "/v1/chat/completions", "bad_request")):
            status, body, _ = post(self.gw, self.token, payload, path)
            self.assertEqual((status, body["error"]["code"]), (400, code), payload if not isinstance(payload, bytes) else "")
        self.assertEqual(self.fake.requests, [])

    def test_models_lists_the_aliases(self) -> None:
        status, body, _ = self.gw.get("/v1/models", {}, {"Authorization": f"Bearer {self.token}"})
        self.assertEqual(status, 200)
        self.assertEqual({m["id"] for m in json.loads(body)["data"]}, {"eye", "gpt-4o-mini", "text-embedding-3-small"})


class FallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeProviders()
        self.gw = make_gateway(self.fake)
        self.token = self.gw.store.add_box("house1")

    def test_5xx_retries_once_then_falls_back(self) -> None:
        self.fake.script = {"api.openai.com": [(503, {"error": {"message": "overloaded"}})]}
        status, _, headers = post(self.gw, self.token, chat_payload())
        self.assertEqual(status, 200)
        self.assertEqual(self.fake.hosts(), ["api.openai.com", "api.openai.com", "openrouter.ai"])
        self.assertEqual(headers["x-homeguard-upstream"], "openrouter:qwen/qwen3-vl-8b-instruct")

    def test_a_429_then_success_stays_on_the_first_upstream(self) -> None:
        self.fake.script = {"api.openai.com": [(429, {"error": {"message": "slow down"}}), (200, completion())]}
        self.assertEqual(post(self.gw, self.token, chat_payload())[0], 200)
        self.assertEqual(self.fake.hosts(), ["api.openai.com", "api.openai.com"])

    def test_timeouts_and_connection_errors_fall_back(self) -> None:
        self.fake.script = {"api.openai.com": [httpx.ReadTimeout("slow"), httpx.ConnectError("refused")]}
        self.assertEqual(post(self.gw, self.token, chat_payload())[0], 200)
        self.assertEqual(self.fake.hosts(), ["api.openai.com", "api.openai.com", "openrouter.ai"])

    def test_our_bad_key_moves_on_without_a_retry(self) -> None:
        self.fake.script = {"api.openai.com": [(401, {"error": {"message": "Incorrect API key sk-abc***"}})]}
        self.assertEqual(post(self.gw, self.token, chat_payload())[0], 200)
        self.assertEqual(self.fake.hosts(), ["api.openai.com", "openrouter.ai"])

    def test_a_missing_key_skips_that_upstream(self) -> None:
        gw = make_gateway(self.fake, env={"OPENROUTER_API_KEY": "sk-or-secret-0002"})
        token = gw.store.add_box("house1")
        self.assertEqual(post(gw, token, chat_payload())[0], 200)
        self.assertEqual(self.fake.hosts(), ["openrouter.ai"])

    def test_the_requests_own_error_goes_back_as_is(self) -> None:
        err = {"error": {"message": "Invalid parameter: 'response_format' of type 'json_schema' is not supported",
                         "type": "invalid_request_error"}}
        self.fake.script = {"api.openai.com": [(400, err)]}
        status, body, headers = post(self.gw, self.token, chat_payload())
        self.assertEqual((status, body), (400, err))          # the box's json_object fallback reads this
        self.assertEqual(headers["x-should-retry"], "false")
        self.assertEqual(self.fake.hosts(), ["api.openai.com"])

    def test_every_upstream_down_is_a_502_the_box_does_not_retry(self) -> None:
        self.fake.script = {"api.openai.com": [(500, {})], "openrouter.ai": [(502, {})]}
        status, body, headers = post(self.gw, self.token, chat_payload())
        self.assertEqual((status, body["error"]["code"]), (502, "upstreams_failed"))
        self.assertEqual(headers["x-should-retry"], "false")
        self.assertEqual(len(self.fake.requests), 4)
        self.assertNotIn("sk-", body["error"]["message"])

    def test_the_deadline_stops_the_chain(self) -> None:
        clock = Clock()

        def slow(request: httpx.Request) -> httpx.Response:
            self.fake.requests.append(request)
            clock.now += 15
            return httpx.Response(500, json={})

        gw = Gateway(parse_config(RAW_CONFIG, env=ENV), Store(":memory:"),
                     httpx.Client(transport=httpx.MockTransport(slow)), clock=clock, sleep=lambda s: None)
        token = gw.store.add_box("house1")
        self.assertEqual(post(gw, token, chat_payload())[0], 502)
        self.assertEqual(len(self.fake.requests), 2)          # 2 x 15 s of a 27 s deadline; no third try

    def test_a_200_that_is_not_json_counts_as_a_failure(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.fake.requests.append(request)
            if request.url.host == "api.openai.com":
                return httpx.Response(200, content=b"<html>proxy error</html>")
            return httpx.Response(200, json=completion())

        gw = Gateway(parse_config(RAW_CONFIG, env=ENV), Store(":memory:"),
                     httpx.Client(transport=httpx.MockTransport(handler)), clock=Clock(), sleep=lambda s: None)
        token = gw.store.add_box("house1")
        self.assertEqual(post(gw, token, chat_payload())[0], 200)
        self.assertEqual(self.fake.hosts()[-1], "openrouter.ai")


class MeteringAndCapTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeProviders()
        self.gw = make_gateway(self.fake)
        self.token = self.gw.store.add_box("house1")

    def test_cost_comes_from_the_shared_price_table(self) -> None:
        _, _, headers = post(self.gw, self.token, chat_payload())
        want = providers.cost_usd("gpt-4o", 1000, 100)
        self.assertAlmostEqual(want, (1000 * 2.50 + 100 * 10.00) / 1e6)
        self.assertAlmostEqual(self.gw.store.spent("house1"), want)
        self.assertEqual(headers["x-homeguard-cost-usd"], f"{want:.6f}")
        row = dict(self.gw.store._db.execute("SELECT * FROM calls").fetchone())
        self.assertEqual((row["box_id"], row["alias"], row["provider"], row["model"], row["status"]),
                         ("house1", "eye", "openai", "gpt-4o", 200))
        self.assertEqual((row["prompt_tokens"], row["completion_tokens"], row["endpoint"]), (1000, 100, "chat"))

    def test_failed_attempts_are_recorded_at_no_cost(self) -> None:
        self.fake.script = {"api.openai.com": [(500, {"error": {"message": "boom " + IMAGE}})]}
        post(self.gw, self.token, chat_payload())
        rows = [dict(r) for r in self.gw.store._db.execute("SELECT provider, status, cost_usd, error FROM calls")]
        self.assertEqual([(r["provider"], r["status"]) for r in rows],
                         [("openai", 500), ("openai", 500), ("openrouter", 200)])
        self.assertEqual(rows[0]["cost_usd"], 0)
        self.assertNotIn("AAAAAAAA", rows[0]["error"])        # no image in the database
        self.assertAlmostEqual(self.gw.store.spent("house1"),
                               providers.cost_usd("qwen/qwen3-vl-8b-instruct", 1000, 100))

    def test_over_the_daily_cap_is_a_402_without_a_provider_call(self) -> None:
        self.fake.script = {"api.openai.com": [(200, completion(p_in=300_000, p_out=50_000))]}   # $1.25
        self.assertEqual(post(self.gw, self.token, chat_payload())[0], 200)
        status, body, headers = post(self.gw, self.token, chat_payload())
        self.assertEqual((status, body["error"]["code"], body["error"]["type"]),
                         (402, "box_daily_cap", "insufficient_quota"))
        self.assertEqual(headers["x-should-retry"], "false")
        self.assertEqual(len(self.fake.requests), 1)
        other = self.gw.store.add_box("house2")              # another box is not affected
        self.assertEqual(post(self.gw, other, chat_payload())[0], 200)

    def test_a_box_own_cap_wins(self) -> None:
        self.gw.store.set_cap("house1", 0.001)
        self.assertEqual(post(self.gw, self.token, chat_payload())[0], 200)    # $0.0035 spent
        self.assertEqual(post(self.gw, self.token, chat_payload())[0], 402)
        self.gw.store.set_cap("house1", 5.0)
        self.assertEqual(post(self.gw, self.token, chat_payload())[0], 200)

    def test_the_fleet_cap(self) -> None:
        gw = make_gateway(self.fake, caps={"box_daily_usd": None, "fleet_daily_usd": 0.005})
        a, b = gw.store.add_box("house1"), gw.store.add_box("house2")
        self.assertEqual(post(gw, a, chat_payload())[0], 200)
        self.assertEqual(post(gw, b, chat_payload())[0], 200)                  # $0.007 > $0.005 now
        status, body, _ = post(gw, b, chat_payload())
        self.assertEqual((status, body["error"]["code"]), (402, "fleet_daily_cap"))

    def test_the_per_box_rate_limit(self) -> None:
        gw = make_gateway(self.fake, box_rpm=2)
        token = gw.store.add_box("house1")
        self.assertEqual([post(gw, token, chat_payload())[0] for _ in range(3)], [200, 200, 429])
        gw._clock.now += 61
        self.assertEqual(post(gw, token, chat_payload())[0], 200)

    def test_nothing_secret_or_visual_is_logged(self) -> None:
        with self.assertLogs("gateway", level="INFO") as logs:
            post(self.gw, self.token, chat_payload())
        text = "\n".join(logs.output)
        for secret in (self.token, ENV["OPENAI_API_KEY"], "AAAAAAAA", "what is at the door"):
            self.assertNotIn(secret, text)
        self.assertIn("box=house1 alias=eye -> openai:gpt-4o status=200", text)


class AdminTest(unittest.TestCase):
    ADMIN = "hga_admin-token"

    def setUp(self) -> None:
        import hashlib

        self.fake = FakeProviders()
        self.gw = make_gateway(self.fake, admin_token_sha256=hashlib.sha256(self.ADMIN.encode()).hexdigest())
        self.token = self.gw.store.add_box("house1", note="Faour")
        self.gw.store.add_box("house2")

    def get(self, path: str, query: Dict[str, List[str]], token: str):
        status, body, _ = self.gw.get(path, query, {"Authorization": f"Bearer {token}"})
        return status, json.loads(body)

    def test_costs_per_box_and_day(self) -> None:
        post(self.gw, self.token, chat_payload())
        post(self.gw, self.token, chat_payload())
        status, body = self.get("/admin/costs", {"days": ["7"]}, self.ADMIN)
        self.assertEqual(status, 200)
        (row,) = body["rows"]
        self.assertEqual((row["box_id"], row["calls"], row["failed"], row["prompt_tokens"]), ("house1", 2, 0, 2000))
        self.assertAlmostEqual(row["cost_usd"], 2 * providers.cost_usd("gpt-4o", 1000, 100))
        status, body = self.get("/admin/costs", {"box": ["house2"]}, self.ADMIN)
        self.assertEqual(body["rows"], [])

    def test_boxes_never_show_hashes(self) -> None:
        status, body = self.get("/admin/boxes", {}, self.ADMIN)
        self.assertEqual([b["box_id"] for b in body["boxes"]], ["house1", "house2"])
        self.assertNotIn("token_sha256", body["boxes"][0])

    def test_a_box_token_is_not_an_admin_token(self) -> None:
        for token in (self.token, "", "wrong"):
            self.assertEqual(self.get("/admin/costs", {}, token)[0], 401)

    def test_admin_is_off_without_a_hash(self) -> None:
        gw = make_gateway(self.fake)
        self.assertEqual(gw.get("/admin/boxes", {}, {"Authorization": "Bearer anything"})[0], 401)


class ConfigTest(unittest.TestCase):
    def test_an_upstream_without_a_price_is_refused(self) -> None:
        raw = {"aliases": {"eye": [{"provider": "vllm", "model": "Qwen/Qwen3-VL-4B-Instruct"}]}}
        with self.assertRaises(ConfigError) as cm:
            parse_config(raw, env={})
        self.assertIn("price", str(cm.exception))
        raw["aliases"]["eye"][0]["price_per_m"] = [0, 0]
        self.assertEqual(parse_config(raw, env={}).aliases["eye"].upstreams[0].price(), (0.0, 0.0))

    def test_bad_configs(self) -> None:
        for raw in ({}, {"aliases": {"eye": []}}, {"aliases": {"eye": [{"provider": "nope", "model": "m"}]}},
                    {"aliases": {"eye": [{"provider": "gateway", "model": "eye"}]}},
                    {"aliases": {"eye": {"kind": "audio", "upstreams": [{"provider": "openai", "model": "gpt-4o"}]}}},
                    {"aliases": {"eye": [{"provider": "openai", "model": "gpt-4o"}]}, "caps": {"box_daily_usd": -1}},
                    {"aliases": {"eye": [{"provider": "openai", "model": "gpt-4o"}]}, "admin_token_sha256": "abc"}):
            with self.assertRaises(ConfigError, msg=str(raw)):
                parse_config(raw, env={})

    def test_key_env_override(self) -> None:
        raw = {"aliases": {"eye": [{"provider": "openai", "model": "gpt-4o", "key_env": "OPENAI_KEY_EYE"}]}}
        up = parse_config(raw, env={}).aliases["eye"].upstreams[0]
        self.assertEqual(up.resolve({"OPENAI_KEY_EYE": "sk-eye", "OPENAI_API_KEY": "sk-main"})[:2],
                         ("sk-eye", "https://api.openai.com/v1"))

    def test_the_example_config_loads(self) -> None:
        from home_guard_project.gateway.settings import load_config

        path = os.path.join(os.path.dirname(router.__file__), "gateway.example.yaml")
        cfg = load_config(path, env={})
        self.assertEqual([u.name for u in cfg.aliases["eye"].upstreams], ["openai:gpt-4o", "openrouter:openai/gpt-4o"])
        self.assertEqual(cfg.aliases["text-embedding-3-small"].kind, "embeddings")

    def test_scrub(self) -> None:
        text = router.scrub(f"bad image {IMAGE} with key sk-proj-abcdef123456 and hgb_xyzxyzxyzxyz")
        self.assertNotIn("AAAA", text)
        self.assertNotIn("abcdef123456", text)
        self.assertNotIn("xyzxyzxyz", text)


class CliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.db = os.path.join(self.dir.name, "gw.sqlite3")
        self.config = os.path.join(self.dir.name, "gateway.yaml")
        with open(self.config, "w", encoding="utf-8") as f:
            json.dump(dict(RAW_CONFIG, db_path=self.db), f)     # JSON is YAML
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def run_cli(self, *args: str) -> tuple:
        out = io.StringIO()
        code = cli.main(["--config", self.config, *args], out=out)
        return code, out.getvalue()

    def test_add_list_rotate_revoke(self) -> None:
        code, out = self.run_cli("add-box", "house1", "--cap", "0.5", "--note", "first")
        self.assertEqual(code, 0)
        token = out.split("HOMEGUARD_BOX_TOKEN=")[1].split()[0]
        self.assertTrue(token.startswith("hgb_"))
        store = Store(self.db)
        self.addCleanup(store.close)
        self.assertEqual(store.box_for_token(token).daily_cap_usd, 0.5)
        self.assertEqual(self.run_cli("add-box", "house1")[0], 1)            # already has one
        code, out = self.run_cli("add-box", "house1", "--rotate")
        new = out.split("HOMEGUARD_BOX_TOKEN=")[1].split()[0]
        self.assertIsNone(store.box_for_token(token))
        self.assertEqual(store.box_for_token(new).box_id, "house1")
        self.assertEqual(self.run_cli("set-cap", "house1", "default")[0], 0)
        self.assertIsNone(store.box_for_token(new).daily_cap_usd)
        code, out = self.run_cli("list-boxes")
        self.assertIn("house1", out)
        self.assertNotIn(new, out)
        self.assertEqual(self.run_cli("revoke-box", "house1")[0], 0)
        self.assertIsNone(store.box_for_token(new))
        self.assertEqual(self.run_cli("revoke-box", "house1")[0], 1)
        db = sqlite3.connect(self.db)
        try:
            self.assertNotIn(new, "\n".join(db.iterdump()))
        finally:
            db.close()

    def test_bad_box_names_and_a_bad_config(self) -> None:
        with self.assertRaises(SystemExit):
            self.run_cli("add-box", "House 1")
        code = cli.main(["--config", os.path.join(self.dir.name, "missing.yaml"), "list-boxes"], out=io.StringIO())
        self.assertEqual(code, 2)

    def test_report_and_admin_token(self) -> None:
        self.assertEqual(self.run_cli("report", "--days", "3"), (0, "[]\n"))
        code, out = self.run_cli("new-admin-token")
        token = out.split(": ")[1].split()[0]
        import hashlib

        self.assertIn(f"admin_token_sha256: {hashlib.sha256(token.encode()).hexdigest()}", out)


class EndToEndTest(unittest.TestCase):
    """The box's own client (inference.GptBackend through providers 'gateway') against the real server."""

    def setUp(self) -> None:
        self.fake = FakeProviders()
        self.gw = make_gateway(self.fake)
        self.gw._clock = __import__("time").monotonic
        self.token = self.gw.store.add_box("house1")
        self.server = make_server(self.gw, "127.0.0.1", 0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        port = self.server.server_address[1]
        self.env = {"HOMEGUARD_GATEWAY_URL": f"http://127.0.0.1:{port}/v1", "HOMEGUARD_BOX_TOKEN": self.token,
                    "NO_PROXY": "127.0.0.1"}
        os.environ.setdefault("NO_PROXY", "127.0.0.1")
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def backend(self):
        from home_guard_project.box import inference

        return inference.build_gpt("gateway", "eye", self.env)

    def frames(self) -> list:
        import numpy as np

        return [np.zeros((48, 64, 3), dtype=np.uint8)]

    def test_the_box_gets_its_answer_through_the_gateway(self) -> None:
        raw, parsed = self.backend().analyze(self.frames(), "front_door", 0, 0, 0)
        self.assertEqual(parsed, {"summary": "a person at the door"})
        sent = self.fake.bodies()[0]
        self.assertEqual(sent["model"], "gpt-4o")
        self.assertEqual(sent["response_format"]["type"], "json_schema")
        self.assertTrue(sent["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertAlmostEqual(self.gw.store.spent("house1"), providers.cost_usd("gpt-4o", 1000, 100))

    def test_over_the_cap_the_box_gets_one_clear_error_and_no_retries(self) -> None:
        import openai

        self.gw.store.set_cap("house1", 0.0)
        hits: List[int] = []
        original = self.gw.post
        self.gw.post = lambda *a: hits.append(1) or original(*a)
        with self.assertRaises(openai.APIStatusError) as cm:
            self.backend().analyze(self.frames(), "front_door", 0, 0, 0)
        self.assertEqual(cm.exception.status_code, 402)
        self.assertIn("box_daily_cap", str(cm.exception))
        self.assertEqual(len(hits), 1)                        # x-should-retry: false
        self.assertEqual(self.fake.requests, [])

    def test_every_upstream_down_is_not_retried_by_the_box(self) -> None:
        import openai

        self.fake.script = {"api.openai.com": [(500, {})], "openrouter.ai": [(500, {})]}
        with self.assertRaises(openai.InternalServerError):
            self.backend().analyze(self.frames(), "front_door", 0, 0, 0)
        self.assertEqual(len(self.fake.requests), 4)          # 2 upstreams x 2 tries, once

    def test_the_json_schema_refusal_reaches_the_box(self) -> None:
        err = {"error": {"message": "response_format json_schema is not supported", "type": "invalid_request_error"}}
        self.fake.script = {"api.openai.com": [(400, err), (200, completion())]}
        backend = self.backend()
        raw, parsed = backend.analyze(self.frames(), "front_door", 0, 0, 0)
        self.assertEqual(parsed, {"summary": "a person at the door"})
        self.assertEqual([b["response_format"]["type"] for b in self.fake.bodies()], ["json_schema", "json_object"])

    def test_a_large_body_is_refused(self) -> None:
        self.gw.cfg = parse_config(dict(RAW_CONFIG, max_body_bytes=1000), env=ENV)
        url = self.env["HOMEGUARD_GATEWAY_URL"] + "/chat/completions"
        with httpx.Client(trust_env=False) as client:
            resp = client.post(url, json=chat_payload(), headers={"Authorization": f"Bearer {self.token}"})
        self.assertEqual(resp.status_code, 413)
        self.assertEqual(resp.json()["error"]["code"], "too_large")

    def test_healthz(self) -> None:
        with httpx.Client(trust_env=False) as client:
            resp = client.get(self.env["HOMEGUARD_GATEWAY_URL"].replace("/v1", "/healthz"))
        self.assertEqual(resp.json(), {"ok": True})


if __name__ == "__main__":
    unittest.main()
