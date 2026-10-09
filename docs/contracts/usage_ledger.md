# Contract: the box's AI usage ledger

For the Admin Center's cost monitor ("for each agent, for each model, the cost every time", owner, 2026-10-09).
Box side: `home_guard_project/box/usage_ledger.py`. Pinned by `tests/box/test_usage_ledger.py` (ContractTests):
a field, agent, route or source added in code without this page fails the tests.

## Where the lines are

| Where | Path | Notes |
|---|---|---|
| On the box | `<state_dir>/usage/<YYYY-MM-DD>.jsonl` | `state_dir` = `python -m home_guard_project.box paths --get state_dir` (`C:\home_guard\data\state` on a migrated box). One file per **local** date of the box. Append-only. |
| In S3 | `s3://security-camera-project-v1/production_<site>/usage/<YYYY-MM-DD>.jsonl` | Copied by `box upload` (the same run, schedule and uploader as the chat, `production_<site>/chat/`). Today's file is re-sent whole each time it grew, so read it again on each upload; past days stop changing. The bucket's lifecycle on `production_*` applies. |
| Heartbeat | `s3://security-camera-project-v1/dataset_<site>/_status/heartbeat.json`, field `usage_today` | Today's totals per agent, see below. |

One JSON object per line, UTF-8, `\n` line ends. A line is written once per AI **request** (an HTTP call to a model),
including failed and timed-out ones. Ignore lines that do not parse (a write cut by a power loss).

## Fields

| Field | Type | Meaning |
|---|---|---|
| `ts` | number | Unix time (seconds, UTC epoch) when the call ended. |
| `box_id` | string | `box_identity.box_id()`: 32 hex characters, the same across site renames. Group history by this. `""` if the box could not read it. |
| `site` | string | The box's site (box.yaml) when the call was made. |
| `agent` | string | Who made the call, one of the agents below. |
| `provider` | string | The provider that answered: `openrouter`, `openai`, `google`, `anthropic`, `ollama`, `vllm`, `dashscope-intl`, or `gateway` (only when the gateway did not name its upstream). |
| `model` | string | The model that ANSWERED (the response's `model`; a gateway answer `openrouter:qwen/...` is split into provider and model). The model asked for when nothing answered. May carry a date suffix (`gpt-4o-mini-2024-07-18`). |
| `route` | string | `direct` (straight to the provider), `gateway` (through our API gateway), `fallback-direct` (the gateway failed and the box asked the provider itself). |
| `prompt_tokens` | integer | Input tokens billed (for Anthropic: input + cache reads + cache writes). 0 when unknown or failed. |
| `completion_tokens` | integer | Output tokens billed. |
| `cached_tokens` | integer | Of `prompt_tokens`, how many were cache hits. |
| `images` | integer | Pictures (frames) sent in the call. |
| `usd` | number | What the call cost in US dollars. Always a number: 0 for a failed call the provider did not bill. |
| `usd_source` | string | `provider`: the provider's own figure (OpenRouter `usage.cost`, or the gateway's bill) - exact. `price_table`: tokens x `providers.PRICES` (OpenAI direct) - an estimate that ignores cache discounts. `unknown`: no price for the model (local models, transcription); `usd` is 0 and means "not known", not "free". |
| `seconds` | number | How long the request took. |
| `ok` | boolean | The provider answered. A call that answered but whose answer was unusable is still `true`. |
| `error_kind` | string | `""` when ok; else `timeout`, `429` (rate limit), `5xx` (provider/gateway server error), or `other` (4xx, refused schema, TLS, no connection). |
| `camera` | string | The camera the call was about (alert calls), else `""`. A raw camera id: show its display name, never the id, to the owner. |
| `alert_id` | string | The alert (clip stem) the call belongs to, else `""`. Matches the alert's meta and the chat's `alert_id`. |

## Agents

| Agent | What it is | Code |
|---|---|---|
| `eye` | The alert's vision model (the Eye), main model | `inference.GptBackend.analyze` |
| `eye_fallback` | The Eye's call on the fallback model after the main one failed | `inference.FallbackBackend` |
| `eye_rescue` | No model answered: the main model again on 768 px frames | `inference.vlm_rescue` |
| `second_look` | The yes/no check before a red alert | `inference.second_look` -> `GptBackend.verify` |
| `activity_look` | The owner's-activity context look on a red | `activity_memory.red_look` |
| `describer` | The per-person description of an alert (P1, CAR1) | `describer.Describer.ask` |
| `translator` | English to the owner's language, main model | `messenger.Messenger` |
| `translator_fast` | The translator's hedge model, raced after a slow or broken answer | `messenger.Messenger._hedged` |
| `brain` | The owner's assistant, main model (also the v1 assistant) | `brain/models.py`, `agent.py` |
| `brain_fast` | The assistant's fast model | `brain/models.py` (box.yaml `agent_fast_model`) |
| `brain_tool_vision` | The assistant's camera look (`look_around` / `ask_vision`) | `brain/vision.py` |
| `embeddings` | Vectors for alert search and case memory | `embeddings.py` |
| `investigator` | Reserved. The lingering investigator is tracker-only today and makes no model call | - |
| `transcription` | The owner's voice messages | `voice.py` |
| `case_judge` | The case-memory judge | `case_memory/judge.py` |
| `other` | Anything else: the live-view describe, the eval tools run on the box | `live_view.py`, `eval_prompt.py`, `eval_translation.py` |

One alert usually makes several lines (eye, maybe second_look, describer, translator...) sharing its `alert_id`:
sum them for "what did this alert cost".

## Heartbeat: `usage_today`

Today's (box local date) totals, built from the day file when the heartbeat is written (every `box upload` run):

```
"usage_today": {
  "describer":  {"calls": 41,  "usd": 0.012310},
  "eye":        {"calls": 118, "usd": 0.104522},
  "translator": {"calls": 40,  "usd": 0.003110},
  "total":      {"calls": 199, "usd": 0.119942}
}
```

Keys are the agents that made a call today, plus `total`. With no calls: `{"total": {"calls": 0, "usd": 0.0}}`.
A heartbeat from a box before 2026-10-09 has no `usage_today`.

## Example line

```json
{"ts": 1791583412.312, "box_id": "9f3c2a51d0b84e6f8a7b1c2d3e4f5a6b", "site": "ameer_tes2", "agent": "eye", "provider": "openrouter", "model": "qwen/qwen3.5-9b", "route": "direct", "prompt_tokens": 9123, "completion_tokens": 180, "cached_tokens": 0, "images": 8, "usd": 0.000939, "usd_source": "provider", "seconds": 3.42, "ok": true, "error_kind": "", "camera": "ameer_tes2_ch6", "alert_id": "ameer_tes2_ch6_1791583405_alert"}
```

## On the box

```
python -m home_guard_project.box.usage_ledger summary                      # today, per agent and per model
python -m home_guard_project.box.usage_ledger summary --date 2026-10-09 --days 7   # the week ending that day + trend
```

`HOMEGUARD_USAGE_LEDGER=off` in the environment turns the ledger off (the test suite does).
