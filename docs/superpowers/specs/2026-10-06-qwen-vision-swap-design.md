# Qwen vision swap (week 0, item 1) — design

Date: 2026-10-06. Plan page: https://claude.ai/artifact/XmTeET2SSoXqQt1UBdVEwW §2, §13 "שבוע 0".
Owner decisions (2026-10-06):
- gpt-4o is replaced as the box's vision model regardless of the eval. It stays only as the fallback when the primary call fails.
- The default replacement is the plan's pick, `qwen/qwen3.5-9b`. A Qwen VL model (`qwen3-vl-*`) replaces it only if it beats it on our eval.
- This is research, not a product for sale, so licence is not a filter.

## Goal

Measure Qwen vision models on our 220 tagged home clips, with real token counts and cost, pick the primary model by the rule in §6, and switch the box to it without touching the alert path.

Reference for the report (2026-10-03, prompt `2026-10-03.tagged-rules-label-animals`, gpt-4o): alerts caught 16/18, normal flagged 21/177 (11.9%), 0 errors.

## Approach

Every candidate is reachable through an OpenAI-compatible `chat.completions` endpoint. `inference.GptBackend` already builds the request (prompt + 5 JPEG frames + strict `json_schema`, falling back to `json_object` when a model refuses the schema). We add a `base_url` and a key name to it; nothing else in the alert path changes.

Rejected: a separate `QwenBackend` class (duplicates frame encoding and the schema fallback), LiteLLM (new dependency on the box for one URL).

## Components

### 1. `box/providers.py` (new)

A table, one entry per provider:

| name | base_url | key env var |
|---|---|---|
| `openai` | (SDK default) | `OPENAI_API_KEY` |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| `dashscope-intl` | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` |

Plus a per-model price table ($ per million input / output tokens) used only for reporting. `price_of(model)` returns `None` for unknown models; the report then shows tokens without dollars.

`resolve(provider, env) -> (api_key, base_url)` raises a clear error naming the missing variable.

### 2. `GptBackend` changes (`box/inference.py`)

- `GptBackend(api_key, model, base_url=None, extra_body=None)`.
- After each call, `self.last_usage = {"prompt_tokens", "completion_tokens"}` from `resp.usage` (zeros when the provider omits it).
- Reasoning off: Qwen3.5/3.6 are hybrid thinking models. For `openrouter`, send `extra_body={"reasoning": {"enabled": False}}`. Thinking costs money and seconds and the task does not need it.
- Timeout: 30 s per call on the client, so a slow provider cannot stall an alert.

### 3. Fallback in production (`box/inference.py`, `alert_settings.py`)

New settings, defaults keep today's behaviour exactly:

```yaml
vlm_provider: openai        # openai | openrouter | dashscope-intl
vlm_model: gpt-4o
vlm_fallback_provider: openai
vlm_fallback_model: gpt-4o  # empty = no fallback
```

`make_backend` builds a `FallbackBackend(primary, fallback)` when a fallback is set and differs from the primary. On any exception or empty/unparseable answer from the primary, the same frames go to the fallback once; the log says `[cam] VLM fallback: <reason>` and the training record stores which model answered (`model_name`). If the primary key is missing, the fallback alone is used with a warning. The alert is never dropped because of the swap.

### 4. Eval (`box/eval_prompt.py`)

- `run --provider <name> --model <id>`; the model id is stored with its provider (`openrouter:qwen/qwen3.5-9b`) so tags and resume keys never collide across providers.
- Each answer row also stores `prompt_tokens`, `completion_tokens`, `cost_usd`, `latency_s`.
- Summary adds:
  - day / night split of every metric (night = clip local time 19:00–06:00, from `clip_local_time`);
  - the clip ids of missed alerts and of false alarms;
  - mean prompt tokens per call, mean latency, $ per call;
  - projected $ per box per month at 150 and 300 calls/day.
- A `compare` subcommand prints one table across tags (the gpt-4o reference tag plus each candidate) and names the chosen model by the §6 rule.

### 5. Candidates (all via one OpenRouter key)

| model | why |
|---|---|
| `qwen/qwen3.5-9b` | plan's pick; same family as the training base (Qwen3.5-4B) |
| `qwen/qwen3.5-flash-02-23` | cheapest Qwen3.5 |
| `qwen/qwen3-vl-8b-instruct` | dedicated VL model, small |
| `qwen/qwen3-vl-32b-instruct` | dedicated VL model, larger, still cheap |
| `qwen/qwen3.6-35b-a3b` | newest open MoE; candidate teacher for tagging (reported, not eligible as primary) |

Thinking variants are excluded. Expected spend: under $1 for all five (≈220 calls × ≈3k input tokens each).

### 6. Choosing the primary model

1. Default: `qwen/qwen3.5-9b`.
2. A VL model (`qwen3-vl-8b-instruct` or `qwen3-vl-32b-instruct`) replaces it only if it **beats** it on the full 220 clips:
   - catches more alerts; or catches the same number with fewer normal clips flagged;
   - and does not miss any alert that `qwen3.5-9b` caught on a forced-entry / climbing-in clip;
   - and has no more errors (unparseable / refused).
   Between two VL models that both beat it, the one with more alerts caught wins, then fewer false alarms, then cheaper.
3. `qwen3.5-flash-02-23` is in the run for cost data only; it becomes primary only if it ties `qwen3.5-9b` on alerts and false alarms and is cheaper.
4. gpt-4o is replaced whatever the numbers say. If the chosen model is worse than the gpt-4o reference (fewer than 16/18 alerts, or more than 21/177 normal flagged), the report says so in its first line and lists the clips, so the prompt work in week 1 starts from them.

18 alerts is a small set (16/18 spans roughly 67–97%); a one-clip difference is noise. The report states this, and the plan's week 1 enlarges the eval set.

## Data flow

Laptop: `eval_prompt.py prepare` (existing, S3 → frames + manifest) → `run --provider openrouter` per candidate → `summary` / `compare`, which applies §6 and names the chosen model. The run happens on the laptop, so the live box keeps its own OpenAI quota (avoids the 429s of 2026-10-03).
Box: once the model is chosen, `config.live.yaml` gets `vlm_provider`/`vlm_model`, `api_key.env` on the box gets `OPENROUTER_API_KEY`, then `update.sh`. The first hour of box logs is checked for fallback lines and parse errors.

## Error handling

- Missing key → the eval exits with the variable name; production falls back to gpt-4o with a warning.
- 429 / 5xx → existing retry in the eval loop; resume never re-pays an answered clip.
- Model ignores the schema → existing `json_object` fallback; still-unparseable answers count as errors in the §6 choice.
- Provider returns no `usage` → tokens recorded as 0 and marked `usage_missing`, cost shown as unknown.

## Testing

Unit tests (pytest, no network, fake OpenAI client):
- `providers.resolve` for each provider and for a missing key;
- `GptBackend` passes `base_url`, `extra_body`, records `last_usage`;
- `FallbackBackend`: primary raises → fallback answers; primary returns junk → fallback; primary fine → fallback never called; both fail → raises (caller keeps today's handling);
- `make_backend` with default settings builds exactly today's single gpt-4o backend;
- eval summary day/night split, cost projection, and `compare` applying the §6 choice rule on fixed rows (default kept, VL wins on alerts, VL wins on false alarms at equal alerts, VL loses by missing a forced-entry clip the default caught).
Then `run --fake` end to end before the first paid call.

## Needs the owner

`OPENROUTER_API_KEY=...` in `api_key.env` at the repo root (laptop), with a few dollars of OpenRouter credit. Everything else is built and tested before that.

## Out of scope

The situation-aware prompt (`eye_prompt.py`, week 1), the bigger eval set (week 1), the gateway (week 1–2), training.
