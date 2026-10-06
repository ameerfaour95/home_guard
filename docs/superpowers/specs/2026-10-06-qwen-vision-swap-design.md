# Qwen vision swap (week 0, item 1) — design

Date: 2026-10-06. Plan page: https://claude.ai/artifact/XmTeET2SSoXqQt1UBdVEwW §2, §13 "שבוע 0".

Owner decisions (2026-10-06):
- gpt-4o is replaced as the box's vision model, and it is **not** the fallback either.
- Default primary: **`Qwen/Qwen3-VL-4B-Instruct`**. `Qwen3.5-4B` replaces it only if it beats it on our eval.
- Fallback is the other Qwen: primary Qwen3-VL-4B → fallback Qwen3.5-4B; primary Qwen3.5-4B → fallback Qwen3-VL-4B.
- Families to measure: Qwen3-VL (instruct **and** thinking), Qwen2.5-VL, Qwen3.5. Both 4B models are candidates.
- This is research, not a product for sale, so licence is not a filter (Qwen2.5-VL-3B's non-commercial licence is fine here).

## Goal

Measure the Qwen vision models on our 220 tagged home clips with real token counts, latency and cost, choose primary and fallback by the rule in §6, and give the box a backend that reaches them without touching the alert path.

Reference for the report: gpt-4o on today's prompt (the 2026-10-03 baseline, 16/18 alerts caught and 21/177 normal flagged, used an older prompt version).

## Where each model runs

OpenRouter (checked 2026-10-06) hosts no 4B, 3B or 7B Qwen vision model. Hugging Face lists Qwen3-VL-4B and Qwen2.5-VL-3B/7B on Featherless only, and Qwen3.5-4B nowhere. So:

| Where | Models | Cost |
|---|---|---|
| **Laptop GPU** (RTX 5070 Ti, 12 GB) through Ollama's OpenAI-compatible endpoint | `qwen3-vl:4b-instruct-bf16`, `qwen3-vl:4b-thinking-bf16`, `qwen3.5:4b-bf16`, `qwen2.5vl:3b-fp16`, `qwen2.5vl:7b-q8_0`, `qwen3-vl:8b-instruct-q8_0` | free |
| **OpenRouter** (one key, needs credit) | `qwen/qwen3-vl-8b-instruct`, `qwen/qwen3-vl-8b-thinking`, `qwen/qwen3-vl-32b-instruct`, `qwen/qwen3.5-9b`, `qwen/qwen2.5-vl-72b-instruct`, `openai/gpt-4o` (reference) | about $4 total, $3.3 of it gpt-4o |

The 4B models run unquantized (bf16), as they would on a rented GPU with vLLM, so their accuracy carries over. The 7B and 8B run at q8_0 to fit 12 GB: near-lossless, and they are reported, not candidates for primary.

The local 4B runs give accuracy and latency on an RTX-class GPU, but no API cost. Cost per call for the 4B models is estimated from tokens at the plan's GPU numbers (§2 of the plan page), not billed.

### Hosting the 4B models for the box

Decided with the eval results in hand, not now. Options to put to the owner then:
- a rented GPU (e.g. RunPod RTX 4090, about $0.34–0.69/hour) running vLLM with both 4B models, which is the plan's stage B setup;
- Featherless (flat monthly fee) for Qwen3-VL-4B; Qwen3.5-4B is not hosted there;
- the laptop serving the box over Tailscale, for research hours only.

Until then the box keeps gpt-4o; the code from this work is merged with defaults that change nothing.

## Approach

Every endpoint above (Ollama, OpenRouter, vLLM, Featherless) speaks the OpenAI `chat.completions` API. `inference.GptBackend` already builds the request (prompt + 5 JPEG frames + strict `json_schema`, falling back to `json_object` when a model refuses the schema). We give it a `base_url`, a key and request extras from a provider table. Nothing else in the alert path changes.

Rejected: a separate `QwenBackend` class (duplicates frame encoding and the schema fallback), LiteLLM (new dependency on the box for one URL), loading the models with `transformers` inside the eval (a second inference code path that production would not use).

## Components

### 1. `box/providers.py` (new)

| name | base_url | key env var | extras |
|---|---|---|---|
| `openai` | (SDK default) | `OPENAI_API_KEY` | — |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | `{"reasoning": {"enabled": false}}` |
| `ollama` | `http://localhost:11434/v1` (or `OLLAMA_BASE_URL`) | none (sends `"ollama"`) | — |
| `vllm` | `VLLM_BASE_URL` (required) | `VLLM_API_KEY` (optional) | — |
| `dashscope-intl` | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` | `{"enable_thinking": false}` |

Thinking is off by default for the hybrid models (Qwen3.5) because the task does not need it; the dedicated `*-thinking` Qwen3-VL models are measured as they are, since thinking is what they are. Whether Ollama honours "thinking off" for `qwen3.5:4b` through the OpenAI endpoint is checked on a 3-clip smoke run; if it does not, the provider sends Ollama's own switch (`think: false`) and the smoke run is repeated.

Plus a price table ($ per million input / output tokens) for hosted models, used only for reports.

### 2. `GptBackend` changes (`box/inference.py`)

- `GptBackend(api_key, model, base_url=None, extra_body=None, timeout=30.0)`.
- After each call, `self.last_usage = {"prompt_tokens", "completion_tokens"}` from `resp.usage` (zeros when absent).
- The eval passes a longer timeout (120 s), because the thinking models and the first call that loads a model into the GPU are slow.

### 3. Fallback (`box/inference.py`)

Settings (box.yaml), defaults keep today's behaviour exactly:

```yaml
vlm_provider: openai
vlm_model: gpt-4o
vlm_fallback_provider: ""   # empty = no fallback
vlm_fallback_model: ""
```

After the switch, for example:

```yaml
vlm_provider: vllm
vlm_model: Qwen/Qwen3-VL-4B-Instruct
vlm_fallback_provider: vllm
vlm_fallback_model: Qwen/Qwen3.5-4B
```

`make_backend` builds `FallbackBackend(primary, fallback)` when a fallback is set and differs from the primary. On any exception or an answer that is not a JSON object, the same frames go to the fallback once; the log says `[cam] VLM fallback to <model>: <reason>`, and the training record stores the model that answered. If the primary cannot be built, the fallback alone is used with a warning; if neither can, `NullBackend` (today's behaviour for a missing key). The alert is never dropped because of the swap.

### 4. Eval (`box/eval_prompt.py`)

- `run --provider <name> --model <id>`; results name the model `provider:model` (`ollama:qwen3-vl:4b-instruct-bf16`), except bare for `openai` so older results still match.
- Each answer row stores `prompt_tokens`, `completion_tokens`, `cost_usd` (hosted models only), `latency_s`.
- Summary adds: day / night split (night = clip local time 19:00–05:59), clip ids of missed alerts and false alarms, mean tokens and latency per call, $ per call and $ per box per month at 150 and 300 calls/day (when priced).
- `compare` prints one table across results files, names primary and fallback by §6, and puts the gpt-4o reference in its first line.

### 5. Candidates

| model | where | role |
|---|---|---|
| Qwen3-VL-4B-Instruct | laptop, bf16 | **default primary** |
| Qwen3.5-4B | laptop, bf16 | challenger; default fallback |
| Qwen3-VL-4B-Thinking | laptop, bf16 | measured (instruct vs thinking) |
| Qwen2.5-VL-3B-Instruct | laptop, fp16 | measured (older family) |
| Qwen2.5-VL-7B-Instruct | laptop, q8_0 | measured |
| Qwen3-VL-8B-Instruct | laptop q8_0 and OpenRouter | measured; quantization check (same model both ways) |
| Qwen3-VL-8B-Thinking | OpenRouter | measured |
| Qwen3-VL-32B-Instruct | OpenRouter | measured; teacher candidate |
| Qwen3.5-9B | OpenRouter | measured; next size up from the training base |
| Qwen2.5-VL-72B-Instruct | OpenRouter | measured |
| gpt-4o | OpenRouter | reference on today's prompt |

### 6. Choosing primary and fallback

1. Primary is one of the two 4B models; the other is the fallback.
2. Default: primary Qwen3-VL-4B-Instruct, fallback Qwen3.5-4B.
3. Qwen3.5-4B becomes primary (and Qwen3-VL-4B the fallback) only if it **beats** Qwen3-VL-4B on the full 220 clips:
   - it catches more alerts, or the same number with fewer normal clips flagged;
   - and it does not miss a forced-entry / climbing-in alert that Qwen3-VL-4B caught;
   - and it has no more errors (unparseable or refused answers).
4. The other models are reported next to the two 4B models. If one of them clearly beats both (more alerts caught with no more false alarms), the report says so as a recommendation for the owner; it does not change the choice by itself.
5. The first line of `compare` says whether the chosen primary is worse than the gpt-4o reference and lists the clips. This doesn't block the switch (the owner has decided), but it tells the week-1 prompt work where to start.

18 alerts is a small set (16/18 spans roughly 67–97%); a one-clip difference is noise. The report says so, and week 1 of the plan enlarges the eval set.

## Data flow

Laptop: the prepared eval set (220 clips, already on the laptop) is copied to `C:\Users\ameer\Ameer\home_guard_eval\eval_set` → `run --provider ollama` for each local model (Ollama serves one model at a time; the runs are sequential) → `run --provider openrouter` for each hosted model once the account has credit → `compare`.
Box: unchanged by this work. The switch waits for the hosting decision above.

## Error handling

- Missing key / base URL → the eval exits naming the variable; production uses the fallback or `NullBackend` with a warning.
- Ollama not running or model not pulled → connection/404 error recorded per clip; resume re-asks only failed clips.
- 429 / 5xx / timeout → recorded as an error; re-running resumes without re-paying answered clips.
- Model ignores the schema → existing `json_object` fallback; still-unparseable answers count as errors in §6.
- No `usage` in the response → tokens recorded as 0, cost unknown.

## Testing

Unit tests (pytest, no network, fake OpenAI client):
- `providers.resolve` for each provider, a missing key, a provider without a key (ollama), `vllm` without `VLLM_BASE_URL`;
- `GptBackend` passes `base_url`, `timeout`, `extra_body`, records `last_usage`; a backend built with `__new__` (as older tests do) still works;
- `FallbackBackend`: primary raises → fallback; primary junk → fallback; primary fine → fallback never called; both fail → raises;
- `make_backend`: default settings → today's single gpt-4o backend; Qwen pair → `FallbackBackend`; missing primary → fallback alone; nothing → `NullBackend`;
- eval: usage/cost/latency columns, day/night split, cost projection, `--provider`; `compare` choice rule (default kept, Qwen3.5-4B wins on alerts, wins on false alarms at equal alerts, loses by missing a forced-entry clip the default caught, loses on errors).
Then `run --fake` end to end, then a 3-clip smoke run per model before each full run.

## Needs the owner

- Credit on the OpenRouter account (it shows $0 on 2026-10-06; about $10 covers every hosted run). The local runs don't need it.
- Later: the hosting choice for the box (above).

## Out of scope

The situation-aware prompt (`eye_prompt.py`, week 1), the bigger eval set (week 1), the gateway (week 1–2), training, renting the GPU.
