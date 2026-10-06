# Home Guard API gateway

Today every box calls OpenAI itself, with our API key in its `api_key.env`: anyone who
can open the box can take the key. The gateway moves the keys to our server.

```
box ──(box token, model "eye")──▶ gateway ──(our key, gpt-4o)──▶ OpenAI
                                    │      └─(fallback)────────▶ OpenRouter / our GPU
                                    └─ SQLite: who called, which model, tokens, $, ms, status
```

- **Keys stay with us.** A box holds only its own token (`hgb_...`); the server stores
  only the token's sha256. A stolen token is revoked with one command and can at most
  spend that box's daily cap.
- **Model aliases.** A box asks for `eye`; `gateway.yaml` says which provider and model
  that is today. Change the YAML and restart: every box switches, none is updated.
- **Fallback.** Each alias is an ordered list of upstreams. 429, 5xx, timeouts and
  connection errors retry once, then go to the next upstream; 401/403/404 (our key, a
  removed model) go straight to the next. Other 4xx are the request's own fault and go
  back to the box unchanged (the box's "no json_schema, use json_object" switch needs that).
  The whole request has a deadline (`deadline_sec`, 27 s) under the box's 30 s timeout.
- **Caps.** Per box per UTC day (`caps.box_daily_usd`, or a box's own `--cap`), and for
  the whole fleet (`caps.fleet_daily_usd`). Over the cap the box gets HTTP 402 with code
  `box_daily_cap` / `fleet_daily_cap`. The box then alerts from the detector alone
  ("a person or vehicle was detected"), exactly as when OpenAI is down: alerts are never
  dropped for money. A cap can be overshot by the calls already in flight (cents).
- **Metering.** One row per upstream attempt in `calls` (box, time, endpoint, alias,
  provider, model, tokens in/out, $, latency, status, short error), and a running total
  per box and day in `spend`. Prices come from `box/providers.py` `PRICES`, the same table
  the eval reports use; an upstream with no price is refused at start-up unless the
  config gives one (`price_per_m: [0, 0]` for our own GPU).
- **Never logged:** prompts, images, keys, tokens. An upstream's error message is kept
  cut to 200 characters with data URLs and key-like strings removed.
- **Endpoints:** `POST /v1/chat/completions`, `POST /v1/embeddings`, `GET /v1/models`
  (box token); `GET /healthz`; `GET /admin/costs?days=7&box=<id>`, `GET /admin/boxes`
  (admin token). No streaming (the box does not stream) and no audio (the box does not
  transcribe). Every error the gateway sends has `x-should-retry: false`, so the box's
  OpenAI SDK does not repeat a call the gateway already retried.

Two dependencies, both already in the repo's venv: `httpx`, `pyyaml`. The server is the
standard library's threaded `http.server`: thousands of homes at about one call per ten
minutes is a few requests a second, each waiting a few seconds on a provider, which a
1-vCPU VPS handles. Above ~20 requests a second, run two containers behind Caddy (the
SQLite file must then become Postgres, or one file per container).

## Run it on the laptop

```bash
cp home_guard_project/gateway/gateway.example.yaml gateway.yaml     # db_path: gateway.sqlite3
export OPENAI_API_KEY=sk-...                                          # the server's keys
uv run python -m home_guard_project.gateway add-box house1           # prints HOMEGUARD_BOX_TOKEN=hgb_...
uv run python -m home_guard_project.gateway serve --host 127.0.0.1 --port 8080
```

(On this laptop, unset `SSLKEYLOGFILE` first: the antivirus's value crashes Python's
OpenSSL with "no OPENSSL_Applink".)

Tests (no network, no paid calls):

```bash
uv run --system-certs --no-sync --with pytest python -m pytest tests/gateway tests/box/test_providers.py -q -p no:cacheprovider
```

## Commands

```
python -m home_guard_project.gateway serve            [--host H] [--port P]
python -m home_guard_project.gateway add-box ID       [--cap 0.50] [--note "..."] [--rotate]
python -m home_guard_project.gateway revoke-box ID
python -m home_guard_project.gateway set-cap ID 0.75  (or "default")
python -m home_guard_project.gateway list-boxes
python -m home_guard_project.gateway report           [--days 7] [--box ID]
python -m home_guard_project.gateway prune --keep-days 90
python -m home_guard_project.gateway new-admin-token
```

The config is `--config`, else `$HOMEGUARD_GATEWAY_CONFIG`, else `./gateway.yaml`. Use the
box's site name as its ID (`registration.json` `site`), so the Admin Center can join the
two. A token is printed once.

## Deploy (the owner does this; nothing here creates cloud resources)

1. **A VPS.** Any 1 vCPU / 1-2 GB Linux box with Docker: Hetzner CX22 (~4 EUR/month) or a
   DigitalOcean / Lightsail $5-6 instance. No GPU. Pick a region near the homes (Europe).
2. **DNS + TLS.** An A record, e.g. `gw.<your-domain>` → the VPS's IP. Put Caddy in front
   for automatic Let's Encrypt certificates; the gateway itself speaks plain HTTP on 8080
   and must not be exposed directly (firewall: only 22, 80, 443 open).

   `Caddyfile`:
   ```
   gw.example.com {
       request_body { max_size 10MB }
       reverse_proxy 127.0.0.1:8080
   }
   ```
3. **Keys on the server only.** `/etc/homeguard/gateway.env`, `chmod 600`, owned by root:
   ```
   OPENAI_API_KEY=sk-...
   OPENROUTER_API_KEY=sk-or-...
   HOMEGUARD_ADMIN_TOKEN_SHA256=<from new-admin-token>
   ```
   Give the gateway its own OpenAI project key with a monthly budget set in the OpenAI
   dashboard, as a last line under the gateway's own caps.
4. **Run it.**
   ```bash
   docker build -f home_guard_project/gateway/Dockerfile -t homeguard-gateway .
   docker run -d --name gateway --restart unless-stopped -p 127.0.0.1:8080:8080 \
     --env-file /etc/homeguard/gateway.env \
     -v /srv/homeguard/gateway.yaml:/config/gateway.yaml:ro -v /srv/homeguard/data:/data \
     homeguard-gateway
   docker exec gateway python -m home_guard_project.gateway add-box house1 --note "..."
   ```
   Back up `/srv/homeguard/data/gateway.sqlite3` (token hashes + metering) daily.
5. **Switch a box.** In the box's `api_key.env`:
   ```
   HOMEGUARD_GATEWAY_URL=https://gw.example.com/v1
   HOMEGUARD_BOX_TOKEN=hgb_...
   # everything else that speaks OpenAI on the box (assistant, live view, embeddings, brain vision):
   OPENAI_BASE_URL=https://gw.example.com/v1
   OPENAI_API_KEY=hgb_...            # the box token, not a real key
   ```
   and in `box.yaml`: `vlm_provider: gateway`, `vlm_model: eye` (clear
   `vlm_fallback_provider` / `vlm_fallback_model`: the gateway falls back now). Restart
   the box's program, watch one alert arrive, check `report --box <id>`, then delete the
   real OpenAI key from that box. A box that was never switched keeps calling OpenAI.

## Not covered yet

- Anthropic (`ANTHROPIC_API_KEY`) and Gemini models in the box's brain call those providers
  directly; they are not OpenAI-shaped and do not go through the gateway.
- The box records `eye` as the teacher model in its training metadata; the model that
  actually answered is in the response's `model` field and the gateway's `calls` table.
- Caps are per UTC day (they reset at 02:00 or 03:00 Israel time).
- Every model name a box asks for needs an alias: add one for `agent_model` and any other
  model set in `box.yaml` before switching that box, or those calls get 400 `model_not_found`.
