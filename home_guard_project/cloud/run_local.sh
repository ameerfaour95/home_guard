#!/usr/bin/env bash
# Run the Admin Center API on this laptop: embedded Postgres (pgserver), loops on, 127.0.0.1:8600.
# Reads S3 with the laptop's AWS profile; writes only under fleet/, admin_cache/, training_exports/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HG_HOME="${HOME}/.homeguard"
PG_DIR="${HG_HOME}/cloud_pg"
ENV_FILE="${HG_HOME}/cloud.env"
mkdir -p "$HG_HOME" "$PG_DIR"

if [ ! -f "$ENV_FILE" ]; then
  umask 077
  secret="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
  {
    echo "HG_CLOUD_JWT_SECRET=${secret}"
    echo "HG_CLOUD_BUCKET=security-camera-project-v1"
    echo "HG_CLOUD_REGION=us-east-1"
  } > "$ENV_FILE"
  chmod 600 "$ENV_FILE" 2>/dev/null || true
  echo "created ${ENV_FILE} (JWT secret generated, not shown)"
fi
set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

unset VIRTUAL_ENV
export UV_SYSTEM_CERTS=1
cd "$ROOT"

# Antivirus TLS interception: drop the key-log/startup hooks, use the exported CA bundle.
RUN=(env -u SSLKEYLOGFILE -u PYTHONSTARTUP AWS_CA_BUNDLE=C:/Users/ameer/.homeguard/ca_bundle.pem
     uv run --group cloud --system-certs)

# Start the embedded Postgres and publish its URI to this shell (no secrets are printed).
HG_CLOUD_DB_URL="$("${RUN[@]}" python -c 'import pgserver,sys; print(pgserver.get_server(sys.argv[1], cleanup_mode=None).get_uri())' "$PG_DIR")"
export HG_CLOUD_DB_URL
export HG_CLOUD_RUN_LOOPS=1

"${RUN[@]}" python -m home_guard_project.cloud.manage init-db
echo "Admin Center API on http://127.0.0.1:8600 (loops on)"
exec "${RUN[@]}" python -m uvicorn home_guard_project.cloud.app:create_app_from_env --factory \
  --host 127.0.0.1 --port 8600
