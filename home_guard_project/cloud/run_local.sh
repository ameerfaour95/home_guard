#!/usr/bin/env bash
# Run the Admin Center API on this laptop: embedded Postgres (pgserver), loops on, 127.0.0.1:8610.
# Reads S3 with the laptop's AWS profile; writes only under fleet/, admin_cache/, training_exports/.
#
# Port: set HG_CLOUD_PORT to listen elsewhere (default 8610), e.g. `HG_CLOUD_PORT=8700 ./run_local.sh`.
# Logs: timestamped INFO lines on stderr ("<time> <LEVEL> <logger> <message>"); a failing background loop shows as
#   "... ERROR home_guard_project.cloud.loops loop <name> iteration failed" with its traceback.
# Manual passes (`manage index-once`, `manage media-once`) wait for the server's loops: while the server is indexing
#   or making media they print "the server is already ..." and do nothing. `manage media-retry [--event ID]` makes
#   the media loop retry clips whose media failed.
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
# Founder's laptop: the Admin Center signs in with no password (loopback only). Override in cloud.env.
export HG_CLOUD_LOCAL_TRUST="${HG_CLOUD_LOCAL_TRUST:-1}"
export HG_CLOUD_LOCAL_ADMIN="${HG_CLOUD_LOCAL_ADMIN:-ameerfaour95@gmail.com}"

unset VIRTUAL_ENV
export UV_SYSTEM_CERTS=1
cd "$ROOT"

# Antivirus TLS interception: drop the key-log/startup hooks, use the exported CA bundle.
RUN=(env -u SSLKEYLOGFILE -u PYTHONSTARTUP AWS_CA_BUNDLE=C:/Users/ameer/.homeguard/ca_bundle.pem
     uv run --group cloud --system-certs)

# Start the embedded Postgres and publish its URI to this shell (no secrets are printed).
# After an unclean shutdown Postgres first recovers (about 30 s on this laptop: it retries a file the antivirus
# holds), longer than pgserver's 10 s start timeout. The server keeps starting, so ask again until it answers.
HG_CLOUD_DB_URL=""
for attempt in 1 2 3 4 5 6; do
  if HG_CLOUD_DB_URL="$("${RUN[@]}" python -c 'import pgserver,sys; print(pgserver.get_server(sys.argv[1], cleanup_mode=None).get_uri())' "$PG_DIR" | tail -n1)" \
     && [ -n "$HG_CLOUD_DB_URL" ]; then
    break
  fi
  echo "database not ready yet (attempt ${attempt}); waiting 10 s"
  HG_CLOUD_DB_URL=""
  sleep 10
done
[ -n "$HG_CLOUD_DB_URL" ] || { echo "the embedded database did not start; see ${PG_DIR}/log" >&2; exit 1; }
export HG_CLOUD_DB_URL
export HG_CLOUD_RUN_LOOPS=1

"${RUN[@]}" python -m home_guard_project.cloud.manage init-db
# The launcher compares this with the folder's current commit and restarts a server that runs older code.
git -C "$ROOT" rev-parse HEAD > "${HG_HOME}/cloud_server.version" 2>/dev/null || true
echo "Admin Center API on http://127.0.0.1:${HG_CLOUD_PORT:-8610} (loops on)"
exec "${RUN[@]}" python -m uvicorn home_guard_project.cloud.app:create_app_from_env --factory \
  --host 127.0.0.1 --port "${HG_CLOUD_PORT:-8610}"
