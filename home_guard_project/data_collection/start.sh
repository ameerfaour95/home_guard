#!/usr/bin/env bash
# ============================================================================
#  start.sh — One-command data collection launcher
#
#  Lives in:  home_guard_project/data_collection/start.sh
#
#  Usage:
#      ./home_guard_project/data_collection/start.sh [--discover]
#
#  What it does:
#    1. Installs uv if missing, ensures Python 3.12 (installs via uv), runs uv sync
#    2. Verifies config.yaml exists
#    3. Checks cameras.yaml — if missing or --discover, runs interactive
#       camera discovery (ONVIF / network scan / manual)
#    4. Launches data_collection.py
#    5. Ctrl+C cleanly shuts everything down
# ============================================================================
set -euo pipefail

# ── Always run from the project root (where pyproject.toml lives) ─────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── Parse CLI args ─────────────────────────────────────────────────────────
FORCE_DISCOVER=false
for arg in "$@"; do
    case "$arg" in
        --discover) FORCE_DISCOVER=true ;;
        -h|--help)
            echo "Usage: $0 [--discover]"
            echo ""
            echo "  --discover   Force camera discovery even if cameras.yaml exists"
            exit 0 ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
step()  { echo -e "\n${BOLD}── $* ──${NC}"; }

# ── Cleanup on exit ────────────────────────────────────────────────────────
COLLECTOR_PID=""
cleanup() {
    echo ""
    info "Shutting down..."
    if [[ -n "$COLLECTOR_PID" ]]; then
        kill "$COLLECTOR_PID" 2>/dev/null && info "Data collector stopped." || true
    fi
    ok "Goodbye."
}
trap cleanup EXIT INT TERM

# ============================================================================
#  STEP 1: Ensure uv is installed
# ============================================================================
step "Checking uv"

if ! command -v uv &>/dev/null; then
    info "uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        err "Failed to install uv. Install it manually: https://docs.astral.sh/uv/"
        exit 1
    fi
fi
ok "$(uv --version 2>&1)"

# ============================================================================
#  STEP 2: Ensure Python 3.12
# ============================================================================
step "Checking Python 3.12"

REQUIRED_PY="3.12"

uv python install "$REQUIRED_PY" 2>/dev/null || true

ACTUAL_VER="$(uv run --python "$REQUIRED_PY" python --version 2>&1)" || {
    err "Python $REQUIRED_PY could not be installed. Install it manually."
    exit 1
}
ok "$ACTUAL_VER (managed by uv)"

# ============================================================================
#  STEP 3: Install dependencies
# ============================================================================
step "Installing dependencies (uv sync)"

uv sync --python "$REQUIRED_PY" --quiet
ok "All dependencies installed."

UVRUN="uv run"

# ============================================================================
#  STEP 4: Verify config.yaml
# ============================================================================
step "Checking config.yaml"

CONFIG_YAML="${SCRIPT_DIR}/config.yaml"
if [[ ! -f "$CONFIG_YAML" ]]; then
    err "config.yaml not found at: $CONFIG_YAML"
    exit 1
fi
ok "config.yaml found."

# ============================================================================
#  STEP 5: Camera setup
# ============================================================================
step "Camera Setup"

CAMERAS_YAML="${SCRIPT_DIR}/cameras.yaml"

need_discover=false

if [[ "$FORCE_DISCOVER" == "true" ]]; then
    info "--discover flag set, running camera discovery..."
    need_discover=true
elif [[ ! -f "$CAMERAS_YAML" ]]; then
    warn "cameras.yaml not found — running camera discovery..."
    need_discover=true
else
    # cameras.yaml exists — check it has cameras
    cam_count=$($UVRUN python -c "
import yaml, sys
with open('$CAMERAS_YAML') as f:
    d = yaml.safe_load(f) or {}
cams = d.get('cameras', {})
print(len(cams))
" 2>/dev/null || echo "0")

    if [[ "$cam_count" == "0" ]]; then
        warn "cameras.yaml has no cameras — running discovery..."
        need_discover=true
    else
        ok "cameras.yaml found with $cam_count camera(s)."
    fi
fi

if [[ "$need_discover" == "true" ]]; then
    $UVRUN python home_guard_project/data_collection/discover.py
    # Re-check after discovery
    if [[ ! -f "$CAMERAS_YAML" ]]; then
        err "cameras.yaml still not found after discovery. Exiting."
        exit 1
    fi
fi

# ============================================================================
#  STEP 6: Region of Interest
# ============================================================================
step "Region of Interest (ROI)"

ZONES_YAML="${SCRIPT_DIR}/zones.yaml"

echo ""
echo "  [1] Full frame (detect everywhere — no ROI)"
echo "  [2] Use ROI zones (only trigger on your property)"
read -rp "Choose [1]: " roi_choice
roi_choice="${roi_choice:-1}"

if [[ "$roi_choice" == "2" ]]; then
    if [[ -f "$ZONES_YAML" ]]; then
        zone_count=$($UVRUN python -c "
import yaml
with open('$ZONES_YAML') as f:
    d = yaml.safe_load(f) or {}
z = d.get('zones', {})
print(len(z))
for k in z:
    print(f'  {k}: {len(z[k])} vertices')
" 2>/dev/null || echo "0")

        if [[ "$zone_count" != "0" ]]; then
            ok "Existing zones.yaml found:"
            $UVRUN python -c "
import yaml
with open('$ZONES_YAML') as f:
    d = yaml.safe_load(f) or {}
for k, v in d.get('zones', {}).items():
    print(f'    {k}: {len(v)} vertices')
" 2>/dev/null || true
            echo ""
            read -rp "  Use existing zones? [Y/n]: " use_existing
            use_existing="${use_existing:-y}"
            if [[ "$use_existing" =~ ^[Yy] ]]; then
                ok "Using existing zones."
            else
                info "Launching ROI editor to overwrite..."
                $UVRUN python home_guard_project/data_collection/roi_editor.py
            fi
        else
            info "zones.yaml has no zones — launching ROI editor..."
            $UVRUN python home_guard_project/data_collection/roi_editor.py
        fi
    else
        info "No zones.yaml found — launching ROI editor..."
        $UVRUN python home_guard_project/data_collection/roi_editor.py
    fi
else
    info "Full-frame detection selected (no ROI filtering)."
    # Remove zones.yaml so the pipeline doesn't use stale zones
    rm -f "$ZONES_YAML" 2>/dev/null || true
fi

# ============================================================================
#  STEP 7: Launch data collection
# ============================================================================
step "Starting data collection"

info "Launching data_collection.py..."
info "Press Ctrl+C to stop.\n"

# Run in the foreground with unbuffered stdout so logs stream live in Git Bash.
# Backgrounding (&) caused log lines to block-buffer and never appear.
PYTHONUNBUFFERED=1 $UVRUN python -u home_guard_project/data_collection/data_collection.py
