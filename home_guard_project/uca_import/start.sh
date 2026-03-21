#!/usr/bin/env bash
# ============================================================================
#  start.sh — One-command UCA dataset import launcher
#
#  Lives in:  home_guard_project/uca_import/start.sh
#
#  Usage:
#      ./home_guard_project/uca_import/start.sh [--zip PATH] [--reencode]
#                                                [--no-yolo] [--splits ...]
#
#  What it does:
#    1. Installs uv if missing, ensures Python 3.12, runs uv sync
#    2. Locates the UCA Dataset zip (--zip flag or common paths)
#    3. Runs the UCA import pipeline
# ============================================================================
set -euo pipefail

# ── Always run from the project root (where pyproject.toml lives) ─────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── Parse CLI args ─────────────────────────────────────────────────────────
ZIP_PATH=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --zip)
            ZIP_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--zip PATH] [--reencode] [--no-yolo] [--splits ...]"
            echo ""
            echo "  --zip PATH      Path to 'UCA Dataset.zip' (auto-detected if omitted)"
            echo "  --reencode      Re-encode clips at CRF 18 for frame-accurate cuts"
            echo "  --no-yolo       Skip YOLO export"
            echo "  --splits ...    Annotation splits to import"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
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

# ============================================================================
#  STEP 1: Ensure uv is installed
# ============================================================================
step "Checking uv"

if ! command -v uv &>/dev/null; then
    info "uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
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
#  STEP 4: Locate the UCA Dataset zip
# ============================================================================
step "Locating UCA Dataset zip"

if [[ -z "$ZIP_PATH" ]]; then
    SEARCH_PATHS=(
        "$HOME/Downloads/UCA Dataset.zip"
        "$HOME/Desktop/UCA Dataset.zip"
        "$PROJECT_ROOT/UCA Dataset.zip"
    )
    for candidate in "${SEARCH_PATHS[@]}"; do
        if [[ -f "$candidate" ]]; then
            ZIP_PATH="$candidate"
            break
        fi
    done
fi

if [[ -z "$ZIP_PATH" ]] || [[ ! -f "$ZIP_PATH" ]]; then
    err "UCA Dataset zip not found."
    err "Provide it with:  $0 --zip /path/to/UCA\\ Dataset.zip"
    if [[ -z "$ZIP_PATH" ]]; then
        err "Searched: ~/Downloads, ~/Desktop, project root"
    else
        err "File not found: $ZIP_PATH"
    fi
    exit 1
fi
ok "Found zip: $ZIP_PATH"

# ============================================================================
#  STEP 5: Run the UCA import pipeline
# ============================================================================
step "Running UCA import pipeline"

$UVRUN python -m home_guard_project.uca_import \
    --zip "$ZIP_PATH" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

ok "UCA import pipeline finished."
