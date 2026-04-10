#!/usr/bin/env bash
# ============================================================================
#  start.sh — One-command SmartHome-Bench dataset import launcher
#
#  Lives in:  home_guard_project/smarthome_import/start.sh
#
#  Usage:
#      ./home_guard_project/smarthome_import/start.sh [--repo PATH] [options]
#
#  What it does:
#    1. Installs uv if missing, ensures Python 3.12, runs uv sync
#    2. Ensures yt-dlp is installed (for YouTube downloads)
#    3. Clones the SmartHome-Bench-LLM repo if not already present
#    4. Runs the import pipeline (download + trim + window + YOLO)
# ============================================================================
set -euo pipefail

# ── Always run from the project root (where pyproject.toml lives) ─────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── Parse CLI args ─────────────────────────────────────────────────────────
REPO_PATH=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)
            REPO_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--repo PATH] [--skip-download] [--reencode] [--no-yolo]"
            echo "         [--window-size N] [--categories ...]"
            echo ""
            echo "  --repo PATH        Path to SmartHome-Bench-LLM repo (auto-cloned if omitted)"
            echo "  --skip-download    Skip YouTube downloads (use existing videos)"
            echo "  --reencode         Re-encode clips at CRF 18"
            echo "  --no-yolo          Skip YOLO export"
            echo "  --window-size N    Window duration in seconds (default: 10)"
            echo "  --categories ...   Categories to include"
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
#  STEP 4: Ensure yt-dlp is available
# ============================================================================
step "Checking yt-dlp"

if $UVRUN python -c "import subprocess; subprocess.run(['yt-dlp','--version'],capture_output=True,check=True)" 2>/dev/null; then
    ok "yt-dlp is available"
else
    info "Installing yt-dlp..."
    uv pip install yt-dlp --quiet 2>/dev/null || {
        warn "Could not install yt-dlp via uv pip. Trying pip directly..."
        pip install yt-dlp --quiet 2>/dev/null || {
            err "Failed to install yt-dlp. Install it manually: pip install yt-dlp"
            exit 1
        }
    }
    ok "yt-dlp installed"
fi

# ============================================================================
#  STEP 5: Clone or locate the SmartHome-Bench-LLM repository
# ============================================================================
step "Locating SmartHome-Bench-LLM repository"

if [[ -z "$REPO_PATH" ]]; then
    DEFAULT_REPO="$PROJECT_ROOT/SmartHome-Bench-LLM"
    if [[ -d "$DEFAULT_REPO" ]] && [[ -f "$DEFAULT_REPO/Videos/Video_Annotation.csv" ]]; then
        REPO_PATH="$DEFAULT_REPO"
        ok "Found existing repo: $REPO_PATH"
    else
        info "Cloning SmartHome-Bench-LLM repository..."
        git clone --depth 1 https://github.com/Xinyi-0724/SmartHome-Bench-LLM.git "$DEFAULT_REPO" || {
            err "Failed to clone repository."
            err "Clone it manually: git clone https://github.com/Xinyi-0724/SmartHome-Bench-LLM.git"
            exit 1
        }
        REPO_PATH="$DEFAULT_REPO"
        ok "Cloned to: $REPO_PATH"
    fi
else
    if [[ ! -f "$REPO_PATH/Videos/Video_Annotation.csv" ]]; then
        err "Video_Annotation.csv not found in $REPO_PATH/Videos/"
        err "Make sure --repo points to the SmartHome-Bench-LLM repository root."
        exit 1
    fi
    ok "Using repo: $REPO_PATH"
fi

# ============================================================================
#  STEP 6: Run the import pipeline
# ============================================================================
step "Running SmartHome-Bench import pipeline"

$UVRUN python -m home_guard_project.smarthome_import \
    --repo "$REPO_PATH" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

ok "SmartHome-Bench import pipeline finished."
