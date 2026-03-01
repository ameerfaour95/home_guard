#!/usr/bin/env bash
# ============================================================================
#  start.sh — One-command external video import launcher
#
#  Lives in:  home_guard_project/video_import/start.sh
#
#  Usage:
#      ./home_guard_project/video_import/start.sh <INPUT_PATH> [options]
#
#  INPUT_PATH can be a single video file or a directory of videos.
#
#  Options:
#      --camera-name NAME   Camera name for imported clips (default from config)
#      --output-dir DIR     Dataset output directory (default from config)
#      --clip-seconds N     Split videos into N-second chunks (default from config)
#      --store-fps N        Output FPS for clips (default from config)
#      --no-yolo            Disable YOLO frame export
#      --no-vlm-crop        Disable VLM crop generation
#
#  What it does:
#    1. Verifies Python >= 3.12 and installs uv if missing
#    2. Runs uv sync to install all dependencies
#    3. Checks ffmpeg + ffprobe are available
#    4. Validates the input path exists
#    5. Launches the import pipeline
# ============================================================================
set -euo pipefail

# ── Always run from the project root (where pyproject.toml lives) ─────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── Colors ────────────────────────────────────────────────────────────────
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

# ── Parse CLI args ────────────────────────────────────────────────────────
if [[ $# -lt 1 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo "Usage: $0 <INPUT_PATH> [--camera-name NAME] [--output-dir DIR]"
    echo "                       [--clip-seconds N] [--store-fps N]"
    echo "                       [--no-yolo] [--no-vlm-crop]"
    echo ""
    echo "  INPUT_PATH     Path to a video file or directory of videos"
    echo ""
    echo "Options:"
    echo "  --camera-name   Camera name for imported clips (default: from config)"
    echo "  --output-dir    Dataset output directory (default: from config)"
    echo "  --clip-seconds  Split videos into N-second chunks (default: from config)"
    echo "  --store-fps     Output FPS for clips (default: from config)"
    echo "  --no-yolo       Disable YOLO frame export"
    echo "  --no-vlm-crop   Disable VLM crop generation"
    exit 0
fi

INPUT_PATH="$1"
shift
EXTRA_ARGS=("$@")

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
#  STEP 4: Check ffmpeg
# ============================================================================
step "Checking ffmpeg"

FFMPEG_PATH=""
if command -v ffmpeg &>/dev/null; then
    FFMPEG_PATH="$(command -v ffmpeg)"
elif [[ -f "/c/ffmpeg/bin/ffmpeg.exe" ]]; then
    FFMPEG_PATH="/c/ffmpeg/bin/ffmpeg.exe"
fi

if [[ -n "$FFMPEG_PATH" ]]; then
    FFMPEG_VER="$("$FFMPEG_PATH" -version 2>&1 | head -1)" || FFMPEG_VER="unknown"
    ok "ffmpeg found: $FFMPEG_PATH ($FFMPEG_VER)"
else
    warn "ffmpeg not found in PATH. The pipeline will attempt auto-detection."
    warn "If it fails, install ffmpeg: https://ffmpeg.org/download.html"
fi

# ============================================================================
#  STEP 5: Validate input path
# ============================================================================
step "Validating input"

if [[ ! -e "$INPUT_PATH" ]]; then
    err "Input path does not exist: $INPUT_PATH"
    exit 1
fi

if [[ -d "$INPUT_PATH" ]]; then
    VIDEO_COUNT=$(find "$INPUT_PATH" -maxdepth 1 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" -o -name "*.mov" -o -name "*.webm" -o -name "*.flv" -o -name "*.ts" -o -name "*.m4v" \) 2>/dev/null | wc -l || echo "0")
    ok "Input directory: $INPUT_PATH ($VIDEO_COUNT video file(s))"
    if [[ "$VIDEO_COUNT" -eq 0 ]]; then
        err "No video files found in $INPUT_PATH"
        exit 1
    fi
else
    ok "Input file: $INPUT_PATH"
fi

# ============================================================================
#  STEP 6: Run the import pipeline
# ============================================================================
step "Starting video import"

info "Input: $INPUT_PATH"
echo ""

$UVRUN python -m home_guard_project.video_import "$INPUT_PATH" "${EXTRA_ARGS[@]}"

echo ""
ok "Video import complete."
