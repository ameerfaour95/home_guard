#!/usr/bin/env bash
# ============================================================================
#  prepare_remote.sh — run on the LAPTOP to prepare remote access to a box
#
#  Lives in:  home_guard_project/box/prepare_remote.sh
#
#  What it does:
#    1. Creates an SSH key pair for the box at ~/.ssh/homeguard_box (once)
#    2. Writes dist/enable_remote.ps1 with the public key filled in
#
#  Copy dist/enable_remote.ps1 to the box and run it there as administrator.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

KEY="$HOME/.ssh/homeguard_box"
OUT="$PROJECT_ROOT/dist/enable_remote.ps1"

mkdir -p "$HOME/.ssh" "$PROJECT_ROOT/dist"
if [[ ! -f "$KEY" ]]; then
    ssh-keygen -t ed25519 -N "" -C "homeguard-laptop" -f "$KEY" >/dev/null
    echo "Created SSH key: $KEY"
fi

PUBLIC_KEY="$(cat "$KEY.pub")"
# The key is base64 plus spaces, so '|' is a safe sed delimiter.
sed "s|__PUBLIC_KEY__|$PUBLIC_KEY|" "$SCRIPT_DIR/enable_remote.ps1" > "$OUT"

echo "Wrote $OUT"
echo "Connect later with:  ssh -i $KEY <user>@<box-ip>"
