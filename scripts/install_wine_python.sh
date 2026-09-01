#!/usr/bin/env bash
# scripts/install_wine_python.sh
# ─────────────────────────────────────────────────────────────────────────────
# One-time setup: installs Windows Python 3.11 + MetaTrader5 + rpyc inside
# the Wine prefix so the RPyC bridge can talk to a running MT5 terminal.
#
# Run this ONCE before using start_mt5_bridge.sh.
# Usage: bash scripts/install_wine_python.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

WINEPREFIX="${WINEPREFIX:-/home/omara/.mt5}"
WINE="${WINE:-wine-stable}"
WINEARCH="${WINEARCH:-win64}"

PYTHON_VERSION="3.11.9"
PYTHON_INSTALLER="python-${PYTHON_VERSION}-amd64.exe"
PYTHON_URL="https://www.python.org/ftp/python/${PYTHON_VERSION}/${PYTHON_INSTALLER}"
DOWNLOAD_DIR="/tmp/wine_python_setup"

export WINEPREFIX WINEARCH

echo "════════════════════════════════════════════════════"
echo "  MT5 Wine Bridge — Windows Python Setup"
echo "  WINEPREFIX: $WINEPREFIX"
echo "════════════════════════════════════════════════════"
echo ""

# ── 1. Check Wine ────────────────────────────────────────────────────────────
if ! command -v "$WINE" &>/dev/null; then
    echo "ERROR: '$WINE' not found. Install Wine first: sudo apt install wine-stable"
    exit 1
fi
echo "✓ Wine: $($WINE --version 2>/dev/null | head -1)"

# ── 2. Download Python installer ─────────────────────────────────────────────
mkdir -p "$DOWNLOAD_DIR"
INSTALLER_PATH="$DOWNLOAD_DIR/$PYTHON_INSTALLER"

if [[ -f "$INSTALLER_PATH" ]]; then
    echo "✓ Python installer already downloaded: $INSTALLER_PATH"
else
    echo "→ Downloading Python $PYTHON_VERSION (Windows AMD64)..."
    curl -L --progress-bar -o "$INSTALLER_PATH" "$PYTHON_URL"
    echo "✓ Downloaded to $INSTALLER_PATH"
fi

# ── 3. Install Python inside Wine ────────────────────────────────────────────
echo ""
echo "→ Installing Python $PYTHON_VERSION inside Wine prefix..."
echo "  (This runs silently — wait for the shell prompt to return)"
echo ""

WINEPREFIX="$WINEPREFIX" "$WINE" "$INSTALLER_PATH" \
    /quiet \
    InstallAllUsers=0 \
    PrependPath=1 \
    Include_test=0 \
    Include_pip=1 \
    Include_launcher=0 \
    2>/dev/null || true

echo "✓ Python installation complete."

# ── 4. Locate Wine Python executable ─────────────────────────────────────────
WHOAMI=$(whoami)
WINE_PYTHON_PATH="$WINEPREFIX/drive_c/users/$WHOAMI/AppData/Local/Programs/Python/Python311/python.exe"
WINE_PYTHON_ALT="$WINEPREFIX/drive_c/Program Files/Python311/python.exe"

if [[ -f "$WINE_PYTHON_PATH" ]]; then
    WPYTHON="$WINE_PYTHON_PATH"
elif [[ -f "$WINE_PYTHON_ALT" ]]; then
    WPYTHON="$WINE_PYTHON_ALT"
else
    WPYTHON=$(find "$WINEPREFIX/drive_c" -name "python.exe" 2>/dev/null \
        | grep -v "__pycache__\|test\|windows\|Windows" | head -1 || true)
fi

if [[ -z "${WPYTHON:-}" ]]; then
    echo ""
    echo "ERROR: Could not find python.exe in Wine prefix."
    echo "Try running the installer manually:"
    echo "  WINEPREFIX=$WINEPREFIX $WINE $INSTALLER_PATH"
    exit 1
fi

echo "✓ Found Wine Python: $WPYTHON"

# ── 5. Install pip packages inside Wine Python ────────────────────────────────
echo ""
echo "→ Installing MetaTrader5 and rpyc inside Wine Python..."
WINEPREFIX="$WINEPREFIX" "$WINE" "$WPYTHON" -m pip install --upgrade pip --quiet
WINEPREFIX="$WINEPREFIX" "$WINE" "$WPYTHON" -m pip install MetaTrader5 rpyc --quiet
echo "✓ MetaTrader5 + rpyc installed inside Wine."

# ── 6. Save path for bridge script ───────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/.wine_python_path"
echo "$WPYTHON" > "$CONFIG_FILE"
echo "✓ Saved Wine Python path → $CONFIG_FILE"

# ── 7. Done ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Start MT5 terminal + bridge:"
echo "     bash scripts/start_mt5_bridge.sh"
echo ""
echo "  2. Test the connection:"
echo "     source .venv/bin/activate"
echo "     python scripts/test_connection.py"
echo "════════════════════════════════════════════════════"
