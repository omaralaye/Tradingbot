#!/usr/bin/env bash
# scripts/start_mt5_bridge.sh
# ─────────────────────────────────────────────────────────────────────────────
# Starts the MetaTrader 5 terminal under Wine, then launches an RPyC classic
# server via Wine Python on port 18812.  The Linux trading bot connects to
# this RPyC server to call the MetaTrader5 API as if it were running on Windows.
#
# Usage:
#   bash scripts/start_mt5_bridge.sh          # start bridge (blocks)
#   bash scripts/start_mt5_bridge.sh --stop   # kill bridge + MT5
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

WINEPREFIX="${WINEPREFIX:-/home/omara/.mt5}"
WINE="${WINE:-wine-stable}"
WINEARCH="${WINEARCH:-win64}"
BRIDGE_PORT="${MT5_BRIDGE_PORT:-18812}"
BRIDGE_HOST="${MT5_BRIDGE_HOST:-127.0.0.1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MT5_TERMINAL="$WINEPREFIX/drive_c/Program Files/MetaTrader 5/terminal64.exe"
WINE_PYTHON_CONFIG="$SCRIPT_DIR/.wine_python_path"
LOG_DIR="$SCRIPT_DIR/../logs"
PID_FILE="/tmp/mt5_bridge_rpyc.pid"

export WINEPREFIX WINEARCH

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

port_in_use() {
    ss -tlnp 2>/dev/null | grep -q ":$BRIDGE_PORT " || \
    nc -z "$BRIDGE_HOST" "$BRIDGE_PORT" 2>/dev/null
}

# ── --stop ────────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--stop" ]]; then
    log "Stopping MT5 bridge..."
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        kill "$PID" 2>/dev/null && log "Killed bridge PID $PID" || true
        rm -f "$PID_FILE"
    fi
    pkill -f "rpyc_classic" 2>/dev/null || true
    pkill -f "terminal64.exe" 2>/dev/null || true
    log "Done."
    exit 0
fi

# ── Validate ──────────────────────────────────────────────────────────────────
[[ -f "$MT5_TERMINAL" ]] || \
    die "MT5 terminal not found: $MT5_TERMINAL\nIs WINEPREFIX=$WINEPREFIX correct?"

[[ -f "$WINE_PYTHON_CONFIG" ]] || \
    die "Wine Python not configured.\nRun first: bash scripts/install_wine_python.sh"

WINE_PYTHON_EXE=$(cat "$WINE_PYTHON_CONFIG")
[[ -f "$WINE_PYTHON_EXE" ]] || \
    die "Wine Python not found: $WINE_PYTHON_EXE\nRe-run: bash scripts/install_wine_python.sh"

mkdir -p "$LOG_DIR"

# ── Already running? ──────────────────────────────────────────────────────────
if port_in_use; then
    log "RPyC bridge already listening on port $BRIDGE_PORT. Nothing to do."
    exit 0
fi

echo "════════════════════════════════════════════════════"
echo "  MT5 Wine Bridge Launcher"
echo "  WINEPREFIX : $WINEPREFIX"
echo "  RPyC port  : $BRIDGE_HOST:$BRIDGE_PORT"
echo "════════════════════════════════════════════════════"

# ── Step 1: Start MT5 terminal ────────────────────────────────────────────────
log "Starting MT5 terminal..."
WINEPREFIX="$WINEPREFIX" "$WINE" "$MT5_TERMINAL" &>/dev/null &
log "MT5 terminal launched. Waiting 8s for it to connect to broker..."
sleep 8

# ── Step 2: Start RPyC classic server via Wine Python ────────────────────────
# Use the rpyc_classic.exe script (installed by rpyc pip package) directly.
# Running via "python -m" silently fails under Wine when stdout is not a tty.
log "Starting RPyC bridge on $BRIDGE_HOST:$BRIDGE_PORT..."

# Derive Scripts/ dir from the configured Wine Python path (e.g. .../Python311/python.exe -> .../Python311/Scripts/rpyc_classic.exe)
WINE_SCRIPTS_DIR="$(dirname "$WINE_PYTHON_EXE")/Scripts"
RPYC_EXE="$WINE_SCRIPTS_DIR/rpyc_classic.exe"
[[ -f "$RPYC_EXE" ]] || die "rpyc_classic.exe not found: $RPYC_EXE\nRun: wine pip install rpyc"

WINEPREFIX="$WINEPREFIX" "$WINE" "$RPYC_EXE" \
    --host "$BRIDGE_HOST" \
    --port "$BRIDGE_PORT" \
    >> "$LOG_DIR/mt5_bridge.log" 2>&1 &
BRIDGE_PID=$!
echo "$BRIDGE_PID" > "$PID_FILE"
log "RPyC server PID: $BRIDGE_PID"

# Wait for it to start
for i in $(seq 1 30); do
    if port_in_use; then
        log "✓ Bridge listening on $BRIDGE_PORT (${i}s)."
        break
    fi
    sleep 1
    if [[ $i -eq 30 ]]; then
        die "Bridge did not start after 30s. See: $LOG_DIR/mt5_bridge.log"
    fi
done

echo ""
echo "════════════════════════════════════════════════════"
echo "  ✓ MT5 bridge is running!"
echo ""
echo "  Test:  source .venv/bin/activate && python scripts/test_connection.py"
echo "  Run:   source .venv/bin/activate && python -m live.run_bot"
echo "  Stop:  bash scripts/start_mt5_bridge.sh --stop"
echo "════════════════════════════════════════════════════"

# Keep alive: poll the port; exit when bridge dies or we receive INT/TERM
_ALIVE=1
cleanup() {
    log "Shutting down bridge..."
    _ALIVE=0
    # Kill the Wine python.exe that hosts the bridge
    pkill -f "rpyc_classic.exe" 2>/dev/null || true
    pkill -f "rpyc_classic" 2>/dev/null || true
    rm -f "$PID_FILE"
}
trap cleanup EXIT INT TERM

log "Bridge keep-alive monitor running (Ctrl-C to stop)..."
while [[ $_ALIVE -eq 1 ]]; do
    sleep 5
    if ! port_in_use; then
        log "Bridge port $BRIDGE_PORT is no longer active. Exiting."
        break
    fi
done
