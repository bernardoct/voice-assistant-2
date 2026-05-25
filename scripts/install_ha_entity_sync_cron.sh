#!/bin/bash
#
# Install (or remove) a per-minute crontab entry that syncs Home Assistant
# entities into the cache file consumed by the voice assistant.
#
# Usage:
#   ./install_ha_entity_sync_cron.sh install    # add cron entry (idempotent)
#   ./install_ha_entity_sync_cron.sh uninstall  # remove cron entry
#   ./install_ha_entity_sync_cron.sh status     # show current cron entry
#

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
[ -x "$PYTHON_BIN" ] || PYTHON_BIN="${PROJECT_DIR}/venv/bin/python"
SCRIPT="${PROJECT_DIR}/scripts/ha_entity_sync.py"
LOG="/tmp/ha_entity_sync.log"
MARKER="# voice-assistant-2 ha_entity_sync"

CRON_LINE="* * * * * ${PYTHON_BIN} ${SCRIPT} >> ${LOG} 2>&1 ${MARKER}"

usage() {
    echo "Usage: $0 [install|uninstall|status]"
    exit 1
}

case "${1:-}" in
    install)
        if [ ! -x "$PYTHON_BIN" ]; then
            echo "ERROR: Python venv not found at ${PROJECT_DIR}/.venv or ${PROJECT_DIR}/venv" >&2
            exit 1
        fi
        # Read existing crontab (suppress error if empty), drop any prior
        # entry with our marker, then append the fresh one.
        current="$(crontab -l 2>/dev/null || true)"
        new="$(printf '%s\n' "$current" | grep -vF "$MARKER" || true)"
        # Trim leading/trailing blank lines then append our line.
        printf '%s\n%s\n' "$(printf '%s' "$new" | sed -e '/./,$!d' -e ':a;/^\n*$/{$d;N;ba' -e '}')" "$CRON_LINE" | crontab -
        echo "Installed cron entry:"
        echo "  $CRON_LINE"
        # Print the effective output path so it's clear where the cron will write.
        effective_path="$(
            "$PYTHON_BIN" -c "
import os, sys
sys.path.insert(0, '${PROJECT_DIR}')
from pathlib import Path
# Honor the .env loader the sync script uses.
env_path = Path('${PROJECT_DIR}/.env')
if env_path.is_file():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, _, v = line.partition('=')
        os.environ.setdefault(k.strip(), v.strip().strip(chr(34)).strip(chr(39)))
print(os.environ.get('HA_ENTITIES_PATH') or str(Path.home() / '.cache' / 'ha_entities.json'))
"
        )"
        echo "Sync will write to: $effective_path"
        echo "Make sure the voice assistant on this machine reads from the same path."
        ;;
    uninstall)
        current="$(crontab -l 2>/dev/null || true)"
        new="$(printf '%s\n' "$current" | grep -vF "$MARKER" || true)"
        if [ -z "$new" ]; then
            crontab -r 2>/dev/null || true
        else
            printf '%s\n' "$new" | crontab -
        fi
        echo "Removed any cron entries marked '$MARKER'"
        ;;
    status)
        crontab -l 2>/dev/null | grep -F "$MARKER" || echo "No ha_entity_sync cron entry installed"
        ;;
    *)
        usage
        ;;
esac
