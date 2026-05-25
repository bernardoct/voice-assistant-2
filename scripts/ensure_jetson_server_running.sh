#!/bin/bash
#
# Monitor and restart jetson.server if it crashes.
# Intended to be run periodically from cron on the Jetson.
#
# Example cron entry (check every 5 minutes):
#   */5 * * * * /home/bernardoct/voice-assistant-2/scripts/ensure_jetson_server_running.sh >> /var/log/jetson_server_watchdog.log 2>&1
#

set -e

PROJECT_DIR="/home/bernardoct/voice-assistant-2"
PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
LOG_FILE="/var/log/jetson_server_watchdog.log"
SERVER_LOG="/var/log/jetson_server.log"

# Override via env if needed (mirrors run_jetson VS Code launch config).
ENTITIES_PATH="${ENTITIES_PATH:-/home/bernardoct/ha_entities.json}"
WHISPER_MODEL="${WHISPER_MODEL:-base.en}"

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check if Python executable exists
if [ ! -f "$PYTHON_BIN" ]; then
    log_message "ERROR: Python executable not found at $PYTHON_BIN"
    exit 1
fi

# Check if jetson.server process is running
if pgrep -f "python.*-m jetson.server" > /dev/null 2>&1; then
    log_message "OK: jetson.server is already running"
    exit 0
fi

# Process is not running, start it
log_message "WARN: jetson.server is not running, attempting to start..."

cd "$PROJECT_DIR"

# Start jetson.server in the background with nohup to prevent terminal hangup
nohup "$PYTHON_BIN" -m jetson.server \
    --entities "$ENTITIES_PATH" \
    --model "$WHISPER_MODEL" \
    > "$SERVER_LOG" 2>&1 &
PID=$!

# Give it a moment to start (CUDA + Whisper load can take a few seconds)
sleep 3

# Verify it started
if pgrep -f "python.*-m jetson.server" > /dev/null 2>&1; then
    log_message "SUCCESS: jetson.server started successfully (PID: $PID)"
    exit 0
else
    log_message "ERROR: Failed to start jetson.server. Check $SERVER_LOG for details"
    exit 1
fi
