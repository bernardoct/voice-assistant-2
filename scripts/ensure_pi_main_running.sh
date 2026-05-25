#!/bin/bash
#
# Monitor and restart pi.main if it crashes.
# Intended to be run periodically from cron.
#
# Example cron entry (check every 5 minutes):
#   */5 * * * * /home/bernardoct/voice-assistant-2/scripts/ensure_pi_main_running.sh >> /var/log/pi_main_watchdog.log 2>&1
#

set -e

PROJECT_DIR="/home/bernardoct/voice-assistant-2"
PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
LOG_FILE="/var/log/pi_main_watchdog.log"

# Function to log messages with timestamps
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check if Python executable exists
if [ ! -f "$PYTHON_BIN" ]; then
    log_message "ERROR: Python executable not found at $PYTHON_BIN"
    exit 1
fi

# Check if pi.main process is running
# Look for python processes running the pi.main module
if pgrep -f "python.*-m pi.main" > /dev/null 2>&1; then
    log_message "OK: pi.main is already running"
    exit 0
fi

# Process is not running, start it
log_message "WARN: pi.main is not running, attempting to start..."

cd "$PROJECT_DIR"

# Start pi.main in the background with nohup to prevent terminal hangup
nohup "$PYTHON_BIN" -m pi.main > /var/log/pi_main.log 2>&1 &
PID=$!

# Give it a moment to start
sleep 2

# Verify it started
if pgrep -f "python.*-m pi.main" > /dev/null 2>&1; then
    log_message "SUCCESS: pi.main started successfully (PID: $PID)"
    exit 0
else
    log_message "ERROR: Failed to start pi.main. Check /var/log/pi_main.log for details"
    exit 1
fi
