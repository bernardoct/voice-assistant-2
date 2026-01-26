#!/bin/bash
# Setup script for Voice Assistant Jetson Server systemd service
# Run with: sudo ./setup_jetson_service.sh install
# Remove with: sudo ./setup_jetson_service.sh uninstall

set -e

SERVICE_NAME="voice-assistant-server"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
JOURNAL_CONF="/etc/systemd/journald.conf.d/voice-assistant.conf"

# Detect current user (the one who invoked sudo)
if [ -n "$SUDO_USER" ]; then
    ACTUAL_USER="$SUDO_USER"
else
    ACTUAL_USER="$(whoami)"
fi

# Default paths - adjust if needed
INSTALL_DIR="/home/${ACTUAL_USER}/voice-assistant-2"
VENV_PATH="${INSTALL_DIR}/venv"
ENTITIES_PATH="${INSTALL_DIR}/ha_entities.json"

print_usage() {
    echo "Usage: sudo $0 [install|uninstall|status|logs]"
    echo ""
    echo "Commands:"
    echo "  install   - Install and start the service"
    echo "  uninstall - Stop and remove the service"
    echo "  status    - Show service status"
    echo "  logs      - Show recent logs"
}

check_root() {
    if [ "$EUID" -ne 0 ]; then
        echo "Error: This script must be run as root (use sudo)"
        exit 1
    fi
}

install_service() {
    echo "=== Installing Voice Assistant Jetson Server ==="

    # Verify paths exist
    if [ ! -d "$INSTALL_DIR" ]; then
        echo "Error: Install directory not found: $INSTALL_DIR"
        exit 1
    fi

    if [ ! -f "${VENV_PATH}/bin/python" ]; then
        echo "Error: Virtual environment not found: $VENV_PATH"
        exit 1
    fi

    if [ ! -f "$ENTITIES_PATH" ]; then
        echo "Warning: Entities file not found: $ENTITIES_PATH"
        echo "Make sure to copy ha_entities.json from the Pi before starting the service."
    fi

    # Create journal config directory if needed
    mkdir -p /etc/systemd/journald.conf.d

    # Configure journal size limits to prevent log bloat
    echo "Configuring journal size limits..."
    cat > "$JOURNAL_CONF" << 'EOF'
# Voice Assistant journal limits
# Prevents transcription logs from filling disk
[Journal]
SystemMaxUse=100M
SystemMaxFileSize=10M
MaxRetentionSec=7day
EOF

    # Restart journald to apply changes
    systemctl restart systemd-journald

    # Create systemd service file
    echo "Creating systemd service..."
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Voice Assistant Jetson Server (Whisper + Intent Parsing)
After=network.target

[Service]
Type=simple
User=${ACTUAL_USER}
Group=${ACTUAL_USER}
WorkingDirectory=${INSTALL_DIR}

# Environment
Environment="PATH=${VENV_PATH}/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
Environment="LD_LIBRARY_PATH=/usr/local/cuda/lib64"
Environment="CUDA_HOME=/usr/local/cuda"

# Command
ExecStart=${VENV_PATH}/bin/python run_jetson.py --entities ${ENTITIES_PATH}

# Restart policy
Restart=always
RestartSec=5

# Logging - limit output to prevent journal bloat
StandardOutput=journal
StandardError=journal
SyslogIdentifier=${SERVICE_NAME}

# Resource limits
Nice=0
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd and enable service
    echo "Enabling service..."
    systemctl daemon-reload
    systemctl enable "$SERVICE_NAME"

    # Start service
    echo "Starting service..."
    systemctl start "$SERVICE_NAME"

    # Show status
    sleep 2
    systemctl status "$SERVICE_NAME" --no-pager || true

    echo ""
    echo "=== Installation Complete ==="
    echo ""
    echo "Useful commands:"
    echo "  sudo systemctl status $SERVICE_NAME   # Check status"
    echo "  sudo systemctl restart $SERVICE_NAME  # Restart"
    echo "  sudo systemctl stop $SERVICE_NAME     # Stop"
    echo "  sudo journalctl -u $SERVICE_NAME -f   # Follow logs"
    echo "  sudo journalctl -u $SERVICE_NAME -n 50 # Last 50 lines"
}

uninstall_service() {
    echo "=== Uninstalling Voice Assistant Jetson Server ==="

    # Stop service if running
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        echo "Stopping service..."
        systemctl stop "$SERVICE_NAME"
    fi

    # Disable service
    if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
        echo "Disabling service..."
        systemctl disable "$SERVICE_NAME"
    fi

    # Remove service file
    if [ -f "$SERVICE_FILE" ]; then
        echo "Removing service file..."
        rm -f "$SERVICE_FILE"
    fi

    # Remove journal config
    if [ -f "$JOURNAL_CONF" ]; then
        echo "Removing journal config..."
        rm -f "$JOURNAL_CONF"
    fi

    # Reload systemd
    systemctl daemon-reload

    echo ""
    echo "=== Uninstallation Complete ==="
    echo "Note: The application files in $INSTALL_DIR were not removed."
}

show_status() {
    systemctl status "$SERVICE_NAME" --no-pager
}

show_logs() {
    journalctl -u "$SERVICE_NAME" -n 100 --no-pager
}

# Main
check_root

case "${1:-}" in
    install)
        install_service
        ;;
    uninstall)
        uninstall_service
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    *)
        print_usage
        exit 1
        ;;
esac
