#!/usr/bin/env bash
# install.sh — install / uninstall HEAL hitch systemd units
#
# Generates user-level systemd units with paths baked in, then enables them.
#
# Usage:
#   ops/install.sh                     # install and enable
#   ops/install.sh uninstall           # disable and remove
#   ops/install.sh status              # show unit status
#
# Prerequisites:
#   - systemd with user session (loginctl enable-linger $USER)
#   - Environment file at ~/.heal/env (see ops/README.md)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

UNIT_DIR="${HOME}/.config/systemd/user"
ENV_FILE="${HOME}/.heal/env"
SERVICE_NAME="heal-fix"

TIMER_INTERVAL="${HEAL_TIMER_INTERVAL:-4h}"

# ---------------------------------------------------------------------------
# Generate units
# ---------------------------------------------------------------------------

generate_service() {
    cat <<EOF
[Unit]
Description=HEAL pattern fix loop (hitch)
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=${PROJECT_ROOT}
ExecStart=${PROJECT_ROOT}/ops/heal-hitch.sh
EnvironmentFile=-${ENV_FILE}
TimeoutStartSec=14400

[Install]
WantedBy=default.target
EOF
}

generate_timer() {
    cat <<EOF
[Unit]
Description=HEAL hitch timer (periodic)

[Timer]
OnBootSec=5min
OnUnitActiveSec=${TIMER_INTERVAL}
Persistent=true

[Install]
WantedBy=timers.target
EOF
}

generate_path() {
    cat <<EOF
[Unit]
Description=HEAL hitch trigger (touch-to-run)

[Path]
PathExists=${PROJECT_ROOT}/.heal-trigger
Unit=${SERVICE_NAME}.service

[Install]
WantedBy=default.target
EOF
}

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

do_install() {
    echo "Installing HEAL hitch systemd units…"
    echo "  Project root : ${PROJECT_ROOT}"
    echo "  Timer interval: ${TIMER_INTERVAL}"
    echo "  Env file     : ${ENV_FILE}"
    echo ""

    # Check env file
    if [[ ! -f "${ENV_FILE}" ]]; then
        echo "⚠  No environment file at ${ENV_FILE}"
        echo "   Create it with your Vertex AI credentials:"
        echo ""
        echo "   mkdir -p ~/.heal"
        echo "   cat > ~/.heal/env <<'ENVEOF'"
        echo "   CLAUDE_CODE_USE_VERTEX=1"
        echo "   ANTHROPIC_VERTEX_PROJECT_ID=your-project-id"
        echo "   CLOUD_ML_REGION=global"
        echo "   VERTEX_LOCATION=global"
        echo "   GOOGLE_CLOUD_PROJECT=your-project-id"
        echo "   ENVEOF"
        echo ""
        echo "   Then re-run this script."
        exit 1
    fi

    # Ensure scripts are executable
    chmod +x "${PROJECT_ROOT}/ops/heal-hitch.sh"
    chmod +x "${PROJECT_ROOT}/ops/heal-update.sh"
    chmod +x "${PROJECT_ROOT}/runners/fix.sh"

    # Write units
    mkdir -p "${UNIT_DIR}"
    generate_service > "${UNIT_DIR}/${SERVICE_NAME}.service"
    generate_timer   > "${UNIT_DIR}/${SERVICE_NAME}.timer"
    generate_path    > "${UNIT_DIR}/${SERVICE_NAME}-trigger.path"

    # Enable
    systemctl --user daemon-reload
    systemctl --user enable --now "${SERVICE_NAME}.timer"
    systemctl --user enable --now "${SERVICE_NAME}-trigger.path"

    echo ""
    echo "✓ Installed and enabled."
    echo ""
    echo "Commands:"
    echo "  systemctl --user status ${SERVICE_NAME}.timer       # timer status"
    echo "  systemctl --user start  ${SERVICE_NAME}.service     # run now"
    echo "  touch ${PROJECT_ROOT}/.heal-trigger                 # trigger run"
    echo "  journalctl --user -u ${SERVICE_NAME}.service -f     # follow logs"
}

# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------

do_uninstall() {
    echo "Removing HEAL hitch systemd units…"

    systemctl --user disable --now "${SERVICE_NAME}.timer"         2>/dev/null || true
    systemctl --user disable --now "${SERVICE_NAME}-trigger.path"  2>/dev/null || true
    systemctl --user stop "${SERVICE_NAME}.service"                2>/dev/null || true

    rm -f "${UNIT_DIR}/${SERVICE_NAME}.service"
    rm -f "${UNIT_DIR}/${SERVICE_NAME}.timer"
    rm -f "${UNIT_DIR}/${SERVICE_NAME}-trigger.path"

    systemctl --user daemon-reload

    echo "✓ Removed."
}

# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

do_status() {
    echo "HEAL hitch status"
    echo "─────────────────"
    systemctl --user status "${SERVICE_NAME}.timer"         2>/dev/null || echo "(timer not installed)"
    echo ""
    systemctl --user status "${SERVICE_NAME}-trigger.path"  2>/dev/null || echo "(trigger not installed)"
    echo ""
    systemctl --user status "${SERVICE_NAME}.service"       2>/dev/null || echo "(service not installed)"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

case "${1:-install}" in
    install)    do_install   ;;
    uninstall)  do_uninstall ;;
    status)     do_status    ;;
    *)
        echo "Usage: $0 [install|uninstall|status]"
        exit 1
        ;;
esac
