#!/usr/bin/env bash
# heal-hitch.sh — lock wrapper for fix.sh
#
# Acquires an exclusive lock (flock) so only one fix loop runs at a time.
# Consumes the trigger file on entry so the systemd path unit resets.
# All arguments are forwarded to runners/fix.sh.
#
# Usage:
#   ops/heal-hitch.sh PATTERN_ID [--quick] [--yolo] ...
#   ops/heal-hitch.sh                          # batch mode (all patterns)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

LOCK_FILE="${PROJECT_ROOT}/.heal.lock"
TRIGGER_FILE="${PROJECT_ROOT}/.heal-trigger"
LOG_TAG="heal-hitch"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [${LOG_TAG}] $*"; }

# --- acquire lock (non-blocking) -------------------------------------------

exec 200>"${LOCK_FILE}"
if ! flock -n 200; then
    log "SKIP: another instance is already running"
    exit 0
fi

# Lock held for the rest of this process — released automatically on exit.

# --- consume trigger --------------------------------------------------------

rm -f "${TRIGGER_FILE}"

# --- run fix.sh -------------------------------------------------------------

log "START: runners/fix.sh $*"

cd "${PROJECT_ROOT}"
EXIT_CODE=0
runners/fix.sh "$@" || EXIT_CODE=$?

log "END: exit ${EXIT_CODE}"
exit "${EXIT_CODE}"
