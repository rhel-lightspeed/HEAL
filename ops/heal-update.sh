#!/usr/bin/env bash
# heal-update.sh — safely update HEAL while respecting the run lock
#
# Acquires the same flock used by heal-hitch.sh.  If fix.sh is running the
# lock will be held and this script exits immediately instead of pulling
# mid-run.
#
# Usage:
#   ops/heal-update.sh              # git pull + uv sync
#   ops/heal-update.sh --check      # just report whether an update is safe

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

LOCK_FILE="${PROJECT_ROOT}/.heal.lock"
LOG_TAG="heal-update"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [${LOG_TAG}] $*"; }

# --- acquire lock (non-blocking) -------------------------------------------

exec 200>"${LOCK_FILE}"
if ! flock -n 200; then
    log "BLOCKED: HEAL is currently running — try again later"
    log "Check status: systemctl --user status heal-fix.service"
    exit 1
fi

# --- check-only mode -------------------------------------------------------

if [[ "${1:-}" == "--check" ]]; then
    log "OK: no run in progress — safe to update"
    exit 0
fi

# --- update -----------------------------------------------------------------

log "Lock acquired — updating"

cd "${PROJECT_ROOT}"

BEFORE=$(git rev-parse HEAD)
git pull --ff-only
AFTER=$(git rev-parse HEAD)

if [[ "${BEFORE}" != "${AFTER}" ]]; then
    log "Updated: ${BEFORE:0:8} → ${AFTER:0:8}"
    git log --oneline "${BEFORE}..${AFTER}"
    log "Syncing dependencies…"
    uv sync --extra dev
else
    log "Already up to date (${BEFORE:0:8})"
fi

log "Done"
