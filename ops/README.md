# ops/ — HEAL Hitch

Infrastructure for running HEAL's fix loop on a schedule.

## Concept

HEAL's `runners/fix.sh` runs a single fix cycle and exits. The **hitch** connects it to a timer so it runs repeatedly — the application is the cart, the hitch connects it to something that makes it move.

## Components

| File | Purpose |
|------|---------|
| `heal-hitch.sh` | Lock wrapper. Acquires `flock`, consumes trigger file, runs `fix.sh`. |
| `heal-update.sh` | Safe update. Acquires the same lock before `git pull + uv sync`. |
| `install.sh` | Generates and installs systemd user units. |

## Setup

### 1. Create the environment file

systemd services don't source `.bashrc`. Put your credentials in a file:

```bash
mkdir -p ~/.heal
cat > ~/.heal/env <<'EOF'
CLAUDE_CODE_USE_VERTEX=1
ANTHROPIC_VERTEX_PROJECT_ID=your-project-id
CLOUD_ML_REGION=global
VERTEX_LOCATION=global
GOOGLE_CLOUD_PROJECT=your-project-id
EOF
```

### 2. Enable lingering

Allows user services to run after logout:

```bash
sudo loginctl enable-linger $USER
```

### 3. Install

```bash
ops/install.sh
```

Default timer interval is 4 hours. Override with:

```bash
HEAL_TIMER_INTERVAL=6h ops/install.sh
```

## Usage

```bash
# Check status
ops/install.sh status

# Trigger a run immediately
touch .heal-trigger

# Or start directly
systemctl --user start heal-fix.service

# Follow logs
journalctl --user -u heal-fix.service -f

# Safe update (waits if running)
ops/heal-update.sh

# Check if safe to update without doing it
ops/heal-update.sh --check

# Uninstall
ops/install.sh uninstall
```

## How the lock works

`heal-hitch.sh` and `heal-update.sh` both use `flock` on `.heal.lock`:

- **fix loop starts** → lock acquired, trigger file consumed
- **another fix loop tries to start** → sees lock, exits with SKIP
- **update tries to run** → sees lock, exits with BLOCKED
- **fix loop finishes** → lock released (fd closed on process exit)

The lock is advisory (`flock`), held for the lifetime of the process. No PID files, no stale lock cleanup needed.

## Files (gitignored)

| File | Purpose |
|------|---------|
| `.heal.lock` | flock target — created automatically |
| `.heal-trigger` | Touch to trigger a run via systemd path unit |
