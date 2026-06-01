# ops/ — HEAL Hitch

Infrastructure for running HEAL's fix loop on a schedule.

## Concept

HEAL's `runners/fix.sh` runs a single fix cycle and exits. The **hitch** connects it to a timer so it runs repeatedly — the application is the cart, the hitch connects it to something that makes it move.

## Components

| File | Purpose |
|------|---------|
| `heal-hitch.sh` | Lock wrapper. Acquires `flock`, consumes trigger file, runs `fix.sh`. |
| `heal-update.sh` | Safe update. Acquires the same lock before `git pull + uv sync`. |
| `install.sh` | Generates and installs systemd user units. Runs preflight checks. |

## Full Bootstrap (new machine)

The fix loop requires pattern files to exist before it can run. Complete these steps in order:

### 1. Clone and set up

```bash
git clone https://github.com/rhel-lightspeed/HEAL.git
cd HEAL
uv sync --group dev
```

### 2. Configure credentials

systemd services don't source `.bashrc`. Put credentials in a dedicated file:

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

### 3. Set up persistent journald (recommended)

Without this, logs are lost on reboot and service failures are invisible:

```bash
sudo mkdir -p /etc/systemd/journald.conf.d
echo -e '[Journal]\nStorage=persistent' | sudo tee /etc/systemd/journald.conf.d/persistent.conf
sudo systemctl restart systemd-journald
sudo journalctl --flush
```

The `journalctl --flush` is required — without it, journald continues writing to volatile storage until the next reboot even after the config change.

Verify: `ls /var/log/journal/` should show a machine-id directory.

### 4. Enable lingering

Allows user services to run after logout:

```bash
sudo loginctl enable-linger $USER
```

### 5. Run the data pipeline

These steps populate `config/patterns/` which the fix loop needs:

```bash
./runners/extract.sh          # Extract tickets from JIRA (needs JIRA creds)
./runners/pattern.sh           # Discover patterns via Claude
./runners/split.sh             # Split into per-pattern YAMLs
```

After this, `config/patterns/` should contain one `.yaml` per pattern.

### 6. Install the timer

```bash
ops/install.sh
```

The installer runs preflight checks and will refuse to install if prerequisites are missing.

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
journalctl _SYSTEMD_USER_UNIT=heal-fix.service -f

# View recent runs
journalctl _SYSTEMD_USER_UNIT=heal-fix.service --since "24 hours ago"

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

## Troubleshooting

**Service keeps failing:**
```bash
journalctl _SYSTEMD_USER_UNIT=heal-fix.service -n 50

# Most common cause: config/patterns/ is empty
ls config/patterns/
# If empty, run the data pipeline (step 5 above)
```

**No journal entries:**
```bash
# Use _SYSTEMD_USER_UNIT (not --user -u, which queries a different journal namespace)
journalctl _SYSTEMD_USER_UNIT=heal-fix.service -n 20

# If still empty, check persistent journald is set up
ls /var/log/journal/
# If missing, see step 3 above
```

**Timer not firing after logout:**
```bash
loginctl show-user $USER -p Linger
# If Linger=no, see step 4 above
```

## Files (gitignored)

| File | Purpose |
|------|---------|
| `.heal.lock` | flock target — created automatically |
| `.heal-trigger` | Touch to trigger a run via systemd path unit |
