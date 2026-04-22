# Portability Fixes - "Works on My Machine" Issues

## Issues Found

### 1. Hardcoded User Paths (`/home/emackey`)

**Files with hardcoded paths:**

- `src/heal/runners/quick_eval_fix.py:158,179` - Hardcoded `/home/emackey/Work/{okp-mcp,lscore-deploy}`
- `src/heal/runners/fix_agent_debugger.py:68,70,432` - Hardcoded `Path.home() / "Work/..."`
- `src/heal/agents/okp_mcp_agent.py:6209-6211` - Hardcoded in main block
- `src/heal/agents/okp_mcp_llm_advisor.py:713` - Default `Path.home() / "Work/okp-mcp"`
- `src/heal/agents/okp_solr_config_analyzer.py:449` - Default `Path.home() / "Work/okp-mcp"`
- `src/heal/agents/okp_mcp_agent.py:437` - Worktree root defaults to `Path.home() / "Work/okp-mcp-worktrees"`
- `src/heal/bootstrap/split_patterns_to_eval_files.py:218,222` - Example commands with hardcoded paths

**Impact:** Code fails for users without this exact directory structure.

**Fix:** Use environment variables with sensible defaults.

---

### 2. Hardcoded Localhost URLs

**Files with localhost assumptions:**

- `src/heal/core/search/solr_rag_expert.py:59` - `http://localhost:8983/solr/portal`
- `src/heal/agents/solr_expert.py:56` - Same
- `src/heal/agents/solr_expert_rag.py:65` - Same  
- `src/heal/agents/simple_solr_agent.py:24` - `http://localhost:8983/solr`
- `src/heal/agents/rag_solr_agent.py:32` - Same
- `src/heal/agents/okp_solr_config_analyzer.py:20` - Default localhost
- `src/heal/agents/okp_solr_checker.py:17,21,218` - Hardcoded localhost

**Current state:** Some have `os.getenv("SOLR_URL")` checks but still default to localhost.

**Impact:** Fails if Solr is running on different host/port or in container/k8s.

**Fix:** Already partially implemented with `SOLR_URL` env var, but needs consistency.

---

### 3. Hardcoded Debug Log Paths (`/tmp/...`)

**Files with /tmp paths:**

- `src/heal/agents/solr_multi_agent.py:176` - `/tmp/solr_multi_agent_debug.log`
- `src/heal/agents/okp_mcp_llm_advisor.py:776` - `/tmp/claude_sdk_debug.log`
- `src/heal/agents/okp_mcp_agent.py:4733` - Reference to `/tmp/claude_sdk_debug.log`

**Impact:** 
- Fails on Windows (no `/tmp`)
- Log clutter in system temp directory
- Permission issues in locked-down environments

**Fix:** Use Python's `tempfile` module or configurable log directory.

---

### 4. Missing Environment Variable Documentation

**What's documented:**
- ✅ `GOOGLE_APPLICATION_CREDENTIALS` (in .env.example)
- ✅ `SOLR_URL` (in .env.example as optional)
- ✅ `OKP_MCP_ROOT`, `LSCORE_DEPLOY_ROOT` (in .env.example as optional)

**What's missing:**
- `ANTHROPIC_VERTEX_PROJECT_ID` (mentioned in README but not in .env.example)
- `HEAL_LOG_DIR` (not implemented - should control debug log location)
- `HEAL_WORKTREE_ROOT` (for git worktrees - currently hardcoded)

---

## Recommended Fixes

### Priority 1: Critical Path Issues

**1. Update .env.example with all required variables:**

```bash
# HEAL Configuration

# Required: Claude agents authentication (ADC)
# Run: gcloud auth application-default login
# Or set: GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json

# Required: Anthropic Vertex AI project
ANTHROPIC_VERTEX_PROJECT_ID=your-gcp-project-id

# Optional: Solr configuration (defaults to localhost:8983)
# SOLR_URL=http://localhost:8983/solr/portal

# Optional: Custom repository paths (auto-detected if not set)
# OKP_MCP_ROOT=/path/to/okp-mcp
# LSCORE_DEPLOY_ROOT=/path/to/lscore-deploy
# LIGHTSPEED_EVAL_ROOT=/path/to/lightspeed-evaluation

# Optional: Git worktree storage (defaults to ~/.heal/worktrees)
# HEAL_WORKTREE_ROOT=/path/to/worktrees

# Optional: Debug log directory (defaults to ~/.heal/logs)
# HEAL_LOG_DIR=/path/to/logs
```

**2. Create config helper module:**

`src/heal/core/config.py`:
```python
"""Centralized configuration from environment variables."""

import os
from pathlib import Path
from typing import Optional

class HEALConfig:
    """HEAL configuration from environment."""
    
    @staticmethod
    def get_okp_mcp_root() -> Optional[Path]:
        """Get OKP-MCP repository root with auto-detection."""
        # 1. Check env var
        if env_path := os.getenv("OKP_MCP_ROOT"):
            return Path(env_path)
        
        # 2. Auto-detect relative to HEAL repo
        candidates = [
            Path("../okp-mcp"),
            Path("../../okp-mcp"),
            Path.home() / "okp-mcp",  # Not Work/okp-mcp - more portable
        ]
        
        for candidate in candidates:
            if candidate.exists() and (candidate / ".git").exists():
                return candidate.resolve()
        
        return None
    
    @staticmethod
    def get_solr_url() -> str:
        """Get Solr URL with localhost default."""
        return os.getenv("SOLR_URL", "http://localhost:8983/solr/portal")
    
    @staticmethod
    def get_log_dir() -> Path:
        """Get debug log directory."""
        if env_path := os.getenv("HEAL_LOG_DIR"):
            log_dir = Path(env_path)
        else:
            log_dir = Path.home() / ".heal" / "logs"
        
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir
    
    @staticmethod
    def get_worktree_root() -> Path:
        """Get git worktree storage directory."""
        if env_path := os.getenv("HEAL_WORKTREE_ROOT"):
            worktree_root = Path(env_path)
        else:
            worktree_root = Path.home() / ".heal" / "worktrees"
        
        worktree_root.mkdir(parents=True, exist_ok=True)
        return worktree_root
```

**3. Replace hardcoded paths throughout:**

Example for `okp_mcp_llm_advisor.py`:
```python
from heal.core.config import HEALConfig

class OkpMcpLLMAdvisor(BaseAgent):
    def __init__(self, okp_mcp_root=None, ...):
        super().__init__(...)
        
        # OLD: self.okp_mcp_root = okp_mcp_root or (Path.home() / "Work/okp-mcp")
        # NEW:
        self.okp_mcp_root = okp_mcp_root or HEALConfig.get_okp_mcp_root()
        
        if not self.okp_mcp_root:
            raise ValueError(
                "OKP-MCP root not found. Set OKP_MCP_ROOT env var or "
                "place okp-mcp repository adjacent to HEAL."
            )
```

**4. Fix debug log paths:**

```python
# OLD: log_file = Path("/tmp/solr_multi_agent_debug.log")
# NEW:
from heal.core.config import HEALConfig
log_file = HEALConfig.get_log_dir() / "solr_multi_agent_debug.log"
```

---

### Priority 2: Documentation & Examples

**1. Add troubleshooting guide:**

`docs/TROUBLESHOOTING.md`:
```markdown
# Troubleshooting

## Path Resolution Issues

**Error:** `OKP-MCP root not found`

**Fix:**
1. Set environment variable:
   ```bash
   export OKP_MCP_ROOT=/path/to/okp-mcp
   ```

2. Or place repository adjacent to HEAL:
   ```
   parent-dir/
   ├── HEAL/
   └── okp-mcp/
   ```

## Solr Connection Issues

**Error:** `Solr is not accessible at http://localhost:8983/solr/portal`

**Fix:**
1. If Solr is on different host:
   ```bash
   export SOLR_URL=http://your-solr-host:8983/solr/portal
   ```

2. If using Docker:
   ```bash
   export SOLR_URL=http://host.docker.internal:8983/solr/portal
   ```

## Debug Logs

**Location:** Debug logs are written to:
- Default: `~/.heal/logs/`
- Custom: Set `HEAL_LOG_DIR=/path/to/logs`

**Files:**
- `solr_multi_agent_debug.log` - Multi-agent system calls
- `claude_sdk_debug.log` - Claude SDK interactions
```

**2. Update README setup section:**

```markdown
### Environment Setup

1. **Required: Google Cloud Authentication**
   ```bash
   gcloud auth application-default login
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env and set:
   # - ANTHROPIC_VERTEX_PROJECT_ID (required)
   # - OKP_MCP_ROOT (optional - auto-detected if adjacent)
   # - SOLR_URL (optional - defaults to localhost:8983)
   ```

3. **Verify Setup**
   ```bash
   uv run python -c "from heal.core.config import HEALConfig; print(f'OKP-MCP: {HEALConfig.get_okp_mcp_root()}')"
   ```
```

---

### Priority 3: Windows Compatibility

**1. Path separator issues:**

Use `Path` objects consistently instead of string concatenation:
```python
# GOOD
config_path = repo_root / "config" / "solr.xml"

# BAD
config_path = f"{repo_root}/config/solr.xml"
```

**2. Shell script alternatives:**

Document PowerShell equivalents for all `.sh` scripts or provide Python entry points:
```bash
# Instead of: ./scripts/demo_heal_workflow.sh
# Provide: 
uv run python -m heal.cli demo --quick
```

---

## Testing Portability

**Test on clean environment:**

```bash
# 1. Create fresh user account or container
docker run -it python:3.11 bash

# 2. Clone and setup
git clone <HEAL-repo>
cd HEAL
uv sync --extra dev

# 3. Try to run without any config
# Should fail with helpful error message, not cryptic path errors
```

**Expected behavior:**
- ❌ No hardcoded `/home/emackey` paths
- ✅ Clear error messages pointing to missing env vars
- ✅ Auto-detection works for relative paths
- ✅ Works on macOS, Linux, Windows (WSL)

---

## Implementation Checklist

- [ ] Create `src/heal/core/config.py` with HEALConfig class
- [ ] Update `.env.example` with all variables
- [ ] Replace all `Path.home() / "Work/..."` with `HEALConfig.get_*()`
- [ ] Replace all `/tmp/` log paths with `HEALConfig.get_log_dir()`
- [ ] Replace all hardcoded `localhost:8983` with `HEALConfig.get_solr_url()`
- [ ] Update README.md setup instructions
- [ ] Add `docs/TROUBLESHOOTING.md`
- [ ] Test on clean environment (Docker container)
- [ ] Add validation in __main__ blocks that fails fast with helpful errors
- [ ] Update all example commands in docstrings to not have hardcoded paths
