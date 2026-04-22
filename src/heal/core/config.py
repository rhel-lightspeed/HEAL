"""Centralized configuration from environment variables.

Provides portable path resolution and configuration for HEAL agents.
All hardcoded paths should go through this module for portability.
"""

import os
from pathlib import Path
from typing import Optional


class HEALConfig:
    """HEAL configuration from environment with sensible defaults."""

    @staticmethod
    def get_okp_mcp_root() -> Optional[Path]:
        """Get OKP-MCP repository root with auto-detection.

        Resolution order:
        1. OKP_MCP_ROOT environment variable
        2. Auto-detect relative to HEAL repo (../okp-mcp, ../../okp-mcp)
        3. Check user's home directory (~/okp-mcp)

        Returns:
            Path to okp-mcp repository, or None if not found
        """
        # 1. Check env var
        if env_path := os.getenv("OKP_MCP_ROOT"):
            path = Path(env_path)
            if path.exists():
                return path.resolve()
            else:
                raise ValueError(f"OKP_MCP_ROOT env var set but path doesn't exist: {env_path}")

        # 2. Auto-detect relative to HEAL repo
        candidates = [
            Path("../okp-mcp"),
            Path("../../okp-mcp"),
            Path.home() / "okp-mcp",  # More portable than ~/Work/okp-mcp
        ]

        for candidate in candidates:
            if candidate.exists() and (candidate / ".git").exists():
                return candidate.resolve()

        return None

    @staticmethod
    def get_lscore_deploy_root() -> Optional[Path]:
        """Get lscore-deploy repository root with auto-detection.

        Resolution order:
        1. LSCORE_DEPLOY_ROOT environment variable
        2. Auto-detect relative to HEAL repo
        3. Check user's home directory

        Returns:
            Path to lscore-deploy repository, or None if not found
        """
        # 1. Check env var
        if env_path := os.getenv("LSCORE_DEPLOY_ROOT"):
            path = Path(env_path)
            if path.exists():
                return path.resolve()
            else:
                raise ValueError(
                    f"LSCORE_DEPLOY_ROOT env var set but path doesn't exist: {env_path}"
                )

        # 2. Auto-detect
        candidates = [
            Path("../lscore-deploy"),
            Path("../../lscore-deploy"),
            Path.home() / "lscore-deploy",
        ]

        for candidate in candidates:
            if candidate.exists() and (candidate / ".git").exists():
                return candidate.resolve()

        return None

    @staticmethod
    def get_lightspeed_eval_root() -> Optional[Path]:
        """Get lightspeed-evaluation repository root with auto-detection.

        Resolution order:
        1. LIGHTSPEED_EVAL_ROOT environment variable
        2. Auto-detect relative to HEAL repo
        3. Check user's home directory

        Returns:
            Path to lightspeed-evaluation repository, or None if not found
        """
        # 1. Check env var
        if env_path := os.getenv("LIGHTSPEED_EVAL_ROOT"):
            path = Path(env_path)
            if path.exists():
                return path.resolve()
            else:
                raise ValueError(
                    f"LIGHTSPEED_EVAL_ROOT env var set but path doesn't exist: {env_path}"
                )

        # 2. Auto-detect
        candidates = [
            Path("../lightspeed-evaluation"),
            Path("../../lightspeed-evaluation"),
            Path.home() / "lightspeed-evaluation",
        ]

        for candidate in candidates:
            if candidate.exists() and (candidate / ".git").exists():
                return candidate.resolve()

        return None

    @staticmethod
    def get_solr_url() -> str:
        """Get Solr URL with localhost default.

        Returns:
            Solr URL (default: http://localhost:8983/solr/portal)
        """
        return os.getenv("SOLR_URL", "http://localhost:8983/solr/portal")

    @staticmethod
    def get_log_dir() -> Path:
        """Get debug log directory with auto-creation.

        Resolution order:
        1. HEAL_LOG_DIR environment variable
        2. Default: ~/.heal/logs

        Returns:
            Path to log directory (created if doesn't exist)
        """
        if env_path := os.getenv("HEAL_LOG_DIR"):
            log_dir = Path(env_path)
        else:
            log_dir = Path.home() / ".heal" / "logs"

        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    @staticmethod
    def get_worktree_root() -> Path:
        """Get git worktree storage directory with auto-creation.

        Resolution order:
        1. HEAL_WORKTREE_ROOT environment variable
        2. Default: ~/.heal/worktrees

        Returns:
            Path to worktree directory (created if doesn't exist)
        """
        if env_path := os.getenv("HEAL_WORKTREE_ROOT"):
            worktree_root = Path(env_path)
        else:
            worktree_root = Path.home() / ".heal" / "worktrees"

        worktree_root.mkdir(parents=True, exist_ok=True)
        return worktree_root

    @staticmethod
    def validate_environment() -> dict[str, bool]:
        """Validate HEAL environment setup.

        Returns:
            Dictionary of validation checks and their pass/fail status
        """
        checks = {}

        # Check required repositories
        checks["okp_mcp_found"] = HEALConfig.get_okp_mcp_root() is not None
        checks["lscore_deploy_found"] = HEALConfig.get_lscore_deploy_root() is not None
        checks["lightspeed_eval_found"] = HEALConfig.get_lightspeed_eval_root() is not None

        # Check write access to log/worktree directories
        try:
            log_dir = HEALConfig.get_log_dir()
            checks["log_dir_writable"] = os.access(log_dir, os.W_OK)
        except Exception:
            checks["log_dir_writable"] = False

        try:
            worktree_root = HEALConfig.get_worktree_root()
            checks["worktree_dir_writable"] = os.access(worktree_root, os.W_OK)
        except Exception:
            checks["worktree_dir_writable"] = False

        # Check Solr URL format
        solr_url = HEALConfig.get_solr_url()
        checks["solr_url_valid"] = solr_url.startswith("http")

        return checks

    @staticmethod
    def print_config_summary():
        """Print configuration summary for debugging."""
        print("HEAL Configuration:")
        print(f"  OKP-MCP root:         {HEALConfig.get_okp_mcp_root()}")
        print(f"  lscore-deploy root:   {HEALConfig.get_lscore_deploy_root()}")
        print(f"  lightspeed-eval root: {HEALConfig.get_lightspeed_eval_root()}")
        print(f"  Solr URL:             {HEALConfig.get_solr_url()}")
        print(f"  Log directory:        {HEALConfig.get_log_dir()}")
        print(f"  Worktree directory:   {HEALConfig.get_worktree_root()}")
        print()

        checks = HEALConfig.validate_environment()
        print("Environment Validation:")
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}")
