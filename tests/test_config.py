"""Tests for HEAL configuration and portability."""

import os
from pathlib import Path

import pytest

from heal.core.config import HEALConfig


class TestHEALConfig:
    """Test configuration resolution and portability."""

    def test_config_module_imports(self):
        """Test config module can be imported."""
        from heal.core.config import HEALConfig

        assert HEALConfig is not None

    def test_solr_url_default(self, monkeypatch):
        """Test Solr URL has sensible default."""
        # Clear any env var
        monkeypatch.delenv("SOLR_URL", raising=False)

        url = HEALConfig.get_solr_url()
        assert url.startswith("http")
        assert "8983" in url
        assert "solr" in url.lower()

    def test_solr_url_from_env(self, monkeypatch):
        """Test Solr URL can be overridden by env var."""
        custom_url = "http://custom-solr:9999/solr/custom"
        monkeypatch.setenv("SOLR_URL", custom_url)

        url = HEALConfig.get_solr_url()
        assert url == custom_url

    def test_log_dir_creation(self, tmp_path, monkeypatch):
        """Test log directory is created if doesn't exist."""
        log_dir = tmp_path / "custom_logs"
        monkeypatch.setenv("HEAL_LOG_DIR", str(log_dir))

        result = HEALConfig.get_log_dir()

        assert result == log_dir
        assert result.exists()
        assert result.is_dir()

    def test_log_dir_default_location(self, monkeypatch):
        """Test log directory defaults to ~/.heal/logs."""
        monkeypatch.delenv("HEAL_LOG_DIR", raising=False)

        log_dir = HEALConfig.get_log_dir()

        assert log_dir == Path.home() / ".heal" / "logs"
        assert log_dir.exists()

    def test_worktree_root_creation(self, tmp_path, monkeypatch):
        """Test worktree directory is created if doesn't exist."""
        worktree_dir = tmp_path / "custom_worktrees"
        monkeypatch.setenv("HEAL_WORKTREE_ROOT", str(worktree_dir))

        result = HEALConfig.get_worktree_root()

        assert result == worktree_dir
        assert result.exists()
        assert result.is_dir()

    def test_worktree_root_default_location(self, monkeypatch):
        """Test worktree directory defaults to ~/.heal/worktrees."""
        monkeypatch.delenv("HEAL_WORKTREE_ROOT", raising=False)

        worktree_dir = HEALConfig.get_worktree_root()

        assert worktree_dir == Path.home() / ".heal" / "worktrees"
        assert worktree_dir.exists()

    def test_okp_mcp_root_from_env(self, tmp_path, monkeypatch):
        """Test OKP-MCP root can be set via env var."""
        # Create fake repo with .git directory
        fake_repo = tmp_path / "okp-mcp"
        fake_repo.mkdir()
        (fake_repo / ".git").mkdir()

        monkeypatch.setenv("OKP_MCP_ROOT", str(fake_repo))

        result = HEALConfig.get_okp_mcp_root()

        assert result == fake_repo.resolve()

    def test_okp_mcp_root_env_var_invalid_path(self, monkeypatch):
        """Test OKP-MCP root raises error if env var points to non-existent path."""
        monkeypatch.setenv("OKP_MCP_ROOT", "/nonexistent/path")

        with pytest.raises(ValueError, match="OKP_MCP_ROOT.*doesn't exist"):
            HEALConfig.get_okp_mcp_root()

    def test_okp_mcp_root_auto_detect_adjacent(self, tmp_path, monkeypatch):
        """Test OKP-MCP root auto-detects adjacent repository."""
        # Create structure:
        # parent/
        #   HEAL/  (cwd)
        #   okp-mcp/
        parent = tmp_path
        heal_dir = parent / "HEAL"
        okp_dir = parent / "okp-mcp"

        heal_dir.mkdir()
        okp_dir.mkdir()
        (okp_dir / ".git").mkdir()

        # Change to HEAL directory
        monkeypatch.chdir(heal_dir)
        monkeypatch.delenv("OKP_MCP_ROOT", raising=False)

        result = HEALConfig.get_okp_mcp_root()

        assert result is not None
        assert result.name == "okp-mcp"

    def test_validate_environment_returns_dict(self):
        """Test environment validation returns dictionary of checks."""
        checks = HEALConfig.validate_environment()

        assert isinstance(checks, dict)
        assert "okp_mcp_found" in checks
        assert "log_dir_writable" in checks
        assert "worktree_dir_writable" in checks
        assert "solr_url_valid" in checks

    def test_validate_environment_checks_are_boolean(self):
        """Test all validation checks return boolean values."""
        checks = HEALConfig.validate_environment()

        for check_name, result in checks.items():
            assert isinstance(result, bool), f"{check_name} should return bool, got {type(result)}"

    def test_print_config_summary_no_errors(self, capsys):
        """Test config summary prints without errors."""
        HEALConfig.print_config_summary()

        captured = capsys.readouterr()
        assert "HEAL Configuration:" in captured.out
        assert "Environment Validation:" in captured.out

    def test_no_hardcoded_home_emackey_paths(self):
        """Test that no source files have hardcoded /home/emackey paths."""
        import subprocess

        # Search for hardcoded paths in src/
        result = subprocess.run(
            ["grep", "-r", "/home/emackey", "src/", "--include=*.py"],
            capture_output=True,
            text=True,
        )

        # Should find nothing (exit code 1) or only in comments/docstrings
        if result.returncode == 0:
            # Found some - check they're only in comments
            for line in result.stdout.splitlines():
                # Allow in docstrings/comments (e.g., example commands)
                if '"""' in line or "#" in line or "Reference:" in line:
                    continue
                pytest.fail(
                    f"Found hardcoded /home/emackey path in source code:\n{line}\n"
                    "Use HEALConfig methods or environment variables instead."
                )

    def test_no_hardcoded_tmp_paths(self):
        """Test that source files use HEALConfig for log paths instead of /tmp."""
        import subprocess

        # Search for /tmp usage in critical files (allow in tests)
        result = subprocess.run(
            [
                "grep",
                "-r",
                'Path("/tmp/',
                "src/heal/agents/",
                "src/heal/runners/",
                "--include=*.py",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            # Found some - should all use HEALConfig.get_log_dir()
            for line in result.stdout.splitlines():
                # Allow in comments
                if "#" in line:
                    continue
                pytest.fail(
                    f"Found hardcoded /tmp path:\n{line}\n"
                    "Use HEALConfig.get_log_dir() instead for portability."
                )


class TestConfigPortability:
    """Test configuration works across different environments."""

    def test_works_without_any_env_vars(self, monkeypatch):
        """Test config works with all env vars unset (minimal setup)."""
        # Clear all HEAL env vars
        monkeypatch.delenv("OKP_MCP_ROOT", raising=False)
        monkeypatch.delenv("LSCORE_DEPLOY_ROOT", raising=False)
        monkeypatch.delenv("LIGHTSPEED_EVAL_ROOT", raising=False)
        monkeypatch.delenv("SOLR_URL", raising=False)
        monkeypatch.delenv("HEAL_LOG_DIR", raising=False)
        monkeypatch.delenv("HEAL_WORKTREE_ROOT", raising=False)

        # Should not raise errors
        solr_url = HEALConfig.get_solr_url()
        log_dir = HEALConfig.get_log_dir()
        worktree_root = HEALConfig.get_worktree_root()

        # Should have sensible defaults
        assert solr_url.startswith("http")
        assert log_dir.exists()
        assert worktree_root.exists()

        # Repos may not be found - that's OK
        okp_root = HEALConfig.get_okp_mcp_root()
        # None is acceptable if repos not present
        assert okp_root is None or okp_root.exists()

    def test_handles_windows_style_paths(self, monkeypatch):
        """Test config handles Windows-style paths (if on Windows)."""
        import platform

        if platform.system() != "Windows":
            pytest.skip("Windows-specific test")

        # Test with Windows path
        monkeypatch.setenv("HEAL_LOG_DIR", r"C:\Users\test\logs")

        # Should handle without errors
        try:
            log_dir = HEALConfig.get_log_dir()
            assert log_dir.exists()
        except Exception as e:
            pytest.fail(f"Failed to handle Windows path: {e}")
