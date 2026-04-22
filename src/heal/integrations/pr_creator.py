"""Creates GitHub PRs for pattern fixes via gh CLI."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from heal.integrations.formatters.pr_description_formatter import PRDescriptionFormatter


@dataclass
class PRResult:
    """Result of PR creation."""

    success: bool
    pr_url: Optional[str] = None
    pr_number: Optional[int] = None
    error: Optional[str] = None


class PRCreator:
    """Creates GitHub PRs for pattern fixes."""

    def __init__(self, dry_run: bool = False):
        """Initialize PR creator.

        Args:
            dry_run: If True, print what would happen but don't create PR
        """
        self.dry_run = dry_run
        self.formatter = PRDescriptionFormatter()

    def create_pattern_pr(
        self,
        pattern_result: Any,
        branch_name: str,
        okp_mcp_root: Path,
    ) -> PRResult:
        """Create GitHub PR for pattern fix.

        Args:
            pattern_result: Complete pattern fix result
            branch_name: Git branch name (e.g., "fix/pattern-eol-container-compat")
            okp_mcp_root: Path to okp-mcp repository

        Returns:
            PRResult with success status and PR URL
        """
        print(f"\n{'='*80}")
        print("PR CREATION: GitHub Pull Request")
        print(f"{'='*80}")
        print(f"   Branch: {branch_name}")
        print(f"   Dry Run: {self.dry_run}")
        print()

        # Format PR title and body
        pr_title = self.formatter.format_pr_title(pattern_result)
        pr_body = self.formatter.format_pr_body(pattern_result)

        # Dry run mode - skip all external commands
        if self.dry_run:
            print("[DRY RUN] Would create PR:")
            print(f"  Title: {pr_title}")
            print(f"  Body: {pr_body[:200]}...")
            print()
            return PRResult(success=True, pr_url="[dry-run-url]")

        # Check prerequisites (only when not dry-run)
        prereq_check = self._check_prerequisites(okp_mcp_root)
        if not prereq_check.success:
            return prereq_check

        # Push branch to remote
        push_result = self._push_branch(branch_name, okp_mcp_root)
        if not push_result.success:
            return push_result

        return self._create_pr_with_gh(pr_title, pr_body, branch_name, okp_mcp_root)

    def _check_prerequisites(self, okp_mcp_root: Path) -> PRResult:
        """Check if gh CLI is installed and authenticated."""
        # Check gh CLI installed
        try:
            subprocess.run(
                ["gh", "--version"],
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return PRResult(
                success=False,
                error="gh CLI not installed. Install with: brew install gh",
            )

        # Check gh authenticated
        try:
            subprocess.run(
                ["gh", "auth", "status"],
                cwd=okp_mcp_root,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            return PRResult(
                success=False,
                error="gh CLI not authenticated. Run: gh auth login",
            )

        return PRResult(success=True)

    def _push_branch(self, branch_name: str, okp_mcp_root: Path) -> PRResult:
        """Push branch to remote."""
        print("Pushing branch to remote...")

        try:
            subprocess.run(
                ["git", "push", "-u", "origin", branch_name],
                cwd=okp_mcp_root,
                capture_output=True,
                text=True,
                check=True,
            )
            print(f"✅ Pushed {branch_name}")
            return PRResult(success=True)

        except subprocess.CalledProcessError as e:
            return PRResult(
                success=False,
                error=f"Failed to push branch: {e.stderr}",
            )

    def _create_pr_with_gh(
        self,
        title: str,
        body: str,
        branch_name: str,
        okp_mcp_root: Path,
    ) -> PRResult:
        """Create PR using gh CLI."""
        print("Creating PR...")

        # Build gh pr create command
        cmd = [
            "gh",
            "pr",
            "create",
            "--title",
            title,
            "--body",
            body,
            "--label",
            "pattern-fix,auto-generated,needs-review",
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=okp_mcp_root,
                capture_output=True,
                text=True,
                check=True,
            )

            # Extract PR URL from output
            pr_url = result.stdout.strip()
            pr_number = self._extract_pr_number(pr_url)

            print(f"✅ PR created: {pr_url}")

            return PRResult(
                success=True,
                pr_url=pr_url,
                pr_number=pr_number,
            )

        except subprocess.CalledProcessError as e:
            return PRResult(
                success=False,
                error=f"gh pr create failed: {e.stderr}",
            )

    def _extract_pr_number(self, pr_url: str) -> Optional[int]:
        """Extract PR number from URL."""
        try:
            # URL format: https://github.com/org/repo/pull/123
            return int(pr_url.rstrip("/").split("/")[-1])
        except (ValueError, IndexError):
            return None


# Synchronous wrapper
def create_pattern_pr(
    pattern_result: Any,
    branch_name: str,
    okp_mcp_root: Path,
    dry_run: bool = False,
) -> PRResult:
    """Create PR for pattern fix.

    Args:
        pattern_result: Complete pattern fix result
        branch_name: Git branch name
        okp_mcp_root: Path to okp-mcp repository
        dry_run: If True, preview what would happen

    Returns:
        PRResult with success status and PR URL
    """
    creator = PRCreator(dry_run=dry_run)
    return creator.create_pattern_pr(pattern_result, branch_name, okp_mcp_root)
