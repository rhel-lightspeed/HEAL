"""Jira integration for automated ticket updates via MCP tools."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from heal.integrations.formatters.jira_comment_formatter import JiraCommentFormatter


@dataclass
class JiraUpdateResult:
    """Result of Jira update operation."""

    success: bool
    tickets_updated: int
    tickets_failed: int
    error: Optional[str] = None
    fallback_file: Optional[Path] = None  # Where comments were saved if Jira failed


class JiraIntegration:
    """Handles Jira ticket updates via MCP Atlassian tools."""

    def __init__(self, dry_run: bool = False):
        """Initialize Jira integration.

        Args:
            dry_run: If True, print what would happen but don't actually update
        """
        self.dry_run = dry_run
        self.formatter = JiraCommentFormatter()

    async def update_tickets_for_pattern(
        self,
        pattern_result: Any,
        pattern_id: str,
        ticket_ids: List[str],
    ) -> JiraUpdateResult:
        """Update all tickets in a pattern with fix results.

        Args:
            pattern_result: Complete pattern fix result
            pattern_id: Pattern identifier (e.g., "EOL_CONTAINER_COMPAT")
            ticket_ids: List of Jira ticket IDs (e.g., ["RSPEED-2482", "RSPEED-2511"])

        Returns:
            JiraUpdateResult with success status
        """
        print(f"\n{'='*80}")
        print("JIRA INTEGRATION: Updating Tickets")
        print(f"{'='*80}")
        print(f"   Pattern: {pattern_id}")
        print(f"   Tickets: {len(ticket_ids)}")
        print(f"   Dry Run: {self.dry_run}")
        print()

        updated_count = 0
        failed_count = 0
        fallback_file = None

        for ticket_id in ticket_ids:
            try:
                # Format comment for this ticket
                comment = self.formatter.format_pattern_comment(
                    pattern_result=pattern_result,
                    pattern_id=pattern_id,
                    all_tickets=ticket_ids,
                    current_ticket=ticket_id,
                )

                # Post to Jira (or dry-run)
                if self.dry_run:
                    print(f"[DRY RUN] Would post to {ticket_id}")
                    print(f"  Preview: {comment[:150]}...")
                    print(
                        f"  Full comment saved to: .diagnostics/{pattern_id}/jira_preview_{ticket_id}.md\n"
                    )

                    # Save full comment for review
                    if not fallback_file:
                        fallback_file = self._create_preview_file(pattern_id)
                    self._append_to_preview(fallback_file, ticket_id, comment)
                    updated_count += 1
                else:
                    success = await self._post_comment(ticket_id, comment)

                    if success:
                        print(f"✅ Updated {ticket_id}")
                        updated_count += 1
                    else:
                        print(f"❌ Failed to update {ticket_id}")
                        failed_count += 1

                        # Save to fallback file
                        if not fallback_file:
                            fallback_file = self._create_fallback_file(pattern_id)

                        self._append_to_fallback(fallback_file, ticket_id, comment)

            except Exception as e:
                print(f"❌ Error updating {ticket_id}: {e}")
                failed_count += 1

                # Save to fallback file
                if not fallback_file:
                    fallback_file = self._create_fallback_file(pattern_id)

                self._append_to_fallback(
                    fallback_file,
                    ticket_id,
                    comment if "comment" in locals() else "Comment generation failed",
                )

        print()
        print(f"{'='*80}")
        if self.dry_run:
            print(f"[DRY RUN] Preview: {updated_count} comments")
            if fallback_file:
                print(f"Full preview saved to: {fallback_file}")
                print("To post these comments, re-run with: --enable-jira")
        else:
            print(f"Updated: {updated_count}, Failed: {failed_count}")
            if fallback_file:
                print(f"Failed comments saved to: {fallback_file}")
        print(f"{'='*80}\n")

        return JiraUpdateResult(
            success=(failed_count == 0),
            tickets_updated=updated_count,
            tickets_failed=failed_count,
            error=f"{failed_count} tickets failed" if failed_count > 0 else None,
            fallback_file=fallback_file,
        )

    async def _post_comment(self, ticket_id: str, comment_body: str) -> bool:
        """Post comment to Jira ticket via MCP tool.

        Args:
            ticket_id: Jira ticket key (e.g., "RSPEED-2482")
            comment_body: Markdown-formatted comment

        Returns:
            True if successful, False otherwise
        """
        try:
            # Try to import Claude Agent SDK
            from claude_agent_sdk import query as claude_query, ClaudeAgentOptions
        except ImportError:
            print("   ⚠️  Claude Agent SDK not available - saving to fallback file")
            return False

        # Build prompt for Claude Agent SDK
        prompt = f"""Use the Jira MCP tool to add a comment to ticket {ticket_id}.

Comment body (in Markdown):
{comment_body}

Use mcp__mcp-atlassian__jira_add_comment with:
- issue_key: "{ticket_id}"
- body: (the full comment body above)

If successful, respond with "SUCCESS".
If it fails, respond with the error message."""

        options = ClaudeAgentOptions(
            model="claude-sonnet-4-6",
            max_turns=3,
            allowed_tools=["mcp__mcp-atlassian__jira_add_comment"],
            permission_mode="auto",
        )

        try:
            final_message = None
            async for message in claude_query(prompt=prompt, options=options):
                final_message = message

            # Check if successful
            if final_message and "SUCCESS" in str(final_message):
                return True
            else:
                print(f"   Jira API response: {final_message}")
                return False

        except Exception as e:
            print(f"   Claude Agent SDK error: {e}")
            return False

    def _create_fallback_file(self, pattern_id: str) -> Path:
        """Create fallback file for comments that couldn't be posted.

        Args:
            pattern_id: Pattern identifier

        Returns:
            Path to fallback file
        """
        fallback_dir = Path(f".diagnostics/{pattern_id}")
        fallback_dir.mkdir(parents=True, exist_ok=True)

        fallback_file = fallback_dir / "JIRA_COMMENTS_FALLBACK.md"

        with open(fallback_file, "w") as f:
            f.write(f"# Jira Comments Fallback - {pattern_id}\n\n")
            f.write("These comments failed to post automatically. ")
            f.write("Copy-paste them manually to Jira tickets.\n\n")
            f.write(f"{'='*80}\n\n")

        return fallback_file

    def _append_to_fallback(self, fallback_file: Path, ticket_id: str, comment: str):
        """Append failed comment to fallback file.

        Args:
            fallback_file: Path to fallback file
            ticket_id: Jira ticket key
            comment: Comment that failed to post
        """
        with open(fallback_file, "a") as f:
            f.write(f"## {ticket_id}\n\n")
            f.write(f"Link: https://redhat.atlassian.net/browse/{ticket_id}\n\n")
            f.write(f"{comment}\n\n")
            f.write(f"{'='*80}\n\n")

    def _create_preview_file(self, pattern_id: str) -> Path:
        """Create preview file for dry-run comments.

        Args:
            pattern_id: Pattern identifier

        Returns:
            Path to preview file
        """
        preview_dir = Path(f".diagnostics/{pattern_id}")
        preview_dir.mkdir(parents=True, exist_ok=True)

        preview_file = preview_dir / "JIRA_COMMENTS_PREVIEW.md"

        with open(preview_file, "w") as f:
            f.write(f"# Jira Comments Preview (DRY RUN) - {pattern_id}\n\n")
            f.write("**NOTE:** This is a preview of what would be posted to Jira.\n")
            f.write("No actual Jira updates were made.\n\n")
            f.write("To post these comments:\n")
            f.write("1. Review the content below\n")
            f.write("2. Re-run with `--enable-jira` flag\n\n")
            f.write(f"{'='*80}\n\n")

        return preview_file

    def _append_to_preview(self, preview_file: Path, ticket_id: str, comment: str):
        """Append comment preview to preview file.

        Args:
            preview_file: Path to preview file
            ticket_id: Jira ticket key
            comment: Comment that would be posted
        """
        with open(preview_file, "a") as f:
            f.write(f"## {ticket_id}\n\n")
            f.write(f"Link: https://redhat.atlassian.net/browse/{ticket_id}\n\n")
            f.write(f"{comment}\n\n")
            f.write(f"{'='*80}\n\n")


# Synchronous wrapper for easier integration
def update_tickets_for_pattern(
    pattern_result: Any,
    pattern_id: str,
    ticket_ids: List[str],
    dry_run: bool = False,
) -> JiraUpdateResult:
    """Synchronous wrapper for updating Jira tickets.

    Args:
        pattern_result: Complete pattern fix result
        pattern_id: Pattern identifier
        ticket_ids: List of Jira ticket IDs
        dry_run: If True, preview what would happen

    Returns:
        JiraUpdateResult with success status
    """
    integration = JiraIntegration(dry_run=dry_run)

    # Run async function
    return asyncio.run(
        integration.update_tickets_for_pattern(
            pattern_result=pattern_result,
            pattern_id=pattern_id,
            ticket_ids=ticket_ids,
        )
    )
