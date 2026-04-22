"""Integration modules for external systems (Jira, GitHub)."""

from heal.integrations.jira_integration import (
    JiraIntegration,
    JiraUpdateResult,
    update_tickets_for_pattern,
)
from heal.integrations.pr_creator import (
    PRCreator,
    PRResult,
    create_pattern_pr,
)

__all__ = [
    "JiraIntegration",
    "JiraUpdateResult",
    "update_tickets_for_pattern",
    "PRCreator",
    "PRResult",
    "create_pattern_pr",
]
