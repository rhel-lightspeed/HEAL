"""Tests for Jira integration functionality."""

from unittest.mock import MagicMock

import pytest

from heal.integrations.jira_integration import JiraIntegration, JiraUpdateResult


class TestJiraIntegration:
    """Test Jira integration functionality."""

    @pytest.fixture
    def mock_pattern_result(self):
        """Create mock pattern result."""
        result = MagicMock()
        result.success = True
        result.pattern_id = "TEST_PATTERN"
        result.branch_name = "fix/pattern-test"
        result.duration_seconds = 1234.5

        # Mock baseline
        result.baseline = MagicMock()
        result.baseline.final_metrics = {
            "answer_correctness": 0.65,
            "url_f1": 0.32,
            "faithfulness": 0.70,
            "context_relevance": 0.45,
            "success_rate": 0.50,
        }

        # Mock optimization
        result.optimization = MagicMock()
        result.optimization.final_metrics = {
            "answer_correctness": 0.92,
            "url_f1": 0.88,
            "faithfulness": 0.85,
            "context_relevance": 0.90,
            "success_rate": 0.875,
        }
        result.optimization.iterations = 5

        # Mock validation (with per-ticket results)
        result.answer_validation = MagicMock()
        result.answer_validation.final_metrics = result.optimization.final_metrics
        result.answer_validation.per_ticket_results = {
            "RSPEED-1": MagicMock(ticket_id="RSPEED-1", answer_correctness=0.95, url_f1=0.88),
            "RSPEED-2": MagicMock(ticket_id="RSPEED-2", answer_correctness=0.89, url_f1=0.85),
        }
        result.answer_validation.rag_bypass_tickets = []
        result.answer_validation.high_variance_tickets = []

        return result

    def test_jira_comment_formatting(self, mock_pattern_result):
        """Test Jira comment is properly formatted."""
        integration = JiraIntegration(dry_run=True)

        comment = integration.formatter.format_pattern_comment(
            pattern_result=mock_pattern_result,
            pattern_id="TEST_PATTERN",
            all_tickets=["RSPEED-1", "RSPEED-2"],
            current_ticket="RSPEED-1",
        )

        assert "## 🤖 Automated Pattern Fix" in comment
        assert "Test Pattern" in comment  # Pattern name should be title-cased
        assert "📊 Results Summary" in comment
        assert "RSPEED-1" in comment
        assert "RSPEED-2" in comment
        assert "**This ticket**" in comment  # Should highlight current ticket
        assert "fix/pattern-test" in comment  # Branch name
        assert "✅" in comment  # Success emoji

    @pytest.mark.asyncio
    async def test_jira_integration_dry_run(self, mock_pattern_result, tmp_path):
        """Test dry-run mode doesn't actually post."""
        integration = JiraIntegration(dry_run=True)

        result = await integration.update_tickets_for_pattern(
            pattern_result=mock_pattern_result,
            pattern_id="TEST",
            ticket_ids=["RSPEED-1", "RSPEED-2"],
        )

        assert result.success
        assert result.tickets_updated == 2
        assert result.tickets_failed == 0
        # Dry-run creates preview file, not fallback file
        assert result.fallback_file is not None
        assert "PREVIEW" in str(result.fallback_file)

    @pytest.mark.asyncio
    async def test_jira_integration_fallback_on_failure(
        self, mock_pattern_result, monkeypatch, tmp_path
    ):
        """Test fallback file created when Jira API fails."""
        integration = JiraIntegration(dry_run=False)

        # Mock _post_comment to always fail (return False asynchronously)
        async def mock_post_comment(ticket_id, comment_body):
            return False

        monkeypatch.setattr(integration, "_post_comment", mock_post_comment)

        # Mock _create_fallback_file to use tmp_path
        fallback_file = tmp_path / ".diagnostics" / "TEST" / "JIRA_COMMENTS_FALLBACK.md"

        def mock_create_fallback(pattern_id):
            fallback_file.parent.mkdir(parents=True, exist_ok=True)
            fallback_file.write_text(f"# Jira Comments Fallback - {pattern_id}\n\n")
            return fallback_file

        monkeypatch.setattr(integration, "_create_fallback_file", mock_create_fallback)

        result = await integration.update_tickets_for_pattern(
            pattern_result=mock_pattern_result,
            pattern_id="TEST",
            ticket_ids=["RSPEED-1"],
        )

        assert not result.success
        assert result.tickets_updated == 0
        assert result.tickets_failed == 1
        assert result.fallback_file == fallback_file
        assert fallback_file.exists()

    def test_fallback_file_format(self, mock_pattern_result, tmp_path):
        """Test fallback file has correct format."""
        integration = JiraIntegration(dry_run=False)

        # Create fallback file
        fallback_file = tmp_path / "JIRA_COMMENTS_FALLBACK.md"
        integration._create_fallback_file = lambda pattern_id: fallback_file
        fallback_file.write_text(
            "# Jira Comments Fallback - TEST\n\n"
            "These comments failed to post automatically. "
            "Copy-paste them manually to Jira tickets.\n\n"
            "=" * 80 + "\n\n"
        )

        # Append a comment
        integration._append_to_fallback(fallback_file, "RSPEED-123", "Test comment content")

        content = fallback_file.read_text()
        assert "RSPEED-123" in content
        assert "https://redhat.atlassian.net/browse/RSPEED-123" in content
        assert "Test comment content" in content

    def test_synchronous_wrapper(self, mock_pattern_result):
        """Test synchronous wrapper function."""
        from heal.integrations.jira_integration import update_tickets_for_pattern

        # Dry run to avoid actual API calls
        result = update_tickets_for_pattern(
            pattern_result=mock_pattern_result,
            pattern_id="TEST",
            ticket_ids=["RSPEED-1", "RSPEED-2"],
            dry_run=True,
        )

        assert isinstance(result, JiraUpdateResult)
        assert result.success
        assert result.tickets_updated == 2
