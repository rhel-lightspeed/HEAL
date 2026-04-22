"""Tests for PR creator functionality."""

from unittest.mock import MagicMock

import pytest

from heal.integrations.pr_creator import PRCreator, PRResult


class TestPRCreator:
    """Test PR creator functionality."""

    @pytest.fixture
    def mock_pattern_result(self):
        """Create mock pattern result."""
        result = MagicMock()
        result.success = True
        result.pattern_id = "CONTAINER_EOL_COMPAT"
        result.branch_name = "fix/pattern-container-eol-compat"
        result.duration_seconds = 2715.0

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
            "RSPEED-2482": MagicMock(ticket_id="RSPEED-2482", answer_correctness=0.95, url_f1=0.88),
            "RSPEED-2511": MagicMock(ticket_id="RSPEED-2511", answer_correctness=0.92, url_f1=0.85),
            "RSPEED-2520": MagicMock(ticket_id="RSPEED-2520", answer_correctness=0.88, url_f1=0.82),
            "RSPEED-2530": MagicMock(ticket_id="RSPEED-2530", answer_correctness=0.90, url_f1=0.84),
            "RSPEED-2545": MagicMock(ticket_id="RSPEED-2545", answer_correctness=0.87, url_f1=0.83),
            "RSPEED-2558": MagicMock(ticket_id="RSPEED-2558", answer_correctness=0.91, url_f1=0.86),
            "RSPEED-2570": MagicMock(ticket_id="RSPEED-2570", answer_correctness=0.86, url_f1=0.80),
        }
        result.answer_validation.num_runs = 3
        result.answer_validation.rag_bypass_tickets = []
        result.answer_validation.high_variance_tickets = ["RSPEED-2530"]

        return result

    def test_pr_title_format(self, mock_pattern_result):
        """Test PR title is properly formatted."""
        creator = PRCreator(dry_run=True)

        title = creator.formatter.format_pr_title(mock_pattern_result)

        assert title.startswith("fix(pattern):")
        assert "Container Eol Compat" in title  # Pattern name formatted
        assert "%" in title  # Success rate
        assert "/" in title  # Passing/total format

    def test_pr_title_with_success_rate(self, mock_pattern_result):
        """Test PR title includes correct success rate."""
        creator = PRCreator(dry_run=True)

        title = creator.formatter.format_pr_title(mock_pattern_result)

        # 7 out of 7 tickets passed (all >= 0.85), success_rate is 87.5%
        assert "88%" in title or "87%" in title  # Success rate percentage
        assert "7/7" in title or "(7/7" in title  # 7 passing out of 7 total

    def test_pr_body_includes_metrics(self, mock_pattern_result):
        """Test PR body includes before/after metrics."""
        creator = PRCreator(dry_run=True)

        body = creator.formatter.format_pr_body(mock_pattern_result)

        assert "📊 Quality Metrics" in body
        assert "Before" in body
        assert "After" in body
        assert "0.65" in body  # Baseline answer_correctness
        assert "0.92" in body  # Final answer_correctness

    def test_pr_body_includes_testing_section(self, mock_pattern_result):
        """Test PR body includes testing performed section."""
        creator = PRCreator(dry_run=True)

        body = creator.formatter.format_pr_body(mock_pattern_result)

        assert "🔬 Testing Performed" in body
        assert "3 validation cycles" in body
        assert "Per-Ticket Results" in body
        assert "RSPEED-2482" in body

    def test_pr_body_includes_warnings(self, mock_pattern_result):
        """Test PR body includes warnings section."""
        creator = PRCreator(dry_run=True)

        body = creator.formatter.format_pr_body(mock_pattern_result)

        assert "⚠️ Warnings & Risks" in body
        # Should mention high variance ticket
        assert "RSPEED-2530" in body or "metric instability" in body

    def test_pr_body_includes_reviewer_checklist(self, mock_pattern_result):
        """Test PR body includes reviewer checklist."""
        creator = PRCreator(dry_run=True)

        body = creator.formatter.format_pr_body(mock_pattern_result)

        assert "✅ Reviewer Checklist" in body
        assert "Metrics Look Good" in body
        assert "Code Quality" in body
        assert "How to Test Locally" in body
        assert "git checkout fix/pattern-container-eol-compat" in body

    def test_pr_creation_dry_run(self, mock_pattern_result, tmp_path):
        """Test PR creation in dry-run mode."""
        creator = PRCreator(dry_run=True)

        result = creator.create_pattern_pr(
            pattern_result=mock_pattern_result,
            branch_name="fix/pattern-test",
            okp_mcp_root=tmp_path,
        )

        assert result.success
        assert result.pr_url == "[dry-run-url]"
        # Dry run should not actually run git commands

    def test_pr_prerequisites_check_no_gh(self, mock_pattern_result, tmp_path, monkeypatch):
        """Test prerequisite check fails when gh CLI not installed."""
        import subprocess

        creator = PRCreator(dry_run=False)

        # Mock subprocess to simulate gh not installed
        def mock_run(*args, **kwargs):
            raise FileNotFoundError()

        monkeypatch.setattr(subprocess, "run", mock_run)

        result = creator.create_pattern_pr(
            pattern_result=mock_pattern_result,
            branch_name="fix/pattern-test",
            okp_mcp_root=tmp_path,
        )

        assert not result.success
        assert "gh CLI not installed" in result.error

    def test_pr_number_extraction(self, mock_pattern_result):
        """Test PR number extraction from URL."""
        creator = PRCreator(dry_run=True)

        # Test valid URLs
        assert creator._extract_pr_number("https://github.com/org/repo/pull/123") == 123
        assert creator._extract_pr_number("https://github.com/org/repo/pull/456/") == 456

        # Test invalid URLs
        assert creator._extract_pr_number("invalid-url") is None
        assert creator._extract_pr_number("https://github.com/org/repo") is None

    def test_synchronous_wrapper(self, mock_pattern_result, tmp_path):
        """Test synchronous wrapper function."""
        from heal.integrations.pr_creator import create_pattern_pr

        # Dry run to avoid actual git/gh commands
        result = create_pattern_pr(
            pattern_result=mock_pattern_result,
            branch_name="fix/pattern-test",
            okp_mcp_root=tmp_path,
            dry_run=True,
        )

        assert isinstance(result, PRResult)
        assert result.success
