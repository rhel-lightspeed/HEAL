"""Tests for token tracking functionality."""

import json
from pathlib import Path

import pytest

from heal.core.token_tracker import TokenTracker


class TestTokenTracker:
    """Test TokenTracker basic functionality."""

    def teardown_method(self):
        """Clear singleton instance after each test."""
        TokenTracker._instance = None

    def test_initialization(self, tmp_path):
        """Test TokenTracker initializes correctly."""
        tracker = TokenTracker(pattern_id="TEST_PATTERN", output_dir=tmp_path)

        assert tracker.pattern_id == "TEST_PATTERN"
        assert tracker.output_dir == tmp_path
        assert tracker.calls == []
        assert tracker.baseline_answer_correctness == 0.0

    def test_get_instance(self, tmp_path):
        """Test singleton get_instance() method."""
        # No instance yet
        assert TokenTracker.get_instance() is None

        # Create instance
        tracker = TokenTracker(pattern_id="TEST", output_dir=tmp_path)

        # Should return the instance
        assert TokenTracker.get_instance() is tracker

    def test_record_tokens(self, tmp_path):
        """Test recording token usage."""
        tracker = TokenTracker(pattern_id="TEST", output_dir=tmp_path)
        tracker.set_iteration(1, 1)

        tracker.record_tokens(
            input_tokens=1000,
            output_tokens=500,
            call_type="multi_agent_solr_expert",
            model="claude-sonnet-4-6",
        )

        assert len(tracker.calls) == 1
        call = tracker.calls[0]
        assert call.input_tokens == 1000
        assert call.output_tokens == 500
        assert call.total_tokens == 1500
        assert call.call_type == "multi_agent_solr_expert"
        assert call.model == "claude-sonnet-4-6"
        assert call.cost_usd > 0  # Should have calculated cost

    def test_by_model_breakdown(self, tmp_path):
        """Test per-model token breakdown."""
        tracker = TokenTracker(pattern_id="TEST", output_dir=tmp_path)
        tracker.set_iteration(1, 1)

        # Record calls with different models
        tracker.record_tokens(
            input_tokens=1000,
            output_tokens=500,
            call_type="multi_agent",
            model="claude-sonnet-4-6",
        )
        tracker.record_tokens(
            input_tokens=2000,
            output_tokens=1000,
            call_type="multi_agent",
            model="claude-opus-4-6",
        )
        tracker.record_tokens(
            input_tokens=500,
            output_tokens=250,
            call_type="multi_agent",
            model="claude-sonnet-4-6",
        )

        summary = tracker.get_summary()

        # Check by_model breakdown exists
        assert "by_model" in summary
        by_model = summary["by_model"]

        # Should have 2 models
        assert len(by_model) == 2
        assert "claude-sonnet-4-6" in by_model
        assert "claude-opus-4-6" in by_model

        # Sonnet should have 2 calls, 1500 + 750 = 2250 tokens
        assert by_model["claude-sonnet-4-6"]["calls"] == 2
        assert by_model["claude-sonnet-4-6"]["tokens"] == 2250

        # Opus should have 1 call, 3000 tokens
        assert by_model["claude-opus-4-6"]["calls"] == 1
        assert by_model["claude-opus-4-6"]["tokens"] == 3000

    def test_iteration_summary(self, tmp_path):
        """Test recording iteration summary."""
        tracker = TokenTracker(pattern_id="TEST", output_dir=tmp_path)
        tracker.set_iteration(1, 1)

        # Record some calls
        tracker.record_tokens(
            input_tokens=1000,
            output_tokens=500,
            call_type="multi_agent_solr_expert",
            model="claude-sonnet-4-6",
        )
        tracker.record_tokens(
            input_tokens=5000,
            output_tokens=2000,
            call_type="ragas_answer_correctness",
            model="claude-sonnet-4-6",
        )

        # Record iteration summary
        tracker.record_iteration_summary(
            iteration=1,
            cycle=1,
            before_answer=0.60,
            after_answer=0.75,
            used_pattern_context=False,
        )

        # Check summary file was created
        summary_file = tmp_path / "TEST_token_summaries.jsonl"
        assert summary_file.exists()

        # Read and verify
        with open(summary_file) as f:
            summary = json.loads(f.read())

        assert summary["iteration"] == 1
        assert summary["cycle"] == 1
        assert summary["multi_agent_tokens"] == 1500  # 1000 + 500
        assert summary["ragas_tokens"] == 7000  # 5000 + 2000
        assert summary["before_answer_correctness"] == 0.60
        assert summary["after_answer_correctness"] == 0.75
        assert summary["answer_improvement"] == pytest.approx(0.15)

    def test_generate_report_with_model_breakdown(self, tmp_path):
        """Test report generation includes per-model breakdown."""
        tracker = TokenTracker(pattern_id="TEST", output_dir=tmp_path)
        tracker.set_baseline(0.60)
        tracker.set_iteration(1, 1)

        # Record calls with multiple models
        tracker.record_tokens(
            input_tokens=1000,
            output_tokens=500,
            call_type="multi_agent",
            model="claude-sonnet-4-6",
        )
        tracker.record_tokens(
            input_tokens=2000,
            output_tokens=1000,
            call_type="ragas",
            model="claude-opus-4-6",
        )

        tracker.record_iteration_summary(
            iteration=1,
            cycle=1,
            before_answer=0.60,
            after_answer=0.75,
            used_pattern_context=False,
        )

        # Generate report
        report = tracker.generate_report()

        # Verify report includes model breakdown
        assert "### By Model" in report
        assert "claude-sonnet-4-6" in report
        assert "claude-opus-4-6" in report
        assert "| Model | Calls | Tokens | Cost | % of Total |" in report
