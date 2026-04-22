"""Tests for per-ticket success criteria."""

import pytest
from pathlib import Path

from heal.agents.okp_mcp_agent import PatternEvaluationResult, EvaluationResult
from heal.runners.run_pattern_fix_poc import PatternFixAgent


class TestPerTicketSuccess:
    """Test per-ticket success criteria logic."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create test agent."""
        # Create mock directories
        eval_root = tmp_path / "eval"
        okp_root = tmp_path / "okp"
        lscore_root = tmp_path / "lscore"
        eval_root.mkdir()
        okp_root.mkdir()
        lscore_root.mkdir()

        return PatternFixAgent(
            pattern_id="TEST",
            eval_root=eval_root,
            okp_mcp_root=okp_root,
            lscore_deploy_root=lscore_root,
        )

    def test_analyze_some_improved_to_passing(self, agent):
        """Test when 1/3 tickets improves to passing."""
        # Baseline: all below threshold
        baseline = {
            "TICKET-1": {"answer": 0.70, "f1": 0.5},
            "TICKET-2": {"answer": 0.65, "f1": 0.4},
            "TICKET-3": {"answer": 0.75, "f1": 0.6},
        }

        # Current: TICKET-1 improved to passing, others unchanged
        current = PatternEvaluationResult(
            pattern_id="TEST",
            num_runs=3,
            per_ticket_results={
                "TICKET-1": EvaluationResult(
                    ticket_id="TICKET-1", answer_correctness=0.92, url_f1=0.8
                ),
                "TICKET-2": EvaluationResult(
                    ticket_id="TICKET-2", answer_correctness=0.67, url_f1=0.42
                ),
                "TICKET-3": EvaluationResult(
                    ticket_id="TICKET-3", answer_correctness=0.76, url_f1=0.61
                ),
            },
        )

        changes = agent._analyze_per_ticket_changes(baseline, current)

        assert len(changes["improved_to_passing"]) == 1
        assert "TICKET-1" in changes["improved_to_passing"]
        assert len(changes["improved"]) == 0  # TICKET-2/3 not meaningfully improved
        assert len(changes["regressed"]) == 0
        assert len(changes["catastrophic"]) == 0
        assert changes["net_improvement"] == 1

    def test_analyze_catastrophic_regression(self, agent):
        """Test detection of catastrophic regression."""
        baseline = {
            "TICKET-1": {"answer": 0.85, "f1": 0.7},
            "TICKET-2": {"answer": 0.80, "f1": 0.6},
        }

        # TICKET-2 catastrophically regressed (0.80 → 0.55 = -0.25 drop)
        current = PatternEvaluationResult(
            pattern_id="TEST",
            num_runs=3,
            per_ticket_results={
                "TICKET-1": EvaluationResult(
                    ticket_id="TICKET-1", answer_correctness=0.87, url_f1=0.72
                ),
                "TICKET-2": EvaluationResult(
                    ticket_id="TICKET-2", answer_correctness=0.55, url_f1=0.4
                ),
            },
        )

        changes = agent._analyze_per_ticket_changes(baseline, current)

        assert len(changes["catastrophic"]) == 1
        assert "TICKET-2" in changes["catastrophic"]
        assert len(changes["unchanged"]) == 1  # TICKET-1 changed by only 0.02 (<0.05 threshold)
        # Note: catastrophic tickets are separate from regressed, so net_improvement doesn't count them
        assert (
            changes["net_improvement"] == 0
        )  # 0 improved - 0 regressed (catastrophic is separate)

    def test_analyze_net_positive_improvements(self, agent):
        """Test net positive improvement (2 improved, 1 regressed)."""
        baseline = {
            "TICKET-1": {"answer": 0.70, "f1": 0.5},
            "TICKET-2": {"answer": 0.65, "f1": 0.4},
            "TICKET-3": {"answer": 0.80, "f1": 0.7},
        }

        # 2 improved meaningfully, 1 regressed slightly
        current = PatternEvaluationResult(
            pattern_id="TEST",
            num_runs=3,
            per_ticket_results={
                "TICKET-1": EvaluationResult(
                    ticket_id="TICKET-1", answer_correctness=0.77, url_f1=0.6
                ),  # +0.07
                "TICKET-2": EvaluationResult(
                    ticket_id="TICKET-2", answer_correctness=0.73, url_f1=0.5
                ),  # +0.08
                "TICKET-3": EvaluationResult(
                    ticket_id="TICKET-3", answer_correctness=0.74, url_f1=0.65
                ),  # -0.06
            },
        )

        changes = agent._analyze_per_ticket_changes(baseline, current)

        assert len(changes["improved"]) == 2
        assert len(changes["regressed"]) == 1
        assert len(changes["catastrophic"]) == 0
        assert changes["net_improvement"] == 1  # 2 - 1 = 1

    def test_analyze_unchanged_tickets(self, agent):
        """Test detection of unchanged tickets."""
        baseline = {
            "TICKET-1": {"answer": 0.70, "f1": 0.5},
            "TICKET-2": {"answer": 0.75, "f1": 0.6},
        }

        # Both changed by <0.05 (unchanged threshold)
        current = PatternEvaluationResult(
            pattern_id="TEST",
            num_runs=3,
            per_ticket_results={
                "TICKET-1": EvaluationResult(
                    ticket_id="TICKET-1", answer_correctness=0.72, url_f1=0.52
                ),  # +0.02
                "TICKET-2": EvaluationResult(
                    ticket_id="TICKET-2", answer_correctness=0.77, url_f1=0.62
                ),  # +0.02
            },
        )

        changes = agent._analyze_per_ticket_changes(baseline, current)

        assert len(changes["unchanged"]) == 2
        assert len(changes["improved"]) == 0
        assert len(changes["regressed"]) == 0
        assert changes["net_improvement"] == 0
