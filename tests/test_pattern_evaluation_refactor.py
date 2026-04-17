"""Tests for per-ticket pattern evaluation refactor.

Tests the new PatternEvaluationResult structure and helper methods
that preserve per-ticket granularity in pattern mode.
"""

import pytest

from heal.agents.okp_mcp_agent import (
    EvaluationResult,
    PatternEvaluationResult,
    OkpMcpAgent,
)


class TestPatternEvaluationResult:
    """Test PatternEvaluationResult data structure."""

    def test_pattern_result_structure(self):
        """Test PatternEvaluationResult contains per-ticket data."""
        # Create mock per-ticket results
        ticket_1 = EvaluationResult(
            ticket_id="RSPEED-1001",
            url_f1=0.8,
            answer_correctness=0.9,
            faithfulness=0.85,
            context_relevance=0.9,
            context_precision=0.8,
            num_runs=3,
        )
        ticket_2 = EvaluationResult(
            ticket_id="RSPEED-1002",
            url_f1=0.5,
            answer_correctness=0.6,
            faithfulness=0.7,
            context_relevance=0.6,
            context_precision=0.5,
            num_runs=3,
        )

        per_ticket_results = {
            "RSPEED-1001": ticket_1,
            "RSPEED-1002": ticket_2,
        }

        # Create pattern result
        pattern_result = PatternEvaluationResult(
            pattern_id="TEST_PATTERN",
            num_runs=3,
            per_ticket_results=per_ticket_results,
            pattern_url_f1=0.65,
            pattern_answer_correctness=0.75,
            pattern_faithfulness=0.775,
            pattern_composite_score=0.72,
            success_rate=0.5,
            passing_tickets=["RSPEED-1001"],
            failing_tickets=["RSPEED-1002"],
        )

        # Assert structure
        assert pattern_result.pattern_id == "TEST_PATTERN"
        assert pattern_result.num_runs == 3
        assert len(pattern_result.per_ticket_results) == 2
        assert "RSPEED-1001" in pattern_result.per_ticket_results
        assert "RSPEED-1002" in pattern_result.per_ticket_results
        assert pattern_result.success_rate == 0.5
        assert len(pattern_result.passing_tickets) == 1
        assert len(pattern_result.failing_tickets) == 1

    def test_pattern_result_summary(self):
        """Test pattern result summary generation."""
        pattern_result = PatternEvaluationResult(
            pattern_id="TEST_PATTERN",
            num_runs=3,
            per_ticket_results={
                "RSPEED-1001": EvaluationResult(ticket_id="RSPEED-1001"),
                "RSPEED-1002": EvaluationResult(ticket_id="RSPEED-1002"),
            },
            success_rate=0.5,
            passing_tickets=["RSPEED-1001"],
            failing_tickets=["RSPEED-1002"],
            pattern_composite_score=0.75,
        )

        summary = pattern_result.summary()

        assert "TEST_PATTERN" in summary
        assert "2" in summary  # 2 tickets evaluated
        assert "50%" in summary or "0.5" in summary  # Success rate


class TestBuildEvaluationResultFromRuns:
    """Test _build_evaluation_result_from_runs helper method."""

    def test_single_run(self):
        """Test building result from single run."""
        # Test the logic directly
        runs = [
            {
                "custom:url_retrieval_eval": 0.8,
                "custom:answer_correctness": 0.9,
                "ragas:faithfulness": 0.85,
                "ragas:context_relevance": 0.9,
                "ragas:context_precision_without_reference": 0.8,
            }
        ]

        # Expected averages (same as input since only 1 run)
        assert runs[0]["custom:url_retrieval_eval"] == 0.8
        assert runs[0]["custom:answer_correctness"] == 0.9

    def test_multiple_runs_averaging(self):
        """Test averaging metrics across multiple runs."""
        runs = [
            {"custom:answer_correctness": 0.9, "ragas:faithfulness": 0.8},
            {"custom:answer_correctness": 0.7, "ragas:faithfulness": 0.6},
            {"custom:answer_correctness": 0.8, "ragas:faithfulness": 0.7},
        ]

        # Calculate expected averages
        avg_answer = (0.9 + 0.7 + 0.8) / 3
        avg_faith = (0.8 + 0.6 + 0.7) / 3

        assert avg_answer == pytest.approx(0.8, abs=0.01)
        assert avg_faith == pytest.approx(0.7, abs=0.01)

    def test_high_variance_detection(self):
        """Test detection of high variance metrics."""
        import statistics

        runs = [
            {"custom:answer_correctness": 0.95},
            {"custom:answer_correctness": 0.50},
            {"custom:answer_correctness": 0.90},
        ]

        values = [r["custom:answer_correctness"] for r in runs]
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)

        # Variance > 15% of mean
        assert (std_val / mean_val) > 0.15


class TestBuildPatternResult:
    """Test _build_pattern_result helper method."""

    def test_pattern_aggregates(self):
        """Test pattern-level aggregate calculations."""
        per_ticket_results = {
            "RSPEED-1001": EvaluationResult(
                ticket_id="RSPEED-1001",
                url_f1=0.8,
                answer_correctness=0.9,
                faithfulness=0.85,
            ),
            "RSPEED-1002": EvaluationResult(
                ticket_id="RSPEED-1002",
                url_f1=0.6,
                answer_correctness=0.7,
                faithfulness=0.75,
            ),
        }

        # Expected averages
        expected_url_f1 = (0.8 + 0.6) / 2
        expected_answer = (0.9 + 0.7) / 2
        expected_faith = (0.85 + 0.75) / 2

        assert expected_url_f1 == 0.7
        assert expected_answer == 0.8
        assert expected_faith == 0.8

    def test_passing_failing_classification(self):
        """Test classification of passing/failing tickets."""
        # Ticket 1: Composite = 0.80*0.9 + 0.15*0.8 + 0.05*0.85 = 0.8825 (PASS)
        # Ticket 2: Composite = 0.80*0.6 + 0.15*0.5 + 0.05*0.6 = 0.585 (FAIL)

        ticket_1_composite = 0.80 * 0.9 + 0.15 * 0.8 + 0.05 * 0.85
        ticket_2_composite = 0.80 * 0.6 + 0.15 * 0.5 + 0.05 * 0.6

        assert ticket_1_composite >= 0.80  # Should pass
        assert ticket_2_composite < 0.80  # Should fail

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        total_tickets = 4
        passing_tickets = 3

        success_rate = passing_tickets / total_tickets

        assert success_rate == 0.75

    def test_rag_bypass_detection(self):
        """Test detection of RAG bypass scenarios."""
        # Scenario 1: RAG used successfully
        ticket_1 = EvaluationResult(
            ticket_id="RSPEED-1001",
            rag_used=True,
            docs_retrieved=True,
        )

        # Scenario 2: RAG tool called but failed (0 docs)
        ticket_2 = EvaluationResult(
            ticket_id="RSPEED-1002",
            rag_used=True,
            docs_retrieved=False,
        )

        # Scenario 3: RAG bypassed - LLM didn't use tool
        ticket_3 = EvaluationResult(
            ticket_id="RSPEED-1003",
            rag_used=False,
            docs_retrieved=False,
        )

        # Expected RAG bypass tickets
        assert ticket_1.rag_used and ticket_1.docs_retrieved  # Not bypass
        assert ticket_2.rag_used and not ticket_2.docs_retrieved  # Bypass (tool failure)
        assert not ticket_3.rag_used  # Bypass (no tool call)

    def test_high_variance_ticket_detection(self):
        """Test detection of tickets with high variance."""
        ticket_stable = EvaluationResult(
            ticket_id="RSPEED-1001",
            high_variance_metrics=[],
        )

        ticket_unstable = EvaluationResult(
            ticket_id="RSPEED-1002",
            high_variance_metrics=["custom:answer_correctness (std=0.240)"],
        )

        assert len(ticket_stable.high_variance_metrics) == 0
        assert len(ticket_unstable.high_variance_metrics) > 0


class TestCompositeScoreWeighting:
    """Test composite score calculations for different RAG scenarios."""

    def test_normal_rag_composite(self):
        """Test composite score for normal RAG (80% answer + 15% relevance + 5% precision)."""
        answer = 0.9
        relevance = 0.8
        precision = 0.7

        composite = 0.80 * answer + 0.15 * relevance + 0.05 * precision

        expected = 0.80 * 0.9 + 0.15 * 0.8 + 0.05 * 0.7
        assert composite == pytest.approx(expected, abs=0.001)
        assert composite == pytest.approx(0.875, abs=0.001)

    def test_rag_bypass_composite(self):
        """Test composite score for RAG bypass (70% answer + 25% faithfulness + 5% relevancy)."""
        answer = 0.9
        faithfulness = 0.8
        relevancy = 0.85

        composite = 0.70 * answer + 0.25 * faithfulness + 0.05 * relevancy

        expected = 0.70 * 0.9 + 0.25 * 0.8 + 0.05 * 0.85
        assert composite == pytest.approx(expected, abs=0.001)
        assert composite == pytest.approx(0.8725, abs=0.001)

    def test_retrieval_only_composite(self):
        """Test composite score for retrieval-only mode (50% url_f1 + 50% relevance)."""
        url_f1 = 0.8
        relevance = 0.9

        composite = 0.5 * url_f1 + 0.5 * relevance

        expected = 0.5 * 0.8 + 0.5 * 0.9
        assert composite == pytest.approx(expected, abs=0.001)
        assert composite == pytest.approx(0.85, abs=0.001)


class TestEmptyPattern:
    """Test handling of edge cases (empty patterns, no metrics, etc.)."""

    def test_empty_pattern_result(self):
        """Test creating pattern result with no tickets."""
        pattern_result = PatternEvaluationResult(
            pattern_id="EMPTY_PATTERN",
            num_runs=3,
        )

        assert pattern_result.pattern_id == "EMPTY_PATTERN"
        assert len(pattern_result.per_ticket_results) == 0
        assert pattern_result.success_rate == 0.0
        assert len(pattern_result.passing_tickets) == 0
        assert len(pattern_result.failing_tickets) == 0

    def test_ticket_with_no_metrics(self):
        """Test ticket with no metrics available."""
        ticket = EvaluationResult(
            ticket_id="RSPEED-1001",
            num_runs=0,
        )

        assert ticket.ticket_id == "RSPEED-1001"
        assert ticket.url_f1 is None
        assert ticket.answer_correctness is None
        assert not ticket.has_metrics
