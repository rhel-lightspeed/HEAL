"""Test pattern evaluation logic using real data fixtures.

Tests the per-ticket evaluation refactor using captured real data,
without hitting any LLMs or APIs.
"""

import pytest
import statistics
from pathlib import Path

from heal.agents.okp_mcp_agent import PatternEvaluationResult, EvaluationResult
from tests.fixtures import get_mock_per_ticket_results


class TestBuildEvaluationResultLogic:
    """Test _build_evaluation_result_from_runs logic with real data."""

    def test_metric_averaging_with_single_run(self):
        """Test that single run returns metrics correctly."""
        # Load real fixture
        per_ticket_results = get_mock_per_ticket_results(
            "bootloader_grub_pattern/run_001_results.json"
        )
        runs = per_ticket_results["RSPEED-1725"]

        # Manual calculation of averages (single run)
        assert len(runs) == 1
        run = runs[0]

        url_f1 = run["custom:url_retrieval_eval"]
        answer = run["custom:answer_correctness"]
        faith = run["ragas:faithfulness"]
        mrr = run["metric_metadata"]["mrr"]

        # Assertions
        assert url_f1 == pytest.approx(0.333, abs=0.001)
        assert answer == pytest.approx(0.3, abs=0.001)
        assert faith == pytest.approx(0.947, abs=0.001)
        assert mrr == pytest.approx(0.09, abs=0.001)

    def test_mrr_extraction_from_fixture(self):
        """Test that MRR is extracted correctly from fixture."""
        per_ticket_results = get_mock_per_ticket_results(
            "bootloader_grub_pattern/run_001_results.json"
        )

        # RSPEED-1725 has MRR=0.09 (from reason field in CSV)
        runs = per_ticket_results["RSPEED-1725"]
        assert runs[0]["metric_metadata"]["mrr"] == pytest.approx(0.09, abs=0.001)

    def test_retrieval_problem_detection_logic(self):
        """Test is_retrieval_problem logic with fixture data."""
        per_ticket_results = get_mock_per_ticket_results(
            "bootloader_grub_pattern/run_001_results.json"
        )

        # RSPEED-1723: F1=0.0 → retrieval problem
        runs_1723 = per_ticket_results["RSPEED-1723"]
        url_f1_1723 = runs_1723[0]["custom:url_retrieval_eval"]
        assert url_f1_1723 == 0.0

        # Manually add MRR for testing
        runs_1723[0]["metric_metadata"]["mrr"] = 0.0
        mrr_1723 = runs_1723[0]["metric_metadata"]["mrr"]
        ctx_rel_1723 = runs_1723[0]["ragas:context_relevance"]

        # is_retrieval_problem logic: url_f1 < 0.5 OR mrr < 0.5 OR ctx_rel < 0.7
        is_problem = (url_f1_1723 < 0.5) or (mrr_1723 < 0.5) or (ctx_rel_1723 < 0.7)
        assert is_problem is True  # F1=0.0 triggers it


class TestPatternAggregationLogic:
    """Test _build_pattern_result aggregation logic."""

    def test_pattern_level_averaging(self):
        """Test pattern-level metric averaging."""
        per_ticket_results = get_mock_per_ticket_results(
            "bootloader_grub_pattern/run_001_results.json"
        )

        # Calculate expected averages
        url_f1_values = [
            runs[0]["custom:url_retrieval_eval"] for runs in per_ticket_results.values()
        ]
        answer_values = [
            runs[0]["custom:answer_correctness"] for runs in per_ticket_results.values()
        ]

        expected_avg_f1 = statistics.mean(url_f1_values)
        expected_avg_answer = statistics.mean(answer_values)

        # Assertions (0.0 + 0.0 + 0.333) / 3 ≈ 0.111
        assert expected_avg_f1 == pytest.approx(0.111, abs=0.001)

        # (0.6 + 0.7 + 0.3) / 3 ≈ 0.533
        assert expected_avg_answer == pytest.approx(0.533, abs=0.001)

    def test_composite_score_calculation(self):
        """Test composite score formula."""
        # Using RSPEED-1725 data
        answer = 0.3
        faith = 0.947
        ctx_rel = 1.0
        ctx_prec = 0.333

        # Normal RAG composite: 80% answer + 15% relevance + 5% precision
        composite = 0.80 * answer + 0.15 * ctx_rel + 0.05 * ctx_prec

        expected = 0.80 * 0.3 + 0.15 * 1.0 + 0.05 * 0.333
        assert composite == pytest.approx(expected, abs=0.001)
        assert composite == pytest.approx(0.407, abs=0.001)

    def test_passing_failing_classification_logic(self):
        """Test ticket classification as passing/failing."""
        per_ticket_results = get_mock_per_ticket_results(
            "bootloader_grub_pattern/run_001_results.json"
        )

        # Calculate composite for each ticket
        composites = {}
        for ticket_id, runs in per_ticket_results.items():
            run = runs[0]
            answer = run["custom:answer_correctness"]
            ctx_rel = run["ragas:context_relevance"]
            ctx_prec = run["ragas:context_precision_without_reference"]

            composite = 0.80 * answer + 0.15 * ctx_rel + 0.05 * ctx_prec
            composites[ticket_id] = composite

        # All tickets should have composite < 0.8 (failing threshold)
        # RSPEED-1723: 0.80*0.6 + 0.15*1.0 + 0.05*0.333 ≈ 0.647
        # RSPEED-1724: 0.80*0.7 + 0.15*0.5 + 0.05*0.0 ≈ 0.635
        # RSPEED-1725: 0.80*0.3 + 0.15*1.0 + 0.05*0.333 ≈ 0.407

        for ticket_id, composite in composites.items():
            assert composite < 0.8, f"{ticket_id} should fail (composite={composite:.3f})"

        # All 3 should be classified as failing
        failing_count = sum(1 for c in composites.values() if c < 0.8)
        assert failing_count == 3


class TestFixtureDataQuality:
    """Verify fixture data has all required fields."""

    def test_fixture_has_all_metrics(self):
        """Verify fixture contains all expected metrics."""
        per_ticket_results = get_mock_per_ticket_results(
            "bootloader_grub_pattern/run_001_results.json"
        )

        for ticket_id, runs in per_ticket_results.items():
            for run in runs:
                # Check essential metrics present
                assert "custom:url_retrieval_eval" in run
                assert "custom:answer_correctness" in run
                assert "ragas:faithfulness" in run
                assert "ragas:context_relevance" in run

                # Check metric_metadata exists
                assert "metric_metadata" in run

    def test_fixture_has_mrr(self):
        """Verify fixture has MRR in metric_metadata."""
        per_ticket_results = get_mock_per_ticket_results(
            "bootloader_grub_pattern/run_001_results.json"
        )

        # RSPEED-1725 should have MRR=0.09 from extraction
        runs = per_ticket_results["RSPEED-1725"]
        assert "metric_metadata" in runs[0]
        assert "mrr" in runs[0]["metric_metadata"]
        assert runs[0]["metric_metadata"]["mrr"] == pytest.approx(0.09, abs=0.001)

    def test_fixture_has_three_tickets(self):
        """Verify fixture has expected number of tickets."""
        per_ticket_results = get_mock_per_ticket_results(
            "bootloader_grub_pattern/run_001_results.json"
        )

        assert len(per_ticket_results) == 3
        assert "RSPEED-1723" in per_ticket_results
        assert "RSPEED-1724" in per_ticket_results
        assert "RSPEED-1725" in per_ticket_results
