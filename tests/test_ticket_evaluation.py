"""Tests for ticket evaluation classes."""

import pytest
from heal.core.ticket_evaluation import PatternEvaluation, TicketEvaluation


class TestTicketEvaluation:
    """Test TicketEvaluation class for individual ticket assessment."""

    def test_basic_properties(self):
        """Test basic property calculations with multiple runs."""
        ticket = TicketEvaluation(
            ticket_id="RSPEED-1234",
            runs=[
                {
                    "answer_correctness": 0.95,
                    "context_relevance": 0.80,
                    "context_precision": 0.75,
                    "url_f1": 0.60,
                },
                {
                    "answer_correctness": 0.90,
                    "context_relevance": 0.85,
                    "context_precision": 0.70,
                    "url_f1": 0.65,
                },
            ],
        )

        assert ticket.ticket_id == "RSPEED-1234"
        assert ticket.num_runs == 2
        assert ticket.mean_answer_correctness == 0.925  # (0.95 + 0.90) / 2
        assert ticket.mean_context_relevance == 0.825  # (0.80 + 0.85) / 2
        assert ticket.mean_context_precision == 0.725  # (0.75 + 0.70) / 2
        assert ticket.mean_url_f1 == 0.625  # (0.60 + 0.65) / 2

    def test_composite_score_calculation(self):
        """Test composite score uses correct weights (80/15/5)."""
        ticket = TicketEvaluation(
            ticket_id="TEST-001",
            runs=[
                {
                    "answer_correctness": 1.0,
                    "context_relevance": 0.0,
                    "context_precision": 0.0,
                }
            ],
        )

        # Composite = 1.0*0.80 + 0.0*0.15 + 0.0*0.05 = 0.80
        assert ticket.composite_score == 0.80

        ticket2 = TicketEvaluation(
            ticket_id="TEST-002",
            runs=[
                {
                    "answer_correctness": 0.0,
                    "context_relevance": 1.0,
                    "context_precision": 0.0,
                }
            ],
        )

        # Composite = 0.0*0.80 + 1.0*0.15 + 0.0*0.05 = 0.15
        assert ticket2.composite_score == 0.15

        ticket3 = TicketEvaluation(
            ticket_id="TEST-003",
            runs=[
                {
                    "answer_correctness": 0.0,
                    "context_relevance": 0.0,
                    "context_precision": 1.0,
                }
            ],
        )

        # Composite = 0.0*0.80 + 0.0*0.15 + 1.0*0.05 = 0.05
        assert ticket3.composite_score == 0.05

    def test_variance_calculation(self):
        """Test variance calculation across runs."""
        # Low variance
        low_var = TicketEvaluation(
            ticket_id="TEST-LOW",
            runs=[
                {"answer_correctness": 0.91},
                {"answer_correctness": 0.92},
                {"answer_correctness": 0.91},
            ],
        )

        assert low_var.variance < 0.001  # Very low variance

        # High variance
        high_var = TicketEvaluation(
            ticket_id="TEST-HIGH",
            runs=[
                {"answer_correctness": 0.50},
                {"answer_correctness": 0.95},
                {"answer_correctness": 0.50},
            ],
        )

        assert high_var.variance > 0.04  # High variance (actual: ~0.045)

    def test_variance_single_run(self):
        """Single run should have zero variance."""
        ticket = TicketEvaluation(ticket_id="TEST-SINGLE", runs=[{"answer_correctness": 0.95}])

        assert ticket.variance == 0.0

    def test_variance_empty_runs(self):
        """Empty runs should have zero variance."""
        ticket = TicketEvaluation(ticket_id="TEST-EMPTY", runs=[])

        assert ticket.variance == 0.0

    def test_status_stable_passing(self):
        """Low variance, high score → STABLE_PASSING."""
        ticket = TicketEvaluation(
            ticket_id="TEST-STABLE",
            runs=[
                {"answer_correctness": 0.92},
                {"answer_correctness": 0.91},
                {"answer_correctness": 0.93},
            ],
        )

        assert ticket.status == "STABLE_PASSING"

    def test_status_consistently_failing(self):
        """Low variance, low score → CONSISTENTLY_FAILING."""
        ticket = TicketEvaluation(
            ticket_id="TEST-FAILING",
            runs=[
                {"answer_correctness": 0.40},
                {"answer_correctness": 0.42},
                {"answer_correctness": 0.38},
            ],
        )

        assert ticket.status == "CONSISTENTLY_FAILING"

    def test_status_erratic(self):
        """High variance → ERRATIC."""
        ticket = TicketEvaluation(
            ticket_id="TEST-ERRATIC",
            runs=[
                {"answer_correctness": 0.95},
                {"answer_correctness": 0.30},  # Lower to increase variance
                {"answer_correctness": 0.95},
                {"answer_correctness": 0.30},  # Add more runs for higher variance
            ],
        )

        assert ticket.status == "ERRATIC"
        assert ticket.variance > 0.05

    def test_status_improving(self):
        """Score improved significantly vs baseline → IMPROVING."""
        baseline = TicketEvaluation(ticket_id="TEST-IMPROVE", runs=[{"answer_correctness": 0.60}])

        current = TicketEvaluation(
            ticket_id="TEST-IMPROVE",
            runs=[{"answer_correctness": 0.85}],
            baseline=baseline,
        )

        assert current.status == "IMPROVING"
        assert current.improvement_over_baseline() > 0.10

    def test_status_regressing(self):
        """Score decreased significantly vs baseline → REGRESSING."""
        baseline = TicketEvaluation(ticket_id="TEST-REGRESS", runs=[{"answer_correctness": 0.95}])

        current = TicketEvaluation(
            ticket_id="TEST-REGRESS",
            runs=[{"answer_correctness": 0.70}],
            baseline=baseline,
        )

        assert current.status == "REGRESSING"
        assert current.improvement_over_baseline() < -0.10

    def test_status_no_data(self):
        """Empty runs → NO_DATA."""
        ticket = TicketEvaluation(ticket_id="TEST-NODATA", runs=[])

        assert ticket.status == "NO_DATA"

    def test_improvement_over_baseline(self):
        """Test baseline comparison calculation."""
        baseline = TicketEvaluation(
            ticket_id="TEST-001",
            runs=[
                {"answer_correctness": 0.70},
                {"answer_correctness": 0.75},
            ],
        )

        current = TicketEvaluation(
            ticket_id="TEST-001",
            runs=[
                {"answer_correctness": 0.90},
                {"answer_correctness": 0.95},
            ],
            baseline=baseline,
        )

        # Baseline avg: (0.70 + 0.75) / 2 = 0.725
        # Current avg: (0.90 + 0.95) / 2 = 0.925
        # Improvement: 0.925 - 0.725 = 0.20
        improvement = current.improvement_over_baseline()
        assert abs(improvement - 0.20) < 0.01

    def test_improvement_no_baseline(self):
        """No baseline → improvement is 0.0."""
        ticket = TicketEvaluation(ticket_id="TEST-001", runs=[{"answer_correctness": 0.95}])

        assert ticket.improvement_over_baseline() == 0.0

    def test_passes_default_threshold(self):
        """Test passes() with default threshold (0.80)."""
        passing = TicketEvaluation(
            ticket_id="PASS",
            runs=[
                {
                    "answer_correctness": 0.95,
                    "context_relevance": 0.85,
                    "context_precision": 0.75,
                }
            ],
        )

        # Composite = 0.95*0.80 + 0.85*0.15 + 0.75*0.05
        #           = 0.76 + 0.1275 + 0.0375 = 0.925
        assert passing.passes()
        assert passing.passes(threshold=0.80)

        failing = TicketEvaluation(
            ticket_id="FAIL",
            runs=[
                {
                    "answer_correctness": 0.60,
                    "context_relevance": 0.50,
                    "context_precision": 0.40,
                }
            ],
        )

        # Composite = 0.60*0.80 + 0.50*0.15 + 0.40*0.05
        #           = 0.48 + 0.075 + 0.02 = 0.575
        assert not failing.passes()

    def test_passes_custom_threshold(self):
        """Test passes() with custom threshold."""
        ticket = TicketEvaluation(
            ticket_id="TEST-CUSTOM",
            runs=[
                {
                    "answer_correctness": 0.85,
                    "context_relevance": 0.80,
                    "context_precision": 0.75,
                }
            ],
        )

        # Composite = 0.85*0.80 + 0.80*0.15 + 0.75*0.05
        #           = 0.68 + 0.12 + 0.0375 = 0.8375
        assert ticket.passes(threshold=0.80)
        assert ticket.passes(threshold=0.83)
        assert not ticket.passes(threshold=0.85)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        ticket = TicketEvaluation(
            ticket_id="TEST-DICT",
            runs=[
                {
                    "answer_correctness": 0.95,
                    "context_relevance": 0.85,
                    "context_precision": 0.75,
                    "url_f1": 0.70,
                }
            ],
        )

        result = ticket.to_dict()

        assert result["ticket_id"] == "TEST-DICT"
        assert "mean_answer_correctness" in result
        assert "mean_context_relevance" in result
        assert "mean_context_precision" in result
        assert "mean_url_f1" in result
        assert "composite_score" in result
        assert "variance" in result
        assert "status" in result
        assert result["status"] == "STABLE_PASSING"

    def test_missing_metrics_in_runs(self):
        """Test handling of runs with missing metrics."""
        ticket = TicketEvaluation(
            ticket_id="TEST-MISSING",
            runs=[
                {"answer_correctness": 0.95},  # Missing context metrics
                {
                    "answer_correctness": 0.90,
                    "context_relevance": 0.80,
                },  # Missing precision
            ],
        )

        # Should handle missing values gracefully
        assert ticket.mean_answer_correctness == 0.925
        # Missing values treated as 0.0, so: (0.0 + 0.80) / 2 = 0.40
        assert ticket.mean_context_relevance == 0.40
        assert ticket.mean_context_precision == 0.0  # No runs have it

    def test_empty_metrics_treated_as_zero(self):
        """Test that missing metrics don't crash composite calculation."""
        ticket = TicketEvaluation(
            ticket_id="TEST-EMPTY-METRICS",
            runs=[{}],  # Empty dict
        )

        # Should not crash
        assert ticket.mean_answer_correctness == 0.0
        assert ticket.composite_score == 0.0
        assert ticket.passes() is False


class TestPatternEvaluation:
    """Test PatternEvaluation class for pattern-level assessment."""

    def test_basic_properties(self):
        """Test basic pattern properties."""
        pattern = PatternEvaluation(pattern_id="TEST_PATTERN")

        ticket1 = TicketEvaluation(
            ticket_id="TICKET-1",
            runs=[
                {
                    "answer_correctness": 0.95,
                    "context_relevance": 0.85,
                    "context_precision": 0.75,
                }
            ],
        )

        ticket2 = TicketEvaluation(
            ticket_id="TICKET-2",
            runs=[
                {
                    "answer_correctness": 0.60,
                    "context_relevance": 0.50,
                    "context_precision": 0.40,
                }
            ],
        )

        pattern.tickets["TICKET-1"] = ticket1
        pattern.tickets["TICKET-2"] = ticket2

        assert pattern.pattern_id == "TEST_PATTERN"
        assert pattern.num_tickets == 2

    def test_passing_and_failing_tickets(self):
        """Test identification of passing/failing tickets."""
        pattern = PatternEvaluation(pattern_id="MIXED_PATTERN")

        # Create passing ticket (composite >= 0.80)
        passing = TicketEvaluation(
            ticket_id="PASS-1",
            runs=[
                {
                    "answer_correctness": 0.95,
                    "context_relevance": 0.85,
                    "context_precision": 0.75,
                }
            ],
        )

        # Create failing ticket (composite < 0.80)
        failing = TicketEvaluation(
            ticket_id="FAIL-1",
            runs=[
                {
                    "answer_correctness": 0.60,
                    "context_relevance": 0.50,
                    "context_precision": 0.40,
                }
            ],
        )

        pattern.tickets["PASS-1"] = passing
        pattern.tickets["FAIL-1"] = failing

        assert "PASS-1" in pattern.passing_tickets
        assert "FAIL-1" in pattern.failing_tickets
        assert "FAIL-1" not in pattern.passing_tickets
        assert "PASS-1" not in pattern.failing_tickets

    def test_success_rate(self):
        """Test success rate calculation."""
        pattern = PatternEvaluation(pattern_id="RATE_TEST")

        # 3 passing, 1 failing
        for i in range(3):
            pattern.tickets[f"PASS-{i}"] = TicketEvaluation(
                ticket_id=f"PASS-{i}",
                runs=[
                    {
                        "answer_correctness": 0.95,
                        "context_relevance": 0.85,
                        "context_precision": 0.75,
                    }
                ],
            )

        pattern.tickets["FAIL-1"] = TicketEvaluation(
            ticket_id="FAIL-1",
            runs=[
                {
                    "answer_correctness": 0.50,
                    "context_relevance": 0.40,
                    "context_precision": 0.30,
                }
            ],
        )

        # Success rate: 3/4 = 0.75
        assert pattern.success_rate == 0.75

    def test_success_rate_empty_pattern(self):
        """Empty pattern has 0.0 success rate."""
        pattern = PatternEvaluation(pattern_id="EMPTY")

        assert pattern.success_rate == 0.0

    def test_mean_composite_score(self):
        """Test mean composite score calculation."""
        pattern = PatternEvaluation(pattern_id="MEAN_TEST")

        ticket1 = TicketEvaluation(
            ticket_id="T1",
            runs=[
                {
                    "answer_correctness": 1.0,
                    "context_relevance": 1.0,
                    "context_precision": 1.0,
                }
            ],
        )
        # Composite = 1.0*0.80 + 1.0*0.15 + 1.0*0.05 = 1.0

        ticket2 = TicketEvaluation(
            ticket_id="T2",
            runs=[
                {
                    "answer_correctness": 0.60,
                    "context_relevance": 0.60,
                    "context_precision": 0.60,
                }
            ],
        )
        # Composite = 0.60*0.80 + 0.60*0.15 + 0.60*0.05 = 0.60

        pattern.tickets["T1"] = ticket1
        pattern.tickets["T2"] = ticket2

        # Mean: (1.0 + 0.60) / 2 = 0.80
        assert pattern.mean_composite_score == 0.80

    def test_mean_composite_score_empty_pattern(self):
        """Empty pattern has 0.0 mean composite score."""
        pattern = PatternEvaluation(pattern_id="EMPTY")

        assert pattern.mean_composite_score == 0.0

    def test_passes_majority_criteria(self):
        """Test passes() with majority criteria (>50%)."""
        pattern = PatternEvaluation(pattern_id="MAJORITY")

        # 3 passing, 2 failing → passes
        for i in range(3):
            pattern.tickets[f"PASS-{i}"] = TicketEvaluation(
                ticket_id=f"PASS-{i}",
                runs=[
                    {
                        "answer_correctness": 0.95,
                        "context_relevance": 0.85,
                        "context_precision": 0.75,
                    }
                ],
            )

        for i in range(2):
            pattern.tickets[f"FAIL-{i}"] = TicketEvaluation(
                ticket_id=f"FAIL-{i}",
                runs=[
                    {
                        "answer_correctness": 0.50,
                        "context_relevance": 0.40,
                        "context_precision": 0.30,
                    }
                ],
            )

        assert pattern.passes(criteria="majority")

        # 2 passing, 3 failing → fails
        pattern2 = PatternEvaluation(pattern_id="MAJORITY_FAIL")

        for i in range(2):
            pattern2.tickets[f"PASS-{i}"] = TicketEvaluation(
                ticket_id=f"PASS-{i}",
                runs=[
                    {
                        "answer_correctness": 0.95,
                        "context_relevance": 0.85,
                        "context_precision": 0.75,
                    }
                ],
            )

        for i in range(3):
            pattern2.tickets[f"FAIL-{i}"] = TicketEvaluation(
                ticket_id=f"FAIL-{i}",
                runs=[
                    {
                        "answer_correctness": 0.50,
                        "context_relevance": 0.40,
                        "context_precision": 0.30,
                    }
                ],
            )

        assert not pattern2.passes(criteria="majority")

    def test_passes_all_criteria(self):
        """Test passes() with all criteria (100%)."""
        # All passing
        pattern_pass = PatternEvaluation(pattern_id="ALL_PASS")

        for i in range(3):
            pattern_pass.tickets[f"PASS-{i}"] = TicketEvaluation(
                ticket_id=f"PASS-{i}",
                runs=[
                    {
                        "answer_correctness": 0.95,
                        "context_relevance": 0.85,
                        "context_precision": 0.75,
                    }
                ],
            )

        assert pattern_pass.passes(criteria="all")

        # One failing → fails
        pattern_fail = PatternEvaluation(pattern_id="NOT_ALL_PASS")

        for i in range(2):
            pattern_fail.tickets[f"PASS-{i}"] = TicketEvaluation(
                ticket_id=f"PASS-{i}",
                runs=[
                    {
                        "answer_correctness": 0.95,
                        "context_relevance": 0.85,
                        "context_precision": 0.75,
                    }
                ],
            )

        pattern_fail.tickets["FAIL-1"] = TicketEvaluation(
            ticket_id="FAIL-1",
            runs=[
                {
                    "answer_correctness": 0.50,
                    "context_relevance": 0.40,
                    "context_precision": 0.30,
                }
            ],
        )

        assert not pattern_fail.passes(criteria="all")

    def test_passes_any_criteria(self):
        """Test passes() with any criteria (at least one passing)."""
        # At least one passing
        pattern_pass = PatternEvaluation(pattern_id="ANY_PASS")

        pattern_pass.tickets["PASS-1"] = TicketEvaluation(
            ticket_id="PASS-1",
            runs=[
                {
                    "answer_correctness": 0.95,
                    "context_relevance": 0.85,
                    "context_precision": 0.75,
                }
            ],
        )

        for i in range(3):
            pattern_pass.tickets[f"FAIL-{i}"] = TicketEvaluation(
                ticket_id=f"FAIL-{i}",
                runs=[
                    {
                        "answer_correctness": 0.50,
                        "context_relevance": 0.40,
                        "context_precision": 0.30,
                    }
                ],
            )

        assert pattern_pass.passes(criteria="any")

        # None passing → fails
        pattern_fail = PatternEvaluation(pattern_id="NONE_PASS")

        for i in range(3):
            pattern_fail.tickets[f"FAIL-{i}"] = TicketEvaluation(
                ticket_id=f"FAIL-{i}",
                runs=[
                    {
                        "answer_correctness": 0.50,
                        "context_relevance": 0.40,
                        "context_precision": 0.30,
                    }
                ],
            )

        assert not pattern_fail.passes(criteria="any")

    def test_passes_average_criteria(self):
        """Test passes() with average criteria (mean composite >= 0.80)."""
        # Mean composite = 0.80 → passes
        pattern_pass = PatternEvaluation(pattern_id="AVG_PASS")

        pattern_pass.tickets["T1"] = TicketEvaluation(
            ticket_id="T1",
            runs=[
                {
                    "answer_correctness": 1.0,
                    "context_relevance": 1.0,
                    "context_precision": 1.0,
                }
            ],
        )  # Composite = 1.0

        pattern_pass.tickets["T2"] = TicketEvaluation(
            ticket_id="T2",
            runs=[
                {
                    "answer_correctness": 0.60,
                    "context_relevance": 0.60,
                    "context_precision": 0.60,
                }
            ],
        )  # Composite = 0.60

        # Mean: (1.0 + 0.60) / 2 = 0.80
        assert pattern_pass.passes(criteria="average")

        # Mean composite < 0.80 → fails
        pattern_fail = PatternEvaluation(pattern_id="AVG_FAIL")

        pattern_fail.tickets["T1"] = TicketEvaluation(
            ticket_id="T1",
            runs=[
                {
                    "answer_correctness": 0.70,
                    "context_relevance": 0.70,
                    "context_precision": 0.70,
                }
            ],
        )  # Composite = 0.70

        pattern_fail.tickets["T2"] = TicketEvaluation(
            ticket_id="T2",
            runs=[
                {
                    "answer_correctness": 0.60,
                    "context_relevance": 0.60,
                    "context_precision": 0.60,
                }
            ],
        )  # Composite = 0.60

        # Mean: (0.70 + 0.60) / 2 = 0.65
        assert not pattern_fail.passes(criteria="average")

    def test_passes_invalid_criteria_raises_error(self):
        """Invalid criteria should raise ValueError."""
        pattern = PatternEvaluation(pattern_id="TEST")

        with pytest.raises(ValueError, match="Unknown criteria"):
            pattern.passes(criteria="invalid")

    def test_get_ticket_by_status(self):
        """Test filtering tickets by status."""
        pattern = PatternEvaluation(pattern_id="STATUS_TEST")

        # Stable passing
        pattern.tickets["STABLE-1"] = TicketEvaluation(
            ticket_id="STABLE-1",
            runs=[
                {"answer_correctness": 0.92},
                {"answer_correctness": 0.91},
                {"answer_correctness": 0.93},
            ],
        )

        # Erratic (need higher variance: var > 0.05)
        pattern.tickets["ERRATIC-1"] = TicketEvaluation(
            ticket_id="ERRATIC-1",
            runs=[
                {"answer_correctness": 0.95},
                {"answer_correctness": 0.30},
                {"answer_correctness": 0.95},
                {"answer_correctness": 0.30},
            ],
        )

        # Consistently failing
        pattern.tickets["FAILING-1"] = TicketEvaluation(
            ticket_id="FAILING-1",
            runs=[
                {"answer_correctness": 0.40},
                {"answer_correctness": 0.42},
                {"answer_correctness": 0.38},
            ],
        )

        stable = pattern.get_ticket_by_status("STABLE_PASSING")
        erratic = pattern.get_ticket_by_status("ERRATIC")
        failing = pattern.get_ticket_by_status("CONSISTENTLY_FAILING")

        assert "STABLE-1" in stable
        assert "ERRATIC-1" in erratic
        assert "FAILING-1" in failing

        # Empty status
        empty = pattern.get_ticket_by_status("IMPROVING")
        assert empty == []

    def test_to_dict(self):
        """Test pattern serialization to dictionary."""
        pattern = PatternEvaluation(pattern_id="DICT_TEST")

        pattern.tickets["T1"] = TicketEvaluation(
            ticket_id="T1",
            runs=[
                {
                    "answer_correctness": 0.95,
                    "context_relevance": 0.85,
                    "context_precision": 0.75,
                }
            ],
        )

        pattern.tickets["T2"] = TicketEvaluation(
            ticket_id="T2",
            runs=[
                {
                    "answer_correctness": 0.50,
                    "context_relevance": 0.40,
                    "context_precision": 0.30,
                }
            ],
        )

        result = pattern.to_dict()

        assert result["pattern_id"] == "DICT_TEST"
        assert result["num_tickets"] == 2
        assert "passing_tickets" in result
        assert "failing_tickets" in result
        assert "success_rate" in result
        assert "mean_composite_score" in result
        assert "tickets" in result
        assert "T1" in result["tickets"]
        assert "T2" in result["tickets"]
        assert isinstance(result["tickets"]["T1"], dict)

    def test_baseline_comparison(self):
        """Test pattern-level baseline comparison."""
        baseline = PatternEvaluation(pattern_id="BASELINE")

        baseline.tickets["T1"] = TicketEvaluation(
            ticket_id="T1", runs=[{"answer_correctness": 0.60}]
        )

        baseline.tickets["T2"] = TicketEvaluation(
            ticket_id="T2", runs=[{"answer_correctness": 0.65}]
        )

        # Create current pattern with baseline reference
        current = PatternEvaluation(pattern_id="CURRENT", baseline=baseline)

        current.tickets["T1"] = TicketEvaluation(
            ticket_id="T1",
            runs=[{"answer_correctness": 0.85}],
            baseline=baseline.tickets["T1"],
        )

        current.tickets["T2"] = TicketEvaluation(
            ticket_id="T2",
            runs=[{"answer_correctness": 0.80}],  # Improved but not passing
            baseline=baseline.tickets["T2"],
        )

        # Both tickets improved
        assert current.tickets["T1"].improvement_over_baseline() > 0.0
        assert current.tickets["T2"].improvement_over_baseline() > 0.0

        # Check statuses reflect improvement
        # T1: improved by 0.25 (>0.10) but not passing (<0.90) → IMPROVING
        assert current.tickets["T1"].status == "IMPROVING"
        # T2: improved by 0.15 (>0.10) but not passing (<0.90) → IMPROVING
        assert current.tickets["T2"].status == "IMPROVING"

    def test_empty_pattern_properties(self):
        """Test empty pattern edge cases."""
        pattern = PatternEvaluation(pattern_id="EMPTY")

        assert pattern.num_tickets == 0
        assert pattern.passing_tickets == []
        assert pattern.failing_tickets == []
        assert pattern.success_rate == 0.0
        assert pattern.mean_composite_score == 0.0
        assert not pattern.passes(criteria="majority")
        assert not pattern.passes(criteria="all")
        assert not pattern.passes(criteria="any")
        assert not pattern.passes(criteria="average")
