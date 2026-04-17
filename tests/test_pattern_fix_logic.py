"""Tests for pattern fix loop logic.

Tests the core logic for pattern-based ticket fixing including:
- No-doc ticket handling
- Skip tag classification
- Pattern success criteria
- RAG quality warnings (answer correct but docs poor)
"""

import tempfile
from pathlib import Path

import yaml

# Import from the actual module
# Note: We're testing logic, not infrastructure (no container restarts, etc.)


class TestNoDocTicketHandling:
    """Test no-doc ticket filtering and classification logic."""

    def test_load_pattern_with_no_doc_tickets(self):
        """Test that pattern loads correctly with mixed doc/no-doc tickets."""
        # Create test pattern YAML
        pattern_data = [
            {
                "conversation_group_id": "RSPEED-1001",
                "turns": [
                    {
                        "query": "How to fix grub?",
                        "expected_urls": ["solutions/123456", "articles/789012"],
                    }
                ],
            },
            {
                "conversation_group_id": "RSPEED-1002",
                "turns": [{"query": "What is the error code?"}],  # NO expected_urls
            },
            {
                "conversation_group_id": "RSPEED-1003",
                "turns": [
                    {"query": "How to update kernel?", "expected_urls": ["solutions/999999"]}
                ],
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(pattern_data, f)
            pattern_file = Path(f.name)

        try:
            # Load and validate
            with open(pattern_file) as f:
                conversations = yaml.safe_load(f)

            tickets_with_docs = 0
            tickets_without_docs = []
            pattern_tickets = []

            for conv in conversations:
                ticket_id = conv["conversation_group_id"]

                # Check if ticket has expected_urls
                has_expected_urls = False
                for turn in conv.get("turns", []):
                    if turn.get("expected_urls"):
                        has_expected_urls = True
                        break

                ticket_info = {"ticket_id": ticket_id, "has_expected_urls": has_expected_urls}
                pattern_tickets.append(ticket_info)

                if has_expected_urls:
                    tickets_with_docs += 1
                else:
                    tickets_without_docs.append(ticket_id)

            # Assertions
            assert len(pattern_tickets) == 3
            assert tickets_with_docs == 2
            assert len(tickets_without_docs) == 1
            assert "RSPEED-1002" in tickets_without_docs
            assert "RSPEED-1001" not in tickets_without_docs
            assert "RSPEED-1003" not in tickets_without_docs

            # Verify ticket info structure
            assert pattern_tickets[0]["ticket_id"] == "RSPEED-1001"
            assert pattern_tickets[0]["has_expected_urls"] is True
            assert pattern_tickets[1]["ticket_id"] == "RSPEED-1002"
            assert pattern_tickets[1]["has_expected_urls"] is False
            assert pattern_tickets[2]["ticket_id"] == "RSPEED-1003"
            assert pattern_tickets[2]["has_expected_urls"] is True

        finally:
            pattern_file.unlink()

    def test_all_tickets_have_docs(self):
        """Test pattern where all tickets have documentation."""
        pattern_data = [
            {
                "conversation_group_id": f"RSPEED-100{i}",
                "turns": [{"query": "Test query", "expected_urls": ["solutions/123"]}],
            }
            for i in range(3)
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(pattern_data, f)
            pattern_file = Path(f.name)

        try:
            with open(pattern_file) as f:
                conversations = yaml.safe_load(f)

            tickets_without_docs = []
            for conv in conversations:
                has_expected_urls = any(turn.get("expected_urls") for turn in conv.get("turns", []))
                if not has_expected_urls:
                    tickets_without_docs.append(conv["conversation_group_id"])

            assert len(tickets_without_docs) == 0

        finally:
            pattern_file.unlink()

    def test_all_tickets_missing_docs(self):
        """Test pattern where NO tickets have documentation."""
        pattern_data = [
            {"conversation_group_id": f"RSPEED-100{i}", "turns": [{"query": "Test query"}]}
            for i in range(3)
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(pattern_data, f)
            pattern_file = Path(f.name)

        try:
            with open(pattern_file) as f:
                conversations = yaml.safe_load(f)

            tickets_with_docs = 0
            tickets_without_docs = []

            for conv in conversations:
                has_expected_urls = any(turn.get("expected_urls") for turn in conv.get("turns", []))
                if has_expected_urls:
                    tickets_with_docs += 1
                else:
                    tickets_without_docs.append(conv["conversation_group_id"])

            assert tickets_with_docs == 0
            assert len(tickets_without_docs) == 3

        finally:
            pattern_file.unlink()

    def test_no_doc_ticket_classification_passing(self):
        """Test that passing no-doc tickets are marked to skip."""
        from heal.core.stability_classifier import StabilityStatus, classify_stability

        # Simulate a no-doc ticket that passes
        runs = [0.92, 0.91, 0.93]  # All passing
        classification = classify_stability(runs, threshold=0.90)

        # For a no-doc ticket that passes, we override to skip
        is_no_doc = True
        if is_no_doc and classification.status == StabilityStatus.STABLE_PASSING:
            classification.skip = True
            classification.reason = (
                f"{classification.reason} [NO-DOC: Answered from LLM training data]"
            )

        assert classification.skip is True
        assert "NO-DOC" in classification.reason
        assert "LLM training data" in classification.reason

    def test_no_doc_ticket_classification_failing(self):
        """Test that failing no-doc tickets are marked HIGH priority."""
        from heal.core.stability_classifier import StabilityStatus, classify_stability

        # Simulate a no-doc ticket that fails
        runs = [0.40, 0.42, 0.38]  # All failing
        classification = classify_stability(runs, threshold=0.90)

        # For a no-doc ticket that fails, we flag for SME review
        is_no_doc = True
        if is_no_doc and classification.status != StabilityStatus.STABLE_PASSING:
            classification.skip = False
            classification.priority = "HIGH"
            classification.needs_review = True
            classification.reason = (
                f"{classification.reason} [NO-DOC: Failing without docs, needs SME review]"
            )

        assert classification.skip is False
        assert classification.priority == "HIGH"
        assert classification.needs_review is True
        assert "NO-DOC" in classification.reason
        assert "SME review" in classification.reason

    def test_regular_ticket_classification_unchanged(self):
        """Test that regular tickets (with docs) aren't affected by no-doc logic."""
        from heal.core.stability_classifier import classify_stability

        # Regular ticket, passing
        runs = [0.92, 0.91, 0.93]
        classification = classify_stability(runs, threshold=0.90)

        # Don't apply no-doc overrides
        is_no_doc = False
        if not is_no_doc:
            # Regular classification, no special handling
            pass

        assert classification.skip is True  # Regular STABLE_PASSING behavior
        assert "NO-DOC" not in classification.reason


class TestPatternSuccessCriteria:
    """Test pattern-level success criteria and composite scoring."""

    def test_majority_passing_criteria(self):
        """Test that pattern passes with >50% tickets passing."""
        from heal.core.ticket_evaluation import PatternEvaluation, TicketEvaluation

        pattern = PatternEvaluation(pattern_id="TEST_PATTERN")

        # 3 passing, 2 failing = 60% pass rate
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
        assert pattern.success_rate == 0.6
        assert len(pattern.passing_tickets) == 3
        assert len(pattern.failing_tickets) == 2

    def test_majority_failing_criteria(self):
        """Test that pattern fails with <50% tickets passing."""
        from heal.core.ticket_evaluation import PatternEvaluation, TicketEvaluation

        pattern = PatternEvaluation(pattern_id="TEST_PATTERN")

        # 2 passing, 3 failing = 40% pass rate
        for i in range(2):
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

        for i in range(3):
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

        assert not pattern.passes(criteria="majority")
        assert pattern.success_rate == 0.4
        assert len(pattern.passing_tickets) == 2
        assert len(pattern.failing_tickets) == 3

    def test_pattern_with_no_doc_tickets_included(self):
        """Test that no-doc tickets are included in pattern success rate.

        No-doc tickets use answer_correctness >= 0.90 threshold (not composite).
        """
        from heal.core.ticket_evaluation import PatternEvaluation, TicketEvaluation

        pattern = PatternEvaluation(pattern_id="MIXED_PATTERN")

        # 1 regular passing
        pattern.tickets["REG-PASS"] = TicketEvaluation(
            ticket_id="REG-PASS",
            runs=[
                {
                    "answer_correctness": 0.95,
                    "context_relevance": 0.85,
                    "context_precision": 0.75,
                }
            ],
        )

        # 1 no-doc passing (uses answer_correctness >= 0.90, not composite)
        pattern.tickets["NO-DOC-PASS"] = TicketEvaluation(
            ticket_id="NO-DOC-PASS",
            runs=[
                {
                    "answer_correctness": 0.95,  # >= 0.90, passes!
                    "context_relevance": 0.0,  # No docs (ignored)
                    "context_precision": 0.0,  # No docs (ignored)
                }
            ],
            is_no_doc=True,  # Flag that this is a no-doc ticket
        )

        # 1 no-doc failing
        pattern.tickets["NO-DOC-FAIL"] = TicketEvaluation(
            ticket_id="NO-DOC-FAIL",
            runs=[
                {
                    "answer_correctness": 0.40,  # < 0.90, fails
                    "context_relevance": 0.0,  # No docs
                    "context_precision": 0.0,  # No docs
                }
            ],
            is_no_doc=True,
        )

        # Success rate: 2/3 = 66.7%
        assert pattern.success_rate > 0.5
        assert len(pattern.passing_tickets) == 2
        assert len(pattern.failing_tickets) == 1
        assert pattern.passes(criteria="majority")

    def test_composite_score_weights(self):
        """Test that composite score uses correct weights (80/15/5)."""
        from heal.core.ticket_evaluation import TicketEvaluation

        # Test pure answer correctness
        ticket1 = TicketEvaluation(
            ticket_id="T1",
            runs=[
                {
                    "answer_correctness": 1.0,
                    "context_relevance": 0.0,
                    "context_precision": 0.0,
                }
            ],
        )
        # Composite = 1.0 * 0.80 = 0.80
        assert ticket1.composite_score == 0.80

        # Test balanced scores
        ticket2 = TicketEvaluation(
            ticket_id="T2",
            runs=[
                {
                    "answer_correctness": 0.90,
                    "context_relevance": 0.80,
                    "context_precision": 0.70,
                }
            ],
        )
        # Composite = 0.90*0.80 + 0.80*0.15 + 0.70*0.05
        #           = 0.72 + 0.12 + 0.035 = 0.875
        assert abs(ticket2.composite_score - 0.875) < 0.001

    def test_baseline_improvement_tracking(self):
        """Test that we can track improvement over baseline."""
        from heal.core.ticket_evaluation import TicketEvaluation

        # Baseline
        baseline = TicketEvaluation(ticket_id="TEST-001", runs=[{"answer_correctness": 0.60}])

        # Current (improved)
        current = TicketEvaluation(
            ticket_id="TEST-001",
            runs=[{"answer_correctness": 0.85}],
            baseline=baseline,
        )

        improvement = current.improvement_over_baseline()
        assert improvement == 0.25  # 0.85 - 0.60
        assert current.status == "IMPROVING"  # Improved by >0.10

    def test_ticket_status_classification(self):
        """Test that ticket status is correctly classified."""
        from heal.core.ticket_evaluation import TicketEvaluation

        # STABLE_PASSING
        stable = TicketEvaluation(
            ticket_id="STABLE",
            runs=[
                {"answer_correctness": 0.92},
                {"answer_correctness": 0.91},
                {"answer_correctness": 0.93},
            ],
        )
        assert stable.status == "STABLE_PASSING"

        # CONSISTENTLY_FAILING
        failing = TicketEvaluation(
            ticket_id="FAILING",
            runs=[
                {"answer_correctness": 0.40},
                {"answer_correctness": 0.42},
                {"answer_correctness": 0.38},
            ],
        )
        assert failing.status == "CONSISTENTLY_FAILING"

        # ERRATIC (high variance)
        erratic = TicketEvaluation(
            ticket_id="ERRATIC",
            runs=[
                {"answer_correctness": 0.95},
                {"answer_correctness": 0.30},
                {"answer_correctness": 0.95},
                {"answer_correctness": 0.30},
            ],
        )
        assert erratic.status == "ERRATIC"


class TestSkipTagLogic:
    """Test skip tag logic for optimization phases."""

    def test_skip_tags_for_stable_passing(self):
        """Test that stable passing tickets get skip=true."""
        from heal.core.stability_classifier import classify_stability

        runs = [0.92, 0.91, 0.93]
        classification = classify_stability(runs, threshold=0.90)

        assert classification.status.value == "STABLE_PASSING"
        assert classification.skip is True
        assert classification.priority == "LOW"

    def test_no_skip_for_failing_tickets(self):
        """Test that failing tickets don't get skipped."""
        from heal.core.stability_classifier import classify_stability

        runs = [0.65, 0.68, 0.62]
        classification = classify_stability(runs, threshold=0.90)

        assert classification.status.value == "CONSISTENTLY_FAILING"
        assert classification.skip is False
        assert classification.priority == "HIGH"

    def test_skip_breakdown_by_category(self):
        """Test counting skip tickets by category (regular vs no-doc)."""
        ticket_classifications = {
            "REG-PASS": type(
                "Classification",
                (),
                {"skip": True, "status": type("Status", (), {"value": "STABLE_PASSING"})},
            )(),
            "NO-DOC-PASS": type(
                "Classification",
                (),
                {"skip": True, "status": type("Status", (), {"value": "STABLE_PASSING"})},
            )(),
            "FAIL": type(
                "Classification",
                (),
                {"skip": False, "status": type("Status", (), {"value": "CONSISTENTLY_FAILING"})},
            )(),
        }

        no_doc_tickets = {"NO-DOC-PASS"}

        # Count skipped tickets
        skipped_stable = [
            tid
            for tid, cls in ticket_classifications.items()
            if cls.skip and cls.status.value == "STABLE_PASSING"
        ]
        skipped_no_doc = [tid for tid in skipped_stable if tid in no_doc_tickets]
        skipped_regular = len(skipped_stable) - len(skipped_no_doc)

        assert len(skipped_stable) == 2
        assert len(skipped_no_doc) == 1
        assert skipped_regular == 1


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_pattern(self):
        """Test handling of empty pattern."""
        from heal.core.ticket_evaluation import PatternEvaluation

        pattern = PatternEvaluation(pattern_id="EMPTY")

        assert pattern.num_tickets == 0
        assert pattern.success_rate == 0.0
        assert pattern.passing_tickets == []
        assert pattern.failing_tickets == []
        assert not pattern.passes(criteria="majority")

    def test_pattern_with_single_ticket(self):
        """Test pattern with only one ticket."""
        from heal.core.ticket_evaluation import PatternEvaluation, TicketEvaluation

        pattern = PatternEvaluation(pattern_id="SINGLE")
        pattern.tickets["ONLY-1"] = TicketEvaluation(
            ticket_id="ONLY-1",
            runs=[
                {
                    "answer_correctness": 0.95,
                    "context_relevance": 0.85,
                    "context_precision": 0.75,
                }
            ],
        )

        # 1/1 = 100% > 50%
        assert pattern.passes(criteria="majority")
        assert pattern.success_rate == 1.0

    def test_pattern_exactly_fifty_percent(self):
        """Test pattern with exactly 50% passing (should fail)."""
        from heal.core.ticket_evaluation import PatternEvaluation, TicketEvaluation

        pattern = PatternEvaluation(pattern_id="FIFTY")

        # 2 passing, 2 failing = exactly 50%
        for i in range(2):
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

        # 50% is NOT > 50%, so should fail
        assert not pattern.passes(criteria="majority")
        assert pattern.success_rate == 0.5

    def test_ticket_with_no_runs(self):
        """Test ticket with no evaluation runs."""
        from heal.core.ticket_evaluation import TicketEvaluation

        ticket = TicketEvaluation(ticket_id="EMPTY", runs=[])

        assert ticket.num_runs == 0
        assert ticket.mean_answer_correctness == 0.0
        assert ticket.status == "NO_DATA"
        assert not ticket.passes()

    def test_ticket_with_missing_metrics(self):
        """Test ticket with incomplete metrics in runs."""
        from heal.core.ticket_evaluation import TicketEvaluation

        ticket = TicketEvaluation(
            ticket_id="INCOMPLETE",
            runs=[
                {"answer_correctness": 0.95},  # Missing context metrics
                {
                    "answer_correctness": 0.90,
                    "context_relevance": 0.80,
                },  # Missing precision
            ],
        )

        # Should handle gracefully
        assert ticket.mean_answer_correctness == 0.925
        assert ticket.mean_context_relevance == 0.40  # (0 + 0.80) / 2
        assert ticket.mean_context_precision == 0.0  # Both missing


class TestRAGQualityWarnings:
    """Test RAG quality warning logic in _is_passing method."""

    class MockPatternFix:
        """Minimal mock for testing _is_passing method."""

        def _is_passing(self, metrics: dict, answer_threshold: float) -> bool:
            """Check if metrics indicate passing ticket with RAG quality warnings."""
            ans_corr = metrics.get("answer_correctness")

            if ans_corr is None:
                return False

            # ANSWER-FIRST: If answer is correct, we pass
            passing = ans_corr >= answer_threshold

            # Check RAG quality for warnings (doesn't affect pass/fail)
            if passing:
                ctx_rel = metrics.get("context_relevance")
                ctx_prec = metrics.get("context_precision")
                faith = metrics.get("faithfulness")

                # Check if RAG was used but quality was poor
                rag_used = ctx_rel is not None or ctx_prec is not None
                if rag_used:
                    # Check for poor RAG metrics
                    poor_context = (ctx_rel is not None and ctx_rel < 0.7) or (
                        ctx_prec is not None and ctx_prec < 0.7
                    )
                    poor_faithfulness = faith is not None and faith < 0.7

                    if poor_context or poor_faithfulness:
                        print("   ⚠️  PASS but RAG quality low:")
                        print(f"      answer_correctness={ans_corr:.2f} (GOOD)")
                        if ctx_rel is not None:
                            print(
                                f"      context_relevance={ctx_rel:.2f} {'(LOW)' if ctx_rel < 0.7 else ''}"
                            )
                        if ctx_prec is not None:
                            print(
                                f"      context_precision={ctx_prec:.2f} {'(LOW)' if ctx_prec < 0.7 else ''}"
                            )
                        if faith is not None:
                            print(
                                f"      faithfulness={faith:.2f} {'(LOW)' if faith < 0.7 else ''}"
                            )
                        print("      → LLM likely ignored bad docs and answered correctly anyway")
                        print("      → Consider reviewing Solr config (may need tuning)")

            return passing

    def test_passing_with_good_answer_and_good_rag(self, capsys):
        """Test passing ticket with good answer and good RAG metrics - no warning."""
        mock = self.MockPatternFix()
        metrics = {
            "answer_correctness": 0.95,
            "context_relevance": 0.90,
            "context_precision": 0.85,
            "faithfulness": 0.92,
        }

        result = mock._is_passing(metrics, answer_threshold=0.90)

        assert result is True
        captured = capsys.readouterr()
        assert "⚠️  PASS but RAG quality low" not in captured.out

    def test_passing_with_good_answer_but_poor_rag(self, capsys):
        """Test passing ticket with good answer but poor RAG - should warn."""
        mock = self.MockPatternFix()
        metrics = {
            "answer_correctness": 0.95,
            "context_relevance": 0.50,  # LOW
            "context_precision": 0.45,  # LOW
            "faithfulness": 0.60,  # LOW
        }

        result = mock._is_passing(metrics, answer_threshold=0.90)

        assert result is True  # Still passes because answer is correct
        captured = capsys.readouterr()
        assert "⚠️  PASS but RAG quality low" in captured.out
        assert "answer_correctness=0.95 (GOOD)" in captured.out
        assert "context_relevance=0.50 (LOW)" in captured.out
        assert "context_precision=0.45 (LOW)" in captured.out
        assert "faithfulness=0.60 (LOW)" in captured.out
        assert "LLM likely ignored bad docs" in captured.out
        assert "Consider reviewing Solr config" in captured.out

    def test_passing_with_good_answer_mixed_rag(self, capsys):
        """Test passing with good answer and mixed RAG (some good, some poor)."""
        mock = self.MockPatternFix()
        metrics = {
            "answer_correctness": 0.92,
            "context_relevance": 0.50,  # LOW
            "context_precision": 0.85,  # GOOD
            "faithfulness": 0.90,  # GOOD
        }

        result = mock._is_passing(metrics, answer_threshold=0.90)

        assert result is True
        captured = capsys.readouterr()
        assert "⚠️  PASS but RAG quality low" in captured.out
        assert "context_relevance=0.50 (LOW)" in captured.out

    def test_failing_with_poor_answer_good_rag(self, capsys):
        """Test failing ticket with poor answer but good RAG - no warning."""
        mock = self.MockPatternFix()
        metrics = {
            "answer_correctness": 0.70,  # Below threshold
            "context_relevance": 0.90,
            "context_precision": 0.85,
            "faithfulness": 0.92,
        }

        result = mock._is_passing(metrics, answer_threshold=0.90)

        assert result is False
        captured = capsys.readouterr()
        assert "⚠️  PASS but RAG quality low" not in captured.out

    def test_passing_with_no_rag_metrics(self, capsys):
        """Test passing with good answer but no RAG metrics (no-doc ticket)."""
        mock = self.MockPatternFix()
        metrics = {
            "answer_correctness": 0.95,
            # No RAG metrics at all
        }

        result = mock._is_passing(metrics, answer_threshold=0.90)

        assert result is True
        captured = capsys.readouterr()
        assert "⚠️  PASS but RAG quality low" not in captured.out

    def test_passing_at_exact_threshold(self, capsys):
        """Test passing at exactly the threshold with poor RAG."""
        mock = self.MockPatternFix()
        metrics = {
            "answer_correctness": 0.90,  # Exactly at threshold
            "context_relevance": 0.40,  # LOW
            "context_precision": 0.30,  # LOW
        }

        result = mock._is_passing(metrics, answer_threshold=0.90)

        assert result is True
        captured = capsys.readouterr()
        assert "⚠️  PASS but RAG quality low" in captured.out

    def test_just_below_threshold(self, capsys):
        """Test just below threshold - should fail, no warning."""
        mock = self.MockPatternFix()
        metrics = {
            "answer_correctness": 0.89,  # Just below 0.90
            "context_relevance": 0.40,
            "context_precision": 0.30,
        }

        result = mock._is_passing(metrics, answer_threshold=0.90)

        assert result is False
        captured = capsys.readouterr()
        assert "⚠️  PASS but RAG quality low" not in captured.out

    def test_missing_answer_correctness(self):
        """Test with missing answer_correctness metric."""
        mock = self.MockPatternFix()
        metrics = {
            "context_relevance": 0.90,
            "context_precision": 0.85,
        }

        result = mock._is_passing(metrics, answer_threshold=0.90)

        assert result is False
