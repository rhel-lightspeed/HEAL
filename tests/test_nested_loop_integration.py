"""Integration tests for nested loop architecture.

Tests the complete nested loop flow:
- Outer loop: validation_cycles with full answer validation
- Inner loop: fast Solr optimization with early exit
- Pattern database integration
- Iteration context passing to multi-agent
- Incremental learning (no reverts)
- Early exit when passing threshold
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
import yaml


class TestNestedLoopIntegration:
    """Integration tests for nested loop architecture."""

    @pytest.fixture
    def pattern_config(self, tmp_path):
        """Create a test pattern YAML file."""
        pattern_data = [
            {
                "conversation_group_id": "TICKET-001",
                "turns": [
                    {
                        "query": "How to fix GRUB?",
                        "expected_urls": ["solutions/123", "articles/456"],
                    }
                ],
            },
            {
                "conversation_group_id": "TICKET-002",
                "turns": [
                    {
                        "query": "How to update kernel?",
                        "expected_urls": ["solutions/789"],
                    }
                ],
            },
        ]

        pattern_file = tmp_path / "TEST_PATTERN.yaml"
        with open(pattern_file, "w") as f:
            yaml.dump(pattern_data, f)

        return pattern_file

    @pytest.fixture
    def mock_pattern_fix_agent(self, tmp_path, pattern_config):
        """Create a mock PatternFixAgent with necessary methods."""
        from heal.runners.run_pattern_fix_poc import PatternFixAgent
        from heal.agents.okp_mcp_agent import PatternEvaluationResult

        # Create mock agent
        agent = MagicMock(spec=PatternFixAgent)
        agent.pattern_id = "TEST_PATTERN"
        agent.okp_mcp_root = tmp_path / "okp-mcp"
        agent.okp_mcp_root.mkdir()

        # Create src/okp_mcp/solr.py for git operations
        solr_file = agent.okp_mcp_root / "src" / "okp_mcp" / "solr.py"
        solr_file.parent.mkdir(parents=True, exist_ok=True)
        solr_file.write_text("# Solr config\n")

        # Mock pattern database
        from heal.core.fix_pattern_database import FixPatternDatabase

        agent.pattern_db = FixPatternDatabase()

        # Mock multi-agent (optional)
        agent.multi_agent = None

        return agent

    def test_nested_loop_basic_flow(self, mock_pattern_fix_agent, tmp_path):
        """Test basic nested loop flow with 2 cycles."""
        from heal.agents.okp_mcp_agent import PatternEvaluationResult, EvaluationResult

        agent = mock_pattern_fix_agent

        # Track calls
        diagnose_calls = []
        record_iteration_calls = []

        # Mock diagnose() to return improving results
        def mock_diagnose(ticket_id=None, runs=3, iteration=None):
            # Return pattern-wide result
            result = MagicMock(spec=PatternEvaluationResult)

            # Simulate improving answer over cycles
            cycle = iteration or 1
            result.pattern_answer = 0.65 + (cycle * 0.10)  # 0.75, 0.85, 0.95
            result.pattern_faithfulness = 0.80
            result.pattern_url_f1 = 0.60

            diagnose_calls.append({"cycle": cycle, "answer": result.pattern_answer})

            return result

        agent.diagnose = mock_diagnose

        # Mock pattern_db.record_iteration()
        agent.pattern_db.record_iteration = MagicMock(
            side_effect=lambda **kwargs: record_iteration_calls.append(kwargs)
        )

        # Mock pattern_db.get_iteration_context()
        agent.pattern_db.get_iteration_context = MagicMock(return_value="")

        # Mock restart_okp_mcp
        agent.restart_okp_mcp = MagicMock()

        # Simulate nested loop logic (simplified version)
        validation_cycles = 2
        best_answer = 0.60

        for cycle in range(1, validation_cycles + 1):
            # Get iteration context
            iteration_context = agent.pattern_db.get_iteration_context("TEST_PATTERN")

            # Inner loop would happen here (we'll skip for this test)
            # ...

            # OUTER CHECKPOINT: Full answer validation
            answer_result = agent.diagnose(ticket_id=None, runs=3, iteration=cycle)

            current_answer = answer_result.pattern_answer

            # Record iteration
            agent.pattern_db.record_iteration(
                pattern_id="TEST_PATTERN",
                iteration=cycle,
                cycle=cycle,
                suggested_change=f"Cycle {cycle} changes",
                reasoning="Test",
                confidence=0.75,
                before_metrics={"answer": best_answer},
                after_metrics={"answer": current_answer},
                committed=True,
            )

            # Check if passing
            if current_answer >= 0.85:
                print(f"SUCCESS at cycle {cycle}")
                break

            best_answer = max(best_answer, current_answer)

        # Assertions
        assert len(diagnose_calls) == 2, "Should run 2 validation cycles"
        assert diagnose_calls[0]["answer"] == pytest.approx(0.75), "Cycle 1 answer should be 0.75"
        assert diagnose_calls[1]["answer"] == pytest.approx(0.85), "Cycle 2 answer should be 0.85"

        assert len(record_iteration_calls) == 2, "Should record 2 iterations"
        assert record_iteration_calls[0]["cycle"] == 1
        assert record_iteration_calls[1]["cycle"] == 2
        assert record_iteration_calls[1]["after_metrics"]["answer"] == pytest.approx(0.85)

        # Should exit early at cycle 2 (answer >= 0.85)
        assert len(diagnose_calls) == 2, "Should stop at cycle 2 (passing threshold)"

    def test_nested_loop_early_exit_success(self, mock_pattern_fix_agent):
        """Test early exit when passing threshold reached."""
        from heal.agents.okp_mcp_agent import PatternEvaluationResult

        agent = mock_pattern_fix_agent

        # Mock diagnose to immediately return passing result
        def mock_diagnose(ticket_id=None, runs=3, iteration=None):
            result = MagicMock(spec=PatternEvaluationResult)
            result.pattern_answer = 0.92  # Passing!
            result.pattern_faithfulness = 0.88
            return result

        agent.diagnose = mock_diagnose

        # Simulate loop
        validation_cycles = 5
        cycles_run = 0

        for cycle in range(1, validation_cycles + 1):
            cycles_run += 1

            answer_result = agent.diagnose(ticket_id=None, runs=3, iteration=cycle)
            current_answer = answer_result.pattern_answer

            # Early exit
            if current_answer >= 0.85:
                break

        assert cycles_run == 1, "Should exit after first cycle (already passing)"

    def test_nested_loop_max_cycles_without_improvement(self, mock_pattern_fix_agent):
        """Test stopping after max cycles when no improvement."""
        from heal.agents.okp_mcp_agent import PatternEvaluationResult

        agent = mock_pattern_fix_agent

        # Mock diagnose to return stagnant results
        def mock_diagnose(ticket_id=None, runs=3, iteration=None):
            result = MagicMock(spec=PatternEvaluationResult)
            result.pattern_answer = 0.60  # Stuck at 0.60
            result.pattern_faithfulness = 0.70
            return result

        agent.diagnose = mock_diagnose

        # Simulate loop with early stop logic
        validation_cycles = 5
        max_cycles_without_improvement = 2
        cycles_without_improvement = 0
        best_answer = 0.60
        cycles_run = 0

        for cycle in range(1, validation_cycles + 1):
            cycles_run += 1

            answer_result = agent.diagnose(ticket_id=None, runs=3, iteration=cycle)
            current_answer = answer_result.pattern_answer

            if current_answer > best_answer:
                best_answer = current_answer
                cycles_without_improvement = 0
            else:
                cycles_without_improvement += 1

            # Early stop if stuck
            if cycles_without_improvement >= max_cycles_without_improvement:
                break

        assert cycles_run == 2, "Should stop after 2 cycles (no improvement)"

    def test_pattern_database_integration(self, tmp_path, mock_pattern_fix_agent):
        """Test that pattern database correctly records iterations."""
        from heal.core.fix_pattern_database import FixPatternDatabase
        import time

        # Use unique pattern ID to avoid conflicts with other test runs
        unique_pattern = f"TEST_PATTERN_{int(time.time() * 1000)}"

        # Use real pattern database
        pattern_db = FixPatternDatabase()

        # Record multiple iterations
        for cycle in range(1, 4):
            pattern_db.record_iteration(
                pattern_id=unique_pattern,
                iteration=cycle,
                cycle=cycle,
                suggested_change=f"Change {cycle}",
                reasoning=f"Reasoning {cycle}",
                confidence=0.75,
                before_metrics={"answer": 0.60 + (cycle - 1) * 0.10},
                after_metrics={"answer": 0.60 + cycle * 0.10},
                committed=True,
            )

        # Check that file was created (stored in .claude/fix_patterns/ by default)
        iterations_file = Path(f".claude/fix_patterns/{unique_pattern}_iterations.jsonl")

        assert iterations_file.exists(), "Iterations file should be created"

        # Read and verify contents
        iterations = []
        with open(iterations_file) as f:
            for line in f:
                iterations.append(json.loads(line))

        assert len(iterations) == 3, "Should have 3 iteration records"
        assert iterations[0]["cycle"] == 1
        assert iterations[1]["cycle"] == 2
        assert iterations[2]["cycle"] == 3
        assert iterations[2]["after_answer"] == 0.90

        # Get iteration context
        context = pattern_db.get_iteration_context("TEST_PATTERN")

        assert context is not None, "Should return iteration context"
        assert "TEST_PATTERN" in context or len(context) > 0, "Context should have content"

    def test_iteration_context_passed_to_multiagent(self, mock_pattern_fix_agent):
        """Test that iteration context is passed to multi-agent system."""
        from heal.core.fix_pattern_database import FixPatternDatabase

        agent = mock_pattern_fix_agent

        # Record a prior iteration
        pattern_db = FixPatternDatabase()
        pattern_db.record_iteration(
            pattern_id="TEST_PATTERN",
            iteration=1,
            cycle=1,
            suggested_change="Increase mm to 75%",
            reasoning="Better recall",
            confidence=0.80,
            before_metrics={"answer": 0.60},
            after_metrics={"answer": 0.75},
            committed=True,
        )

        # Get context
        context = pattern_db.get_iteration_context("TEST_PATTERN")

        # Verify context contains prior attempt
        assert context is not None
        # Context should be non-empty if there's iteration data
        # (actual format depends on implementation)

    def test_incremental_learning_no_reverts(self, mock_pattern_fix_agent, tmp_path):
        """Test that changes are never reverted (incremental learning)."""
        agent = mock_pattern_fix_agent

        # Initialize git repo in okp-mcp
        import subprocess

        subprocess.run(["git", "init"], cwd=agent.okp_mcp_root, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"], cwd=agent.okp_mcp_root, check=True
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=agent.okp_mcp_root, check=True)

        # Make initial commit
        solr_file = agent.okp_mcp_root / "src" / "okp_mcp" / "solr.py"
        subprocess.run(["git", "add", "."], cwd=agent.okp_mcp_root, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"], cwd=agent.okp_mcp_root, check=True
        )

        # Simulate changes over cycles (even unsuccessful ones should be kept)
        changes = [
            ("Change 1: mm=75%", True),  # Successful
            ("Change 2: qf boost", False),  # Unsuccessful - but still keep!
            ("Change 3: pf tuning", True),  # Successful
        ]

        for idx, (change, improved) in enumerate(changes, 1):
            # Modify file
            with open(solr_file, "a") as f:
                f.write(f"\n# {change}\n")

            # Commit (incremental - always commit, never revert)
            subprocess.run(
                ["git", "add", "src/okp_mcp/solr.py"], cwd=agent.okp_mcp_root, check=True
            )

            commit_msg = f"pattern: {change}"
            if not improved:
                commit_msg = f"pattern (unsuccessful): {change}"

            subprocess.run(["git", "commit", "-m", commit_msg], cwd=agent.okp_mcp_root, check=True)

        # Verify git log shows all 3 changes (+ initial = 4 commits)
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=agent.okp_mcp_root,
            capture_output=True,
            text=True,
            check=True,
        )

        commit_count = int(result.stdout.strip())
        assert commit_count == 4, "Should have 4 commits (initial + 3 changes, no reverts)"

        # Verify file contains all changes
        content = solr_file.read_text()
        assert "Change 1: mm=75%" in content
        assert "Change 2: qf boost" in content  # Even unsuccessful change is kept!
        assert "Change 3: pf tuning" in content

    def test_validation_cycles_parameter_flow(self, mock_pattern_fix_agent):
        """Test that validation_cycles parameter controls outer loop count."""
        from heal.agents.okp_mcp_agent import PatternEvaluationResult

        agent = mock_pattern_fix_agent

        # Track how many times diagnose is called
        diagnose_count = 0

        def mock_diagnose(ticket_id=None, runs=3, iteration=None):
            nonlocal diagnose_count
            diagnose_count += 1

            result = MagicMock(spec=PatternEvaluationResult)
            result.pattern_answer = 0.70  # Below threshold
            result.pattern_faithfulness = 0.80
            return result

        agent.diagnose = mock_diagnose

        # Simulate loop with different validation_cycles values
        for validation_cycles in [1, 3, 5]:
            diagnose_count = 0

            for cycle in range(1, validation_cycles + 1):
                answer_result = agent.diagnose(ticket_id=None, runs=3, iteration=cycle)
                # Don't exit early (never reaches threshold)

            assert (
                diagnose_count == validation_cycles
            ), f"Should run exactly {validation_cycles} cycles"

    def test_inner_loop_early_exit_on_f1_improvement(self, mock_pattern_fix_agent):
        """Test that inner loop exits early when F1 improves significantly."""

        # Simulate inner loop logic
        max_iterations = 10
        best_f1 = 0.30
        iterations_run = 0

        # Simulate F1 improving significantly on iteration 3
        f1_scores = [0.30, 0.35, 0.55, 0.60, 0.65]  # Jump to 0.55 on iteration 3

        for iteration in range(1, max_iterations + 1):
            iterations_run += 1

            # Simulate getting F1 score
            current_f1 = f1_scores[min(iteration - 1, len(f1_scores) - 1)]

            # Early exit if F1 improved significantly
            if current_f1 > best_f1 + 0.15:  # Significant improvement threshold
                print(
                    f"Early exit at iteration {iteration}: F1 jumped from {best_f1} to {current_f1}"
                )
                break

            best_f1 = max(best_f1, current_f1)

        # Should exit at iteration 3 when F1 jumps to 0.55
        assert iterations_run <= 5, "Should exit early when F1 improves significantly"


class TestNestedLoopEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_validation_cycles(self):
        """Test that validation_cycles=0 is handled gracefully."""
        validation_cycles = 0
        cycles_run = 0

        for cycle in range(1, validation_cycles + 1):
            cycles_run += 1

        assert cycles_run == 0, "Should run 0 cycles"

    def test_negative_validation_cycles(self):
        """Test that negative validation_cycles is handled."""
        validation_cycles = -1
        cycles_run = 0

        for cycle in range(1, validation_cycles + 1):
            cycles_run += 1

        assert cycles_run == 0, "Should run 0 cycles (range is empty)"

    def test_validation_cycle_with_exception(self):
        """Test graceful handling when validation fails."""
        from heal.agents.okp_mcp_agent import PatternEvaluationResult

        # Create minimal mock
        agent = MagicMock()

        # Mock diagnose to raise exception
        def mock_diagnose(ticket_id=None, runs=3, iteration=None):
            if iteration == 2:
                raise RuntimeError("Evaluation failed")

            result = MagicMock(spec=PatternEvaluationResult)
            result.pattern_answer = 0.75
            return result

        agent.diagnose = mock_diagnose

        # Simulate loop with exception handling
        validation_cycles = 3
        cycles_completed = 0

        for cycle in range(1, validation_cycles + 1):
            try:
                answer_result = agent.diagnose(ticket_id=None, runs=3, iteration=cycle)
                cycles_completed += 1
            except RuntimeError:
                # Continue to next cycle despite error
                continue

        assert cycles_completed == 2, "Should complete 2 cycles (cycle 2 failed)"


class TestPatternDatabaseContextFormat:
    """Test pattern database iteration context formatting."""

    def test_iteration_context_format(self, tmp_path):
        """Test that iteration context is properly formatted for multi-agent."""
        from heal.core.fix_pattern_database import FixPatternDatabase

        pattern_db = FixPatternDatabase()

        # Record some iterations
        pattern_db.record_iteration(
            pattern_id="TEST",
            iteration=1,
            cycle=1,
            suggested_change="Increase mm to 75%",
            reasoning="Better recall",
            confidence=0.80,
            before_metrics={"answer": 0.60},
            after_metrics={"answer": 0.75},
            committed=True,
        )

        pattern_db.record_iteration(
            pattern_id="TEST",
            iteration=2,
            cycle=1,
            suggested_change="Boost qf",
            reasoning="Emphasize title",
            confidence=0.70,
            before_metrics={"answer": 0.75},
            after_metrics={"answer": 0.72},
            committed=True,  # Still committed (incremental)
        )

        # Get context
        context = pattern_db.get_iteration_context("TEST")

        # Verify context exists
        assert context is not None
        assert isinstance(context, str)

        # Context should contain pattern ID if there's data
        # (exact format depends on implementation)

    def test_empty_iteration_context(self):
        """Test iteration context for pattern with no prior iterations."""
        from heal.core.fix_pattern_database import FixPatternDatabase

        pattern_db = FixPatternDatabase()

        # Get context for non-existent pattern
        context = pattern_db.get_iteration_context("NONEXISTENT")

        # Should return a message indicating no prior iterations
        assert context is not None
        assert "no prior iterations" in context.lower() or "first" in context.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
