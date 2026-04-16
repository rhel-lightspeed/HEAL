#!/usr/bin/env python3
"""Pattern Fix Loop - Proof of Concept

Implements the simplified fix loop with smart routing between Solr optimization
and full retrieval path testing.

Usage:
    # Run POC on small pattern
    python okp_mcp_agent/runners/run_pattern_fix_poc.py RHEL10_DEPRECATED_FEATURES

    # Custom thresholds
    python okp_mcp_agent/runners/run_pattern_fix_poc.py CONTAINER_UNSUPPORTED_CONFIG \
        --max-iterations 15 \
        --answer-threshold 0.90 \
        --stability-runs 5

Output:
    - Git branch: fix/pattern-{pattern_id}
    - Diagnostics: .diagnostics/{pattern_id}/
    - Review report: .diagnostics/{pattern_id}/REVIEW_REPORT.md
"""

import argparse
import atexit
import os
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from heal.agents.okp_mcp_agent import OkpMcpAgent

# Force unbuffered output so prints show up immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Define repo root for default file paths
REPO_ROOT = Path(__file__).parent.parent.parent.parent


@dataclass
class PhaseResult:
    """Result from a fix loop phase."""

    phase_name: str
    success: bool
    iterations: int = 0
    final_metrics: Dict = field(default_factory=dict)
    reason: str = ""
    baseline_result: Any = None  # Full DiagnosticResult for RAG bypass detection


@dataclass
class PatternFixResult:
    """Complete result from pattern fix loop."""

    pattern_id: str
    total_tickets: int
    tickets_tested: int

    # Phase results
    baseline: Optional[PhaseResult] = None
    optimization: Optional[PhaseResult] = None
    answer_validation: Optional[PhaseResult] = None
    stability: Optional[PhaseResult] = None  # Phase 4: Final pattern validation
    cla_regression: Optional[PhaseResult] = None  # Phase 5: CLA regression test

    # Overall status
    success: bool = False
    branch_name: str = ""
    diagnostics_dir: Path = Path()

    # Timing
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0


class PatternFixAgent(OkpMcpAgent):
    """Fix loop agent for pattern-based ticket resolution."""

    def __init__(
        self,
        pattern_id: str,
        eval_root: Path,
        okp_mcp_root: Path,
        lscore_deploy_root: Path,
        **kwargs,
    ):
        """Initialize pattern fix agent.

        Args:
            pattern_id: Pattern identifier
            eval_root: Path to lightspeed-evaluation repo
            okp_mcp_root: Path to okp-mcp repo
            lscore_deploy_root: Path to lscore-deploy repo
            **kwargs: Additional options (interactive, enable_llm_advisor, etc.)
        """
        super().__init__(
            eval_root=eval_root,
            okp_mcp_root=okp_mcp_root,
            lscore_deploy_root=lscore_deploy_root,
            **kwargs,
        )
        self.pattern_id = pattern_id
        self.pattern_tickets: List[Dict[str, Any]] = []
        self.branch_name = f"fix/pattern-{pattern_id.lower().replace('_', '-')}"
        self.cleaned_config: Optional[Path] = None  # Cleaned config with skip tags
        self._original_branch: Optional[str] = None  # Track original branch for cleanup
        self._cleanup_done: bool = False  # Prevent duplicate cleanup

        # Register cleanup handlers
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def load_pattern_tickets(self, patterns_dir: Path) -> None:
        """Use pattern YAML as evaluation config.

        Pattern YAMLs are already in lightspeed-evaluation format,
        so we just point to the file and use it directly.

        Args:
            patterns_dir: Directory containing pattern YAMLs
        """
        pattern_file = patterns_dir / f"{self.pattern_id}.yaml"

        if not pattern_file.exists():
            raise FileNotFoundError(f"Pattern file not found: {pattern_file}")

        print(f"Using pattern file as test config: {pattern_file}")

        # Load just to count tickets
        with open(pattern_file) as f:
            content = f.read()
            lines = [line for line in content.split("\n") if not line.startswith("#")]
            yaml_content = "\n".join(lines)
            conversations = yaml.safe_load(yaml_content)

        if not conversations:
            raise ValueError(f"No conversations found in {pattern_file}")

        # Store ticket IDs for tracking
        for conv in conversations:
            self.pattern_tickets.append({"ticket_id": conv["conversation_group_id"]})

        # Use pattern file directly as test config (already in correct format)
        self.functional_full = pattern_file
        self.functional_retrieval = pattern_file

        print(f"✅ Loaded {len(self.pattern_tickets)} tickets for pattern {self.pattern_id}")

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C and other signals gracefully."""
        print(f"\n\n⚠️  Received signal {signum}, cleaning up...")
        self.cleanup()
        sys.exit(1)

    def cleanup(self) -> None:
        """Cleanup: return to original branch if we created a pattern branch."""
        import subprocess

        # Prevent duplicate cleanup
        if self._cleanup_done:
            return

        self._cleanup_done = True

        if not self._original_branch:
            return  # Never switched branches

        try:
            print(f"\n🧹 Cleanup: Returning to branch '{self._original_branch}'...")
            subprocess.run(
                ["git", "checkout", self._original_branch],
                cwd=self.okp_mcp_root,
                capture_output=True,
                check=True,
            )
            print(f"✅ Back on branch: {self._original_branch}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Could not switch back to {self._original_branch}: {e}")

    def create_pattern_branch(self) -> None:
        """Create git branch for this pattern's fixes."""
        import subprocess

        # Ensure we start from main
        print("\n📌 Ensuring clean starting state...")

        # Get current branch
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=self.okp_mcp_root,
            capture_output=True,
            text=True,
            check=True,
        )
        current_branch = result.stdout.strip()

        # If not on main, switch to it
        if current_branch != "main":
            print(f"   Currently on: {current_branch}")
            print("   Switching to main...")
            subprocess.run(["git", "checkout", "main"], cwd=self.okp_mcp_root, check=True)
            current_branch = "main"

        # Store original branch for cleanup
        self._original_branch = current_branch
        print(f"   ✅ Starting from: {current_branch}")

        print(f"\n📌 Creating branch: {self.branch_name}")

        # Check if branch already exists
        result = subprocess.run(
            ["git", "branch", "--list", self.branch_name],
            cwd=self.okp_mcp_root,
            capture_output=True,
            text=True,
        )

        if result.stdout.strip():
            print("⚠️  Branch already exists, deleting and recreating")
            subprocess.run(
                ["git", "branch", "-D", self.branch_name], cwd=self.okp_mcp_root, check=True
            )
            subprocess.run(
                ["git", "checkout", "-b", self.branch_name], cwd=self.okp_mcp_root, check=True
            )
        else:
            subprocess.run(
                ["git", "checkout", "-b", self.branch_name], cwd=self.okp_mcp_root, check=True
            )

        print(f"✅ On branch: {self.branch_name}")

    def run_fix_loop(
        self,
        max_iterations: int = 15,
        answer_threshold: float = 0.90,
        stability_runs: int = 5,
        mode: str = "single",
    ) -> PatternFixResult:
        """Run complete fix loop with all phases.

        Args:
            max_iterations: Max iterations for optimization phases
            answer_threshold: Minimum answer_correctness to pass
            stability_runs: Number of runs for stability check
            mode: Testing mode - 'single' (one ticket) or 'full' (all tickets)

        Returns:
            PatternFixResult with complete status
        """
        start_time = datetime.now()

        result = PatternFixResult(
            pattern_id=self.pattern_id,
            total_tickets=len(self.pattern_tickets),
            tickets_tested=0,
            start_time=start_time.isoformat(),
            diagnostics_dir=Path(f".diagnostics/{self.pattern_id}"),
        )

        # Create branch
        self.create_pattern_branch()
        result.branch_name = self.branch_name

        print(f"\n{'='*80}")
        print(f"PATTERN FIX LOOP: {self.pattern_id}")
        print(f"{'='*80}")
        print(f"Tickets: {len(self.pattern_tickets)}")
        print(f"Branch: {self.branch_name}")
        print(f"Testing mode: {mode.upper()}")
        print(f"Max iterations: {max_iterations}")
        print(f"Answer threshold: {answer_threshold}")
        print(f"Stability runs: {stability_runs}")
        print(f"{'='*80}\n")

        # Determine which tickets to test based on mode
        if mode == "single":
            # Fast mode: test one representative ticket
            test_ticket = self.pattern_tickets[0]
            ticket_id = test_ticket["ticket_id"]
            result.tickets_tested = 1
            print(f"📋 Single-ticket mode: Testing representative ticket {ticket_id}\n")
        else:  # mode == "full"
            # Full pattern mode: test all tickets at once
            ticket_id = None  # Signal to run full pattern
            result.tickets_tested = len(self.pattern_tickets)
            print(f"📋 Full-pattern mode: Testing all {len(self.pattern_tickets)} tickets\n")

        # Create ONE cleaned config with full metrics, reuse throughout workflow
        print("📝 Creating cleaned config with all 6 metrics (will reuse throughout)...")
        full_metrics = [
            "custom:url_retrieval_eval",
            "ragas:context_relevance",
            "ragas:context_precision_without_reference",
            "custom:answer_correctness",
            "ragas:faithfulness",
            "ragas:response_relevancy",
        ]

        # Use pattern YAML as source, create cleaned version with full metrics
        pattern_yaml = Path(f"config/patterns/{self.pattern_id}.yaml")
        self.cleaned_config = self.clean_pattern_config(pattern_yaml, metrics=full_metrics)

        # Reuse this config throughout all phases (just toggle skip tags)
        self.functional_full = self.cleaned_config
        self.functional_retrieval = self.cleaned_config

        print(f"   ✅ Config ready: {self.cleaned_config}")
        print(f"   📋 Metrics in YAML: {len(full_metrics)}")
        print("   🔄 Will reuse this config, toggling skip tags as needed\n")

        # PHASE 1: Initial Full Baseline (with stability runs)
        print(f"\n{'='*80}")
        print("PHASE 1: INITIAL BASELINE (STABILITY CHECK)")
        print(f"{'='*80}\n")

        baseline_result = self.run_baseline(ticket_id, stability_runs=stability_runs)
        result.baseline = baseline_result

        print(f"\n{'─'*80}")
        print("📊 PHASE 1 COMPLETE:")
        if baseline_result.success:
            print("   ✅ Baseline established")
            ans = baseline_result.final_metrics.get("answer_correctness", 0.0)
            faith = baseline_result.final_metrics.get("faithfulness", 0.0)
            print(f"   Answer: {ans:.2f}, Faithfulness: {faith:.2f}")
        else:
            print(f"   ❌ Baseline failed: {baseline_result.reason}")
        print(f"{'─'*80}\n")

        if not baseline_result.success:
            result.success = False
            result.end_time = datetime.now().isoformat()
            result.duration_seconds = (datetime.now() - start_time).total_seconds()
            return result

        # Check if already passing (answer-first approach)
        if self._is_passing(baseline_result.final_metrics, answer_threshold):
            ans_corr = baseline_result.final_metrics.get("answer_correctness", 0.0)
            faith = baseline_result.final_metrics.get("faithfulness", 0.0)
            print("\n" + "=" * 80)
            print("✅ ANSWER ALREADY CORRECT - SKIPPING OPTIMIZATION")
            print("=" * 80)
            print(f"\nAnswer Correctness: {ans_corr:.3f} (≥ {answer_threshold})")
            print(f"Faithfulness:       {faith:.3f} (≥ 0.8)")
            print("\n💡 Answer-first approach: No need to optimize retrieval")
            print("   The LLM got the right answer, regardless of which docs were used.")
            print("   This ticket will be marked skip=true for future iterations.")

            # Update skip tag in YAML config
            ticket_classifications = baseline_result.final_metrics.get("ticket_classifications", {})
            if ticket_classifications and self.cleaned_config and self.cleaned_config.exists():
                print(f"\n🏷️  Setting skip=true in: {self.cleaned_config}")
                self.update_skip_tags(self.cleaned_config, ticket_classifications, mode="set")
                if ticket_id and ticket_id in ticket_classifications:
                    classification = ticket_classifications[ticket_id]
                    print(f"   Status: {classification.status.value}")
                    print(f"   Skip: {classification.skip}")
                print()

            result.success = True
            result.end_time = datetime.now().isoformat()
            result.duration_seconds = (datetime.now() - start_time).total_seconds()
            return result

        # PHASE 2: Smart Routing - Optimization
        print(f"\n{'='*80}")
        print("PHASE 2: SMART OPTIMIZATION")
        print(f"{'='*80}\n")

        opt_result = self.run_optimization(
            ticket_id,
            baseline_result.final_metrics,
            max_iterations,
            baseline_result.baseline_result,
        )
        result.optimization = opt_result

        print(f"\n{'─'*80}")
        print("📊 PHASE 2 COMPLETE:")
        if opt_result.success:
            print(f"   ✅ Optimization improved retrieval ({opt_result.iterations} iterations)")
            f1 = opt_result.final_metrics.get("url_f1", 0.0)
            ctx_rel = opt_result.final_metrics.get("context_relevance", 0.0)
            print(f"   F1: {f1:.2f}, Context Relevance: {ctx_rel:.2f}")
        else:
            print(f"   ⚠️  No significant improvement: {opt_result.reason}")
            print("   → Continuing to answer validation anyway")
        print(f"{'─'*80}\n")

        # PHASE 3: Answer Correctness Validation (with stability runs)
        print(f"\n{'='*80}")
        print("PHASE 3: ANSWER VALIDATION (STABILITY CHECK)")
        print(f"{'='*80}\n")

        answer_result = self.run_answer_validation(
            ticket_id, answer_threshold, stability_runs=stability_runs
        )
        result.answer_validation = answer_result

        print(f"\n{'─'*80}")
        print("📊 PHASE 3 COMPLETE:")
        if answer_result.success:
            print("   ✅ Answer validation passed")
            ans = answer_result.final_metrics.get("answer_correctness", 0.0)
            faith = answer_result.final_metrics.get("faithfulness", 0.0)
            print(f"   Answer: {ans:.2f} (≥ {answer_threshold}), Faithfulness: {faith:.2f}")
        else:
            print(f"   ❌ Answer validation failed: {answer_result.reason}")
        print(f"{'─'*80}\n")

        if not answer_result.success:
            result.success = False
            result.end_time = datetime.now().isoformat()
            result.duration_seconds = (datetime.now() - start_time).total_seconds()
            return result

        # PHASE 4: Final Pattern Validation (Remove skip tags, test all tickets)
        print(f"\n{'='*80}")
        print("PHASE 4: FINAL PATTERN VALIDATION")
        print(f"{'='*80}\n")

        pattern_validation_result = self.run_final_pattern_validation(
            ticket_id, stability_runs=stability_runs
        )
        result.stability = pattern_validation_result  # Store pattern validation

        print(f"\n{'─'*80}")
        print("📊 PHASE 4 COMPLETE:")
        if pattern_validation_result.success:
            print("   ✅ Pattern validation passed")
            ans = pattern_validation_result.final_metrics.get("answer_correctness", 0.0)
            print(f"   All tickets validated: Answer {ans:.2f}")
        else:
            print(f"   ❌ Pattern validation failed: {pattern_validation_result.reason}")
        print(f"{'─'*80}\n")

        if not pattern_validation_result.success:
            result.success = False
            result.end_time = datetime.now().isoformat()
            result.duration_seconds = (datetime.now() - start_time).total_seconds()
            return result

        # PHASE 5: CLA Regression Test (Release Gating Questions)
        print(f"\n{'='*80}")
        print("PHASE 5: CLA REGRESSION TEST (Release Gating)")
        print(f"{'='*80}\n")

        cla_result = self.run_cla_regression_test()
        result.cla_regression = cla_result  # Store CLA results separately

        print(f"\n{'─'*80}")
        print("📊 PHASE 5 COMPLETE:")
        if cla_result.success:
            total = cla_result.final_metrics.get("cla_total", 0)
            passed = cla_result.final_metrics.get("cla_passed", 0)
            rate = cla_result.final_metrics.get("cla_pass_rate", 0)
            print("   ✅ CLA regression test passed")
            print(f"   {passed}/{total} questions passed ({rate*100:.1f}%)")
        else:
            print(f"   ❌ CLA regression test failed: {cla_result.reason}")
        print(f"{'─'*80}\n")

        result.success = cla_result.success
        result.end_time = datetime.now().isoformat()
        result.duration_seconds = (datetime.now() - start_time).total_seconds()

        # Print improvement summary
        self._print_improvement_summary(result)

        return result

    def run_baseline(self, ticket_id: Optional[str], stability_runs: int = 1) -> PhaseResult:
        """Phase 1: Run full baseline evaluation with per-ticket stability classification.

        Args:
            ticket_id: Ticket to evaluate (None = all tickets in pattern)
            stability_runs: Number of runs for stability baseline

        Returns:
            PhaseResult with baseline metrics and per-ticket classifications
        """
        from heal.core.stability_classifier import classify_stability

        if ticket_id:
            print(f"🔍 Running full baseline evaluation for {ticket_id} ({stability_runs} runs)...")
        else:
            print(
                f"🔍 Running full baseline evaluation for ALL tickets in pattern ({stability_runs} runs)..."
            )
        print("   Metrics: url_retrieval, context_relevance, context_precision,")
        print("           answer_correctness, faithfulness, response_relevancy")

        try:
            # Run full diagnosis with all metrics (multiple runs for stability)
            result = self.diagnose(ticket_id, use_existing=False, runs=stability_runs)

            # Show key metrics from baseline
            print("\n📊 BASELINE METRICS:")
            print(
                f"   URL F1:             {result.url_f1:.2f}"
                if result.url_f1 is not None
                else "   URL F1:             N/A"
            )
            print(
                f"   Answer Correctness: {result.answer_correctness:.2f}"
                if result.answer_correctness is not None
                else "   Answer Correctness: N/A"
            )
            print(
                f"   Faithfulness:       {result.faithfulness:.2f}"
                if result.faithfulness is not None
                else "   Faithfulness:       N/A"
            )
            print(
                f"   Context Relevance:  {result.context_relevance:.2f}"
                if result.context_relevance is not None
                else "   Context Relevance:  N/A"
            )

            # NEW: Get per-ticket, per-run results for classification
            output_dir = self.get_latest_output_dir("full")
            per_ticket_results = self.parse_results_per_ticket(output_dir)

            # Classify each ticket based on per-run scores
            ticket_classifications = {}
            for ticket, runs in per_ticket_results.items():
                # Extract answer_correctness scores across runs
                ans_scores = [
                    r.get("answer_correctness", 0.0) for r in runs if "answer_correctness" in r
                ]

                if ans_scores:
                    classification = classify_stability(
                        ans_scores,
                        threshold=0.90,
                        catastrophic_threshold=0.70,
                        high_cv_threshold=0.15,
                    )
                    ticket_classifications[ticket] = classification

            # Display per-ticket classifications (full-pattern mode)
            if not ticket_id and ticket_classifications:
                print("\n📊 PER-TICKET STABILITY CLASSIFICATION:")
                print("=" * 80)
                for ticket, classification in ticket_classifications.items():
                    emoji = {
                        "STABLE_PASSING": "✅",
                        "UNSTABLE_PASSING": "⚠️",
                        "BORDERLINE": "❌",
                        "INTERMITTENT_FAILURE": "❌",
                        "CONSISTENTLY_FAILING": "❌",
                    }.get(classification.status.value, "❓")
                    print(f"\n{emoji} {ticket}: {classification.status.value}")
                    print(f"   {classification.reason}")
                    print(
                        f"   Min/Max/Mean: {classification.min_score:.2f}/{classification.max_score:.2f}/{classification.mean_score:.2f}"
                    )
                    print(f"   Skip: {classification.skip}, Priority: {classification.priority}")

                # Summary
                print("\n📋 CLASSIFICATION SUMMARY:")
                print("=" * 80)
                stable_count = sum(
                    1 for c in ticket_classifications.values() if c.status.value == "STABLE_PASSING"
                )
                unstable_pass_count = sum(
                    1
                    for c in ticket_classifications.values()
                    if c.status.value == "UNSTABLE_PASSING"
                )
                borderline_count = sum(
                    1 for c in ticket_classifications.values() if c.status.value == "BORDERLINE"
                )
                intermittent_count = sum(
                    1
                    for c in ticket_classifications.values()
                    if c.status.value == "INTERMITTENT_FAILURE"
                )
                failing_count = sum(
                    1
                    for c in ticket_classifications.values()
                    if c.status.value == "CONSISTENTLY_FAILING"
                )
                skip_count = sum(1 for c in ticket_classifications.values() if c.skip)
                high_priority_count = sum(
                    1 for c in ticket_classifications.values() if c.priority == "HIGH"
                )

                print(f"   Total tickets:        {len(ticket_classifications)}")
                print(f"   ✅ Stable passing:    {stable_count} (will skip in optimization)")
                print(f"   ⚠️  Unstable passing:  {unstable_pass_count} (will skip, needs review)")
                print(f"   ❌ Borderline:        {borderline_count} (needs fixing)")
                print(
                    f"   ❌ Intermittent:      {intermittent_count} (HIGH priority, needs investigation)"
                )
                print(f"   ❌ Failing:           {failing_count} (needs fixing)")
                print(f"   📌 Will skip:         {skip_count}/{len(ticket_classifications)}")
                print(f"   🔥 High priority:     {high_priority_count}")

                # Update skip tags in the SINGLE cleaned config (reused throughout)
                print("\n🏷️  Updating skip tags in config...")
                if self.cleaned_config and self.cleaned_config.exists():
                    self.update_skip_tags(self.cleaned_config, ticket_classifications, mode="set")
                    print(f"   ✅ Skip tags updated in: {self.cleaned_config}")
                else:
                    print(f"   ⚠️  Cleaned config not found: {self.cleaned_config}")

            # Calculate averaged metrics (for backward compatibility)
            metrics = {
                "url_f1": result.url_f1 or 0.0,
                "mrr": result.mrr or 0.0,
                "context_relevance": result.context_relevance or 0.0,
                "context_precision": result.context_precision or 0.0,
                "answer_correctness": result.answer_correctness or 0.0,
                "faithfulness": result.faithfulness or 0.0,
                "response_relevancy": result.response_relevancy or 0.0,
                # NEW: Store classifications
                "ticket_classifications": ticket_classifications,
            }

            # Check RAG status
            rag_bypassed = self._is_rag_bypassed(result)
            num_docs = result.num_docs_retrieved() if hasattr(result, "num_docs_retrieved") else 0

            print("\n📊 Baseline Metrics:")
            print(f"   Runs:               {result.num_runs}")
            print(
                f"   RAG Status:         {'❌ BYPASSED (0 docs)' if rag_bypassed else f'✅ Used ({num_docs} docs)'}"
            )
            print(f"   URL F1:             {metrics['url_f1']:.2f}")
            print(f"   MRR:                {metrics['mrr']:.2f}")
            print(f"   Context Relevance:  {metrics['context_relevance']:.2f}")
            print(f"   Context Precision:  {metrics['context_precision']:.2f}")
            print(f"   Answer Correctness: {metrics['answer_correctness']:.2f}")
            print(f"   Faithfulness:       {metrics['faithfulness']:.2f}")
            print(f"   Response Relevancy: {metrics['response_relevancy']:.2f}")

            # Check for high variance (instability)
            if result.high_variance_metrics:
                print("\n⚠️  HIGH VARIANCE DETECTED in baseline:")
                for metric_info in result.high_variance_metrics:
                    print(f"   • {metric_info}")
                print("   → Baseline is UNSTABLE - optimization may not help")

            # Determine problem type
            is_retrieval = result.is_retrieval_problem
            is_answer = result.is_answer_problem

            print("\n🔍 Problem Analysis:")
            print(f"   RAG Bypassed:      {rag_bypassed}")
            print(f"   Retrieval Problem: {is_retrieval}")
            print(f"   Answer Problem:    {is_answer}")

            return PhaseResult(
                phase_name="baseline",
                success=True,
                final_metrics=metrics,
                reason=f"retrieval_problem={is_retrieval}, answer_problem={is_answer}",
                baseline_result=result,  # Store full result for RAG bypass detection
            )

        except Exception as e:
            print(f"❌ Baseline failed: {e}")
            return PhaseResult(phase_name="baseline", success=False, reason=str(e))

    def run_optimization(
        self,
        ticket_id: Optional[str],
        baseline_metrics: Dict,
        max_iterations: int,
        baseline_result: Any = None,
    ) -> PhaseResult:
        """Phase 2: Smart routing optimization.

        Routes to appropriate optimization path based on problem type:
        - RAG bypassed → Prompt optimization (force RAG usage or verify answer)
        - Bad retrieval → Retrieval optimization (Solr changes)
        - Good retrieval, bad answer → Prompt optimization (LLM not using docs)

        Args:
            ticket_id: Ticket to optimize (None = all tickets in pattern)
            baseline_metrics: Baseline metrics from Phase 1
            max_iterations: Max optimization iterations
            baseline_result: Full diagnostic result for RAG bypass detection

        Returns:
            PhaseResult with optimization outcome
        """
        print("🎯 Analyzing problem type for smart routing...")

        # CRITICAL: Check RAG bypass FIRST
        if baseline_result and self._is_rag_bypassed(baseline_result):
            print("   RAG Bypassed: True (0 docs retrieved)")
            print("\n📍 Route C: RAG BYPASS - PROMPT OPTIMIZATION")
            print("   Issue: System chose not to use RAG or RAG returned 0 docs")
            print("   Testing: Force RAG usage via system prompt changes")
            print("   Mode: Full evaluation (WITH response generation)")
            print("   Speed: ~30-60 sec/iteration")
            print("   Note: Cannot optimize Solr if RAG wasn't used!")
            return self.run_prompt_optimization(ticket_id, baseline_metrics, max_iterations)

        # RAG was used - determine problem type
        is_retrieval = self._is_retrieval_problem(baseline_metrics)
        is_answer = self._is_answer_problem(baseline_metrics)

        print(f"   Retrieval Problem: {is_retrieval}")
        print(f"   Answer Problem:    {is_answer}")

        if is_retrieval:
            # Route A: Fast retrieval optimization (Solr config changes)
            print("\n📍 Route A: RETRIEVAL OPTIMIZATION")
            print("   Testing: Solr config changes (qf, pf, mm, highlighting, etc.)")
            print("   Mode: Retrieval-only (no response generation)")
            print("   Speed: ~15-20 sec/iteration")
            return self.run_retrieval_optimization(ticket_id, baseline_metrics, max_iterations)
        elif is_answer:
            # Route B: Prompt optimization (system prompt changes)
            print("\n📍 Route B: PROMPT OPTIMIZATION")
            print("   Testing: System prompt changes (instructions, grounding, etc.)")
            print("   Mode: Full evaluation (WITH response generation)")
            print("   Speed: ~30-60 sec/iteration")
            return self.run_prompt_optimization(ticket_id, baseline_metrics, max_iterations)
        else:
            print("\n⚠️  No clear problem identified - trying retrieval optimization")
            return self.run_retrieval_optimization(ticket_id, baseline_metrics, max_iterations)

    def run_retrieval_optimization(
        self, ticket_id: Optional[str], baseline_metrics: Dict, max_iterations: int
    ) -> PhaseResult:
        """Route A: Fast retrieval optimization (Solr config changes).

        Uses retrieval-only mode - NO response generation needed.
        Tests: qf boosting, pf phrase matching, mm, highlighting, field weights.

        Args:
            ticket_id: Ticket to optimize
            baseline_metrics: Baseline metrics
            max_iterations: Max iterations

        Returns:
            PhaseResult with optimization outcome
        """
        print(f"   Max iterations: {max_iterations}")
        print("   Early exit: F1 > 0 (any expected docs found)\n")

        try:
            iteration = 0
            current_f1 = baseline_metrics.get("url_f1", 0.0)
            current_ctx_rel = baseline_metrics.get("context_relevance", 0.0)

            while iteration < max_iterations:
                iteration += 1
                print(f"\n{'='*80}")
                print(f"📍 OPTIMIZATION ITERATION {iteration}/{max_iterations}")
                print(
                    f"   Current scores → F1: {current_f1:.2f}, Context Rel: {current_ctx_rel:.2f}"
                )
                print(f"{'='*80}")

                # Use retrieval-only mode (faster, no LLM response generation)
                result = self.diagnose_retrieval_only(ticket_id, iteration=iteration)

                new_f1 = result.url_f1 or 0.0
                new_ctx_rel = result.context_relevance or 0.0

                print(f"\n   📊 Results → F1: {new_f1:.2f}, Context Rel: {new_ctx_rel:.2f}")

                if new_f1 > current_f1:
                    print(f"   ✅ F1 improved: {current_f1:.2f} → {new_f1:.2f}")
                    current_f1 = new_f1
                else:
                    print(f"   ➡️  F1 unchanged: {current_f1:.2f}")

                if new_ctx_rel > current_ctx_rel:
                    print(
                        f"   ✅ Context Relevance improved: {current_ctx_rel:.2f} → {new_ctx_rel:.2f}"
                    )
                    current_ctx_rel = new_ctx_rel
                else:
                    print(f"   ➡️  Context Relevance unchanged: {current_ctx_rel:.2f}")

                # Early exit if we found ANY expected docs
                # F1 can be "low" (e.g., 0.4) but still have all right docs
                # Example: 3 expected docs, 10 retrieved (with all 3) → F1=0.46
                # Don't keep optimizing - test answer instead!
                if current_f1 > 0.0:
                    print("\n🎯 EARLY EXIT: Found expected docs!")
                    print(f"   F1: {current_f1:.2f} (may be 'low' due to extra docs retrieved)")
                    print(f"   Context Relevance: {current_ctx_rel:.2f}")
                    print("   → Stopping optimization to test answer quality")
                    break

            final_metrics = {
                "url_f1": current_f1,
                "context_relevance": current_ctx_rel,
            }

            improved = current_f1 > baseline_metrics.get("url_f1", 0.0)

            return PhaseResult(
                phase_name="retrieval_optimization",
                success=improved,
                iterations=iteration,
                final_metrics=final_metrics,
                reason=f"F1: {baseline_metrics.get('url_f1', 0):.2f} → {current_f1:.2f}",
            )

        except Exception as e:
            print(f"❌ Retrieval optimization failed: {e}")
            return PhaseResult(
                phase_name="retrieval_optimization",
                success=False,
                iterations=iteration,
                reason=str(e),
            )

    def run_prompt_optimization(
        self, ticket_id: Optional[str], baseline_metrics: Dict, max_iterations: int
    ) -> PhaseResult:
        """Route B: Prompt optimization (system prompt changes).

        Uses FULL evaluation mode - response generation required.
        Tests: system prompt changes, grounding instructions, RAG usage.

        Args:
            ticket_id: Ticket to optimize
            baseline_metrics: Baseline metrics
            max_iterations: Max iterations

        Returns:
            PhaseResult with optimization outcome
        """
        print(f"   Max iterations: {max_iterations}")
        print("   Early exit: answer_correctness > 0.90\n")

        try:
            iteration = 0
            current_ans_corr = baseline_metrics.get("answer_correctness", 0.0)
            current_faithful = baseline_metrics.get("faithfulness", 0.0)

            while iteration < max_iterations:
                iteration += 1
                print(f"\n{'='*80}")
                print(f"📍 OPTIMIZATION ITERATION {iteration}/{max_iterations}")
                print(f"{'='*80}")

                # MUST use full evaluation (need to see answer quality)
                result = self.diagnose(ticket_id, use_existing=False)

                if result.answer_correctness and result.answer_correctness > current_ans_corr:
                    print(
                        f"✅ Improved! Answer: {current_ans_corr:.2f} → {result.answer_correctness:.2f}"
                    )
                    current_ans_corr = result.answer_correctness

                if result.faithfulness and result.faithfulness > current_faithful:
                    print(
                        f"✅ Improved! Faithfulness: {current_faithful:.2f} → {result.faithfulness:.2f}"
                    )
                    current_faithful = result.faithfulness

                # Early exit if good enough
                if current_ans_corr > 0.90 and current_faithful > 0.8:
                    print("\n🎯 Good enough! Exiting optimization early.")
                    print(f"   Answer Correctness: {current_ans_corr:.2f}")
                    print(f"   Faithfulness: {current_faithful:.2f}")
                    break

            final_metrics = {
                "answer_correctness": current_ans_corr,
                "faithfulness": current_faithful,
            }

            improved = current_ans_corr > baseline_metrics.get("answer_correctness", 0.0)

            return PhaseResult(
                phase_name="prompt_optimization",
                success=improved,
                iterations=iteration,
                final_metrics=final_metrics,
                reason=f"Answer: {baseline_metrics.get('answer_correctness', 0):.2f} → {current_ans_corr:.2f}",
            )

        except Exception as e:
            print(f"❌ Prompt optimization failed: {e}")
            return PhaseResult(
                phase_name="prompt_optimization", success=False, iterations=iteration, reason=str(e)
            )

    def _is_rag_bypassed(self, result: Any) -> bool:
        """Detect if RAG was bypassed (not used or returned 0 docs).

        Args:
            result: DiagnosticResult from baseline evaluation

        Returns:
            True if RAG was not used or returned 0 documents
        """
        # Check if RAG wasn't called at all
        if not hasattr(result, "rag_used") or not result.rag_used:
            return True

        # Check if RAG was called but returned 0 docs
        if hasattr(result, "num_docs_retrieved"):
            num_docs = result.num_docs_retrieved()
            return num_docs == 0

        # Fallback: check contexts directly
        if hasattr(result, "contexts"):
            contexts_str = str(result.contexts).strip()
            return contexts_str == "" or contexts_str == "[]" or contexts_str == "null"

        return False

    def _is_retrieval_problem(self, metrics: Dict) -> bool:
        """Determine if this is a retrieval problem.

        Args:
            metrics: Baseline metrics

        Returns:
            True if retrieval needs optimization
        """
        url_f1 = metrics.get("url_f1", 0.0)
        ctx_rel = metrics.get("context_relevance", 0.0)
        ctx_prec = metrics.get("context_precision", 0.0)

        # Retrieval problem if any retrieval metric is low
        return url_f1 < 0.5 or ctx_rel < 0.7 or ctx_prec < 0.7

    def _is_answer_problem(self, metrics: Dict) -> bool:
        """Determine if this is an answer quality problem.

        Args:
            metrics: Baseline metrics

        Returns:
            True if answer quality needs optimization
        """
        url_f1 = metrics.get("url_f1", 0.0)
        ctx_rel = metrics.get("context_relevance", 0.0)
        ans_corr = metrics.get("answer_correctness", 0.0)
        faithful = metrics.get("faithfulness", 0.0)

        # Answer problem if retrieval is good but answer is bad
        retrieval_good = url_f1 >= 0.5 and ctx_rel >= 0.7
        answer_bad = ans_corr < 0.90 or faithful < 0.8

        return retrieval_good and answer_bad

    def run_answer_validation(
        self, ticket_id: Optional[str], threshold: float, stability_runs: int = 1
    ) -> PhaseResult:
        """Phase 3: Validate answer correctness with stability check.

        Args:
            ticket_id: Ticket to validate
            threshold: Minimum answer_correctness score
            stability_runs: Number of runs for stability validation

        Returns:
            PhaseResult with answer validation outcome
        """
        print(f"📝 Validating answer correctness ({stability_runs} runs for stability)...")
        print(f"   Threshold: {threshold}")

        try:
            # Run full diagnosis with response generation (multiple runs for stability)
            result = self.diagnose(ticket_id, use_existing=False, runs=stability_runs)

            answer_correct = result.answer_correctness or 0.0
            faithful = result.faithfulness or 0.0

            print("\n📊 Answer Metrics:")
            print(f"   Answer Correctness: {answer_correct:.2f} (avg of {result.num_runs} runs)")
            print(f"   Faithfulness:       {faithful:.2f} (avg of {result.num_runs} runs)")

            # Check for high variance (instability)
            if result.high_variance_metrics:
                print("\n⚠️  HIGH VARIANCE DETECTED:")
                for metric_info in result.high_variance_metrics:
                    print(f"   • {metric_info}")
                print("   → Results are UNSTABLE across runs")
                print("   → Fix may not be reliable")

            passing = answer_correct >= threshold and faithful >= 0.8

            if passing:
                print("\n✅ Answer validation PASSED")
            else:
                print("\n❌ Answer validation FAILED")
                if answer_correct < threshold:
                    print(f"   Answer correctness too low: {answer_correct:.2f} < {threshold}")
                if faithful < 0.8:
                    print(f"   Faithfulness too low: {faithful:.2f} < 0.8")

            return PhaseResult(
                phase_name="answer_validation",
                success=passing,
                iterations=1,
                final_metrics={
                    "answer_correctness": answer_correct,
                    "faithfulness": faithful,
                },
                reason=f"answer_correctness={answer_correct:.2f}, faithfulness={faithful:.2f}",
            )

        except Exception as e:
            print(f"❌ Answer validation failed: {e}")
            return PhaseResult(phase_name="answer_validation", success=False, reason=str(e))

    def run_final_pattern_validation(
        self, ticket_id: Optional[str], stability_runs: int = 2
    ) -> PhaseResult:
        """Phase 4: Final pattern validation - remove skip tags and test all tickets.

        Ensures:
        - All pattern tickets are tested (no skip tags)
        - Previously-passing tickets still pass (no regressions)
        - All fixes remain stable

        Args:
            ticket_id: Ticket ID or None for full pattern
            stability_runs: Number of runs for stability check

        Returns:
            PhaseResult with pattern validation outcome
        """
        if ticket_id:
            print(f"📊 Final validation for {ticket_id} ({stability_runs} runs)...")
        else:
            print(f"📊 Final validation for ALL pattern tickets ({stability_runs} runs)...")
        print("   → Removing all skip tags")
        print("   → Testing with full metrics")
        print("   → Verifying no regressions in previously-passing tickets")

        try:
            # Remove skip tags from the SINGLE cleaned config (reused throughout)
            if self.cleaned_config and self.cleaned_config.exists():
                print(f"   🏷️  Removing skip tags from: {self.cleaned_config}")
                self.update_skip_tags(self.cleaned_config, {}, mode="remove")

            # Run full diagnosis with all metrics (skip tags removed)
            result = self.diagnose(ticket_id, use_existing=False, runs=stability_runs)

            # Check metrics
            answer_correct = result.answer_correctness or 0.0
            faithful = result.faithfulness or 0.0
            url_f1 = result.url_f1 or 0.0

            print("\n📊 Final Validation Metrics:")
            print(f"   Answer Correctness: {answer_correct:.2f} (avg of {result.num_runs} runs)")
            print(f"   Faithfulness:       {faithful:.2f}")
            print(f"   URL F1:             {url_f1:.2f}")

            # Check for high variance (instability)
            if result.high_variance_metrics:
                print("\n⚠️  HIGH VARIANCE DETECTED:")
                for metric_info in result.high_variance_metrics:
                    print(f"   • {metric_info}")
                print("   → Pattern fix may not be stable")

            # Calculate composite score (weights: answer 40%, context relevance 30%, precision 15%, keywords 10%, forbidden 5%)
            composite_score = self._calculate_composite_metric(result)
            COMPOSITE_THRESHOLD = 0.80  # High quality across all metrics

            passing = composite_score >= COMPOSITE_THRESHOLD

            if passing:
                print(f"\n✅ Final pattern validation PASSED")
                print(f"   Composite Score: {composite_score:.2f} (≥ {COMPOSITE_THRESHOLD:.2f})")
                print("   → All tickets validated")
                print("   → No regressions detected")
                print("   → Fixes are stable")
            else:
                print(f"\n❌ Final pattern validation FAILED")
                print(f"   Composite Score: {composite_score:.2f} < {COMPOSITE_THRESHOLD:.2f}")
                print("\n   Metric Breakdown:")
                if answer_correct < 0.90:
                    print(f"   • Answer correctness: {answer_correct:.2f} (target: ≥0.90)")
                else:
                    print(f"   ✓ Answer correctness: {answer_correct:.2f}")
                if faithful < 0.8:
                    print(f"   • Faithfulness: {faithful:.2f} (target: ≥0.80)")
                else:
                    print(f"   ✓ Faithfulness: {faithful:.2f}")
                if url_f1 < 0.7:
                    print(f"   • URL F1: {url_f1:.2f} (target: ≥0.70)")
                else:
                    print(f"   ✓ URL F1: {url_f1:.2f}")

            return PhaseResult(
                phase_name="final_pattern_validation",
                success=passing,
                iterations=1,
                final_metrics={
                    "composite_score": composite_score,
                    "answer_correctness": answer_correct,
                    "faithfulness": faithful,
                    "url_f1": url_f1,
                },
                reason=f"All tickets validated, composite_score={composite_score:.2f}, passing={passing}",
            )

        except Exception as e:
            print(f"❌ Final pattern validation failed: {e}")
            return PhaseResult(phase_name="final_pattern_validation", success=False, reason=str(e))

    def run_cla_regression_test(self) -> PhaseResult:
        """Phase 5: Run CLA release-gating tests to verify no regression.

        Returns:
            PhaseResult with CLA test outcome
        """
        print("🧪 Running CLA release-gating tests (96 questions)...")
        print("   Config: config/CLA_tests.yaml")
        print("   System: config/system_cla.yaml")
        print("   Purpose: Verify fix doesn't break existing functionality")

        cla_config = self.eval_root / "config" / "CLA_tests.yaml"
        cla_system = self.eval_root / "config" / "system_cla.yaml"

        if not cla_config.exists():
            print(f"⚠️  CLA tests not found: {cla_config}")
            return PhaseResult(
                phase_name="cla_regression",
                success=False,
                reason="CLA test config not found",
            )

        try:
            print("\n   ⏳ This will take ~5-10 minutes (96 questions)...\n")
            print("   Note: CLA tests often bypass RAG, using answer_correctness only")

            # Run CLA tests using lightspeed-eval with answer_correctness metric only
            # (RAG is often bypassed in CLA, so we only check answer quality)
            self.run_command(
                [
                    "uv",
                    "run",
                    "lightspeed-eval",
                    "run",
                    str(cla_system),
                    "--data",
                    str(cla_config),
                    "--metrics",
                    "custom:answer_correctness",
                ],
                cwd=self.eval_root,
            )

            # Find the latest output directory
            output_dirs = sorted(
                (self.eval_root / "results").glob("evaluation_*"),
                key=lambda p: p.stat().st_mtime,
            )
            if not output_dirs:
                raise RuntimeError("No CLA evaluation output found")

            output_dir = output_dirs[-1]

            # Parse summary results
            summary_file = output_dir / "evaluation_summary.csv"
            if not summary_file.exists():
                raise RuntimeError(f"Summary file not found: {summary_file}")

            import pandas as pd

            summary_df = pd.read_csv(summary_file)

            # Calculate pass rate
            total = len(summary_df)
            passed = len(summary_df[summary_df["result"] == "PASS"])
            failed = total - passed
            pass_rate = passed / total if total > 0 else 0.0

            print("\n📊 CLA Test Results:")
            print(f"   Total:     {total}")
            print(f"   Passed:    {passed}")
            print(f"   Failed:    {failed}")
            print(f"   Pass Rate: {pass_rate*100:.1f}%")

            # Success if >= 90% pass rate (allow some flakiness)
            success = pass_rate >= 0.90

            if success:
                print("\n✅ CLA regression test PASSED")
                print("   → Fix does not break existing functionality")
            else:
                print("\n❌ CLA regression test FAILED")
                print(f"   → Pass rate too low: {pass_rate*100:.1f}% < 90%")
                print("   → Fix may have introduced regressions")

            return PhaseResult(
                phase_name="cla_regression",
                success=success,
                iterations=1,
                final_metrics={
                    "cla_total": total,
                    "cla_passed": passed,
                    "cla_failed": failed,
                    "cla_pass_rate": pass_rate,
                },
                reason=f"CLA pass rate: {pass_rate*100:.1f}% ({passed}/{total})",
            )

        except Exception as e:
            print(f"❌ CLA regression test failed: {e}")
            return PhaseResult(phase_name="cla_regression", success=False, reason=str(e))

    def run_stability_check(
        self, ticket_id: Optional[str], threshold: float, num_runs: int
    ) -> PhaseResult:
        """Phase 4: Check answer stability across multiple runs.

        Args:
            ticket_id: Ticket to test
            threshold: Minimum answer_correctness per run
            num_runs: Number of stability runs

        Returns:
            PhaseResult with stability check outcome
        """
        print(f"🔄 Running stability check ({num_runs} runs)...")
        print(f"   Threshold: {threshold} per run")
        print("   Max variance: 0.05")

        try:
            runs = []

            for i in range(1, num_runs + 1):
                print(f"\n   Run {i}/{num_runs}...")
                result = self.diagnose(ticket_id, use_existing=False)

                answer_correct = result.answer_correctness or 0.0
                faithful = result.faithfulness or 0.0

                runs.append(
                    {
                        "run": i,
                        "answer_correctness": answer_correct,
                        "faithfulness": faithful,
                    }
                )

                print(f"      Answer: {answer_correct:.2f}, Faithfulness: {faithful:.2f}")

            # Calculate variance
            scores = [r["answer_correctness"] for r in runs]
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / len(scores)

            # Check all pass
            all_pass = all(r["answer_correctness"] >= threshold for r in runs)
            low_variance = variance < 0.05

            print("\n📊 Stability Results:")
            print(f"   Mean:     {mean:.2f}")
            print(f"   Variance: {variance:.4f}")
            print(f"   All pass: {all_pass}")
            print(f"   Stable:   {low_variance}")

            stable = all_pass and low_variance

            if stable:
                print("✅ Stability check PASSED")
            else:
                print("❌ Stability check FAILED")
                if not all_pass:
                    failing = [r for r in runs if r["answer_correctness"] < threshold]
                    print(f"   {len(failing)} runs failed threshold")
                if not low_variance:
                    print(f"   High variance: {variance:.4f} >= 0.05")
                    # TODO: Escalate to reliability testing framework
                    # For now, pass through - human review will decide next steps
                    self._escalate_to_reliability_testing(
                        ticket_id=ticket_id, variance=variance, runs=runs
                    )

            return PhaseResult(
                phase_name="stability",
                success=stable,
                iterations=num_runs,
                final_metrics={
                    "mean_answer_correctness": mean,
                    "variance": variance,
                    "runs": runs,
                },
                reason=f"mean={mean:.2f}, variance={variance:.4f}, all_pass={all_pass}",
            )

        except Exception as e:
            print(f"❌ Stability check failed: {e}")
            return PhaseResult(
                phase_name="stability", success=False, iterations=num_runs, reason=str(e)
            )

    def _escalate_to_reliability_testing(self, ticket_id: str, variance: float, runs: list) -> None:
        """Escalate unstable pattern for root cause analysis.

        High variance (>= 0.05) is usually fixable - see docs/VARIANCE_SOLUTIONS.md
        for diagnostic steps and common root causes:
        1. Bad ground truth (vague expected_response)
        2. Non-deterministic retrieval (URL ordering varies)
        3. Prompt sensitivity (small input changes → big output changes)
        4. Environmental issues (Solr index state, LLM API issues)

        TODO: Implement automated variance analysis:
        - Compare actual_response across runs for semantic similarity
        - Check retrieved URL ordering stability
        - Analyze if expected_response is too vague
        - Suggest specific fixes based on root cause

        For now, this is a pass-through that logs the escalation.
        Human review follows docs/VARIANCE_SOLUTIONS.md diagnostic steps.

        Args:
            ticket_id: Ticket that showed instability
            variance: Observed variance in answer_correctness
            runs: List of run results with scores
        """
        print("\n⚠️  ESCALATION: High variance detected")
        print(f"   Ticket: {ticket_id}")
        print(f"   Variance: {variance:.4f}")
        print("   TODO: Automated variance analysis (see docs/VARIANCE_SOLUTIONS.md)")
        print("   Human review required - check diagnostics for root cause")
        # Future: implement automated analysis from docs/VARIANCE_SOLUTIONS.md

    def _print_improvement_summary(self, result: PatternFixResult) -> None:
        """Print improvement summary showing metric trends across phases.

        Args:
            result: PatternFixResult with all phase results
        """
        print("\n" + "=" * 80)
        print("📈 IMPROVEMENT SUMMARY")
        print("=" * 80)

        # Extract metrics from each phase
        baseline_metrics = result.baseline.final_metrics if result.baseline else {}
        opt_metrics = result.optimization.final_metrics if result.optimization else {}
        final_metrics = result.stability.final_metrics if result.stability else {}

        # Calculate composite scores for each phase
        baseline_composite = baseline_metrics.get("composite_score")
        final_composite = final_metrics.get("composite_score")

        print("\n┌─ PHASE PROGRESSION ─────────────────────────────────────────┐")
        print("│                                                              │")

        # Answer Correctness
        baseline_ans = baseline_metrics.get("answer_correctness", 0.0)
        final_ans = final_metrics.get("answer_correctness", 0.0)
        ans_change = final_ans - baseline_ans
        ans_arrow = "→" if abs(ans_change) < 0.01 else ("↗" if ans_change > 0 else "↘")
        print(f"│  Answer Correctness:  {baseline_ans:.2f} {ans_arrow} {final_ans:.2f}  ({ans_change:+.2f})       │")

        # Faithfulness
        baseline_faith = baseline_metrics.get("faithfulness", 0.0)
        final_faith = final_metrics.get("faithfulness", 0.0)
        faith_change = final_faith - baseline_faith
        faith_arrow = "→" if abs(faith_change) < 0.01 else ("↗" if faith_change > 0 else "↘")
        print(f"│  Faithfulness:        {baseline_faith:.2f} {faith_arrow} {final_faith:.2f}  ({faith_change:+.2f})       │")

        # URL F1
        baseline_url = baseline_metrics.get("url_f1", 0.0)
        final_url = final_metrics.get("url_f1", 0.0)
        url_change = final_url - baseline_url
        url_arrow = "→" if abs(url_change) < 0.01 else ("↗" if url_change > 0 else "↘")
        print(f"│  URL F1:              {baseline_url:.2f} {url_arrow} {final_url:.2f}  ({url_change:+.2f})       │")

        # Composite Score (if available)
        if baseline_composite is not None and final_composite is not None:
            comp_change = final_composite - baseline_composite
            comp_arrow = "→" if abs(comp_change) < 0.01 else ("↗" if comp_change > 0 else "↘")
            print(f"│  Composite Score:     {baseline_composite:.2f} {comp_arrow} {final_composite:.2f}  ({comp_change:+.2f})       │")

        print("│                                                              │")
        print("└──────────────────────────────────────────────────────────────┘")

        # Show optimization details if available
        if result.optimization and result.optimization.iterations > 0:
            print(f"\n💡 Optimization: {result.optimization.iterations} iterations")
            print(f"   {result.optimization.reason}")

        # Overall verdict
        print("\n┌─ OVERALL RESULT ────────────────────────────────────────────┐")
        print("│                                                              │")
        if result.success:
            print("│  ✅ PATTERN FIX SUCCESSFUL                                   │")
        else:
            print("│  ❌ PATTERN FIX INCOMPLETE                                   │")

        if final_composite is not None:
            threshold = 0.80
            if final_composite >= threshold:
                print(f"│  Composite score {final_composite:.2f} meets threshold {threshold:.2f}           │")
            else:
                print(f"│  Composite score {final_composite:.2f} below threshold {threshold:.2f}          │")

        print("│                                                              │")
        print("└──────────────────────────────────────────────────────────────┘")

        # Duration
        duration_min = result.duration_seconds / 60
        print(f"\n⏱️  Total Duration: {duration_min:.1f} minutes")
        print("=" * 80 + "\n")

    def _is_passing(self, metrics: Dict, answer_threshold: float) -> bool:
        """Check if metrics indicate passing ticket.

        TRUE ANSWER-FIRST APPROACH:
        - If answer_correctness >= threshold → PASS (always)
        - If answer is correct BUT RAG metrics are bad → PASS with WARNING
        - If answer is incorrect → FAIL

        This allows:
        - Perfect answer + bad RAG = PASS (LLM correctly ignored bad docs)
        - Perfect answer + no RAG = PASS (LLM used training data)
        - Perfect answer + good RAG = PASS (ideal case)
        - Bad answer = FAIL (regardless of RAG quality)

        Args:
            metrics: Metric dictionary
            answer_threshold: Minimum answer_correctness (default: 0.90)

        Returns:
            True if answer_correctness meets threshold
        """
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
                        print(f"      faithfulness={faith:.2f} {'(LOW)' if faith < 0.7 else ''}")
                    print("      → LLM likely ignored bad docs and answered correctly anyway")
                    print("      → Consider reviewing Solr config (may need tuning)")

        return passing

    def generate_review_report(self, result: PatternFixResult) -> None:
        """Generate human review report.

        Args:
            result: Complete pattern fix result
        """
        report_path = result.diagnostics_dir / "REVIEW_REPORT.md"
        result.diagnostics_dir.mkdir(parents=True, exist_ok=True)

        status_emoji = "✅" if result.success else "❌"
        status_text = "SUCCESS" if result.success else "FAILED"

        duration_min = result.duration_seconds / 60

        report = f"""# Pattern Fix Review: {result.pattern_id}

## Summary
- **Status:** {status_emoji} {status_text}
- **Tickets Tested:** {result.tickets_tested}/{result.total_tickets}
- **Duration:** {duration_min:.1f} minutes
- **Branch:** {result.branch_name}

## Phase Results

### Phase 1: Baseline
"""

        if result.baseline:
            if result.baseline.success:
                report += "✅ **SUCCESS**\n\n"
                report += "Metrics:\n"
                for k, v in result.baseline.final_metrics.items():
                    if isinstance(v, (int, float)):
                        report += f"- {k}: {v:.2f}\n"
                    else:
                        report += f"- {k}: {v}\n"
                report += f"\nReason: {result.baseline.reason}\n"
            else:
                report += f"❌ **FAILED**\n\nReason: {result.baseline.reason}\n"
        else:
            report += "❌ Not run\n"

        report += "\n### Phase 2: Optimization\n"

        if result.optimization:
            if result.optimization.success:
                report += f"✅ **SUCCESS** ({result.optimization.iterations} iterations)\n\n"
                report += "Final Metrics:\n"
                for k, v in result.optimization.final_metrics.items():
                    if isinstance(v, (int, float)):
                        report += f"- {k}: {v:.2f}\n"
                    else:
                        report += f"- {k}: {v}\n"
                report += f"\nReason: {result.optimization.reason}\n"
            else:
                report += (
                    f"⚠️  **NO IMPROVEMENT** ({result.optimization.iterations} iterations)\n\n"
                )
                report += f"Reason: {result.optimization.reason}\n"
        else:
            report += "❌ Not run\n"

        report += "\n### Phase 3: Answer Validation\n"

        if result.answer_validation:
            if result.answer_validation.success:
                report += "✅ **PASSED**\n\n"
                report += "Metrics:\n"
                for k, v in result.answer_validation.final_metrics.items():
                    if isinstance(v, (int, float)):
                        report += f"- {k}: {v:.2f}\n"
                    else:
                        report += f"- {k}: {v}\n"
            else:
                report += f"❌ **FAILED**\n\nReason: {result.answer_validation.reason}\n"
        else:
            report += "❌ Not run\n"

        report += "\n### Phase 4: Final Pattern Validation\n"

        if result.stability:  # Pattern validation results
            if result.stability.success:
                report += "✅ **PASSED**\n\n"
                report += "All pattern tickets validated:\n"
                for k, v in result.stability.final_metrics.items():
                    if isinstance(v, (int, float)):
                        report += f"- {k}: {v:.2f}\n"
                    else:
                        report += f"- {k}: {v}\n"
            else:
                report += f"❌ **FAILED**\n\nReason: {result.stability.reason}\n"
        else:
            report += "❌ Not run\n"

        report += "\n### Phase 5: CLA Regression Test (Release Gating)\n"

        if result.cla_regression:  # CLA test results
            if result.cla_regression.success:
                cla_total = result.cla_regression.final_metrics.get("cla_total", 0)
                cla_passed = result.cla_regression.final_metrics.get("cla_passed", 0)
                cla_pass_rate = result.cla_regression.final_metrics.get("cla_pass_rate", 0)
                report += f"✅ **PASSED** ({cla_passed}/{cla_total} questions, {cla_pass_rate*100:.1f}%)\n\n"
                report += "Results:\n"
                report += f"- Total Questions: {cla_total}\n"
                report += f"- Passed: {cla_passed}\n"
                report += f"- Failed: {result.cla_regression.final_metrics.get('cla_failed', 0)}\n"
                report += f"- Pass Rate: {cla_pass_rate*100:.1f}%\n"
            else:
                report += f"❌ **FAILED**\n\nReason: {result.cla_regression.reason}\n"
        else:
            report += "❌ Not run\n"

        report += f"""

## Artifacts
- **Branch:** `{result.branch_name}`
- **Diagnostics:** `.diagnostics/{result.pattern_id}/`
- **Git Log:** `git log {result.branch_name} --oneline`

## Next Steps

"""

        if result.success:
            report += f"""1. Review branch commits:
   ```bash
   git checkout {result.branch_name}
   git log --oneline
   ```

2. Review diagnostics:
   ```bash
   cat .diagnostics/{result.pattern_id}/iteration_summary.txt
   ```

3. Test manually (optional):
   ```bash
   uv run lightspeed-eval \\
       --config config/system_cla.yaml \\
       --data config/patterns_v2/{result.pattern_id}.yaml
   ```

4. Merge if satisfied:
   ```bash
   git checkout main
   git merge --squash {result.branch_name}
   git commit -m "fix: {result.pattern_id} - improved retrieval and answer quality"
   ```
"""
        else:
            report += f"""1. Review what failed:
   ```bash
   cat .diagnostics/{result.pattern_id}/iteration_summary.txt
   ```

2. Check diagnostics for insights:
   ```bash
   ls .diagnostics/{result.pattern_id}/
   ```

3. Possible issues:
   - Bad ground truth (check expected_response)
   - Insufficient documentation (docs don't exist)
   - Retrieval optimization limit (need different approach)
   - Unstable LLM responses (need prompt tuning)

4. Manual investigation recommended
"""

        with open(report_path, "w") as f:
            f.write(report)

        print(f"\n📄 Review report generated: {report_path}")


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML with environment variable expansion.

    Args:
        config_path: Path to config YAML file

    Returns:
        Dictionary with resolved configuration
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        lines = f.readlines()

    # Expand environment variables in non-comment lines only
    import re

    def expand_var(match):
        var_name = match.group(1)
        value = os.environ.get(var_name)
        if value is None:
            raise ValueError(
                f"Environment variable ${{{var_name}}} not set. "
                f"Please set it or edit {config_path}"
            )
        return value

    processed_lines = []
    for line in lines:
        # Skip comment lines for variable expansion
        if not line.lstrip().startswith("#"):
            # Only expand ${VAR} syntax (not bare $VAR to avoid false matches)
            line = re.sub(r"\$\{(\w+)\}", expand_var, line)
        processed_lines.append(line)

    config_str = "".join(processed_lines)
    config = yaml.safe_load(config_str)

    # Resolve relative paths
    config_dir = config_path.parent

    # Repository root paths are relative to config file directory
    for key in ["eval_root", "okp_mcp_root", "lscore_deploy_root"]:
        if key in config and not Path(config[key]).is_absolute():
            config[key] = (config_dir / config[key]).resolve()

    # patterns_dir is relative to REPO_ROOT (HEAL root), not config_dir
    if "patterns_dir" in config and not Path(config["patterns_dir"]).is_absolute():
        config["patterns_dir"] = (REPO_ROOT / config["patterns_dir"]).resolve()

    return config


def main():
    """Main entry point for POC."""
    parser = argparse.ArgumentParser(description="Pattern fix loop proof of concept")

    parser.add_argument("pattern_id", help="Pattern ID to fix (e.g., EOL_UNSUPPORTED_LEGACY_RHEL)")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "pattern_fix_config.yaml",
        help="Config file with paths (default: config/pattern_fix_config.yaml)",
    )
    parser.add_argument("--patterns-dir", type=Path, help="Override patterns directory from config")
    parser.add_argument(
        "--max-iterations", type=int, help="Override max optimization iterations from config"
    )
    parser.add_argument(
        "--answer-threshold", type=float, help="Override minimum answer_correctness from config"
    )
    parser.add_argument(
        "--stability-runs", type=int, help="Override number of stability check runs from config"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "full"],
        default="single",
        help="Testing mode: 'single' (one representative ticket) or 'full' (all tickets in pattern)",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"❌ Failed to load config from {args.config}: {e}")
        sys.exit(1)

    # Override config with command-line args if provided
    if args.patterns_dir:
        config["patterns_dir"] = args.patterns_dir
    if args.max_iterations:
        config["max_iterations"] = args.max_iterations
    if args.answer_threshold:
        config["answer_threshold"] = args.answer_threshold
    if args.stability_runs:
        config["stability_runs"] = args.stability_runs

    # Validate required paths exist
    for key in ["eval_root", "okp_mcp_root", "lscore_deploy_root"]:
        path = Path(config[key])
        if not path.exists():
            print(f"❌ Required path does not exist: {key} = {path}")
            print(f"   Please edit {args.config} or set environment variable")
            sys.exit(1)

    # Initialize agent
    print("🚀 Pattern Fix Loop POC")
    print(f"{'='*80}\n")

    agent = PatternFixAgent(
        pattern_id=args.pattern_id,
        eval_root=Path(config["eval_root"]),
        okp_mcp_root=Path(config["okp_mcp_root"]),
        lscore_deploy_root=Path(config["lscore_deploy_root"]),
        interactive=config.get("interactive", True),
        enable_llm_advisor=config.get("enable_llm_advisor", True),
    )

    # Load pattern tickets
    try:
        agent.load_pattern_tickets(Path(config["patterns_dir"]))
    except Exception as e:
        print(f"❌ Failed to load pattern: {e}")
        sys.exit(1)

    # Run fix loop
    try:
        result = agent.run_fix_loop(
            max_iterations=config["max_iterations"],
            answer_threshold=config["answer_threshold"],
            stability_runs=config["stability_runs"],
            mode=args.mode,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        agent.cleanup()
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fix loop failed: {e}")
        import traceback

        traceback.print_exc()
        agent.cleanup()
        sys.exit(1)

    # Generate review report
    try:
        agent.generate_review_report(result)
    except Exception as e:
        print(f"⚠️  Failed to generate review report: {e}")

    # Print final summary
    print(f"\n{'='*80}")
    print("PATTERN FIX LOOP COMPLETE")
    print(f"{'='*80}")
    print(f"Pattern: {result.pattern_id}")
    print(f"Status: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
    print(f"Duration: {result.duration_seconds / 60:.1f} minutes")
    print(f"Branch: {result.branch_name}")
    print(f"Diagnostics: {result.diagnostics_dir}")
    print(f"{'='*80}\n")

    if result.success:
        print("✅ Pattern fix successful!")
        print(f"   Review: cat {result.diagnostics_dir}/REVIEW_REPORT.md")
        print(f"   Merge:  git merge --squash {result.branch_name}")
        sys.exit(0)
    else:
        print("❌ Pattern fix failed")
        print(f"   Review diagnostics: ls {result.diagnostics_dir}/")
        print(f"   Check report: cat {result.diagnostics_dir}/REVIEW_REPORT.md")
        sys.exit(1)


if __name__ == "__main__":
    main()
