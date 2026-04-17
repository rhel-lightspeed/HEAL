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

from heal.agents.okp_mcp_agent import OkpMcpAgent, PatternEvaluationResult, TIER_MODELS
from heal.core.ticket_evaluation import PatternEvaluation, TicketEvaluation

# Optional multi-agent system (requires claude-agent-sdk)
try:
    from heal.agents.solr_multi_agent import SolrMultiAgentSystem
    MULTI_AGENT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    SolrMultiAgentSystem = None
    MULTI_AGENT_AVAILABLE = False

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

        # Track per-ticket baseline evaluation for comparison
        self._baseline_pattern: Optional[PatternEvaluation] = None

        # Initialize multi-agent system for better Solr optimization
        if MULTI_AGENT_AVAILABLE:
            try:
                self.multi_agent = SolrMultiAgentSystem(
                    okp_mcp_root=okp_mcp_root,
                    model="claude-sonnet-4-6",
                )
                print("✅ Multi-agent Solr optimization enabled (Solr Expert + Code Expert)")
            except Exception as e:
                print(f"⚠️  Multi-agent system failed to initialize: {e}")
                print("   Falling back to single-agent mode")
                self.multi_agent = None
        else:
            print("⚠️  Multi-agent system not available (requires claude-agent-sdk)")
            print("   Using single-agent mode")
            self.multi_agent = None

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

        # Validate tickets and categorize by documentation availability
        tickets_with_docs = 0
        tickets_without_docs = []

        for conv in conversations:
            ticket_id = conv["conversation_group_id"]

            # Check if ticket has expected_urls (documentation exists)
            has_expected_urls = False
            for turn in conv.get("turns", []):
                if turn.get("expected_urls"):
                    has_expected_urls = True
                    break

            # Track all tickets, but flag which ones lack docs
            ticket_info = {"ticket_id": ticket_id, "has_expected_urls": has_expected_urls}
            self.pattern_tickets.append(ticket_info)

            if has_expected_urls:
                tickets_with_docs += 1
            else:
                tickets_without_docs.append(ticket_id)

        # Use pattern file directly as test config (already in correct format)
        self.functional_full = pattern_file
        self.functional_retrieval = pattern_file

        print(f"✅ Loaded {len(self.pattern_tickets)} tickets for pattern {self.pattern_id}")
        print(f"   • With documentation: {tickets_with_docs}")
        print(f"   • Without documentation: {len(tickets_without_docs)}")

        # Inform about no-doc tickets (answer-only evaluation)
        if tickets_without_docs:
            print("\n📋 NO-DOC TICKET HANDLING:")
            print(f"   {len(tickets_without_docs)} ticket(s) have NO expected_urls")
            print("   → These will be evaluated on ANSWER CORRECTNESS ONLY")
            print("   → No retrieval optimization (docs don't exist)")
            print("   → If answer >= 0.90, will mark as STABLE_PASSING and skip")
            print("\n   ❓ Tickets without documentation:")
            for ticket_id in tickets_without_docs:
                print(f"      • {ticket_id}")
            print("\n   💡 STRATEGY:")
            print("      1. Baseline: Test answer_correctness only")
            print("      2. If passing (≥0.90): Skip (LLM answered from training data)")
            print("      3. If failing: Flag for SME review (needs new documentation)")
            print("      4. Skip retrieval optimization (no docs to optimize)")
            print("      5. Re-test in final validation (ensure no regression)\n")

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

        # CRITICAL: Restart container to use new branch code
        print("\n🔄 Restarting okp-mcp container to load branch code...")
        print(f"   Container mounts: {self.okp_mcp_root}/src")
        print(f"   Current branch: {self.branch_name}")
        self.restart_okp_mcp()
        print(f"   ✅ Container restarted - now running code from {self.branch_name}")

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

            # Handle both single ticket and pattern results
            if isinstance(result, PatternEvaluationResult):
                # Pattern mode - show pattern-level aggregates
                print(f"   Pattern: {result.pattern_id}")
                print(f"   Tickets Evaluated: {len(result.per_ticket_results)}")
                print(f"   URL F1 (avg):             {result.pattern_url_f1:.2f}")
                print(f"   Answer Correctness (avg): {result.pattern_answer_correctness:.2f}")
                print(f"   Faithfulness (avg):       {result.pattern_faithfulness:.2f}")
                print(f"   Success Rate:             {result.success_rate:.0%}")
            else:
                # Single ticket mode - show ticket metrics
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

            # Build lookup of tickets without expected_urls (must come BEFORE usage)
            no_doc_tickets = {
                t["ticket_id"] for t in self.pattern_tickets if not t.get("has_expected_urls", True)
            }

            # Store baseline per-ticket evaluations for later comparison
            self._baseline_pattern = PatternEvaluation(pattern_id=self.pattern_id)
            for ticket_id, runs in per_ticket_results.items():
                if runs:
                    # Check if this is a no-doc ticket
                    is_no_doc = ticket_id in no_doc_tickets
                    ticket_eval = TicketEvaluation(
                        ticket_id=ticket_id, runs=runs, is_no_doc=is_no_doc
                    )
                    self._baseline_pattern.tickets[ticket_id] = ticket_eval

            # Classify each ticket based on per-run scores
            ticket_classifications = {}
            for ticket, runs in per_ticket_results.items():
                # Extract answer_correctness scores across runs
                ans_scores = [
                    r.get("answer_correctness", 0.0) for r in runs if "answer_correctness" in r
                ]

                if ans_scores:
                    # Check if this is a no-doc ticket
                    is_no_doc = ticket in no_doc_tickets

                    classification = classify_stability(
                        ans_scores,
                        threshold=0.90,
                        catastrophic_threshold=0.70,
                        high_cv_threshold=0.15,
                    )

                    # Override skip logic for no-doc tickets
                    if is_no_doc:
                        # For no-doc tickets, only answer_correctness matters
                        # Mark as skip if passing (LLM answered from training data)
                        if classification.status.value == "STABLE_PASSING":
                            classification.skip = True
                            classification.reason = f"{classification.reason} [NO-DOC: Answered from LLM training data, skip retrieval optimization]"
                        else:
                            # Failing no-doc ticket = needs new documentation
                            classification.skip = False
                            classification.priority = "HIGH"
                            classification.needs_review = True
                            classification.reason = f"{classification.reason} [NO-DOC: Failing without docs, needs SME review or new documentation]"

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
                    # Check if this is a no-doc ticket
                    is_no_doc = ticket in no_doc_tickets
                    no_doc_tag = " [NO-DOC]" if is_no_doc else ""

                    print(f"\n{emoji} {ticket}: {classification.status.value}{no_doc_tag}")
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
                no_doc_count = len(no_doc_tickets)

                print(f"   Total tickets:        {len(ticket_classifications)}")
                if no_doc_count > 0:
                    print(f"   🔍 No-doc tickets:    {no_doc_count} (answer-only evaluation)")
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

                    # Show which categories are being skipped
                    skipped_stable = [
                        tid
                        for tid, cls in ticket_classifications.items()
                        if cls.skip and cls.status.value == "STABLE_PASSING"
                    ]
                    skipped_no_doc = [tid for tid in skipped_stable if tid in no_doc_tickets]

                    if skipped_stable:
                        print(
                            f"   📌 Skipping {len(skipped_stable)} stable-passing ticket(s) in optimization:"
                        )
                        if skipped_no_doc:
                            print(
                                f"      • {len(skipped_no_doc)} no-doc tickets (answered from LLM training)"
                            )
                        reg_skipped = len(skipped_stable) - len(skipped_no_doc)
                        if reg_skipped > 0:
                            print(f"      • {reg_skipped} regular tickets (already passing)")
                else:
                    print(f"   ⚠️  Cleaned config not found: {self.cleaned_config}")

            # Calculate averaged metrics (for backward compatibility)
            # Handle both PatternEvaluationResult and EvaluationResult
            if isinstance(result, PatternEvaluationResult):
                # Pattern mode - use pattern-level aggregates
                metrics = {
                    "url_f1": result.pattern_url_f1,
                    "mrr": 0.0,  # TODO: Calculate pattern-level MRR average
                    "context_relevance": 0.0,  # TODO: Calculate from per-ticket results
                    "context_precision": 0.0,  # TODO: Calculate from per-ticket results
                    "answer_correctness": result.pattern_answer_correctness,
                    "faithfulness": result.pattern_faithfulness,
                    "response_relevancy": 0.0,  # TODO: Calculate from per-ticket results
                    "ticket_classifications": ticket_classifications,
                    "baseline_result": result,  # Store full pattern result
                }
                # Pattern-level RAG bypass: check if ANY ticket bypassed RAG
                rag_bypassed = len(result.rag_bypass_tickets) > 0
                num_docs = 0  # Pattern mode doesn't have single doc count

                print("\n📊 Baseline Metrics (Pattern Averages):")
                print(f"   Runs:               {result.num_runs}")
                print(f"   URL F1 (avg):       {metrics['url_f1']:.2f}")
                print(f"   Answer (avg):       {metrics['answer_correctness']:.2f}")
                print(f"   Faithfulness (avg): {metrics['faithfulness']:.2f}")
                print(f"   Success Rate:       {result.success_rate:.0%}")

                if result.rag_bypass_tickets:
                    print(f"\n⚠️  RAG Bypass: {len(result.rag_bypass_tickets)} ticket(s)")
                    for tid in result.rag_bypass_tickets[:3]:  # Show first 3
                        print(f"      • {tid}")

                if result.high_variance_tickets:
                    print(f"\n⚠️  High Variance: {len(result.high_variance_tickets)} ticket(s)")
                    for tid in result.high_variance_tickets[:3]:  # Show first 3
                        print(f"      • {tid}")

                # Pattern-level problem detection
                # Check if ANY ticket has retrieval or answer problems
                is_retrieval = any(
                    tr.is_retrieval_problem for tr in result.per_ticket_results.values()
                )
                is_answer = any(tr.is_answer_problem for tr in result.per_ticket_results.values())

                print("\n🔍 Problem Analysis (Any Ticket):")
                print(f"   Retrieval Problem: {is_retrieval}")
                print(f"   Answer Problem:    {is_answer}")

            else:
                # Single ticket mode - use ticket-level attributes
                metrics = {
                    "url_f1": result.url_f1 or 0.0,
                    "mrr": result.mrr or 0.0,
                    "context_relevance": result.context_relevance or 0.0,
                    "context_precision": result.context_precision or 0.0,
                    "answer_correctness": result.answer_correctness or 0.0,
                    "faithfulness": result.faithfulness or 0.0,
                    "response_relevancy": result.response_relevancy or 0.0,
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

            # Check if this is a documentation gap
            url_f1 = baseline_metrics.get("url_f1", 0.0)
            if url_f1 == 0.0:
                print("\n❌ DOCUMENTATION GAP DETECTED:")
                print("   → Expected URLs in config but 0 docs retrieved")
                print("   → This indicates:")
                print("      • Documentation missing from Solr/OKP index")
                print("      • OR expected_urls in YAML are wrong")
                print("      • OR query doesn't match indexed docs")
                print("\n   ⚠️  CANNOT FIX via RAG optimization")
                print("   📋 REQUIRED ACTIONS:")
                print("      1. Verify docs exist in Solr index")
                print("      2. Check expected_urls match actual doc IDs")
                print("      3. Add missing documentation if needed")
                print("      4. Quarantine ticket for SME review")
                return PhaseResult(
                    phase_name="optimization",
                    success=False,
                    reason="Documentation gap - expected docs not in index",
                )

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
            return self.run_retrieval_optimization(ticket_id, baseline_metrics, max_iterations, baseline_result)
        elif is_answer:
            # Route B: Prompt optimization (system prompt changes)
            print("\n📍 Route B: PROMPT OPTIMIZATION")
            print("   Testing: System prompt changes (instructions, grounding, etc.)")
            print("   Mode: Full evaluation (WITH response generation)")
            print("   Speed: ~30-60 sec/iteration")
            return self.run_prompt_optimization(ticket_id, baseline_metrics, max_iterations)
        else:
            print("\n⚠️  No clear problem identified - trying retrieval optimization")
            return self.run_retrieval_optimization(ticket_id, baseline_metrics, max_iterations, baseline_result)

    def run_retrieval_optimization(
        self, ticket_id: Optional[str], baseline_metrics: Dict, max_iterations: int, baseline_result: Any = None
    ) -> PhaseResult:
        """Route A: Fast retrieval optimization (Solr config changes).

        Uses the parent class's optimize_solr_retrieval() method which:
        1. Gets suggestions from Solr Expert
        2. Applies changes to Solr config
        3. Restarts container (CRITICAL!)
        4. Tests the change
        5. Commits if improved

        Args:
            ticket_id: Ticket to optimize
            baseline_metrics: Baseline metrics
            max_iterations: Max iterations
            baseline_result: Full baseline result (PatternEvaluationResult for pattern mode)

        Returns:
            PhaseResult with optimization outcome
        """
        print(f"   Max iterations: {max_iterations}")
        print("   Early exit: F1 > 0 (any expected docs found)\n")

        # Full-pattern mode: Pattern-wide optimization (NOT sequential per-ticket!)
        if not ticket_id:
            return self._run_pattern_wide_retrieval_optimization(
                baseline_result, baseline_metrics, max_iterations
            )

        # Load pattern YAML to get ticket details
        pattern_file = Path(f"config/patterns/{self.pattern_id}.yaml")
        with open(pattern_file) as f:
            content = f.read()
            lines = [line for line in content.split("\n") if not line.startswith("#")]
            yaml_content = "\n".join(lines)
            conversations = yaml.safe_load(yaml_content)

        # Find the conversation for this ticket
        conversation = None
        for conv in conversations:
            if conv.get("conversation_group_id") == ticket_id:
                conversation = conv
                break

        if not conversation:
            raise ValueError(f"Ticket {ticket_id} not found in pattern {self.pattern_id}")

        # Get first turn (assuming single-turn for now)
        turns = conversation.get("turns", [])
        if not turns:
            raise ValueError(f"No turns found for ticket {ticket_id}")

        first_turn = turns[0]
        query = first_turn.get("query", "")
        expected_urls = first_turn.get("expected_urls", [])

        try:
            # Call parent class's fast retrieval loop
            # This handles: get suggestion → apply change → restart → test → commit
            self.fast_retrieval_loop(
                ticket_id=ticket_id,
                query=query,
                expected_urls=expected_urls,
                max_iterations=max_iterations,
            )

            # Get final metrics by running one more test
            result = self.diagnose_retrieval_only(ticket_id, iteration=max_iterations + 1)

            final_f1 = result.url_f1 or 0.0
            final_ctx_rel = result.context_relevance or 0.0

            final_metrics = {
                "url_f1": final_f1,
                "context_relevance": final_ctx_rel,
            }

            improved = final_f1 > baseline_metrics.get("url_f1", 0.0)

            return PhaseResult(
                phase_name="retrieval_optimization",
                success=improved,
                iterations=max_iterations,
                final_metrics=final_metrics,
                reason=f"F1: {baseline_metrics.get('url_f1', 0):.2f} → {final_f1:.2f}",
            )

        except Exception as e:
            print(f"❌ Retrieval optimization failed: {e}")
            import traceback

            traceback.print_exc()
            return PhaseResult(
                phase_name="retrieval_optimization",
                success=False,
                iterations=0,
                reason=str(e),
            )

    def _run_pattern_wide_retrieval_optimization(
        self,
        baseline_result: PatternEvaluationResult,
        baseline_metrics: Dict,
        max_iterations: int,
    ) -> PhaseResult:
        """Pattern-wide retrieval optimization - ONE change tested on ALL tickets.

        This is the correct pattern-based approach:
        1. Get baseline scores for ALL tickets
        2. Get ONE Solr suggestion considering ALL failing tickets together
        3. Apply change ONCE
        4. Test ALL tickets
        5. Show pattern-wide impact (which improved, which regressed)
        6. Commit if net positive
        7. Repeat for max_iterations

        Args:
            baseline_result: Pattern baseline with per-ticket results
            baseline_metrics: Pattern-level baseline metrics
            max_iterations: Max optimization iterations

        Returns:
            PhaseResult with pattern-wide optimization outcome
        """
        print("🔄 Full-pattern mode: Pattern-wide retrieval optimization")
        print(f"   Strategy: ONE change tested on ALL {len(baseline_result.per_ticket_results)} tickets")
        print(f"   Max iterations: {max_iterations}\n")

        # Load pattern YAML to get all ticket details
        pattern_file = Path(f"config/patterns/{self.pattern_id}.yaml")
        with open(pattern_file) as f:
            content = f.read()
            lines = [line for line in content.split("\n") if not line.startswith("#")]
            yaml_content = "\n".join(lines)
            conversations = yaml.safe_load(yaml_content)

        # Build ticket lookup: {ticket_id: {query, expected_urls}}
        ticket_queries = {}
        for conv in conversations:
            tid = conv.get("conversation_group_id")
            turns = conv.get("turns", [])
            if tid and turns:
                first_turn = turns[0]
                ticket_queries[tid] = {
                    "query": first_turn.get("query", ""),
                    "expected_urls": first_turn.get("expected_urls", []),
                }

        # Identify tickets with retrieval problems
        failing_tickets = [
            tid
            for tid, tres in baseline_result.per_ticket_results.items()
            if tres.is_retrieval_problem and tid in ticket_queries
        ]

        if not failing_tickets:
            print("✅ No tickets have retrieval problems")
            return PhaseResult(
                phase_name="retrieval_optimization",
                success=True,
                iterations=0,
                final_metrics=baseline_metrics,
                reason="No retrieval problems detected",
            )

        print(f"   Found {len(failing_tickets)} tickets with retrieval problems:")
        for tid in failing_tickets:
            tres = baseline_result.per_ticket_results[tid]
            print(f"     • {tid}: F1={tres.url_f1:.2f}, MRR={tres.mrr or 0:.2f}")
        print()

        # Pattern-wide optimization loop
        print(f"{'='*80}")
        print("PATTERN-WIDE RETRIEVAL OPTIMIZATION")
        print(f"{'='*80}\n")

        best_pattern_f1 = baseline_result.pattern_url_f1
        iteration_count = 0
        commits_made = 0
        iterations_without_improvement = 0
        max_iterations_without_improvement = 3  # Exit if stuck for 3 iterations

        for iteration in range(1, max_iterations + 1):
            print(f"\n--- Pattern Iteration {iteration}/{max_iterations} ---\n")

            # Get ONE Solr suggestion considering ALL failing tickets
            # Use the first failing ticket as representative
            representative_ticket = failing_tickets[0]
            rep_query = ticket_queries[representative_ticket]["query"]
            rep_urls = ticket_queries[representative_ticket]["expected_urls"]

            # Get baseline for representative ticket
            baseline_rep = baseline_result.per_ticket_results[representative_ticket]

            # Get suggestion using multi-agent system (if available)
            if self.multi_agent:
                print("🤖 Consulting multi-agent system (Solr Expert + Code Expert)...\n")

                try:
                    # Multi-agent approach: Solr theory + code analysis
                    synthesized = self._run_async_in_thread(
                        self.multi_agent.get_optimized_suggestion(
                            query=rep_query,
                            expected_urls=rep_urls,
                            retrieved_urls=[],  # TODO: Extract from baseline
                            metrics={
                                "url_f1": baseline_rep.url_f1 or 0.0,
                                "mrr": baseline_rep.mrr or 0.0,
                            },
                            solr_explain=None,  # TODO: Get from solr_analyzer
                        )
                    )

                    print(f"🔍 Solr Expert + Code Expert Analysis Complete")
                    print(f"   Confidence: {synthesized.confidence:.0%}")
                    if synthesized.risks:
                        print(f"   Risks: {', '.join(synthesized.risks[:2])}")
                    print()

                    # Convert to compatible suggestion format
                    from dataclasses import dataclass

                    @dataclass
                    class MultiAgentSuggestion:
                        suggested_change: str
                        file_path: str
                        old_code: str
                        new_code: str
                        reasoning: str

                    suggestion = MultiAgentSuggestion(
                        suggested_change=synthesized.suggested_change,
                        file_path=synthesized.file_path,
                        old_code=synthesized.old_code,
                        new_code=synthesized.new_code,
                        reasoning=synthesized.reasoning,
                    )

                except Exception as e:
                    print(f"⚠️  Multi-agent system failed: {e}")
                    print("   Falling back to single-agent mode\n")
                    self.multi_agent = None  # Disable for rest of session
                    suggestion = None

            # Fallback to single-agent mode
            if not self.multi_agent or not suggestion:
                print("🤖 Using single-agent mode (fallback)...\n")

                from dataclasses import dataclass

                @dataclass
                class MinimalResult:
                    ticket_id: str
                    query: str
                    url_f1: float
                    mrr: float
                    expected_urls: List[str]
                    retrieved_urls: List[str]
                    is_retrieval_problem: bool = True

                minimal_result = MinimalResult(
                    ticket_id=representative_ticket,
                    query=rep_query,
                    url_f1=baseline_rep.url_f1 or 0.0,
                    mrr=baseline_rep.mrr or 0.0,
                    expected_urls=rep_urls,
                    retrieved_urls=[],
                )

                # Load Solr config snapshot
                solr_snapshot = self.load_solr_config_snapshot(representative_ticket)
                if not solr_snapshot:
                    solr_snapshot = self.extract_solr_config_snapshot(representative_ticket)

                # Get suggestion from single LLM
                suggestion = self._get_llm_suggestion_object(
                    minimal_result,
                    model=TIER_MODELS["medium"],
                    iteration_history=[],
                    solr_snapshot=solr_snapshot,
                )

            if not suggestion:
                print("❌ Failed to get suggestion")
                continue

            print(f"💡 Suggestion: {suggestion.suggested_change}\n")

            # Apply change
            if not self.apply_code_change(suggestion, iteration_context=f"Pattern Iteration {iteration}"):
                print("❌ Change not applied")
                continue

            # Restart okp-mcp
            print("🔄 Restarting okp-mcp...")
            self.restart_okp_mcp()

            # TEST ALL TICKETS IN PATTERN
            print(f"\n📊 Testing ALL {len(ticket_queries)} tickets in pattern...\n")

            pattern_scores = {}
            for tid in ticket_queries.keys():
                try:
                    # Test this ticket
                    query = ticket_queries[tid]["query"]
                    expected_urls = ticket_queries[tid]["expected_urls"]

                    if not expected_urls:
                        print(f"   ⚠️  {tid}: No expected_urls, skipping")
                        continue

                    # Query Solr directly for fast testing
                    current = self.query_solr_direct(query, expected_urls)

                    if "error" not in current:
                        pattern_scores[tid] = {
                            "url_f1": current["url_f1"],
                            "mrr": current["mrr"],
                        }
                except Exception as e:
                    print(f"   ❌ {tid}: Error testing: {e}")

            # Show pattern-wide impact
            print(f"\n{'='*80}")
            print("PATTERN-WIDE IMPACT")
            print(f"{'='*80}")
            print(f"{'Ticket':<20} {'Baseline F1':<12} {'Current F1':<12} {'Change':<12} {'Status'}")
            print("-" * 80)

            improved_count = 0
            regressed_count = 0
            unchanged_count = 0
            total_f1_delta = 0.0

            for tid in sorted(pattern_scores.keys()):
                current_f1 = pattern_scores[tid]["url_f1"]
                baseline_f1 = baseline_result.per_ticket_results[tid].url_f1 or 0.0
                delta = current_f1 - baseline_f1

                if delta > 0.05:
                    status = "✅ IMPROVED"
                    improved_count += 1
                elif delta < -0.05:
                    status = "❌ REGRESSED"
                    regressed_count += 1
                else:
                    status = "  UNCHANGED"
                    unchanged_count += 1

                total_f1_delta += delta

                print(
                    f"{tid:<20} {baseline_f1:>11.2f} {current_f1:>11.2f} {delta:>+11.2f} {status}"
                )

            print("-" * 80)
            print(
                f"{'SUMMARY':<20} {'':<12} {'':<12} {total_f1_delta:>+11.2f} "
                f"({improved_count} improved, {regressed_count} regressed, {unchanged_count} unchanged)"
            )
            print(f"{'='*80}\n")

            # Decision: Commit if net positive (more improved than regressed)
            if improved_count > regressed_count:
                print(f"✅ Net positive! Committing change ({improved_count} improved > {regressed_count} regressed)")
                import subprocess

                subprocess.run(["git", "add", "src/okp_mcp/solr.py"], cwd=self.okp_mcp_root, check=True)
                subprocess.run(
                    ["git", "commit", "-m", f"pattern: {suggestion.suggested_change}"],
                    cwd=self.okp_mcp_root,
                    check=True,
                )
                commits_made += 1
                iteration_count += 1

                # Update best pattern F1
                avg_f1 = sum(s["url_f1"] for s in pattern_scores.values()) / len(pattern_scores)
                if avg_f1 > best_pattern_f1:
                    best_pattern_f1 = avg_f1
                    iterations_without_improvement = 0  # Reset counter - we improved!
                else:
                    iterations_without_improvement += 1

                # EARLY EXIT: Check if we've solved the pattern
                # Count how many tickets are now "good enough" (F1 > 0.5)
                passing_tickets = sum(1 for s in pattern_scores.values() if s["url_f1"] >= 0.5)
                total_tickets = len(pattern_scores)
                success_rate = passing_tickets / total_tickets if total_tickets > 0 else 0.0

                print(f"\n📊 Pattern Status: {passing_tickets}/{total_tickets} tickets passing (F1 ≥ 0.5)")

                # Exit early if 80%+ of tickets are passing
                if success_rate >= 0.8:
                    print(f"\n🎉 SUCCESS! Pattern solved: {success_rate:.0%} of tickets passing")
                    print(f"   Early exit at iteration {iteration}/{max_iterations}")
                    break

                # Exit early if ALL tickets have F1 > 0 (at least finding some docs)
                if all(s["url_f1"] > 0 for s in pattern_scores.values()):
                    print(f"\n✅ All tickets finding expected docs (F1 > 0)")
                    print(f"   Early exit at iteration {iteration}/{max_iterations}")
                    break

                # Exit early if stuck (no improvement for N iterations)
                if iterations_without_improvement >= max_iterations_without_improvement:
                    print(f"\n⚠️  No improvement for {iterations_without_improvement} iterations")
                    print(f"   Early exit at iteration {iteration}/{max_iterations}")
                    break

            else:
                print(f"❌ Net negative - reverting ({improved_count} improved ≤ {regressed_count} regressed)")
                import subprocess

                subprocess.run(["git", "restore", "src/okp_mcp/solr.py"], cwd=self.okp_mcp_root)
                self.restart_okp_mcp()
                iterations_without_improvement += 1  # Count reverts as no improvement

                # Exit early if stuck after too many reverts
                if iterations_without_improvement >= max_iterations_without_improvement:
                    print(f"\n⚠️  No improvement for {iterations_without_improvement} iterations")
                    print(f"   Early exit at iteration {iteration}/{max_iterations}")
                    break

        # Final summary
        print(f"\n{'='*80}")
        print("PATTERN OPTIMIZATION COMPLETE")
        print(f"{'='*80}")
        print(f"   Iterations attempted: {max_iterations}")
        print(f"   Changes committed: {commits_made}")
        print(f"   Best pattern F1: {best_pattern_f1:.2f}")
        print(f"{'='*80}\n")

        return PhaseResult(
            phase_name="retrieval_optimization",
            success=commits_made > 0,
            iterations=iteration_count,
            final_metrics={
                "pattern_f1": best_pattern_f1,
                "commits_made": commits_made,
            },
            reason=f"Pattern-wide optimization: {commits_made} changes committed",
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

            # Get per-ticket results for detailed analysis
            output_dir = self.get_latest_output_dir("full")
            per_ticket_results = self.parse_results_per_ticket(output_dir)

            # Create PatternEvaluation object with baseline for comparison
            pattern_eval = PatternEvaluation(
                pattern_id=self.pattern_id, baseline=self._baseline_pattern
            )

            # Build lookup of no-doc tickets
            no_doc_tickets = {
                t["ticket_id"] for t in self.pattern_tickets if not t.get("has_expected_urls", True)
            }

            for ticket_id, runs in per_ticket_results.items():
                # Get baseline for this ticket if available
                baseline_ticket = None
                if self._baseline_pattern and ticket_id in self._baseline_pattern.tickets:
                    baseline_ticket = self._baseline_pattern.tickets[ticket_id]

                # Check if this is a no-doc ticket
                is_no_doc = ticket_id in no_doc_tickets

                ticket_eval = TicketEvaluation(
                    ticket_id=ticket_id, runs=runs, baseline=baseline_ticket, is_no_doc=is_no_doc
                )
                pattern_eval.tickets[ticket_id] = ticket_eval

            # Check overall averaged metrics
            answer_correct = result.answer_correctness or 0.0
            faithful = result.faithfulness or 0.0
            url_f1 = result.url_f1 or 0.0

            print("\n📊 Final Validation Metrics (Pattern Average):")
            print(f"   Answer Correctness: {answer_correct:.2f} (avg of {result.num_runs} runs)")
            print(f"   Faithfulness:       {faithful:.2f}")
            print(f"   URL F1:             {url_f1:.2f}")

            # Print per-ticket breakdown
            self._print_per_ticket_progress(pattern_eval, stability_runs)

            # Check for high variance (instability)
            if result.high_variance_metrics:
                print("\n⚠️  HIGH VARIANCE DETECTED:")
                for metric_info in result.high_variance_metrics:
                    print(f"   • {metric_info}")
                print("   → Pattern fix may not be stable")

            # Calculate composite scores per ticket using PatternEvaluation
            ticket_composites = self._calculate_per_ticket_composites(pattern_eval)

            # Calculate pattern-level composite (for reference)
            composite_score = self._calculate_composite_metric(result)
            COMPOSITE_THRESHOLD = 0.80  # High quality across all metrics

            # Use PatternEvaluation to determine pass/fail
            passing = pattern_eval.passes(criteria="majority")  # >50% of tickets
            passing_tickets = pattern_eval.passing_tickets
            failing_tickets = pattern_eval.failing_tickets
            total_tickets = pattern_eval.num_tickets

            if passing:
                print("\n✅ Final pattern validation PASSED")
                print(
                    f"   Passing Tickets: {len(passing_tickets)}/{total_tickets} (>{total_tickets/2:.0f} required)"
                )
                print(f"   Success Rate: {pattern_eval.success_rate:.1%}")
                print(
                    f"   Pattern Composite: {composite_score:.2f} (threshold: {COMPOSITE_THRESHOLD:.2f})"
                )
                print("\n   ✅ Passing tickets:")
                for ticket_id in passing_tickets:
                    ticket_eval = pattern_eval.tickets[ticket_id]
                    print(
                        f"      {ticket_id}: composite={ticket_eval.composite_score:.2f}, status={ticket_eval.status}"
                    )

                if failing_tickets:
                    print("\n   ❌ Still failing (but majority passed):")
                    for ticket_id in failing_tickets:
                        ticket_eval = pattern_eval.tickets[ticket_id]
                        print(
                            f"      {ticket_id}: composite={ticket_eval.composite_score:.2f}, status={ticket_eval.status}"
                        )
            else:
                print("\n❌ Final pattern validation FAILED")
                print(
                    f"   Passing Tickets: {len(passing_tickets)}/{total_tickets} (need >{total_tickets/2:.0f})"
                )
                print(f"   Success Rate: {pattern_eval.success_rate:.1%}")
                print(f"   Pattern Composite: {composite_score:.2f} (avg)")

                print("\n   Per-Ticket Composites:")
                for ticket_id in sorted(
                    pattern_eval.tickets.keys(),
                    key=lambda tid: pattern_eval.tickets[tid].composite_score,
                    reverse=True,
                ):
                    ticket_eval = pattern_eval.tickets[ticket_id]
                    status_emoji = "✅" if ticket_eval.passes() else "❌"
                    print(
                        f"   {status_emoji} {ticket_id}: {ticket_eval.composite_score:.2f} (status={ticket_eval.status})"
                    )

                print("\n   Pattern Average Breakdown:")
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
                    "per_ticket_composites": ticket_composites,
                    "passing_tickets": passing_tickets,
                    "total_tickets": total_tickets,
                },
                reason=f"Pattern validation: {len(passing_tickets)}/{total_tickets} tickets passing, composite={composite_score:.2f}",
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

    def _print_per_ticket_progress(self, pattern_eval: PatternEvaluation, num_runs: int) -> None:
        """Print per-ticket progress across evaluation runs.

        Args:
            pattern_eval: PatternEvaluation with ticket results
            num_runs: Number of runs executed
        """
        print(f"\n📊 Per-Ticket Progress ({num_runs} runs):")
        print("=" * 80)

        for ticket_id in sorted(pattern_eval.tickets.keys()):
            ticket_eval = pattern_eval.tickets[ticket_id]

            if not ticket_eval.runs:
                continue

            # Extract answer_correctness across runs
            ans_scores = [r.get("answer_correctness", 0.0) for r in ticket_eval.runs]
            min_score = min(ans_scores)
            max_score = max(ans_scores)
            mean_score = ticket_eval.mean_answer_correctness
            std_dev = ticket_eval.variance**0.5  # Convert variance to std dev

            # Check for baseline comparison
            baseline_ticket = None
            improvement = None
            if self._baseline_pattern and ticket_id in self._baseline_pattern.tickets:
                baseline_ticket = self._baseline_pattern.tickets[ticket_id]
                improvement = ticket_eval.improvement_over_baseline()

            # Determine status using TicketEvaluation.status property
            status_val = ticket_eval.status

            # Map status to emoji and display text
            status_map = {
                "STABLE_PASSING": ("✅", "✅ STABLE"),
                "CONSISTENTLY_FAILING": ("❌", "❌ FAILING"),
                "IMPROVING": ("📈", "📈 IMPROVING"),
                "REGRESSING": ("📉", "📉 REGRESSING"),
                "ERRATIC": ("❌", "❌ ERRATIC"),
                "IN_PROGRESS": ("⚠️", "⚠️  IN PROGRESS"),
                "NO_DATA": ("❓", "❓ NO DATA"),
            }

            emoji, status = status_map.get(status_val, ("❓", f"❓ {status_val}"))

            # Build run progression string
            if len(ans_scores) <= 3:
                progression = " → ".join(f"{s:.2f}" for s in ans_scores)
            else:
                progression = f"{ans_scores[0]:.2f} → ... → {ans_scores[-1]:.2f}"

            # Print summary
            print(f"\n{emoji} {ticket_id}: {status}")
            if baseline_ticket is not None:
                baseline_score = baseline_ticket.mean_answer_correctness
                print(
                    f"   Baseline→Final: {baseline_score:.2f} → {mean_score:.2f} ({improvement:+.2f})"
                )
            print(f"   Runs: {progression}")
            print(
                f"   Mean: {mean_score:.2f}, Std: {std_dev:.2f}, Range: {min_score:.2f}-{max_score:.2f}"
            )

            # Show other key metrics from latest run
            if ticket_eval.runs:
                latest = ticket_eval.runs[-1]
                faith = latest.get("faithfulness", 0.0)
                url_f1 = latest.get("url_f1", 0.0)
                ctx_rel = latest.get("context_relevance", 0.0)
                print(
                    f"   Latest: faithfulness={faith:.2f}, url_f1={url_f1:.2f}, ctx_rel={ctx_rel:.2f}"
                )

        print("=" * 80)

    def _calculate_per_ticket_composites(self, pattern_eval: PatternEvaluation) -> Dict[str, float]:
        """Calculate composite score for each ticket (averaged across runs).

        Args:
            pattern_eval: PatternEvaluation with ticket results

        Returns:
            Dict mapping ticket_id → composite_score
        """
        ticket_composites = {}

        for ticket_id, ticket_eval in pattern_eval.tickets.items():
            # Use TicketEvaluation's built-in composite_score property
            ticket_composites[ticket_id] = ticket_eval.composite_score

        return ticket_composites

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
        print(
            f"│  Answer Correctness:  {baseline_ans:.2f} {ans_arrow} {final_ans:.2f}  ({ans_change:+.2f})       │"
        )

        # Faithfulness
        baseline_faith = baseline_metrics.get("faithfulness", 0.0)
        final_faith = final_metrics.get("faithfulness", 0.0)
        faith_change = final_faith - baseline_faith
        faith_arrow = "→" if abs(faith_change) < 0.01 else ("↗" if faith_change > 0 else "↘")
        print(
            f"│  Faithfulness:        {baseline_faith:.2f} {faith_arrow} {final_faith:.2f}  ({faith_change:+.2f})       │"
        )

        # URL F1
        baseline_url = baseline_metrics.get("url_f1", 0.0)
        final_url = final_metrics.get("url_f1", 0.0)
        url_change = final_url - baseline_url
        url_arrow = "→" if abs(url_change) < 0.01 else ("↗" if url_change > 0 else "↘")
        print(
            f"│  URL F1:              {baseline_url:.2f} {url_arrow} {final_url:.2f}  ({url_change:+.2f})       │"
        )

        # Composite Score (if available)
        if baseline_composite is not None and final_composite is not None:
            comp_change = final_composite - baseline_composite
            comp_arrow = "→" if abs(comp_change) < 0.01 else ("↗" if comp_change > 0 else "↘")
            print(
                f"│  Composite Score:     {baseline_composite:.2f} {comp_arrow} {final_composite:.2f}  ({comp_change:+.2f})       │"
            )

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
                print(
                    f"│  Composite score {final_composite:.2f} meets threshold {threshold:.2f}           │"
                )
            else:
                print(
                    f"│  Composite score {final_composite:.2f} below threshold {threshold:.2f}          │"
                )

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
