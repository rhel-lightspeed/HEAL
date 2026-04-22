#!/usr/bin/env python3
"""Debug runner for testing multi-agent optimization logic with fixtures.

This script creates a CHECKPOINT system for rapid iteration:

1. ✅ SKIP baseline evaluation (20+ minutes)
2. ✅ LOAD fixture data from previous run (instant)
3. ✅ START from optimization phase (multi-agent)
4. ✅ TEST actual Solr suggestions
5. ✅ ITERATE on multi-agent logic without waiting

Usage:
    # Use existing fixture (okp-mcp path auto-detected from env or ../okp-mcp)
    python src/heal/runners/fix_agent_debugger.py \
        --fixture tests/fixtures/bootloader_grub_pattern/baseline_FIXED.json

    # Or specify custom path
    python src/heal/runners/fix_agent_debugger.py \
        --fixture tests/fixtures/bootloader_grub_pattern/baseline_FIXED.json \
        --okp-mcp-root /path/to/okp-mcp

    # Or use wrapper script
    ./runners/debug.sh BOOTLOADER_GRUB_ISSUES

Benefits:
    - Test multi-agent logic in SECONDS instead of 30+ minutes
    - Iterate on Solr Expert + Code Expert + Synthesizer
    - No need to re-run slow baseline evaluations
    - Perfect for debugging pattern fix loop logic
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from heal.agents.okp_mcp_agent import (
    OkpMcpAgent,
    PatternEvaluationResult,
    EvaluationResult,
)
from heal.agents.solr_multi_agent import SolrMultiAgentSystem


class FixAgentDebugger:
    """Debug runner for testing multi-agent optimization with fixtures."""

    def __init__(
        self,
        fixture_path: Path,
        okp_mcp_root: Path,
        eval_root: Optional[Path] = None,
        lscore_deploy_root: Optional[Path] = None,
    ):
        """Initialize debug runner.

        Args:
            fixture_path: Path to baseline fixture JSON
            okp_mcp_root: Path to okp-mcp repository
            eval_root: Path to lightspeed-evaluation (optional)
            lscore_deploy_root: Path to lscore-deploy (optional)
        """
        self.fixture_path = fixture_path
        self.okp_mcp_root = okp_mcp_root

        # Set defaults using HEALConfig
        from heal.core.config import HEALConfig

        if eval_root is None:
            eval_root = HEALConfig.get_lightspeed_eval_root()
        if lscore_deploy_root is None:
            lscore_deploy_root = HEALConfig.get_lscore_deploy_root()

        self.eval_root = eval_root
        self.lscore_deploy_root = lscore_deploy_root

        # Load fixture
        print(f"📂 Loading fixture: {fixture_path}")
        with open(fixture_path) as f:
            self.fixture = json.load(f)

        self.pattern_id = self.fixture["pattern_id"]
        self.num_runs = self.fixture["num_runs"]
        self.per_ticket_data = self.fixture["per_ticket_data"]

        print(f"✅ Loaded fixture for pattern: {self.pattern_id}")
        print(f"   Runs: {self.num_runs}")
        print(f"   Tickets: {len(self.per_ticket_data)}")

        # Initialize multi-agent system
        print("\n🤖 Initializing multi-agent Solr optimization system...")
        try:
            self.multi_agent = SolrMultiAgentSystem(
                okp_mcp_root=okp_mcp_root,
                model="claude-sonnet-4-6",
            )
            print("✅ Multi-agent system ready (Solr Expert + Code Expert + Synthesizer)")
        except Exception as e:
            print(f"⚠️  Multi-agent system failed to initialize: {e}")
            print("   Continuing without multi-agent optimization")
            self.multi_agent = None

    def build_baseline_result(self) -> PatternEvaluationResult:
        """Build PatternEvaluationResult from fixture data.

        Returns:
            PatternEvaluationResult as if baseline just completed
        """
        print("\n📊 Building baseline result from fixture...")

        # Build EvaluationResult for each ticket
        per_ticket_results = {}

        for ticket_id, ticket_data in self.per_ticket_data.items():
            runs = ticket_data["runs"]
            metadata = ticket_data["metadata"]

            # Calculate averages across runs
            averages = {}
            metrics_by_name: Dict[str, list] = {}

            for run in runs:
                for metric_name, value in run.items():
                    if value is not None:
                        if metric_name not in metrics_by_name:
                            metrics_by_name[metric_name] = []
                        metrics_by_name[metric_name].append(value)

            # Average each metric
            for name, values in metrics_by_name.items():
                averages[name] = sum(values) / len(values)

            # Build EvaluationResult
            result = EvaluationResult(
                ticket_id=ticket_id,
                # Averaged metrics
                url_f1=averages.get("url_f1"),
                mrr=averages.get("mrr"),
                context_relevance=averages.get("context_relevance"),
                context_precision=averages.get("context_precision"),
                faithfulness=averages.get("faithfulness"),
                answer_correctness=averages.get("answer_correctness"),
                response_relevancy=averages.get("response_relevancy"),
                num_runs=len(runs),
                # Metadata
                tool_calls=metadata.get("tool_calls"),
                contexts=metadata.get("contexts"),
                expected_urls=metadata.get("expected_urls", []),
                retrieved_urls=metadata.get("retrieved_urls", []),
                rag_used=metadata.get("rag_used", False),
                docs_retrieved=metadata.get("docs_retrieved", False),
            )

            per_ticket_results[ticket_id] = result

        # Calculate pattern-level aggregates
        url_f1_values = [r.url_f1 for r in per_ticket_results.values() if r.url_f1 is not None]
        answer_values = [
            r.answer_correctness
            for r in per_ticket_results.values()
            if r.answer_correctness is not None
        ]
        faith_values = [
            r.faithfulness for r in per_ticket_results.values() if r.faithfulness is not None
        ]

        pattern_url_f1 = sum(url_f1_values) / len(url_f1_values) if url_f1_values else 0.0
        pattern_answer = sum(answer_values) / len(answer_values) if answer_values else 0.0
        pattern_faith = sum(faith_values) / len(faith_values) if faith_values else 0.0

        # Classify tickets
        passing_tickets = []
        failing_tickets = []
        rag_bypass_tickets = []

        composite_threshold = 0.80

        for ticket_id, result in per_ticket_results.items():
            # Calculate composite
            if result.answer_correctness is not None:
                composite = (
                    0.80 * (result.answer_correctness or 0)
                    + 0.15 * (result.context_relevance or 0)
                    + 0.05 * (result.context_precision or 0)
                )
            else:
                composite = (result.url_f1 or 0) * 0.5 + (result.context_relevance or 0) * 0.5

            if composite >= composite_threshold:
                passing_tickets.append(ticket_id)
            else:
                failing_tickets.append(ticket_id)

            # RAG bypass detection
            if not result.rag_used or (result.rag_used and not result.docs_retrieved):
                rag_bypass_tickets.append(ticket_id)

        success_rate = len(passing_tickets) / len(per_ticket_results) if per_ticket_results else 0.0

        # Pattern composite
        pattern_composite = (
            0.80 * pattern_answer + 0.15 * pattern_url_f1 + 0.05 * pattern_faith
            if pattern_answer > 0
            else pattern_url_f1 * 0.5
        )

        baseline = PatternEvaluationResult(
            pattern_id=self.pattern_id,
            num_runs=self.num_runs,
            per_ticket_results=per_ticket_results,
            pattern_url_f1=pattern_url_f1,
            pattern_answer_correctness=pattern_answer,
            pattern_faithfulness=pattern_faith,
            pattern_composite_score=pattern_composite,
            success_rate=success_rate,
            passing_tickets=passing_tickets,
            failing_tickets=failing_tickets,
            rag_bypass_tickets=rag_bypass_tickets,
            high_variance_tickets=[],
            high_variance_metrics=[],
        )

        return baseline

    def display_baseline(self, baseline: PatternEvaluationResult):
        """Display baseline metrics (as if from real evaluation)."""
        print("\n" + "=" * 80)
        print(f"📊 BASELINE METRICS (from fixture)")
        print("=" * 80)
        print(f"   Pattern: {baseline.pattern_id}")
        print(f"   Tickets Evaluated: {len(baseline.per_ticket_results)}")
        print(f"   URL F1 (avg):             {baseline.pattern_url_f1:.2f}")
        print(f"   Answer Correctness (avg): {baseline.pattern_answer_correctness:.2f}")
        print(f"   Faithfulness (avg):       {baseline.pattern_faithfulness:.2f}")
        print(f"   Success Rate:             {baseline.success_rate:.0%}")
        print()

        if baseline.rag_bypass_tickets:
            print(f"⚠️  RAG Bypass: {len(baseline.rag_bypass_tickets)} ticket(s)")
            for ticket_id in baseline.rag_bypass_tickets:
                print(f"      • {ticket_id}")
            print()

        # Show per-ticket breakdown
        print("📋 Per-Ticket Breakdown:")
        print(f"{'Ticket':<15} {'URL F1':<8} {'Answer':<8} {'Faith':<8} {'RAG':<6}")
        print("-" * 80)

        for ticket_id, result in baseline.per_ticket_results.items():
            url_f1_str = f"{result.url_f1:.2f}" if result.url_f1 is not None else "N/A"
            answer_str = (
                f"{result.answer_correctness:.2f}"
                if result.answer_correctness is not None
                else "N/A"
            )
            faith_str = f"{result.faithfulness:.2f}" if result.faithfulness is not None else "N/A"
            rag_str = "✅" if result.docs_retrieved else "❌"

            print(f"{ticket_id:<15} {url_f1_str:<8} {answer_str:<8} {faith_str:<8} {rag_str:<6}")

        print("=" * 80 + "\n")

    async def test_multi_agent_optimization(self, baseline: PatternEvaluationResult):
        """Test multi-agent Solr optimization logic.

        This is the CHECKPOINT - start optimization phase directly!
        """
        print("\n" + "=" * 80)
        print("🎯 PHASE 2: MULTI-AGENT PATTERN OPTIMIZATION (DEBUG MODE)")
        print("=" * 80)

        if not self.multi_agent:
            print("⚠️  Multi-agent system not available - skipping")
            return

        if not baseline.failing_tickets:
            print("✅ No failing tickets - pattern already passing!")
            return

        print(f"\n🎯 Analyzing ENTIRE PATTERN: {self.pattern_id}")
        print(f"   Failing tickets: {len(baseline.failing_tickets)}")
        for ticket_id in baseline.failing_tickets:
            result = baseline.per_ticket_results[ticket_id]
            f1_str = f"{result.url_f1:.2f}" if result.url_f1 is not None else "N/A"
            mrr_str = f"{result.mrr:.2f}" if result.mrr is not None else "N/A"
            print(f"      • {ticket_id}: F1={f1_str}, MRR={mrr_str}")

        # Build TicketData for ALL failing tickets
        from heal.agents.solr_multi_agent import TicketData

        failing_ticket_data = []
        for ticket_id in baseline.failing_tickets:
            result = baseline.per_ticket_results[ticket_id]

            # Get query from fixture metadata (would normally come from test config YAML)
            # For now, use placeholder - TODO: Load from test config
            query = f"How to fix {ticket_id.replace('RSPEED-', '')} issue?"

            ticket_data = TicketData(
                ticket_id=ticket_id,
                query=query,
                expected_urls=result.expected_urls,
                retrieved_urls=result.retrieved_urls,
                metrics={
                    "url_f1": result.url_f1 or 0.0,
                    "mrr": result.mrr or 0.0,
                    "context_relevance": result.context_relevance or 0.0,
                },
            )
            failing_ticket_data.append(ticket_data)

        print(f"\n🤖 Consulting multi-agent system for PATTERN analysis...")
        print("   Phase 1: Solr Expert (pattern-level theory)...")
        print("   Phase 2: Code Expert (implementation constraints)...")
        print("   Phase 3: Synthesizer (pattern-level fix)...")

        # Call multi-agent system with ALL failing tickets
        try:
            suggestion = await self.multi_agent.get_optimized_suggestion(
                pattern_id=self.pattern_id,
                failing_tickets=failing_ticket_data,
            )

            print("\n✅ Multi-agent PATTERN analysis complete!")
            print("=" * 80)
            print(f"📊 Pattern: {self.pattern_id}")
            print(f"🎯 Tickets Addressed: {len(baseline.failing_tickets)}")
            print()
            print(f"📝 Suggested Change: {suggestion.suggested_change}")
            print(f"📁 File: {suggestion.file_path}")
            print(f"🎯 Confidence: {suggestion.confidence:.0%}")
            print()
            print("Old Code:")
            print("-" * 40)
            print(suggestion.old_code)
            print()
            print("New Code:")
            print("-" * 40)
            print(suggestion.new_code)
            print()
            print("Reasoning (Pattern-Level):")
            print("-" * 40)
            print(suggestion.reasoning)
            print()
            if suggestion.risks:
                print("⚠️  Risks:")
                for risk in suggestion.risks:
                    print(f"   • {risk}")
            print()
            print(
                f"💡 This fix should improve retrieval for ALL {len(baseline.failing_tickets)} tickets in the pattern"
            )
            print("=" * 80)

            # Save suggestion to JSON for automatic application
            suggestion_dir = Path(".diagnostics") / self.pattern_id
            suggestion_dir.mkdir(parents=True, exist_ok=True)
            suggestion_file = suggestion_dir / "suggestion.json"

            suggestion_data = {
                "pattern_id": self.pattern_id,
                "suggested_change": suggestion.suggested_change,
                "file_path": suggestion.file_path,
                "old_code": suggestion.old_code,
                "new_code": suggestion.new_code,
                "reasoning": suggestion.reasoning,
                "confidence": suggestion.confidence,
                "risks": suggestion.risks,
            }

            with open(suggestion_file, "w") as f:
                json.dump(suggestion_data, f, indent=2)

            print(f"\n💾 Suggestion saved to: {suggestion_file}")
            print(
                f"   Use './runners/eval_fix.sh {self.pattern_id} --apply' to apply automatically"
            )

        except Exception as e:
            print(f"❌ Multi-agent optimization failed: {e}")
            import traceback

            traceback.print_exc()

    def run(self):
        """Run the debug session.

        Flow:
        1. Load baseline from fixture (instant)
        2. Display baseline metrics
        3. Run multi-agent optimization (REAL)
        4. Show suggested changes
        """
        print("\n" + "=" * 80)
        print("🔧 FIX AGENT DEBUGGER - Pattern Fix Loop Checkpoint")
        print("=" * 80)
        print(f"Pattern: {self.pattern_id}")
        print(f"Fixture: {self.fixture_path.name}")
        print(f"Mode: DEBUG (skip baseline, load from fixture)")
        print("=" * 80)

        # Step 1: Build baseline from fixture
        baseline = self.build_baseline_result()

        # Step 2: Display baseline
        self.display_baseline(baseline)

        # Step 3: Test multi-agent optimization
        import asyncio

        asyncio.run(self.test_multi_agent_optimization(baseline))

        print("\n✅ Debug session complete!")
        print("\n💡 Next steps:")
        print("   1. Review multi-agent suggestion above")
        print("   2. Iterate on Solr Expert / Code Expert / Synthesizer prompts")
        print("   3. Re-run this script to test changes instantly")
        print("   4. When ready, run full pattern fix loop with real evaluations")


def main():
    parser = argparse.ArgumentParser(
        description="Debug runner for multi-agent optimization (checkpoint mode)"
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        required=True,
        help="Path to baseline fixture JSON",
    )
    parser.add_argument(
        "--okp-mcp-root",
        type=Path,
        default=None,
        help="Path to okp-mcp repository (auto-detected from OKP_MCP_ROOT env var or ../okp-mcp)",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=None,
        help="Path to lightspeed-evaluation (optional)",
    )
    parser.add_argument(
        "--lscore-deploy-root",
        type=Path,
        default=None,
        help="Path to lscore-deploy (optional)",
    )

    args = parser.parse_args()

    # Auto-detect paths if not provided
    from heal.core.config import HEALConfig

    if args.okp_mcp_root is None:
        args.okp_mcp_root = HEALConfig.get_okp_mcp_root()
        if args.okp_mcp_root is None:
            print("❌ OKP-MCP repository not found")
            print("   Set OKP_MCP_ROOT env var or use --okp-mcp-root")
            sys.exit(1)

    # Validate paths
    if not args.fixture.exists():
        print(f"❌ Fixture not found: {args.fixture}")
        print("\n💡 To create a fixture, run:")
        print("   python tests/fixtures/extract_fixtures.py \\")
        print("       --suite-dir okp_mcp_full_output/suite_20260416_233950 \\")
        print("       --output tests/fixtures/my_pattern/baseline.json \\")
        print("       --pattern-id MY_PATTERN")
        sys.exit(1)

    if not args.okp_mcp_root.exists():
        print(f"❌ okp-mcp root not found: {args.okp_mcp_root}")
        sys.exit(1)

    # Run debug session
    debugger = FixAgentDebugger(
        fixture_path=args.fixture,
        okp_mcp_root=args.okp_mcp_root,
        eval_root=args.eval_root,
        lscore_deploy_root=args.lscore_deploy_root,
    )

    debugger.run()


if __name__ == "__main__":
    main()
