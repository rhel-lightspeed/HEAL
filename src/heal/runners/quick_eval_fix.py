#!/usr/bin/env python3
"""Quick fix evaluation - test if code change improves answer correctness.

This is the Python component called by runners/eval_fix.sh

Usage:
    python src/heal/runners/quick_eval_fix.py PATTERN_ID \\
        --baseline-fixture tests/fixtures/.../baseline_FIXED.json \\
        --runs 2
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from heal.agents.okp_mcp_agent import OkpMcpAgent, PatternEvaluationResult


async def load_baseline_metrics(fixture_path: Path) -> Dict[str, float]:
    """Load baseline metrics from fixture.

    Returns:
        Dict with pattern-level baseline metrics
    """
    with open(fixture_path) as f:
        data = json.load(f)

    per_ticket = data["per_ticket_data"]

    # Calculate pattern-level averages
    answer_scores = []
    url_f1_scores = []

    for ticket_id, ticket_data in per_ticket.items():
        runs = ticket_data["runs"]

        # Average answer_correctness across runs for this ticket
        answer_values = [
            r.get("answer_correctness") for r in runs if r.get("answer_correctness") is not None
        ]
        if answer_values:
            avg_answer = sum(answer_values) / len(answer_values)
            answer_scores.append(avg_answer)
            print(f"  Baseline {ticket_id}: Answer={avg_answer:.2f}")

        # Average url_f1 across runs for this ticket
        url_f1_values = [r.get("url_f1") for r in runs if r.get("url_f1") is not None]
        if url_f1_values:
            avg_url_f1 = sum(url_f1_values) / len(url_f1_values)
            url_f1_scores.append(avg_url_f1)

    baseline = {}
    if answer_scores:
        baseline["answer_correctness"] = sum(answer_scores) / len(answer_scores)
    if url_f1_scores:
        baseline["url_f1"] = sum(url_f1_scores) / len(url_f1_scores)

    return baseline


async def run_quick_eval(
    pattern_id: str,
    num_runs: int,
    okp_mcp_root: Path,
    eval_root: Path,
    lscore_deploy_root: Path,
) -> Dict[str, float]:
    """Run quick evaluation and return metrics.

    Returns:
        Dict with pattern-level metrics from new evaluation
    """
    print(f"\n🔄 Running evaluation with {num_runs} runs...")
    print(f"   Pattern: {pattern_id}")
    print()

    agent = OkpMcpAgent(
        pattern_id=pattern_id,
        okp_mcp_root=okp_mcp_root,
        eval_root=eval_root,
        lscore_deploy_root=lscore_deploy_root,
    )

    # Run retrieval-only diagnosis (faster - no LLM judge evals)
    result = agent.diagnose_retrieval_only(
        ticket_id=None,  # Full pattern
        runs=num_runs,
    )

    # Extract metrics
    new_metrics = {}

    if isinstance(result, PatternEvaluationResult):
        # Pattern-level result
        new_metrics["answer_correctness"] = result.pattern_answer_correctness or 0.0
        new_metrics["url_f1"] = result.pattern_url_f1 or 0.0

        # Show per-ticket breakdown
        print("\n📊 Per-Ticket Results:")
        for ticket_id, ticket_result in result.per_ticket_results.items():
            answer = ticket_result.answer_correctness or 0.0
            url_f1 = ticket_result.url_f1 or 0.0
            print(f"  {ticket_id}: Answer={answer:.2f}, URL F1={url_f1:.2f}")
    else:
        # Single ticket result (shouldn't happen with ticket_id=None)
        new_metrics["answer_correctness"] = result.answer_correctness or 0.0
        new_metrics["url_f1"] = result.url_f1 or 0.0

    return new_metrics


async def main():
    parser = argparse.ArgumentParser(
        description="Quick fix evaluation - test if code change improves metrics"
    )
    parser.add_argument("pattern_id", help="Pattern ID (e.g., BOOTLOADER_GRUB_ISSUES)")
    parser.add_argument(
        "--baseline-fixture",
        required=True,
        type=Path,
        help="Path to baseline fixture JSON file",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Number of evaluation runs (default: 2 for speed)",
    )
    parser.add_argument(
        "--okp-mcp-root",
        type=Path,
        help="Path to okp-mcp repository",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        help="Path to lightspeed-evaluation repository",
    )
    parser.add_argument(
        "--lscore-deploy-root",
        type=Path,
        help="Path to lscore-deploy repository",
    )

    args = parser.parse_args()

    # Auto-detect paths if not provided using HEALConfig
    from heal.core.config import HEALConfig

    if not args.okp_mcp_root:
        args.okp_mcp_root = HEALConfig.get_okp_mcp_root()

    if not args.eval_root:
        args.eval_root = HEALConfig.get_lightspeed_eval_root()

    if not args.lscore_deploy_root:
        args.lscore_deploy_root = HEALConfig.get_lscore_deploy_root()

    # Validate paths
    if not args.baseline_fixture.exists():
        print(f"❌ Baseline fixture not found: {args.baseline_fixture}", file=sys.stderr)
        sys.exit(1)

    if not args.okp_mcp_root or not args.okp_mcp_root.exists():
        print("❌ okp-mcp root not found", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("QUICK FIX EVALUATION")
    print("=" * 80)
    print(f"Pattern: {args.pattern_id}")
    print(f"Runs: {args.runs}")
    print(f"Baseline: {args.baseline_fixture}")
    print(f"OKP-MCP: {args.okp_mcp_root}")
    print("=" * 80)

    # Step 1: Load baseline
    print("\n📊 Step 1: Load baseline metrics from fixture")
    baseline = await load_baseline_metrics(args.baseline_fixture)

    print("\n" + "-" * 80)
    print("BASELINE METRICS")
    print("-" * 80)
    if "answer_correctness" in baseline:
        print(f"  Answer Correctness: {baseline['answer_correctness']:.3f}")
    if "url_f1" in baseline:
        print(f"  URL F1: {baseline['url_f1']:.3f}")
    print("-" * 80)

    # Step 2: Run new evaluation
    print("\n📊 Step 2: Run evaluation with modified okp-mcp")
    new_metrics = await run_quick_eval(
        pattern_id=args.pattern_id,
        num_runs=args.runs,
        okp_mcp_root=args.okp_mcp_root,
        eval_root=args.eval_root,
        lscore_deploy_root=args.lscore_deploy_root,
    )

    print("\n" + "-" * 80)
    print("NEW METRICS")
    print("-" * 80)
    if "answer_correctness" in new_metrics:
        print(f"  Answer Correctness: {new_metrics['answer_correctness']:.3f}")
    if "url_f1" in new_metrics:
        print(f"  URL F1: {new_metrics['url_f1']:.3f}")
    print("-" * 80)

    # Step 3: Compare
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    if "answer_correctness" in baseline and "answer_correctness" in new_metrics:
        delta_answer = new_metrics["answer_correctness"] - baseline["answer_correctness"]
        print(f"\nAnswer Correctness:")
        print(f"  Baseline: {baseline['answer_correctness']:.3f}")
        print(f"  New:      {new_metrics['answer_correctness']:.3f}")
        print(f"  Delta:    {delta_answer:+.3f}")

        if delta_answer > 0.05:
            print(f"  ✅ IMPROVEMENT (+{delta_answer:.3f})")
        elif delta_answer < -0.05:
            print(f"  ❌ REGRESSION ({delta_answer:.3f})")
        else:
            print(f"  ➖ NO SIGNIFICANT CHANGE ({delta_answer:+.3f})")

    if "url_f1" in baseline and "url_f1" in new_metrics:
        delta_f1 = new_metrics["url_f1"] - baseline["url_f1"]
        print(f"\nURL F1:")
        print(f"  Baseline: {baseline['url_f1']:.3f}")
        print(f"  New:      {new_metrics['url_f1']:.3f}")
        print(f"  Delta:    {delta_f1:+.3f}")

        if delta_f1 > 0.05:
            print(f"  ✅ IMPROVEMENT (+{delta_f1:.3f})")
        elif delta_f1 < -0.05:
            print(f"  ❌ REGRESSION ({delta_f1:.3f})")
        else:
            print(f"  ➖ NO SIGNIFICANT CHANGE ({delta_f1:+.3f})")

    print("\n" + "=" * 80)
    print()


if __name__ == "__main__":
    asyncio.run(main())
