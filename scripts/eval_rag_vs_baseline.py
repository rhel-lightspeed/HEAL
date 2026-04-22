#!/usr/bin/env python3
"""Run lightspeed-evaluation on baseline vs RAG-extracted tickets for comparison.

This script:
1. Finds common tickets between baseline and RAG YAMLs
2. Creates separate eval files for each
3. Runs lightspeed-evaluation on both
4. Compares the metrics (answer correctness, URL F1, faithfulness)

Usage:
    python scripts/eval_rag_vs_baseline.py
    python scripts/eval_rag_vs_baseline.py --tickets RSPEED-1930,RSPEED-1929
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Paths
SCRIPT_DIR = Path(__file__).parent
HEAL_ROOT = SCRIPT_DIR.parent
CONFIG_DIR = HEAL_ROOT / "config"

BASELINE_YAML = CONFIG_DIR / "extracted_tickets_backup_20260421_123412.yaml"
RAG_YAML = CONFIG_DIR / "extracted_tickets_rag.yaml"

# Output paths for test files
TEST_DIR = HEAL_ROOT / ".test_rag_comparison"
BASELINE_TEST = TEST_DIR / "baseline_test.yaml"
RAG_TEST = TEST_DIR / "rag_test.yaml"


def load_tickets(yaml_path: Path) -> List[Dict[str, Any]]:
    """Load tickets from YAML file."""
    if not yaml_path.exists():
        print(f"❌ File not found: {yaml_path}")
        return []

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    tickets = data.get("tickets", [])
    print(f"Loaded {len(tickets)} tickets from {yaml_path.name}")
    return tickets


def find_common_tickets(
    baseline_tickets: List[Dict[str, Any]],
    rag_tickets: List[Dict[str, Any]],
    specific_tickets: List[str] = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Find tickets that exist in both baseline and RAG extractions.

    Args:
        baseline_tickets: Tickets from baseline extraction
        rag_tickets: Tickets from RAG extraction
        specific_tickets: Optional list of specific ticket IDs to compare

    Returns:
        (baseline_common, rag_common) - matching tickets from each YAML
    """
    # Build lookup by ticket ID
    baseline_map = {
        t.get("conversation_group_id"): t for t in baseline_tickets
    }
    rag_map = {
        t.get("conversation_group_id"): t for t in rag_tickets
    }

    # Find intersection
    if specific_tickets:
        common_ids = set(specific_tickets)
        print(f"\nUsing specific tickets: {common_ids}")
    else:
        common_ids = set(baseline_map.keys()) & set(rag_map.keys())
        print(f"\nFound {len(common_ids)} common tickets:")
        for ticket_id in sorted(common_ids):
            print(f"  - {ticket_id}")

    # Extract common tickets in same order
    baseline_common = [baseline_map[tid] for tid in sorted(common_ids) if tid in baseline_map]
    rag_common = [rag_map[tid] for tid in sorted(common_ids) if tid in rag_map]

    return baseline_common, rag_common


def strip_heal_fields(ticket: Dict[str, Any]) -> Dict[str, Any]:
    """Strip HEAL-specific fields to make ticket compatible with lightspeed-evaluation.

    Args:
        ticket: HEAL ticket dictionary

    Returns:
        Cleaned ticket with only lightspeed-evaluation compatible fields
    """
    # Fields to exclude (HEAL-specific)
    exclude_fields = {
        "review_score", "review_notes", "url_validation",
        "iteration", "feedback_loop", "pattern_id", "jira_key",
        "api_input_tokens", "api_output_tokens"
    }

    # Clean conversation level
    cleaned = {k: v for k, v in ticket.items() if k not in exclude_fields}

    # Clean turn level
    if "turns" in cleaned:
        cleaned_turns = []
        for turn in cleaned["turns"]:
            cleaned_turn = {k: v for k, v in turn.items() if k not in exclude_fields}
            cleaned_turns.append(cleaned_turn)
        cleaned["turns"] = cleaned_turns

    return cleaned


def create_eval_yaml(tickets: List[Dict[str, Any]], output_path: Path, label: str) -> None:
    """Create evaluation YAML file from tickets.

    Args:
        tickets: List of ticket dictionaries
        output_path: Where to save the YAML
        label: Label for metadata (baseline or rag)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip HEAL-specific fields (review_score, review_notes, url_validation, etc.)
    cleaned_tickets = [strip_heal_fields(t) for t in tickets]

    # lightspeed-evaluation expects a flat list, not wrapped in metadata/tickets
    with open(output_path, "w") as f:
        yaml.dump(cleaned_tickets, f, default_flow_style=False, sort_keys=False)

    print(f"Created {output_path} with {len(tickets)} tickets")


def run_evaluation(
    yaml_path: Path,
    label: str,
    eval_root: Path,
) -> Path:
    """Run lightspeed-evaluation on a YAML file.

    Args:
        yaml_path: Path to evaluation YAML
        label: Label for this run (baseline or rag)
        eval_root: Path to lightspeed-evaluation repo

    Returns:
        Path to results CSV
    """
    print(f"\n{'='*80}")
    print(f"Running evaluation: {label}")
    print(f"{'='*80}")

    # Output directory for results
    results_dir = HEAL_ROOT / ".test_rag_comparison" / f"{label}_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # System config from lightspeed-evaluation repo
    system_config = eval_root / "config" / "system.yaml"

    # Build command - use lightspeed-eval script entry point
    cmd = [
        "uv", "run", "lightspeed-eval",
        "--system-config", str(system_config),
        "--eval-data", str(yaml_path),
        "--output-dir", str(results_dir),
        "--cache-warmup",  # Force fresh evaluation (don't reuse cached results)
    ]

    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {eval_root}")

    # Set GOOGLE_APPLICATION_CREDENTIALS for lightspeed-evaluation subprocess only
    # (Claude Agent SDK uses ADC from gcloud auth, but ls-eval needs explicit path)
    # Load from lightspeed-evaluation's .env file to isolate from parent process
    import os
    from dotenv import dotenv_values

    # Start with clean environment copy
    eval_env = os.environ.copy()

    # CRITICAL: Remove GOOGLE_APPLICATION_CREDENTIALS if present in parent env
    # (prevents breaking Claude Agent SDK which uses ADC)
    if "GOOGLE_APPLICATION_CREDENTIALS" in eval_env:
        del eval_env["GOOGLE_APPLICATION_CREDENTIALS"]

    # Load lightspeed-evaluation's .env file
    ls_eval_env_file = eval_root / ".env"
    if ls_eval_env_file.exists():
        ls_eval_env = dotenv_values(ls_eval_env_file)

        # Apply only GOOGLE_APPLICATION_CREDENTIALS from .env to subprocess
        if "GOOGLE_APPLICATION_CREDENTIALS" in ls_eval_env:
            eval_env["GOOGLE_APPLICATION_CREDENTIALS"] = ls_eval_env["GOOGLE_APPLICATION_CREDENTIALS"]
            print(f"✅ Using GOOGLE_APPLICATION_CREDENTIALS from {ls_eval_env_file} (subprocess only)")
        else:
            print(f"⚠️  No GOOGLE_APPLICATION_CREDENTIALS in {ls_eval_env_file}")
    else:
        print(f"⚠️  No .env file found at {ls_eval_env_file}")
        print("   lightspeed-evaluation may fail without credentials")

    try:
        result = subprocess.run(
            cmd,
            cwd=eval_root,
            capture_output=True,
            text=True,
            check=True,
            env=eval_env,  # Pass environment with GOOGLE_APPLICATION_CREDENTIALS
        )
        print(result.stdout)

        # Find results CSV (lightspeed-evaluation creates *_detailed.csv)
        csv_files = list(results_dir.glob("**/*_detailed.csv"))
        if csv_files:
            results_csv = csv_files[0]
            print(f"✅ Results saved to: {results_csv}")
            return results_csv
        else:
            print(f"⚠️  No results CSV found in {results_dir}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None


def compare_results(baseline_csv: Path, rag_csv: Path) -> None:
    """Compare evaluation results from baseline vs RAG.

    Args:
        baseline_csv: Path to baseline results CSV
        rag_csv: Path to RAG results CSV
    """
    import pandas as pd

    print(f"\n{'='*80}")
    print("COMPARISON: Baseline vs RAG-Enhanced")
    print(f"{'='*80}\n")

    if not baseline_csv or not baseline_csv.exists():
        print("❌ Baseline results not found")
        return
    if not rag_csv or not rag_csv.exists():
        print("❌ RAG results not found")
        return

    # Load CSVs (long format: one row per metric per conversation)
    baseline_df = pd.read_csv(baseline_csv)
    rag_df = pd.read_csv(rag_csv)

    # Pivot to wide format: one row per conversation with metric columns
    def pivot_metrics(df):
        """Pivot long-format metrics to wide format."""
        return df.pivot_table(
            index='conversation_group_id',
            columns='metric_identifier',
            values='score',
            aggfunc='first'
        ).reset_index()

    baseline_wide = pivot_metrics(baseline_df)
    rag_wide = pivot_metrics(rag_df)

    # Metrics to compare (using full identifiers from CSV)
    metrics = {
        "custom:answer_correctness": "Answer Correctness",
        "ragas:faithfulness": "Faithfulness",
        "ragas:context_relevance": "Context Relevance",
        "ragas:response_relevancy": "Response Relevancy",
    }

    print("Average Metrics:")
    print("-" * 80)

    for metric_id, metric_name in metrics.items():
        if metric_id in baseline_wide.columns and metric_id in rag_wide.columns:
            baseline_avg = baseline_wide[metric_id].mean()
            rag_avg = rag_wide[metric_id].mean()
            delta = rag_avg - baseline_avg

            print(f"\n{metric_name}:")
            print(f"  Baseline: {baseline_avg:.3f}")
            print(f"  RAG:      {rag_avg:.3f}")
            print(f"  Δ:        {delta:+.3f} ({delta/baseline_avg*100:+.1f}%)")

            if delta > 0.05:
                print(f"  ✅ RAG significantly better")
            elif delta < -0.05:
                print(f"  ❌ RAG significantly worse")
            else:
                print(f"  ➡️  Similar performance")

    # Per-ticket comparison
    print(f"\n{'='*80}")
    print("Per-Ticket Comparison:")
    print("-" * 80)

    # Merge baseline and RAG on conversation_group_id
    merged = baseline_wide.merge(
        rag_wide,
        on='conversation_group_id',
        suffixes=('_baseline', '_rag')
    )

    for _, row in merged.iterrows():
        ticket_id = row['conversation_group_id']
        print(f"\n{ticket_id}:")

        for metric_id, metric_name in metrics.items():
            baseline_col = f"{metric_id}_baseline" if f"{metric_id}_baseline" in row else metric_id
            rag_col = f"{metric_id}_rag" if f"{metric_id}_rag" in row else metric_id

            if baseline_col in row and rag_col in row:
                baseline_val = row[baseline_col]
                rag_val = row[rag_col]

                if pd.notna(baseline_val) and pd.notna(rag_val):
                    delta = rag_val - baseline_val
                    print(f"  {metric_name}: {baseline_val:.3f} → {rag_val:.3f} ({delta:+.3f})")

    print(f"\n{'='*80}")
    print("Summary:")
    print("-" * 80)
    print(f"Baseline CSV: {baseline_csv}")
    print(f"RAG CSV:      {rag_csv}")


def main():
    """Main comparison workflow."""
    parser = argparse.ArgumentParser(
        description="Compare baseline vs RAG-enhanced extraction quality"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=BASELINE_YAML,
        help="Baseline YAML path",
    )
    parser.add_argument(
        "--rag",
        type=Path,
        default=RAG_YAML,
        help="RAG YAML path",
    )
    parser.add_argument(
        "--tickets",
        type=str,
        help="Comma-separated list of specific tickets to compare",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path.home() / "Work/lightspeed-core/lightspeed-evaluation",
        help="Path to lightspeed-evaluation repo",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip running evaluation, just compare existing results",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.baseline.exists():
        print(f"❌ Baseline YAML not found: {args.baseline}")
        return 1
    if not args.rag.exists():
        print(f"❌ RAG YAML not found: {args.rag}")
        return 1
    if not args.eval_root.exists():
        print(f"❌ lightspeed-evaluation not found: {args.eval_root}")
        return 1

    # Parse specific tickets if provided
    specific_tickets = None
    if args.tickets:
        specific_tickets = [t.strip() for t in args.tickets.split(",")]

    # Step 1: Load tickets
    print(f"\n{'='*80}")
    print("STEP 1: Loading tickets")
    print(f"{'='*80}")

    baseline_tickets = load_tickets(args.baseline)
    rag_tickets = load_tickets(args.rag)

    if not baseline_tickets or not rag_tickets:
        print("❌ Failed to load tickets")
        return 1

    # Step 2: Find common tickets
    print(f"\n{'='*80}")
    print("STEP 2: Finding common tickets")
    print(f"{'='*80}")

    baseline_common, rag_common = find_common_tickets(
        baseline_tickets,
        rag_tickets,
        specific_tickets,
    )

    if not baseline_common or not rag_common:
        print("❌ No common tickets found")
        return 1

    # Step 3: Create test YAMLs
    print(f"\n{'='*80}")
    print("STEP 3: Creating test YAMLs")
    print(f"{'='*80}")

    create_eval_yaml(baseline_common, BASELINE_TEST, "baseline")
    create_eval_yaml(rag_common, RAG_TEST, "rag")

    if args.skip_eval:
        print("\nSkipping evaluation (--skip-eval)")
        # Try to find existing results
        baseline_results = list((TEST_DIR / "baseline_results").glob("**/*_detailed.csv"))
        rag_results = list((TEST_DIR / "rag_results").glob("**/*_detailed.csv"))

        if baseline_results and rag_results:
            compare_results(baseline_results[0], rag_results[0])
        else:
            print("❌ No existing results found to compare")
        return 0

    # Step 4: Run evaluations
    print(f"\n{'='*80}")
    print("STEP 4: Running evaluations")
    print(f"{'='*80}")

    baseline_csv = run_evaluation(BASELINE_TEST, "baseline", args.eval_root)
    rag_csv = run_evaluation(RAG_TEST, "rag", args.eval_root)

    # Step 5: Compare results
    if baseline_csv and rag_csv:
        compare_results(baseline_csv, rag_csv)
    else:
        print("❌ Could not compare - missing results")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
