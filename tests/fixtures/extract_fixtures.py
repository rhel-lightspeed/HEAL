#!/usr/bin/env python3
"""Extract test fixtures from real evaluation runs (NEW FORMAT with metadata).

Matches the data structure expected by parse_results_per_ticket() after
the scalar/vector bug fixes.

Usage:
    python tests/fixtures/extract_fixtures.py \
        --suite-dir okp_mcp_full_output/suite_20260416_233950 \
        --output tests/fixtures/bootloader_grub_pattern/baseline_FIXED.json \
        --pattern-id BOOTLOADER_GRUB_ISSUES
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def parse_metric_metadata(metadata_str: str) -> dict:
    """Parse metric_metadata JSON string."""
    try:
        return json.loads(metadata_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def extract_mrr_from_reason(reason: str) -> float | None:
    """Extract MRR value from reason text.

    Example: "URL retrieval: F1=0.33, ... Ranking: MRR=0.090, ..."
    """
    import re

    match = re.search(r"MRR=([\d.]+)", reason)
    if match:
        return float(match.group(1))
    return None


def load_expected_urls_from_pattern_yaml(pattern_id: str, ticket_id: str) -> List[str]:
    """Load expected URLs for a ticket from pattern YAML config.

    Args:
        pattern_id: Pattern ID (e.g., "BOOTLOADER_GRUB_ISSUES")
        ticket_id: Ticket ID (e.g., "RSPEED-1723")

    Returns:
        List of expected URLs, or empty list if not found
    """
    import yaml

    # Find pattern YAML file
    pattern_yaml = Path(f"config/patterns/{pattern_id}.yaml")
    if not pattern_yaml.exists():
        print(f"⚠️  Pattern YAML not found: {pattern_yaml}")
        return []

    try:
        with open(pattern_yaml) as f:
            pattern_data = yaml.safe_load(f)

        # Pattern YAML is a list of tickets
        if not isinstance(pattern_data, list):
            print(f"⚠️  Unexpected pattern YAML format (not a list): {pattern_yaml}")
            return []

        # Find ticket by conversation_group_id
        for ticket in pattern_data:
            if ticket.get("conversation_group_id") == ticket_id:
                # Get first turn's expected_urls
                turns = ticket.get("turns", [])
                if turns and isinstance(turns, list) and len(turns) > 0:
                    first_turn = turns[0]
                    expected_urls = first_turn.get("expected_urls", [])
                    return expected_urls if expected_urls else []

        print(f"⚠️  Ticket {ticket_id} not found in pattern YAML: {pattern_yaml}")
        return []

    except Exception as e:
        print(f"⚠️  Error loading pattern YAML {pattern_yaml}: {e}")
        return []


def extract_fixtures_from_suite(suite_dir: Path, pattern_id: str) -> Dict[str, Any]:
    """Extract fixture data from suite directory with multiple runs.

    Returns NEW data structure matching parse_results_per_ticket():
    {
        "pattern_id": str,
        "num_runs": int,
        "per_ticket_data": {
            "RSPEED-1234": {
                "runs": [
                    {  # Run 1 - simplified keys
                        "url_f1": 0.15,
                        "mrr": 0.09,
                        "answer_correctness": 0.62,
                        "faithfulness": 0.48,
                        ...
                    },
                    {...},  # Run 2
                    {...},  # Run 3
                ],
                "metadata": {
                    "tool_calls": "...",
                    "contexts": "...",
                    "expected_urls": [...],
                    "retrieved_urls": [...],
                    "rag_used": bool,
                    "docs_retrieved": bool,
                }
            },
            "RSPEED-5678": {...},
        }
    }
    """
    # Find all run directories
    run_dirs = sorted(suite_dir.glob("run_*"))
    if not run_dirs:
        raise RuntimeError(f"No run directories found in {suite_dir}")

    num_runs = len(run_dirs)

    # Collect data per ticket, per run
    ticket_data = defaultdict(lambda: {"runs": [], "metadata": None})

    for run_idx, run_dir in enumerate(run_dirs):
        csv_files = list(run_dir.glob("evaluation_*_detailed.csv"))
        if not csv_files:
            continue

        csv_path = csv_files[0]

        with open(csv_path) as f:
            reader = csv.DictReader(f)

            # Group rows by ticket
            ticket_rows = defaultdict(list)
            for row in reader:
                ticket_id = row["conversation_group_id"]
                ticket_rows[ticket_id].append(row)

            # Process each ticket
            for ticket_id, rows in ticket_rows.items():
                # Extract metadata from first row (same across all runs)
                if ticket_data[ticket_id]["metadata"] is None:
                    first_row = rows[0]

                    # Extract tool calls and contexts
                    tool_calls = first_row.get("tool_calls", "")
                    contexts = first_row.get("contexts", "")

                    # Parse retrieved URLs from tool_calls JSON
                    retrieved_urls = []
                    if tool_calls:
                        try:
                            tool_calls_data = json.loads(tool_calls)
                            if isinstance(tool_calls_data, list):
                                for turn_calls in tool_calls_data:
                                    if isinstance(turn_calls, list):
                                        for call in turn_calls:
                                            if isinstance(call, dict) and "result" in call:
                                                result = call["result"]
                                                if (
                                                    isinstance(result, dict)
                                                    and "contexts" in result
                                                ):
                                                    ctxs = result["contexts"]
                                                    if isinstance(ctxs, list):
                                                        for ctx in ctxs:
                                                            if isinstance(ctx, dict):
                                                                url = ctx.get("url", "")
                                                                if url:
                                                                    url_normalized = url.replace(
                                                                        "https://", ""
                                                                    ).replace("http://", "")
                                                                    retrieved_urls.append(
                                                                        url_normalized
                                                                    )
                        except (json.JSONDecodeError, TypeError, KeyError):
                            pass

                    # Check RAG usage
                    rag_used = False
                    if tool_calls:
                        tool_calls_lower = tool_calls.lower()
                        rag_used = any(
                            kw in tool_calls_lower for kw in ["search", "portal", "retrieve", "mcp"]
                        )

                    # Check docs retrieved
                    docs_retrieved = False
                    if contexts:
                        contexts_str = str(contexts).strip()
                        docs_retrieved = (
                            contexts_str != "" and contexts_str != "[]" and contexts_str != "null"
                        )

                    # Load expected URLs from pattern YAML
                    expected_urls = load_expected_urls_from_pattern_yaml(pattern_id, ticket_id)

                    # Store metadata (same across all runs)
                    ticket_data[ticket_id]["metadata"] = {
                        "tool_calls": tool_calls,
                        "contexts": contexts,
                        "expected_urls": expected_urls,
                        "retrieved_urls": retrieved_urls,
                        "rag_used": rag_used,
                        "docs_retrieved": docs_retrieved,
                    }

                # Extract metrics for this run (simplified keys matching NEW format)
                run_metrics = {}

                for row in rows:
                    metric_id = row["metric_identifier"]
                    score_str = row.get("score", "")
                    score = float(score_str) if score_str else None

                    # Map to simplified keys (matches parse_results_per_ticket NEW format)
                    if metric_id == "custom:url_retrieval_eval":
                        run_metrics["url_f1"] = score

                        # Extract MRR from reason
                        reason = row.get("reason", "")
                        if reason:
                            mrr = extract_mrr_from_reason(reason)
                            if mrr is not None:
                                run_metrics["mrr"] = mrr

                    elif metric_id == "ragas:context_relevance":
                        run_metrics["context_relevance"] = score
                    elif metric_id == "ragas:context_precision_without_reference":
                        run_metrics["context_precision"] = score
                    elif metric_id == "ragas:faithfulness":
                        run_metrics["faithfulness"] = score
                    elif metric_id == "custom:answer_correctness":
                        run_metrics["answer_correctness"] = score
                    elif metric_id == "ragas:response_relevancy":
                        run_metrics["response_relevancy"] = score

                ticket_data[ticket_id]["runs"].append(run_metrics)

    return {
        "pattern_id": pattern_id,
        "num_runs": num_runs,
        "per_ticket_data": dict(ticket_data),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract test fixtures (NEW format with metadata)")
    parser.add_argument(
        "--suite-dir",
        type=Path,
        required=True,
        help="Path to suite directory (e.g., okp_mcp_full_output/suite_20260416_233950)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON fixture path",
    )
    parser.add_argument(
        "--pattern-id",
        type=str,
        required=True,
        help="Pattern ID",
    )

    args = parser.parse_args()

    # Extract fixture
    print(f"📊 Extracting fixtures from: {args.suite_dir}")
    fixture = extract_fixtures_from_suite(args.suite_dir, args.pattern_id)

    # Ensure output dir exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write fixture
    with open(args.output, "w") as f:
        json.dump(fixture, f, indent=2)

    print(f"✅ Fixture written to: {args.output}")
    print(f"   Pattern: {fixture['pattern_id']}")
    print(f"   Runs: {fixture['num_runs']}")
    print(f"   Tickets: {len(fixture['per_ticket_data'])}")
    for ticket_id, data in fixture["per_ticket_data"].items():
        print(f"     • {ticket_id}: {len(data['runs'])} run(s)")
        print(f"       - RAG used: {data['metadata']['rag_used']}")
        print(f"       - Docs retrieved: {data['metadata']['docs_retrieved']}")


if __name__ == "__main__":
    main()
