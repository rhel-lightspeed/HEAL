#!/usr/bin/env python3
"""Convert baseline fixture data to validation checkpoint format.

This allows us to test correlation analysis without re-running expensive pattern fixes.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def convert_fixture_to_checkpoints(fixture_path: Path, pattern_id: str) -> Path:
    """Convert baseline fixture to validation checkpoint format.

    Args:
        fixture_path: Path to baseline_FIXED.json fixture file
        pattern_id: Pattern ID (e.g., "BOOTLOADER_GRUB_ISSUES")

    Returns:
        Path to created validation checkpoint file
    """
    # Load fixture
    with open(fixture_path) as f:
        fixture_data = json.load(f)

    # Output file
    output_dir = Path(".claude/fix_patterns")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{pattern_id}_validation_checkpoints.jsonl"

    # Extract per-ticket runs and convert to checkpoint format
    per_ticket_data = fixture_data.get("per_ticket_data", {})

    if not per_ticket_data:
        print(f"❌ No per_ticket_data found in {fixture_path}")
        return None

    checkpoints_written = 0

    with open(output_file, "w") as out:
        for ticket_id, ticket_data in per_ticket_data.items():
            runs = ticket_data.get("runs", [])

            if not runs:
                continue

            # Treat each run as a "cycle" (simulating optimization iterations)
            for cycle, run_data in enumerate(runs, start=1):
                # Skip runs with null/missing answer_correctness
                if run_data.get("answer_correctness") is None:
                    continue

                # Build checkpoint record
                checkpoint = {
                    "cycle": cycle,
                    "timestamp": datetime.now().isoformat(),
                    "ticket_id": ticket_id,
                    # Baseline is run 1, current is this run
                    "baseline_url_f1": runs[0].get("url_f1", 0.0),
                    "current_url_f1": run_data.get("url_f1", 0.0),
                    "url_f1_delta": run_data.get("url_f1", 0.0) - runs[0].get("url_f1", 0.0),
                    "baseline_answer_correctness": runs[0].get("answer_correctness", 0.0),
                    "current_answer_correctness": run_data.get("answer_correctness", 0.0),
                    "answer_correctness_delta": run_data.get("answer_correctness", 0.0) - runs[0].get("answer_correctness", 0.0),
                    "faithfulness": run_data.get("faithfulness"),
                }

                # Add optional metrics
                if run_data.get("context_relevance") is not None:
                    checkpoint["context_relevance"] = run_data["context_relevance"]
                if run_data.get("context_precision") is not None:
                    checkpoint["context_precision"] = run_data["context_precision"]
                if run_data.get("mrr") is not None:
                    checkpoint["baseline_mrr"] = runs[0].get("mrr", 0.0)
                    checkpoint["current_mrr"] = run_data["mrr"]
                    checkpoint["mrr_delta"] = run_data["mrr"] - runs[0].get("mrr", 0.0)

                # Write checkpoint
                out.write(json.dumps(checkpoint) + "\n")
                checkpoints_written += 1

    print(f"✅ Created {output_file}")
    print(f"   Wrote {checkpoints_written} validation checkpoints")
    print(f"   Tickets: {len(per_ticket_data)}")

    return output_file


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python convert_fixture_to_validation_checkpoints.py PATTERN_ID [FIXTURE_FILE]")
        print()
        print("Examples:")
        print("  python scripts/convert_fixture_to_validation_checkpoints.py BOOTLOADER_GRUB_ISSUES")
        print("  python scripts/convert_fixture_to_validation_checkpoints.py BOOTLOADER_GRUB_ISSUES tests/fixtures/bootloader_grub_pattern/baseline_FIXED.json")
        sys.exit(1)

    pattern_id = sys.argv[1]

    # Find fixture file
    if len(sys.argv) >= 3:
        fixture_path = Path(sys.argv[2])
    else:
        # Auto-detect fixture path
        pattern_lower = pattern_id.lower().replace("-", "_")
        fixture_path = Path(f"tests/fixtures/{pattern_lower}_pattern/baseline_FIXED.json")

        # Try alternative naming
        if not fixture_path.exists():
            pattern_lower = pattern_id.lower().replace("_", "_")
            fixture_path = Path(f"tests/fixtures/bootloader_grub_pattern/baseline_FIXED.json")

    if not fixture_path.exists():
        print(f"❌ Fixture not found: {fixture_path}")
        print()
        print("Available fixtures:")
        for p in Path("tests/fixtures").glob("*_pattern/baseline_FIXED.json"):
            print(f"  - {p}")
        sys.exit(1)

    print(f"Converting fixture: {fixture_path}")
    print(f"Pattern ID: {pattern_id}")
    print()

    output_file = convert_fixture_to_checkpoints(fixture_path, pattern_id)

    if output_file:
        print()
        print(f"✅ Done! Now run correlation analysis:")
        print(f"   python scripts/analyze_metric_correlations.py {pattern_id}")


if __name__ == "__main__":
    main()
