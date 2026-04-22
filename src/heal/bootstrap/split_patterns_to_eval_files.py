#!/usr/bin/env python3
"""Split pattern-tagged tickets into separate YAML files for evaluation.

Takes tickets_with_patterns.yaml (already in evaluation format) and splits
into one YAML per pattern, ready for lightspeed-evaluation.

Usage:
    python src/heal/bootstrap/split_patterns_to_eval_files.py \
        --input config/tickets_with_patterns.yaml \
        --patterns config/patterns_report.json \
        --output-dir config/patterns/

Output:
    - One YAML per pattern (e.g., PATTERN_ID.yaml)
    - UNGROUPED.yaml (tickets not in any pattern)

No LLM calls - just reads YAML and reorganizes by pattern_id.
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_inputs(tagged_file: Path, patterns_file: Path) -> tuple:
    """Load input files.

    Args:
        tagged_file: Path to tickets_with_patterns.yaml
        patterns_file: Path to patterns_report.json

    Returns:
        (tagged_tickets, patterns_data)
    """
    logger.info("Loading inputs...")

    if not tagged_file.exists():
        logger.error(f"Tagged tickets file not found: {tagged_file}")
        logger.error("Run pattern discovery first: ./runners/pattern.sh")
        sys.exit(1)

    if not patterns_file.exists():
        logger.error(f"Patterns report file not found: {patterns_file}")
        logger.error("Run pattern discovery first: ./runners/pattern.sh")
        sys.exit(1)

    with open(tagged_file, encoding="utf-8") as f:
        tagged_data = yaml.safe_load(f)
        tagged_tickets = tagged_data.get("tickets", [])

    with open(patterns_file, encoding="utf-8") as f:
        patterns_data = json.load(f)

    logger.info(f"  Loaded {len(tagged_tickets)} tagged tickets")
    logger.info(f"  Loaded {len(patterns_data.get('patterns', []))} patterns")

    return tagged_tickets, patterns_data


def group_tickets_by_pattern(tickets: list) -> dict:
    """Group tickets by pattern_id.

    Args:
        tickets: List of tagged tickets (already in evaluation format)

    Returns:
        Dict mapping pattern_id → List[ticket]
    """
    grouped = defaultdict(list)

    for ticket in tickets:
        pattern_id = ticket.get("pattern_id")

        if pattern_id:
            # Has pattern assignment
            grouped[pattern_id].append(ticket)
        else:
            # Ungrouped
            grouped["UNGROUPED"].append(ticket)

    return grouped


def write_pattern_yaml(
    pattern_id: str,
    tickets: list,
    output_dir: Path,
    patterns_data: dict | None = None,
):
    """Write pattern-specific YAML file.

    Args:
        pattern_id: Pattern identifier (or UNGROUPED)
        tickets: List of tickets in this pattern (already in eval format)
        output_dir: Output directory
        patterns_data: Pattern metadata from patterns_report.json
    """
    # Remove pattern_id field from each ticket (not part of eval format)
    clean_tickets = []
    for ticket in tickets:
        clean_ticket = {k: v for k, v in ticket.items() if k != "pattern_id"}
        clean_tickets.append(clean_ticket)

    # Build header comment
    if pattern_id == "UNGROUPED":
        header = f"# Ungrouped Tickets - No Pattern Match\n# Total tickets: {len(tickets)}\n"
    else:
        # Find pattern metadata
        pattern_meta = None
        if patterns_data:
            for p in patterns_data.get("patterns", []):
                if p["pattern_id"] == pattern_id:
                    pattern_meta = p
                    break

        if pattern_meta:
            header = (
                f"# Pattern: {pattern_id}\n"
                f"# Description: {pattern_meta.get('description', 'N/A')}\n"
                f"# Total tickets: {len(tickets)}\n"
                f"# Problem Type: {pattern_meta.get('common_problem_type', 'N/A')}\n"
                f"# Components: {', '.join(pattern_meta.get('common_components', []))}\n"
            )
        else:
            header = f"# Pattern: {pattern_id}\n# Total tickets: {len(tickets)}\n"

    # Write YAML
    output_file = output_dir / f"{pattern_id}.yaml"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n")
        yaml.dump(clean_tickets, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info(f"  ✅ {output_file.name}: {len(tickets)} tickets")


def main():
    """Main split workflow."""
    parser = argparse.ArgumentParser(
        description="Split pattern-tagged tickets into separate YAML files for evaluation"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("config/tickets_with_patterns.yaml"),
        help="Path to tickets_with_patterns.yaml (default: config/tickets_with_patterns.yaml)",
    )
    parser.add_argument(
        "--patterns",
        type=Path,
        default=Path("config/patterns_report.json"),
        help="Path to patterns_report.json (default: config/patterns_report.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("config/patterns"),
        help="Output directory for pattern YAMLs (default: config/patterns)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    tagged_tickets, patterns_data = load_inputs(args.input, args.patterns)

    # Group tickets by pattern
    logger.info("Grouping tickets by pattern...")
    grouped = group_tickets_by_pattern(tagged_tickets)

    pattern_count = len([k for k in grouped.keys() if k != "UNGROUPED"])
    ungrouped_count = len(grouped.get("UNGROUPED", []))

    logger.info(f"  Patterns: {pattern_count}")
    logger.info(f"  Ungrouped: {ungrouped_count}")

    # Write pattern YAMLs
    logger.info("Writing pattern YAMLs...")

    for pattern_id, pattern_tickets in sorted(grouped.items()):
        write_pattern_yaml(pattern_id, pattern_tickets, args.output_dir, patterns_data)

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SPLIT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Total files created: {len(grouped)}")
    logger.info("")
    logger.info("Pattern files:")
    for pattern_id, pattern_tickets in sorted(grouped.items()):
        if pattern_id != "UNGROUPED":
            logger.info(f"  - {pattern_id}.yaml ({len(pattern_tickets)} tickets)")
    logger.info("")
    if "UNGROUPED" in grouped:
        logger.info(f"  - UNGROUPED.yaml ({len(grouped['UNGROUPED'])} tickets)")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review pattern YAMLs in config/patterns/")
    logger.info("  2. Test with lightspeed-evaluation:")
    logger.info("       cd $LIGHTSPEED_EVAL_ROOT")
    logger.info("       uv run python -m lightspeed_evaluation.runner \\")
    logger.info("         --config config/system_okp_mcp_agent.yaml \\")
    logger.info(f"         --data <HEAL_ROOT>/{args.output_dir}/<PATTERN_ID>.yaml \\")
    logger.info("         --metrics ragas:context_relevance,custom:answer_correctness")
    logger.info("")
    logger.info("  (Set LIGHTSPEED_EVAL_ROOT and HEAL_ROOT in your .env or use absolute paths)")
    logger.info("")


if __name__ == "__main__":
    main()
