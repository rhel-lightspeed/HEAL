#!/usr/bin/env python3
"""Compare quality of extracted_tickets.yaml vs extracted_tickets_rag.yaml.

Analyzes differences to determine if RAG-enhanced retrieval improves
the quality of expected answers in the bootstrapping process.

Metrics:
- Expected answer length (more detailed = better?)
- Number of retrieved URLs
- Unique vs duplicate answers
- Refinement iterations needed
- Review scores (if available)

Usage:
    python scripts/compare_extracted_yamls.py
    python scripts/compare_extracted_yamls.py --details
    python scripts/compare_extracted_yamls.py --ticket RSPEED-2482
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Add src/ to sys.path
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def load_yaml(path: Path) -> List[Dict[str, Any]]:
    """Load tickets from YAML file."""
    if not path.exists():
        print(f"❌ File not found: {path}")
        return []

    with open(path) as f:
        data = yaml.safe_load(f)

    return data.get("tickets", [])


def analyze_ticket(ticket: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze quality metrics for a single ticket."""
    turns = ticket.get("turns", [])

    # Get expected answer (last turn)
    expected_answer = ""
    retrieved_urls = []
    if turns:
        last_turn = turns[-1]
        expected_answer = last_turn.get("expected_response", "")
        retrieved_urls = last_turn.get("retrieved_urls", [])

    # Metadata
    metadata = ticket.get("metadata", {})
    refinement_iterations = metadata.get("refinement_iterations", 0)
    final_score = metadata.get("final_review_score", 0.0)

    return {
        "ticket_key": ticket.get("conversation_group_id", "UNKNOWN"),
        "answer_length": len(expected_answer),
        "answer_word_count": len(expected_answer.split()),
        "num_urls": len(retrieved_urls),
        "unique_urls": len(set(retrieved_urls)),
        "refinement_iterations": refinement_iterations,
        "final_score": final_score,
        "expected_answer": expected_answer,
        "retrieved_urls": retrieved_urls,
    }


def compare_tickets(
    baseline: List[Dict[str, Any]],
    rag: List[Dict[str, Any]],
    show_details: bool = False,
) -> None:
    """Compare tickets from baseline vs RAG extraction."""
    # Build lookup by ticket key
    baseline_map = {t.get("conversation_group_id"): t for t in baseline}
    rag_map = {t.get("conversation_group_id"): t for t in rag}

    # Find common tickets
    common_keys = set(baseline_map.keys()) & set(rag_map.keys())

    if not common_keys:
        print("❌ No common tickets found for comparison")
        print(f"   Baseline: {len(baseline)} tickets")
        print(f"   RAG: {len(rag)} tickets")
        return

    print(f"\n{'='*80}")
    print(f"YAML COMPARISON: Baseline vs RAG-Enhanced")
    print(f"{'='*80}")
    print(f"Common tickets: {len(common_keys)}")
    print()

    # Aggregate metrics
    baseline_metrics = []
    rag_metrics = []

    for key in sorted(common_keys):
        baseline_analysis = analyze_ticket(baseline_map[key])
        rag_analysis = analyze_ticket(rag_map[key])

        baseline_metrics.append(baseline_analysis)
        rag_metrics.append(rag_analysis)

        if show_details:
            print(f"\n{key}")
            print(f"  Answer Length:")
            print(f"    Baseline: {baseline_analysis['answer_length']} chars, "
                  f"{baseline_analysis['answer_word_count']} words")
            print(f"    RAG:      {rag_analysis['answer_length']} chars, "
                  f"{rag_analysis['answer_word_count']} words")
            print(f"    Δ: {rag_analysis['answer_length'] - baseline_analysis['answer_length']:+d} chars")
            print(f"  URLs Retrieved:")
            print(f"    Baseline: {baseline_analysis['num_urls']} "
                  f"({baseline_analysis['unique_urls']} unique)")
            print(f"    RAG:      {rag_analysis['num_urls']} "
                  f"({rag_analysis['unique_urls']} unique)")
            print(f"  Refinement Iterations:")
            print(f"    Baseline: {baseline_analysis['refinement_iterations']}")
            print(f"    RAG:      {rag_analysis['refinement_iterations']}")
            if baseline_analysis['final_score'] > 0 or rag_analysis['final_score'] > 0:
                print(f"  Final Review Score:")
                print(f"    Baseline: {baseline_analysis['final_score']:.2f}")
                print(f"    RAG:      {rag_analysis['final_score']:.2f}")

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")

    def avg(values, key):
        return sum(v[key] for v in values) / len(values) if values else 0

    print(f"\nAverage Answer Length:")
    print(f"  Baseline: {avg(baseline_metrics, 'answer_length'):.1f} chars, "
          f"{avg(baseline_metrics, 'answer_word_count'):.1f} words")
    print(f"  RAG:      {avg(rag_metrics, 'answer_length'):.1f} chars, "
          f"{avg(rag_metrics, 'answer_word_count'):.1f} words")
    delta_chars = avg(rag_metrics, 'answer_length') - avg(baseline_metrics, 'answer_length')
    print(f"  Δ: {delta_chars:+.1f} chars ({delta_chars/avg(baseline_metrics, 'answer_length')*100:+.1f}%)")

    print(f"\nAverage URLs Retrieved:")
    print(f"  Baseline: {avg(baseline_metrics, 'num_urls'):.1f}")
    print(f"  RAG:      {avg(rag_metrics, 'num_urls'):.1f}")
    delta_urls = avg(rag_metrics, 'num_urls') - avg(baseline_metrics, 'num_urls')
    print(f"  Δ: {delta_urls:+.1f}")

    print(f"\nAverage Refinement Iterations:")
    print(f"  Baseline: {avg(baseline_metrics, 'refinement_iterations'):.2f}")
    print(f"  RAG:      {avg(rag_metrics, 'refinement_iterations'):.2f}")
    delta_iter = avg(rag_metrics, 'refinement_iterations') - avg(baseline_metrics, 'refinement_iterations')
    print(f"  Δ: {delta_iter:+.2f}")

    # Check if review scores available
    baseline_scores = [m['final_score'] for m in baseline_metrics if m['final_score'] > 0]
    rag_scores = [m['final_score'] for m in rag_metrics if m['final_score'] > 0]

    if baseline_scores or rag_scores:
        print(f"\nAverage Final Review Score:")
        if baseline_scores:
            print(f"  Baseline: {sum(baseline_scores)/len(baseline_scores):.2f} "
                  f"({len(baseline_scores)}/{len(baseline_metrics)} tickets)")
        else:
            print(f"  Baseline: N/A")
        if rag_scores:
            print(f"  RAG:      {sum(rag_scores)/len(rag_scores):.2f} "
                  f"({len(rag_scores)}/{len(rag_metrics)} tickets)")
        else:
            print(f"  RAG:      N/A")

    # Interpretation
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")

    if delta_chars > 50:
        print("✅ RAG produces longer (more detailed) answers")
    elif delta_chars < -50:
        print("⚠️  RAG produces shorter answers (may be more concise or less detailed)")
    else:
        print("➡️  Answer length similar")

    if delta_urls > 0.5:
        print("✅ RAG retrieves more URLs (broader context)")
    elif delta_urls < -0.5:
        print("⚠️  RAG retrieves fewer URLs (may be more precise)")
    else:
        print("➡️  URL retrieval similar")

    if delta_iter < -0.2:
        print("✅ RAG requires fewer refinement iterations (better first-pass quality)")
    elif delta_iter > 0.2:
        print("⚠️  RAG requires more refinement iterations")
    else:
        print("➡️  Refinement iterations similar")


def compare_single_ticket(
    baseline: List[Dict[str, Any]],
    rag: List[Dict[str, Any]],
    ticket_key: str,
) -> None:
    """Detailed comparison of a single ticket."""
    # Find ticket in both
    baseline_ticket = next((t for t in baseline if t.get("conversation_group_id") == ticket_key), None)
    rag_ticket = next((t for t in rag if t.get("conversation_group_id") == ticket_key), None)

    if not baseline_ticket:
        print(f"❌ Ticket {ticket_key} not found in baseline")
        return
    if not rag_ticket:
        print(f"❌ Ticket {ticket_key} not found in RAG extraction")
        return

    print(f"\n{'='*80}")
    print(f"TICKET COMPARISON: {ticket_key}")
    print(f"{'='*80}\n")

    baseline_analysis = analyze_ticket(baseline_ticket)
    rag_analysis = analyze_ticket(rag_ticket)

    print("BASELINE EXTRACTION")
    print("-" * 80)
    print(f"Answer Length: {baseline_analysis['answer_length']} chars, "
          f"{baseline_analysis['answer_word_count']} words")
    print(f"URLs Retrieved: {baseline_analysis['num_urls']} ({baseline_analysis['unique_urls']} unique)")
    print(f"Refinement Iterations: {baseline_analysis['refinement_iterations']}")
    print(f"URLs: {', '.join(baseline_analysis['retrieved_urls'][:3])}...")
    print(f"\nExpected Answer (first 200 chars):")
    print(baseline_analysis['expected_answer'][:200])

    print(f"\n{'='*80}")
    print("RAG EXTRACTION")
    print("-" * 80)
    print(f"Answer Length: {rag_analysis['answer_length']} chars, "
          f"{rag_analysis['answer_word_count']} words")
    print(f"URLs Retrieved: {rag_analysis['num_urls']} ({rag_analysis['unique_urls']} unique)")
    print(f"Refinement Iterations: {rag_analysis['refinement_iterations']}")
    print(f"URLs: {', '.join(rag_analysis['retrieved_urls'][:3])}...")
    print(f"\nExpected Answer (first 200 chars):")
    print(rag_analysis['expected_answer'][:200])

    print(f"\n{'='*80}")
    print("DIFFERENCES")
    print("-" * 80)
    print(f"Δ Answer Length: {rag_analysis['answer_length'] - baseline_analysis['answer_length']:+d} chars")
    print(f"Δ URLs: {rag_analysis['num_urls'] - baseline_analysis['num_urls']:+d}")
    print(f"Δ Refinement: {rag_analysis['refinement_iterations'] - baseline_analysis['refinement_iterations']:+d}")

    # URL differences
    baseline_urls = set(baseline_analysis['retrieved_urls'])
    rag_urls = set(rag_analysis['retrieved_urls'])

    only_in_baseline = baseline_urls - rag_urls
    only_in_rag = rag_urls - baseline_urls
    common_urls = baseline_urls & rag_urls

    print(f"\nURL Comparison:")
    print(f"  Common: {len(common_urls)}")
    print(f"  Only in baseline: {len(only_in_baseline)}")
    print(f"  Only in RAG: {len(only_in_rag)}")

    if only_in_rag:
        print(f"\n  New URLs from RAG:")
        for url in list(only_in_rag)[:3]:
            print(f"    + {url}")


def main():
    """Main comparison workflow."""
    parser = argparse.ArgumentParser(
        description="Compare baseline vs RAG-enhanced YAML quality"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=REPO_ROOT / "config" / "extracted_tickets.yaml",
        help="Baseline YAML path",
    )
    parser.add_argument(
        "--rag",
        type=Path,
        default=REPO_ROOT / "config" / "extracted_tickets_rag.yaml",
        help="RAG YAML path",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show per-ticket details",
    )
    parser.add_argument(
        "--ticket",
        type=str,
        help="Compare single ticket in detail",
    )

    args = parser.parse_args()

    # Load YAMLs
    print("Loading YAMLs...")
    baseline_tickets = load_yaml(args.baseline)
    rag_tickets = load_yaml(args.rag)

    if not baseline_tickets:
        print(f"❌ No baseline tickets found in {args.baseline}")
        return

    if not rag_tickets:
        print(f"❌ No RAG tickets found in {args.rag}")
        print(f"\nTo generate RAG extraction:")
        print(f"  python src/heal/bootstrap/extract_jira_tickets_rag.py --tickets RSPEED-XXXX")
        return

    # Compare
    if args.ticket:
        compare_single_ticket(baseline_tickets, rag_tickets, args.ticket)
    else:
        compare_tickets(baseline_tickets, rag_tickets, show_details=args.details)


if __name__ == "__main__":
    main()
