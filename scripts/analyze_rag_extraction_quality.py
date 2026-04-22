#!/usr/bin/env python3
"""Comprehensive analysis of RAG vs Baseline extraction quality.

Compares 4 "LLM responses" for each ticket:
1. Baseline expected_response (extracted with non-RAG Solr)
2. RAG expected_response (extracted with RAG Solr)
3. CLA actual response (baseline eval run)
4. CLA actual response (RAG eval run)

For each response, we measure:
- Content relevance of retrieved docs
- Answer correctness (judge evaluation)
- Faithfulness to sources

This tells us:
- Which extraction method produces better expected answers
- Whether expected answers are better than actual CLA responses
- Whether better retrieval leads to better answers
"""

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

# Add src to path
SCRIPT_DIR = Path(__file__).parent
HEAL_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(HEAL_ROOT))

from src.heal.agents.answer_review_agent import AnswerReviewAgent
from src.heal.agents.linux_expert import LinuxExpertAgent


@dataclass
class ResponseVariant:
    """One response variant to analyze."""

    source: str  # "baseline_expected", "rag_expected", "baseline_actual", "rag_actual"
    query: str
    response: str
    contexts: List[str]
    urls: List[str]

    # Scores from AnswerReviewAgent
    reviewer_score: float = 0.0
    reviewer_correctness: float = 0.0
    reviewer_faithfulness: float = 0.0
    reviewer_notes: str = ""

    # Scores from LinuxExpertAgent
    expert_score: float = 0.0
    expert_correctness: float = 0.0
    expert_faithfulness: float = 0.0
    expert_notes: str = ""

    # Combined scores (average of both judges)
    combined_score: float = 0.0
    combined_correctness: float = 0.0
    combined_faithfulness: float = 0.0

    # Other metrics
    content_relevance: float = 0.0


@dataclass
class TicketAnalysis:
    """Analysis for one ticket across all 4 response variants."""

    ticket_id: str
    query: str
    variants: List[ResponseVariant]

    def best_by_metric(self, metric: str = 'combined_score') -> ResponseVariant:
        """Get best variant by given metric."""
        return max(self.variants, key=lambda v: getattr(v, metric))

    def rank_by_metric(self, metric: str = 'combined_score') -> List[tuple[str, float]]:
        """Rank variants by metric."""
        return sorted(
            [(v.source, getattr(v, metric)) for v in self.variants],
            key=lambda x: x[1],
            reverse=True
        )


def load_comparison_data(
    baseline_yaml: Path,
    rag_yaml: Path,
    baseline_csv: Path,
    rag_csv: Path,
) -> List[TicketAnalysis]:
    """Load all comparison data and organize by ticket.

    Args:
        baseline_yaml: Baseline extracted tickets
        rag_yaml: RAG extracted tickets
        baseline_csv: Baseline evaluation results
        rag_csv: RAG evaluation results

    Returns:
        List of TicketAnalysis objects
    """
    # Load YAMLs (expected answers)
    with open(baseline_yaml) as f:
        baseline_data = yaml.safe_load(f)
    with open(rag_yaml) as f:
        rag_data = yaml.safe_load(f)

    baseline_tickets = {t['conversation_group_id']: t for t in baseline_data['tickets']}
    rag_tickets = {t['conversation_group_id']: t for t in rag_data['tickets']}

    # Load CSVs (actual CLA responses)
    baseline_df = pd.read_csv(baseline_csv)
    rag_df = pd.read_csv(rag_csv)

    # Get one row per conversation
    baseline_convs = baseline_df.groupby('conversation_group_id').first().reset_index()
    rag_convs = rag_df.groupby('conversation_group_id').first().reset_index()

    # Build analysis objects
    analyses = []

    common_ids = set(baseline_tickets.keys()) & set(rag_tickets.keys())

    for ticket_id in sorted(common_ids):
        baseline_ticket = baseline_tickets[ticket_id]
        rag_ticket = rag_tickets[ticket_id]

        baseline_conv = baseline_convs[baseline_convs['conversation_group_id'] == ticket_id].iloc[0]
        rag_conv = rag_convs[rag_convs['conversation_group_id'] == ticket_id].iloc[0]

        query = baseline_ticket['turns'][0]['query']

        variants = []

        # Variant 1: Baseline expected
        variants.append(ResponseVariant(
            source="baseline_expected",
            query=query,
            response=baseline_ticket['turns'][0]['expected_response'],
            contexts=parse_contexts(baseline_ticket['turns'][0].get('contexts', [])),
            urls=baseline_ticket['turns'][0].get('expected_urls', []),
        ))

        # Variant 2: RAG expected
        variants.append(ResponseVariant(
            source="rag_expected",
            query=query,
            response=rag_ticket['turns'][0]['expected_response'],
            contexts=parse_contexts(rag_ticket['turns'][0].get('contexts', [])),
            urls=rag_ticket['turns'][0].get('expected_urls', []),
        ))

        # Variant 3: Baseline actual CLA
        variants.append(ResponseVariant(
            source="baseline_actual_cla",
            query=query,
            response=str(baseline_conv['response']),
            contexts=parse_contexts_from_csv(baseline_conv['contexts']),
            urls=extract_urls_from_contexts(parse_contexts_from_csv(baseline_conv['contexts'])),
        ))

        # Variant 4: RAG actual CLA
        variants.append(ResponseVariant(
            source="rag_actual_cla",
            query=query,
            response=str(rag_conv['response']),
            contexts=parse_contexts_from_csv(rag_conv['contexts']),
            urls=extract_urls_from_contexts(parse_contexts_from_csv(rag_conv['contexts'])),
        ))

        analyses.append(TicketAnalysis(
            ticket_id=ticket_id,
            query=query,
            variants=variants,
        ))

    return analyses


def parse_contexts(contexts: Any) -> List[str]:
    """Parse contexts from YAML (already a list)."""
    if isinstance(contexts, list):
        return contexts
    return []


def parse_contexts_from_csv(contexts_str: Any) -> List[str]:
    """Parse contexts from CSV (JSON string)."""
    if pd.isna(contexts_str):
        return []
    try:
        return eval(str(contexts_str))
    except:
        return []


def extract_urls_from_contexts(contexts: List[str]) -> List[str]:
    """Extract URLs from context strings."""
    urls = []
    for ctx in contexts:
        # Context format includes "URL: https://..."
        for line in str(ctx).split('\n'):
            if line.startswith('URL: '):
                urls.append(line.replace('URL: ', '').strip())
    return urls


async def analyze_variant(
    variant: ResponseVariant,
    reviewer: AnswerReviewAgent,
    linux_expert: LinuxExpertAgent
) -> None:
    """Analyze one response variant with panel of judges.

    Uses two judges:
    1. AnswerReviewAgent - Specialized answer quality reviewer
    2. LinuxExpertAgent - Technical domain expert

    Args:
        variant: ResponseVariant to analyze
        reviewer: AnswerReviewAgent instance
        linux_expert: LinuxExpertAgent instance
    """
    # Judge 1: AnswerReviewAgent
    try:
        review = await reviewer.review_answer(
            query=variant.query,
            expected_response=variant.response,
            sources=variant.urls,  # Use URLs not contexts
        )

        # ReviewResult is a dataclass with pass_fail, score, issues, etc.
        variant.reviewer_score = review.score
        variant.reviewer_correctness = review.score  # AnswerReviewAgent doesn't split these out
        variant.reviewer_faithfulness = review.score
        variant.reviewer_notes = "; ".join(review.issues[:2]) if review.issues else "Pass"[:100]

    except Exception as e:
        print(f"      ⚠️  AnswerReviewAgent error: {e}")
        variant.reviewer_score = 0.5
        variant.reviewer_correctness = 0.5
        variant.reviewer_faithfulness = 0.5
        variant.reviewer_notes = f"Error: {str(e)[:50]}"

    # Judge 2: LinuxExpertAgent
    try:
        evaluation = await linux_expert.evaluate_answer(
            query=variant.query,
            answer=variant.response,
            contexts=variant.contexts,
        )

        variant.expert_score = evaluation.get('overall_score', 0.5)
        variant.expert_correctness = evaluation.get('correctness', 0.5)
        variant.expert_faithfulness = evaluation.get('faithfulness', 0.5)
        variant.expert_notes = evaluation.get('notes', '')[:100]

        # Check if we got fallback scores (parsing failed)
        if variant.expert_score == 0.5 and variant.expert_notes == "Failed to parse evaluation response":
            print(f"      ⚠️  LinuxExpert parsing failed (use --debug to see response)")

    except Exception as e:
        print(f"      ⚠️  LinuxExpertAgent error: {e}")
        variant.expert_score = 0.5
        variant.expert_correctness = 0.5
        variant.expert_faithfulness = 0.5
        variant.expert_notes = f"Error: {str(e)[:50]}"

    # Combine scores (average of both judges)
    variant.combined_score = (variant.reviewer_score + variant.expert_score) / 2
    variant.combined_correctness = (variant.reviewer_correctness + variant.expert_correctness) / 2
    variant.combined_faithfulness = (variant.reviewer_faithfulness + variant.expert_faithfulness) / 2

    # Content relevance (simplified)
    if variant.contexts and len(variant.contexts) > 0:
        total_length = sum(len(str(c)) for c in variant.contexts)
        variant.content_relevance = min(0.9, total_length / 1000)  # Scale by content length
    else:
        variant.content_relevance = 0.0


async def analyze_all_tickets(analyses: List[TicketAnalysis]) -> None:
    """Run analysis on all tickets with panel of judges.

    Args:
        analyses: List of TicketAnalysis objects to populate
    """
    print(f"\n{'='*80}")
    print("ANALYZING RESPONSE VARIANTS WITH PANEL OF JUDGES")
    print(f"{'='*80}")
    print("  Judge 1: AnswerReviewAgent (specialized answer reviewer)")
    print("  Judge 2: LinuxExpertAgent (technical domain expert)")
    print()

    reviewer = AnswerReviewAgent()
    linux_expert = LinuxExpertAgent()

    total_variants = len(analyses) * 4
    processed = 0

    for analysis in analyses:
        print(f"\nAnalyzing {analysis.ticket_id}...")

        for variant in analysis.variants:
            await analyze_variant(variant, reviewer, linux_expert)
            processed += 1
            print(f"  [{processed}/{total_variants}] {variant.source}: "
                  f"Reviewer={variant.reviewer_score:.2f} "
                  f"Expert={variant.expert_score:.2f} "
                  f"Combined={variant.combined_score:.2f}")


def print_results(analyses: List[TicketAnalysis]) -> None:
    """Print comprehensive results.

    Args:
        analyses: List of completed TicketAnalysis objects
    """
    print(f"\n{'='*80}")
    print("COMPREHENSIVE ANALYSIS RESULTS")
    print(f"{'='*80}\n")

    # Overall averages by source
    print("="*80)
    print("OVERALL AVERAGES BY SOURCE")
    print("="*80)

    sources = ["baseline_expected", "rag_expected", "baseline_actual_cla", "rag_actual_cla"]
    source_labels = {
        "baseline_expected": "Baseline Expected (No-RAG Solr)",
        "rag_expected": "RAG Expected (RAG Solr)",
        "baseline_actual_cla": "CLA Actual (Baseline Run)",
        "rag_actual_cla": "CLA Actual (RAG Run)",
    }

    for source in sources:
        variants = [v for a in analyses for v in a.variants if v.source == source]

        # Combined scores (average of both judges)
        avg_combined = sum(v.combined_score for v in variants) / len(variants)
        avg_correctness = sum(v.combined_correctness for v in variants) / len(variants)
        avg_faithfulness = sum(v.combined_faithfulness for v in variants) / len(variants)
        avg_relevance = sum(v.content_relevance for v in variants) / len(variants)

        # Individual judge scores
        avg_reviewer = sum(v.reviewer_score for v in variants) / len(variants)
        avg_expert = sum(v.expert_score for v in variants) / len(variants)

        print(f"\n{source_labels[source]}:")
        print(f"  Combined Score (avg of judges): {avg_combined:.3f}")
        print(f"    ├─ AnswerReviewer:   {avg_reviewer:.3f}")
        print(f"    └─ LinuxExpert:      {avg_expert:.3f}")
        print(f"  Answer Correctness: {avg_correctness:.3f}")
        print(f"  Faithfulness:       {avg_faithfulness:.3f}")
        print(f"  Content Relevance:  {avg_relevance:.3f}")

    # Per-ticket winners
    print(f"\n{'='*80}")
    print("PER-TICKET WINNERS (by Combined Judge Score)")
    print(f"{'='*80}\n")

    for analysis in analyses:
        print(f"\n{analysis.ticket_id}:")
        print(f"  Query: {analysis.query[:80]}...")

        rankings = analysis.rank_by_metric('combined_score')
        print(f"\n  Rankings (Combined Score = avg of both judges):")
        for i, (source, score) in enumerate(rankings, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            # Get variant to show individual judge scores
            variant = next(v for v in analysis.variants if v.source == source)
            print(f"    {emoji} {i}. {source_labels[source]}: {score:.3f}")
            print(f"         (Reviewer={variant.reviewer_score:.2f}, Expert={variant.expert_score:.2f})")

    # Win counts
    print(f"\n{'='*80}")
    print("WIN COUNTS (Best Combined Score Per Ticket)")
    print(f"{'='*80}\n")

    win_counts = {source: 0 for source in sources}

    for analysis in analyses:
        best = analysis.best_by_metric('combined_score')
        win_counts[best.source] += 1

    for source in sources:
        print(f"{source_labels[source]}: {win_counts[source]} wins")

    # Judge agreement analysis
    print(f"\n{'='*80}")
    print("JUDGE AGREEMENT ANALYSIS")
    print(f"{'='*80}\n")

    agreements = 0
    disagreements = 0

    for analysis in analyses:
        reviewer_best = analysis.best_by_metric('reviewer_score')
        expert_best = analysis.best_by_metric('expert_score')

        if reviewer_best.source == expert_best.source:
            agreements += 1
        else:
            disagreements += 1
            print(f"{analysis.ticket_id}: Judges disagree!")
            print(f"  AnswerReviewer picks: {source_labels[reviewer_best.source]}")
            print(f"  LinuxExpert picks:    {source_labels[expert_best.source]}")

    agreement_rate = agreements / len(analyses) * 100
    print(f"\nJudge Agreement: {agreements}/{len(analyses)} ({agreement_rate:.1f}%)")
    print(f"Judge Disagreement: {disagreements}/{len(analyses)} ({100-agreement_rate:.1f}%)")


async def main():
    """Main analysis workflow."""
    parser = argparse.ArgumentParser(
        description="Analyze RAG vs Baseline extraction quality"
    )
    parser.add_argument(
        "--baseline-yaml",
        type=Path,
        default=HEAL_ROOT / "config" / "extracted_tickets_backup_20260421_123412.yaml",
        help="Baseline extracted tickets YAML",
    )
    parser.add_argument(
        "--rag-yaml",
        type=Path,
        default=HEAL_ROOT / "config" / "extracted_tickets_rag.yaml",
        help="RAG extracted tickets YAML",
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=HEAL_ROOT / ".test_rag_comparison/baseline_results/evaluation_20260421_141730_detailed.csv",
        help="Baseline evaluation CSV",
    )
    parser.add_argument(
        "--rag-csv",
        type=Path,
        default=HEAL_ROOT / ".test_rag_comparison/rag_results/evaluation_20260421_142139_detailed.csv",
        help="RAG evaluation CSV",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to see LLM responses",
    )

    args = parser.parse_args()

    # Configure logging based on debug flag
    import logging
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        print("🐛 Debug logging enabled\n")
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s - %(message)s"
        )

    # Load data
    print("Loading comparison data...")
    analyses = load_comparison_data(
        args.baseline_yaml,
        args.rag_yaml,
        args.baseline_csv,
        args.rag_csv,
    )
    print(f"Loaded {len(analyses)} tickets for analysis")

    # Run analysis
    await analyze_all_tickets(analyses)

    # Print results
    print_results(analyses)


if __name__ == "__main__":
    asyncio.run(main())
