#!/usr/bin/env python3
"""Compare okp-mcp (from cached JSON) vs simple baseline vs RAG-enhanced search.

Uses extracted okp-mcp results and compares to:
1. Simple SolrExpert (non-agentic optimizations)
2. RAG SolrExpert (grounded in Apache Solr docs)

Usage:
    # First extract cache to JSON (run from eval directory with litellm):
    uv run python scripts/extract_cache_to_json.py

    # Then run comparison (no litellm needed):
    cd HEAL && uv run python scripts/compare_okp_vs_baseline.py
    uv run python scripts/compare_okp_vs_baseline.py --verbose
"""

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Add paths
SCRIPT_DIR = Path(__file__).parent
HEAL_ROOT = SCRIPT_DIR.parent
EVAL_ROOT = Path("/home/emackey/Work/lightspeed-core/lightspeed-evaluation")

sys.path.insert(0, str(HEAL_ROOT / "src"))
sys.path.insert(0, str(EVAL_ROOT / "src"))
sys.path.insert(0, str(EVAL_ROOT))

from heal.agents.url_validation_agent import URLValidationAgent
from heal.agents.solr_expert import SolrExpertAgent, VerificationQuery
from heal.agents.solr_multi_agent import SolrMultiAgent
from heal.core.search.solr_rag_expert import SolrRAGExpert


@dataclass
class CachedOkpResult:
    """okp-mcp result from extracted JSON."""
    query: str
    contexts: List[str]  # Retrieved doc texts
    conversation_id: str


@dataclass
class ComparisonResult:
    """Comparison between okp-mcp, simple baseline, and RAG."""
    query: str
    okp_score: float
    simple_score: float
    rag_score: float
    okp_tokens: int  # Estimated tokens for validation LLM call
    simple_tokens: int  # Always 0 (no LLM)
    rag_tokens: int  # Always 0 (hardcoded rules)
    winner: str


def load_okp_results_from_json() -> List[CachedOkpResult]:
    """Load okp-mcp results from extracted JSON file.

    Returns:
        List of cached okp-mcp results
    """
    json_file = SCRIPT_DIR / "okp_mcp_cached_results.json"

    if not json_file.exists():
        print(f"JSON file not found: {json_file}")
        print("Run: uv run python scripts/extract_cache_to_json.py")
        return []

    with open(json_file) as f:
        data = json.load(f)

    results = []
    for item in data:
        results.append(CachedOkpResult(
            query=item['query'],
            contexts=item['contexts'],
            conversation_id=item['conversation_id'],
        ))

    return results


async def compare_approaches(
    okp_results: List[CachedOkpResult],
    verbose: bool = False,
) -> List[ComparisonResult]:
    """Compare okp-mcp vs simple baseline vs RAG expert.

    Args:
        okp_results: Cached okp-mcp results
        verbose: Print detailed output

    Returns:
        List of comparison results
    """
    # Initialize agents
    url_validator = URLValidationAgent()
    simple_agent = SolrExpertAgent(solr_url="http://localhost:8983/solr/portal")
    rag_agent = SolrRAGExpert(solr_url="http://localhost:8983/solr/portal")

    comparisons = []

    for i, okp_result in enumerate(okp_results, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}/{len(okp_results)}: {okp_result.query}")
        print(f"{'='*80}")

        # Prepare okp-mcp docs for validation
        okp_docs = []
        for j, context in enumerate(okp_result.contexts):
            okp_docs.append({
                'title': f"Doc {j+1} from okp-mcp",
                'url': f"https://access.redhat.com/doc{j+1}",  # We don't have URLs in cache
                'content': context[:500],
                'documentKind': 'unknown',
            })

        print(f"\n[okp-mcp from cache]")
        print(f"  Retrieved: {len(okp_docs)} docs")

        # Validate okp-mcp results
        okp_validation = await url_validator.validate_urls(
            query=okp_result.query,
            hypothesis="Expected answer for this query",
            retrieved_docs=okp_docs,
        )
        print(f"  Score: {okp_validation.score:.2f}")

        # okp-mcp token cost (production estimate, not cached):
        # - Search: ~1500 (MCP tool use: query parsing, Solr call, result formatting)
        #   Breakdown: Tool call ~500, Solr results parsing ~1000
        # - Validation: ~2000 (URLValidationAgent LLM call)
        okp_search_tokens = 1500  # MCP tool overhead
        okp_validation_tokens = 2000
        okp_total_tokens = okp_search_tokens + okp_validation_tokens
        print(f"  Tokens: ~{okp_total_tokens} (search: ~1500 MCP, validation: ~2000)")

        if verbose and okp_validation.issues:
            print(f"  Issue: {okp_validation.issues[0][:100]}...")

        # Test simple baseline
        print(f"\n[Simple Baseline]")
        vq = VerificationQuery(
            query=okp_result.query,
            context="Need to find relevant RHEL documentation",
            expected_doc_type="solution",
        )

        simple_result = await simple_agent.search_for_verification([vq])
        print(f"  Retrieved: {len(simple_result.found_docs)} docs")

        # Validate simple results
        simple_validation = await url_validator.validate_urls(
            query=okp_result.query,
            hypothesis="Expected answer for this query",
            retrieved_docs=simple_result.found_docs,
        )
        print(f"  Score: {simple_validation.score:.2f}")

        # Simple baseline token cost:
        # - Search: 0 (pure Solr HTTP queries)
        # - Validation: ~2000 (URLValidationAgent LLM call)
        simple_search_tokens = 0
        simple_validation_tokens = 2000
        simple_total_tokens = simple_search_tokens + simple_validation_tokens
        print(f"  Tokens: ~{simple_total_tokens} (search: 0, validation: ~2000)")

        if verbose and simple_validation.issues:
            print(f"  Issue: {simple_validation.issues[0][:100]}...")

        # Test RAG expert
        print(f"\n[RAG Expert]")
        rag_result = await rag_agent.search_for_verification([vq])
        print(f"  Retrieved: {len(rag_result.found_docs)} docs")

        # Validate RAG results
        rag_validation = await url_validator.validate_urls(
            query=okp_result.query,
            hypothesis="Expected answer for this query",
            retrieved_docs=rag_result.found_docs,
        )
        print(f"  Score: {rag_validation.score:.2f}")

        # RAG expert token cost:
        # - Search: 0 (hardcoded rules from Solr docs, no LLM)
        # - Validation: ~2000 (URLValidationAgent LLM call)
        rag_search_tokens = 0
        rag_validation_tokens = 2000
        rag_total_tokens = rag_search_tokens + rag_validation_tokens
        print(f"  Tokens: ~{rag_total_tokens} (search: 0, validation: ~2000)")

        if verbose and rag_validation.issues:
            print(f"  Issue: {rag_validation.issues[0][:100]}...")

        # Determine winner (three-way)
        scores = {
            "okp-mcp": okp_validation.score,
            "simple": simple_validation.score,
            "rag": rag_validation.score,
        }
        winner = max(scores, key=scores.get)

        # Check for ties
        max_score = scores[winner]
        tied = [k for k, v in scores.items() if v == max_score]
        if len(tied) > 1:
            winner = "TIE (" + ", ".join(tied) + ")"

        print(f"\n  🏆 Winner: {winner}")
        print(f"     okp-mcp: {okp_validation.score:.2f}")
        print(f"     simple:  {simple_validation.score:.2f}")
        print(f"     rag:     {rag_validation.score:.2f}")

        # Store comparison
        comparisons.append(ComparisonResult(
            query=okp_result.query,
            okp_score=okp_validation.score,
            simple_score=simple_validation.score,
            rag_score=rag_validation.score,
            okp_tokens=okp_total_tokens,
            simple_tokens=simple_total_tokens,
            rag_tokens=rag_total_tokens,
            winner=winner,
        ))

    return comparisons


async def main():
    """Run comparison."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare okp-mcp vs simple vs RAG")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("Loading okp-mcp results from JSON...")
    okp_results = load_okp_results_from_json()

    if not okp_results:
        print("No results found. Run extract_cache_to_json.py first.")
        return

    print(f"Found {len(okp_results)} cached okp-mcp results")

    # Run comparison
    comparisons = await compare_approaches(okp_results, verbose=args.verbose)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    okp_wins = sum(1 for c in comparisons if c.winner == "okp-mcp")
    simple_wins = sum(1 for c in comparisons if c.winner == "simple")
    rag_wins = sum(1 for c in comparisons if c.winner == "rag")
    ties = sum(1 for c in comparisons if "TIE" in c.winner)

    avg_okp = sum(c.okp_score for c in comparisons) / len(comparisons)
    avg_simple = sum(c.simple_score for c in comparisons) / len(comparisons)
    avg_rag = sum(c.rag_score for c in comparisons) / len(comparisons)

    print(f"\nWin Count:")
    print(f"  okp-mcp: {okp_wins}/{len(comparisons)}")
    print(f"  simple:  {simple_wins}/{len(comparisons)}")
    print(f"  rag:     {rag_wins}/{len(comparisons)}")
    print(f"  ties:    {ties}/{len(comparisons)}")

    print(f"\nAverage Scores:")
    print(f"  okp-mcp: {avg_okp:.2f}")
    print(f"  simple:  {avg_simple:.2f}")
    print(f"  rag:     {avg_rag:.2f}")

    total_okp_tokens = sum(c.okp_tokens for c in comparisons)
    total_simple_tokens = sum(c.simple_tokens for c in comparisons)
    total_rag_tokens = sum(c.rag_tokens for c in comparisons)

    print(f"\nToken Usage (for {len(comparisons)} queries):")
    print(f"  okp-mcp: ~{total_okp_tokens:,} tokens")
    print(f"  simple:  ~{total_simple_tokens:,} tokens")
    print(f"  rag:     ~{total_rag_tokens:,} tokens")

    print(f"\nPer-Query Token Cost:")
    print(f"  okp-mcp: ~{total_okp_tokens // len(comparisons):,} tokens/query")
    print(f"  simple:  ~{total_simple_tokens // len(comparisons):,} tokens/query")
    print(f"  rag:     ~{total_rag_tokens // len(comparisons):,} tokens/query")

    print(f"\nRECOMMENDATION:")
    # Find best approach
    scores = {"okp-mcp": avg_okp, "simple": avg_simple, "rag": avg_rag}
    best = max(scores, key=scores.get)
    best_score = scores[best]

    if best == "rag":
        delta_vs_okp = avg_rag - avg_okp
        delta_vs_simple = avg_rag - avg_simple
        print(f"✅ Use RAG EXPERT for bootstrap")
        print(f"   Beats okp-mcp by {delta_vs_okp:+.2f}")
        print(f"   Beats simple by {delta_vs_simple:+.2f}")
        print(f"   Grounded in Apache Solr documentation!")
    elif best == "simple":
        delta_vs_okp = avg_simple - avg_okp
        delta_vs_rag = avg_simple - avg_rag
        print(f"✅ Use SIMPLE BASELINE for bootstrap")
        print(f"   Beats okp-mcp by {delta_vs_okp:+.2f}")
        print(f"   Beats rag by {delta_vs_rag:+.2f}")
        print(f"   No token costs, simpler is better!")
    elif best == "okp-mcp":
        delta_vs_simple = avg_okp - avg_simple
        delta_vs_rag = avg_okp - avg_rag
        print(f"✅ Use okp-mcp for bootstrap (self-healing architecture)")
        print(f"   Beats simple by {delta_vs_simple:+.2f}")
        print(f"   Beats rag by {delta_vs_rag:+.2f}")
        print(f"   Production-tuned config is better")

    # Check if close
    max_delta = max(abs(avg_okp - avg_simple), abs(avg_okp - avg_rag), abs(avg_simple - avg_rag))
    if max_delta < 0.05:
        print(f"\n⚠️  Note: All approaches within 0.05 - effectively tied")
        print(f"   Choose based on other factors (token cost, simplicity, etc.)")


if __name__ == "__main__":
    asyncio.run(main())
