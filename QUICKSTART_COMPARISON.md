# Quick Start: Retrieval Strategy Comparison

## Files Added to HEAL (2026-04-21)

```
scripts/compare_okp_vs_baseline.py  ← Main comparison script
src/heal/agents/
  ├── simple_solr_agent.py          ← Baseline keyword search
  ├── rag_solr_agent.py              ← Enhanced search (edismax)
  ├── content_relevance_agent.py     ← Semantic scoring
  └── query_parser.py                ← Query reformulation
docs/
  ├── README_COMPARISON.md           ← Full user guide
  └── RETRIEVAL_COMPARISON_SUMMARY.md ← Key findings
```

## Run It Now

```bash
cd ~/Work/rhel-lightspeed/HEAL

# Basic comparison (BOOTLOADER_GRUB_ISSUES pattern)
uv run python scripts/compare_okp_vs_baseline.py

# Show iteration details
uv run python scripts/compare_okp_vs_baseline.py --details

# Try query parser
uv run python src/heal/agents/query_parser.py
```

## What It Shows

- **URL F1**: Exact URL matching (0.0-1.0)
- **Content Relevance**: Semantic keyword overlap (0.0-1.0)
- **Iterations**: Feedback loop refinement attempts
- **Ground Truth**: Which expected URLs exist in Solr

## Key Finding

RAG agent gets **68.8% content relevance** but **2.8% URL F1** → retrieving relevant docs with different URLs!

Problem: Query formulation, not search algorithm.

## Next Session

When you resume in HEAL:
1. We want to load okp-mcp results from evaluation CSVs (not cache)
2. Add query reformulation to feedback loops
3. Compare with syntax parsing (spaCy dependency parsing)

See `docs/RETRIEVAL_COMPARISON_SUMMARY.md` for full details.
