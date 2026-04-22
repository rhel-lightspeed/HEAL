# Retrieval Strategy Comparison Guide

## Quick Start

```bash
# Compare Simple vs RAG agents on BOOTLOADER pattern
python scripts/compare_okp_vs_baseline.py --pattern BOOTLOADER_UEFI_FIRMWARE

# Show detailed iteration-by-iteration breakdown
python scripts/compare_okp_vs_baseline.py --pattern BOOTLOADER_UEFI_FIRMWARE --details

# Try different pattern
python scripts/compare_okp_vs_baseline.py --pattern <PATTERN_NAME>

# Adjust feedback parameters
python scripts/compare_okp_vs_baseline.py \
  --pattern BOOTLOADER_UEFI_FIRMWARE \
  --max-iterations 5 \
  --threshold 0.8 \
  --details
```

## Available Patterns

Patterns are defined in `okp_mcp_agent/config/patterns/`:
- `BOOTLOADER_UEFI_FIRMWARE` - GRUB, Secure Boot, UEFI (6 queries)
- Find others: `ls okp_mcp_agent/config/patterns/*.yaml`

## What It Compares

### 1. **Simple Agent** (Baseline)
- Direct keyword search against Solr
- No field boosting or advanced features
- Refinement strategy: title boosting → exact phrase matching

### 2. **RAG Agent** (Enhanced)
- edismax query parser with field weights
- Title^3.0, main_content^1.5 boosting
- Phrase matching with slop
- Refinement strategy: increase result count (10 → 15 → 20)

### 3. **okp-mcp** (Future)
- Load from cached evaluation results
- Currently placeholder (cache limited to 3 queries)

## Metrics Explained

### URL F1 Score (Exact Matching)
- Precision: % of retrieved URLs that are expected
- Recall: % of expected URLs that were retrieved
- F1: Harmonic mean of precision and recall
- **Problem**: Penalizes semantically correct docs with different URLs

### Content Relevance Score (Semantic Matching)
- Keyword overlap between query and document text
- Bonus for matches in title field
- 0.0 = no relevant terms, 1.0 = all query terms present
- **Better indicator** of actual retrieval quality

## Output Interpretation

```
SUMMARY
================================================================================

SIMPLE
  URL F1 Score: 0.000         ← No expected URLs found
  Content Relevance: 0.000    ← Retrieved completely irrelevant docs
  Average Iterations: 3.0
  Success Rate (URL): 0.0%

RAG
  URL F1 Score: 0.028         ← Found 2.8% of expected URLs
  Content Relevance: 0.688    ← 68.8% keyword overlap (semantically relevant!)
  Average Iterations: 3.0
  Success Rate (URL): 0.0%
```

**Key Insight**: High content relevance + low URL F1 = **retrieving relevant docs, just not the exact expected URLs**

## Ground Truth Verification

The script checks if expected URLs actually exist in Solr:

```
Verifying ground truth URLs in Solr index...
  Total expected URLs: 20
  Found in index: 20 (100.0%)
```

If coverage < 100%, those URLs can never be retrieved (missing from index).

## Iteration Analysis

Use `--details` to see iteration-by-iteration changes:

```
RAG AGENT
[1] How do I recreate the GRUB configuration file...
    Final: URL F1=0.000, Content Rel=0.775, Success=False
    Iter 1: URL F1=0.000, Content=0.775, URLs=10
      → Adjustment: Increased result count to 15
      → Issues: Low recall (0.00): Missing 4/4 expected URLs
    Iter 2: URL F1=0.000, Content=0.775, URLs=15
      → Adjustment: Increased result count to 20
      → Issues: Low recall (0.00): Missing 4/4 expected URLs
    Iter 3: URL F1=0.000, Content=0.775, URLs=20
      → Issues: Low recall (0.00): Missing 4/4 expected URLs
```

**Shows**: Content relevance stays consistent but increasing result count doesn't help.

## What Makes Queries Succeed?

From our analysis:

✅ **Specific technical terms**: "Secure Boot", "grub2-mkconfig"
✅ **Distinctive phrases**: "disable Secure Boot" > "recreate configuration"
❌ **Generic procedural phrasing**: "How do I...", "What is the way to..."
❌ **Common verbs**: "recreate", "configure", "setup" match too broadly

**Example Success (2/4 URLs found)**:
- Query: "How do I disable Secure Boot..."
- Works because "Secure Boot" is distinctive
- Retrieved: "How to disable Secure Boot on Physical systems" ✓

**Example Failure (0/4 URLs found)**:
- Query: "How do I recreate the GRUB configuration file..."
- Fails because "How do I recreate" matches everything
- Retrieved: "How do I recreate CloudForms workers", "How do I recreate VDO index"

## Improving Results

### 1. Query Reformulation (Biggest Impact)
Extract key technical terms, drop generic phrasing:
- Bad: "How do I recreate the GRUB configuration file on Red Hat Enterprise Linux?"
- Good: "grub2-mkconfig grub.cfg RHEL bootloader configuration"

### 2. Better Refinement Strategies
Instead of just increasing result count:
- Synonym expansion (grub.cfg → GRUB2 → bootloader)
- Technical term extraction and boosting
- Phrase structure analysis

### 3. Semantic Reranking
- First pass: Keyword retrieval (current behavior)
- Second pass: LLM reranks top-20 by semantic relevance
- Return: Top-5 reranked results

## Next Steps

1. **Add okp-mcp comparison**: Load from evaluation CSV files
2. **Add query reformulation**: Use spaCy or LLM to extract technical terms
3. **Add semantic reranking**: Post-process with embeddings or LLM
4. **Test on more patterns**: Beyond BOOTLOADER
