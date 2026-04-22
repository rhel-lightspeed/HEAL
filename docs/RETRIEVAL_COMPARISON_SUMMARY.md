# Retrieval Strategy Comparison - Summary

## What Was Added (2026-04-21)

### New Files

1. **`scripts/compare_okp_vs_baseline.py`** - Main comparison script
   - Compares Simple, RAG, and okp-mcp retrieval strategies
   - Implements feedback loops with iterative refinement
   - Tracks URL F1 and content relevance scores
   - Ground truth verification

2. **`src/heal/agents/simple_solr_agent.py`** - Baseline keyword search
   - Direct Solr queries, no advanced features
   - Refinement: title boosting → exact phrase matching

3. **`src/heal/agents/rag_solr_agent.py`** - Enhanced search
   - edismax query parser with field boosting
   - Title^3.0, main_content^1.5 weights
   - Refinement: increase result count (10 → 15 → 20)

4. **`src/heal/agents/content_relevance_agent.py`** - Semantic scoring
   - Keyword overlap heuristic (can be replaced with LLM)
   - Complements URL F1 with semantic relevance

5. **`src/heal/agents/query_parser.py`** - Query reformulation
   - Extracts technical terms, action verbs, core concepts
   - Uses rule-based parsing (similar to AST for queries)
   - Can reformulate verbose queries into concise search terms

6. **`docs/README_COMPARISON.md`** - User guide
   - How to run comparisons
   - Metric explanations
   - Interpretation guidelines

### Backed Up Files

- **`scripts/compare_okp_vs_baseline_OLD.py`** - Previous version (kept for reference)

## Key Findings

### Ground Truth Coverage
- ✅ All expected URLs exist in Solr (100% coverage)
- Problem is NOT missing documents

### URL F1 vs Content Relevance
- **Simple agent**: 0% URL F1, 0% content relevance (broken)
- **RAG agent**: 2.8% URL F1, **68.8% content relevance**
- **Insight**: RAG retrieves semantically relevant docs, just not exact expected URLs

### What Makes Queries Fail?

**Bad (0% success)**:
```
"How do I recreate the GRUB configuration file..."
→ Matches "How do I recreate CloudForms", "How do I recreate VDO"
→ Problem: Generic phrase "How do I recreate" dominates
```

**Good (50% success)**:
```
"How do I disable Secure Boot..."
→ Matches "How to disable Secure Boot on Physical systems" ✓
→ Works: "Secure Boot" is distinctive technical term
```

### Iteration Analysis
- **Simple agent**: Exact phrase matching too restrictive (0 results on iter 3)
- **RAG agent**: Increasing result count doesn't improve ranking
- **Key insight**: More results ≠ better quality

## What Actually Works

✅ **Specific technical terms**: "Secure Boot", "grub2-mkconfig"  
✅ **Distinctive phrases**: Less common = better matching  
❌ **Generic procedural phrasing**: "How do I...", "What is..."  
❌ **More results**: Precision drops as recall increases

## Recommended Improvements

### 1. Query Reformulation (Biggest Impact)
Use `query_parser.py` to extract key terms:
```python
parser = QueryParser()
result = parser.parse("How do I recreate the GRUB configuration file...")
# → "GRUB configuration file recreate RHEL"
```

### 2. Semantic Reranking
- Stage 1: Keyword retrieval (current)
- Stage 2: LLM reranks top-20 by relevance
- Stage 3: Return top-5

### 3. Better Refinement Strategies
Instead of just increasing count:
- Synonym expansion
- Technical term boosting
- Phrase structure analysis

## Usage

```bash
cd ~/Work/rhel-lightspeed/HEAL

# Basic comparison
uv run python scripts/compare_okp_vs_baseline.py

# With details
uv run python scripts/compare_okp_vs_baseline.py --details

# Different pattern
uv run python scripts/compare_okp_vs_baseline.py --pattern BOOTLOADER_GRUB_ISSUES
```

## Next Steps

1. **Load okp-mcp from evaluation CSVs** (currently placeholder)
2. **Add query reformulation** to refinement strategies
3. **Test semantic reranking** with LLM or embeddings
4. **Compare on more patterns** beyond BOOTLOADER

## AST/Dependency Parsing Question

User asked: "Could something like an abstract syntax tree pull out subject verb predicate?"

**Answer**: Yes! `query_parser.py` uses a similar concept:
- **AST for code**: Parses code into syntax tree
- **Dependency parsing for queries**: Parses sentences into grammatical structure
- Current implementation: Rule-based (no external deps)
- Production: Use spaCy dependency parser for better accuracy

Example:
```python
# spaCy approach (not implemented yet)
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("How do I disable Secure Boot?")

for token in doc:
    print(token.text, token.dep_, token.head.text)
# → Subject: "Boot", Verb: "disable", Object: None
```

This would extract semantic structure more accurately than regex.
