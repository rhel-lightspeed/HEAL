# Multi-Agent Solr Optimization

## Overview

The multi-agent Solr optimization system uses **3 specialized AI agents** working together to create better Solr configuration suggestions than a single LLM could achieve alone.

Inspired by the successful multi-agent ticket-to-YAML pipeline, this system combines:
- **Solr theory expertise** (what SHOULD work)
- **Code implementation reality** (what CAN work)  
- **Practical synthesis** (what WILL work)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Pattern Fix Loop                            │
│                  (Orchestrator)                              │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   │ Query: "RHEL 9 bootloader"
                   │ Metrics: F1=0.00, MRR=0.00
                   │
                   ▼
    ┌──────────────────────────────────────────────┐
    │      Multi-Agent Solr Optimization           │
    └──────┬───────────────────┬──────────────┬───┘
           │                   │              │
           ▼                   ▼              ▼
    ┌──────────┐        ┌──────────┐   ┌────────────┐
    │  Phase 1 │        │  Phase 2 │   │  Phase 3   │
    │  Solr    │        │ OKP-MCP  │   │ Synthesizer│
    │  Expert  │        │  Code    │   │            │
    │          │        │  Expert  │   │            │
    └────┬─────┘        └────┬─────┘   └─────┬──────┘
         │                   │               │
         │                   │               │
         ▼                   ▼               ▼
   Ideal Config      Implementation      Practical
   (Theory)          Analysis            Suggestion
                     (Reality)           (Works!)
```

## The Three Agents

### 1. Solr Expert Agent

**Role:** Solr/Lucene theory specialist

**Knowledge:**
- Apache Solr edismax query parser
- BM25 and TF-IDF scoring algorithms
- Query analysis, tokenization, stopwords
- Field boosting strategies (qf, pf, pf2, pf3)
- Minimum match (mm) patterns
- Phrase slop (ps) tuning
- Highlighting and snippet extraction

**Input:**
- User query
- Expected documents (should retrieve)
- Actually retrieved documents
- Evaluation metrics (F1, MRR)
- Solr explain output (optional)

**Output:**
```json
{
  "problem_analysis": "Query has stopwords ('how', 'do', 'I') that Solr ignores, reducing effective terms from 6 to 3. Current mm=75% requires 4.5 terms to match, causing all docs to fail.",
  
  "ideal_config": {
    "mm": "2<-1 5<60%",
    "qf": "title^8 main_content^2 heading_h1^5",
    "pf": "title^12 main_content^6",
    "ps": "5"
  },
  
  "reasoning": "Reducing mm to 60% allows docs with 3/5 terms (60%) to match. Boosting title helps since expected docs likely have query terms in title. Increasing phrase slop to 5 helps with scattered terms.",
  
  "relevant_docs": [
    "edismax minimum match patterns",
    "BM25 field boosting",
    "Handling stopwords in queries"
  ]
}
```

**Strength:** Knows what SHOULD work in theory
**Limitation:** Doesn't know actual okp-mcp implementation

---

### 2. OKP-MCP Code Expert Agent

**Role:** okp-mcp codebase specialist

**Knowledge:**
- Actual Python implementation in `src/okp_mcp/solr.py`
- How queries are preprocessed
- BM25 re-ranking logic
- Highlighting snippet selection
- Special handling for query patterns
- Bugs and edge cases

**Access:**
- **Reads actual code files** from okp-mcp repository
- Can analyze implementation details
- Understands constraints and quirks

**Input:**
- User query
- Solr Expert's ideal config
- Solr Expert's reasoning

**Output:**
```json
{
  "current_implementation": "okp-mcp builds Solr queries at line 140. The mm parameter is set conditionally: for deprecation queries (line 147), mm='100%' for strict matching. For regular queries (line 152), mm='2<-1 5<75%'.",
  
  "constraints": [
    "mm is query-type dependent (lines 147-152) - can't change globally",
    "Title boost conflicts with deprecation boost logic (line 267)",
    "BM25 re-ranking happens AFTER Solr (lines 301-357)",
    "Boost keywords must be in src/okp_mcp/solr.py list (lines 45-250)"
  ],
  
  "bugs_found": [
    "Line 312: ps (phrase slop) is hardcoded to 3, ignoring the ps parameter!",
    "Line 267: title_boost is applied twice in some cases"
  ],
  
  "relevant_code_sections": {
    "src/okp_mcp/solr.py:147-152": "mm query type handling",
    "src/okp_mcp/solr.py:312": "ps hardcoded bug"
  },
  
  "warnings": [
    "Changing mm for regular queries won't affect deprecation queries",
    "Title boost interacts with BM25 re-ranking multipliers"
  ]
}
```

**Strength:** Knows how okp-mcp ACTUALLY works
**Limitation:** Doesn't know Solr theory deeply

---

### 3. Synthesizer Agent

**Role:** Senior engineer combining theory + reality

**Knowledge:**
- Software engineering best practices
- When to prioritize theory vs constraints
- Risk assessment for code changes
- Practical judgment

**Input:**
- User query and metrics
- Solr Expert's ideal config + reasoning
- Code Expert's implementation analysis + constraints

**Output:**
```json
{
  "suggested_change": "Fix ps hardcoding bug and adjust mm for RHEL queries",
  
  "file_path": "src/okp_mcp/solr.py",
  
  "old_code": "        # Build Solr params\n        params['mm'] = '2<-1 5<75%'\n        params['ps'] = 3  # Hardcoded",
  
  "new_code": "        # Build Solr params\n        if 'RHEL' in query:\n            params['mm'] = '2<-1 5<60%'  # More lenient for RHEL queries\n        else:\n            params['mm'] = '2<-1 5<75%'\n        params['ps'] = params.get('ps', 5)  # Fix: respect ps parameter",
  
  "reasoning": "Combines Solr Expert's mm=60% recommendation with Code Expert's finding that RHEL queries need special handling. Also fixes the ps hardcoding bug that Code Expert found. Doesn't touch title boost to avoid conflict with existing deprecation logic.",
  
  "confidence": 0.85,
  
  "risks": [
    "60% mm might be too lenient for non-RHEL queries (keeping 75% for those)",
    "ps=5 increases recall but might decrease precision"
  ]
}
```

**Strength:** Practical suggestions that work with actual code
**Limitation:** None - has best of both worlds!

---

## Example: Full Multi-Agent Flow

### Problem
```
Query: "How do I configure bootloader in RHEL 9?"
Expected docs: solutions/7013132, solutions/5344601, ...
Retrieved: (none)
F1: 0.00, MRR: 0.00
```

### Phase 1: Solr Expert Analysis

**Thinks:** "This query has stopwords ('how', 'do', 'I') that Solr strips. After stopword removal: 'configure bootloader RHEL 9' = 4 effective terms. Current mm='2<-1 5<75%' means 5+ terms need 75% match = 3.75 terms minimum. With only 4 effective terms after stopword removal, this barely meets the threshold. Any variation in tokenization could fail."

**Suggests:** 
- mm='2<-1 5<60%' (more lenient)
- Boost title^8 (expected docs likely have "bootloader" in title)
- ps=5 (scattered terms)

### Phase 2: Code Expert Analysis

**Reads:** `src/okp_mcp/solr.py`

**Finds:**
- Line 147-152: mm varies by query type
- Line 312: **BUG** - ps is hardcoded to 3, ignoring parameter!
- Line 267: title boost conflicts with deprecation boost

**Reports:** "Can implement mm=60% for RHEL queries specifically. Must fix ps hardcoding bug first. Don't change title boost globally - it interacts with deprecation logic."

### Phase 3: Synthesizer Decision

**Combines:**
- Solr theory: mm=60% + ps=5 should work
- Code reality: Can do mm=60% for RHEL queries only, ps bug must be fixed

**Produces:**
```python
# Before
params['mm'] = '2<-1 5<75%'
params['ps'] = 3  # Bug!

# After  
if 'RHEL' in query:
    params['mm'] = '2<-1 5<60%'  # Solr Expert's recommendation
else:
    params['mm'] = '2<-1 5<75%'  # Keep existing for other queries
params['ps'] = params.get('ps', 5)  # Fix Code Expert's bug
```

**Result:** ✅ Combines best of both + fixes bug!

---

## Benefits Over Single-Agent

| Single Agent | Multi-Agent |
|--------------|-------------|
| Generic Solr advice | Solr theory + okp-mcp specifics |
| Outdated hardcoded docs | Reads actual current code |
| Might suggest impossible changes | Respects implementation constraints |
| Misses bugs | Catches and fixes bugs |
| "Increase mm to 80%" | "Increase mm to 80% for deprecation queries only (line 147)" |
| Confidence: 60% | Confidence: 85% |

---

## Usage

### Automatic (Enabled by Default)

The multi-agent system is **automatically used** in pattern-wide optimization:

```bash
./runners/fix.sh BOOTLOADER_GRUB_ISSUES
```

Output:
```
✅ Multi-agent Solr optimization enabled (Solr Expert + Code Expert)

--- Pattern Iteration 1/10 ---

🤖 Consulting multi-agent system (Solr Expert + Code Expert)...

🔍 Solr Expert + Code Expert Analysis Complete
   Confidence: 85%
   Risks: 60% mm might be too lenient

💡 Suggestion: Fix ps hardcoding and adjust mm for RHEL queries
```

### Fallback to Single-Agent

If multi-agent system fails or is unavailable:

```
⚠️  Multi-agent system not available (requires claude-agent-sdk)
   Using single-agent mode

🤖 Using single-agent mode (fallback)...
```

---

## Requirements

- `claude-agent-sdk` (installed via `uv pip install claude-agent-sdk`)
- Access to okp-mcp repository
- Claude API key

---

## Implementation Files

- `src/heal/agents/solr_multi_agent.py` - Multi-agent system
- `src/heal/runners/run_pattern_fix_poc.py` - Integration with pattern fix loop

---

## Future Enhancements

1. **Pattern-aware suggestions**: Currently uses first failing ticket as representative. Could analyze ALL failing tickets simultaneously.

2. **More agents**: Could add:
   - Query Analysis Agent (preprocessing, tokenization)
   - BM25 Re-ranking Agent (boost/demote keywords)
   - Historical Performance Agent (what worked before?)

3. **Learning from pattern results**: After testing ALL tickets, use feedback to improve next suggestion.

4. **Confidence-based model selection**: High confidence → commit immediately, low confidence → escalate to Opus for review.

---

## Success Metrics

Track:
- **Suggestion quality**: F1 improvement per suggestion
- **Bug detection rate**: Bugs caught by Code Expert
- **Confidence calibration**: Does 85% confidence actually improve F1 85% of the time?
- **Time savings**: Multi-agent vs single-agent iteration count

Expected improvement: **30-50% better suggestions** vs single-agent mode.
