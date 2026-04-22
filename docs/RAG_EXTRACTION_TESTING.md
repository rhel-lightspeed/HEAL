# RAG-Enhanced Extraction Testing

**Goal**: Test if RAG-enhanced Solr retrieval (87.4% content relevance) improves the quality of expected answers in the bootstrapping process.

## Background

From retrieval optimization research (`RETRIEVAL_OPTIMIZATION_FINDINGS.md`):
- **Current**: SolrExpertAgent uses basic edismax: `qf="title^5 main_content^2 product"`
- **RAG**: SolrExpertRAGAgent uses optimized params: `qf="title^3.0 main_content^1.5 id^2.0"` + phrase boosting
- **Result**: RAG achieves **87.4% content relevance** vs 63.3% for baseline

**Hypothesis**: Better document retrieval → Better expected answers for LinuxExpert to synthesize.

## Files Created

### New Agents
- **`src/heal/agents/solr_expert_rag.py`** - RAG-enhanced Solr Expert (drop-in replacement)
  - Same interface as `SolrExpertAgent`
  - Uses proven edismax config from optimization research
  - Logs to search intelligence with `retrieval_method: rag_enhanced_edismax`

### Testing Scripts
- **`src/heal/bootstrap/extract_jira_tickets_rag.py`** - RAG variant of extraction script
  - Uses `SolrExpertRAGAgent` instead of `SolrExpertAgent`
  - Outputs to `config/extracted_tickets_rag.yaml`
  - Otherwise identical workflow

- **`scripts/compare_extracted_yamls.py`** - Quality comparison tool
  - Compares baseline vs RAG YAML files
  - Metrics: answer length, URL count, refinement iterations, review scores
  - Supports per-ticket and aggregate analysis

### Backups
- **`config/extracted_tickets_backup_YYYYMMDD_HHMMSS.yaml`** - Automatic backup before testing

## Testing Workflow

### Step 1: Extract with RAG Agent (Test on 1-2 tickets first)

```bash
cd ~/Work/rhel-lightspeed/HEAL

# Test on a single ticket
python src/heal/bootstrap/extract_jira_tickets_rag.py --tickets RSPEED-2482

# Or test on 2-3 recent tickets
python src/heal/bootstrap/extract_jira_tickets_rag.py --limit 3
```

**Output**: `config/extracted_tickets_rag.yaml`

### Step 2: Compare Quality

```bash
# Summary statistics
python scripts/compare_extracted_yamls.py

# Detailed per-ticket comparison
python scripts/compare_extracted_yamls.py --details

# Compare single ticket in depth
python scripts/compare_extracted_yamls.py --ticket RSPEED-2482
```

### Step 3: Analyze Results

**Metrics to Check**:

1. **Answer Length**
   - Longer = more detailed (good)
   - Shorter = more concise or missing info (check manually)

2. **URLs Retrieved**
   - More URLs = broader context (could be good or bad)
   - Different URLs = RAG finding alternative valid docs

3. **Refinement Iterations**
   - Fewer iterations = better first-pass quality ✅
   - Same iterations = no improvement
   - More iterations = worse quality ❌

4. **Review Scores** (if available)
   - Higher score = better quality ✅

5. **Manual Spot-Check**
   - Read expected answers side-by-side
   - Do RAG answers have better technical depth?
   - Are URLs more relevant?

### Step 4: Decide

**If RAG is better**:
```bash
# Replace baseline with RAG agent
cd src/heal/bootstrap
mv extract_jira_tickets.py extract_jira_tickets_baseline.py
mv extract_jira_tickets_rag.py extract_jira_tickets.py

# Update SolrExpertAgent import in extract_jira_tickets.py
# Change: from heal.core import SolrExpertAgent
# To: from heal.agents.solr_expert_rag import SolrExpertRAGAgent as SolrExpertAgent
```

**If baseline is better**:
- Document findings
- Keep current approach
- Investigate why RAG underperformed

**If mixed results**:
- Keep both options
- Use RAG for specific ticket types
- Add CLI flag to choose agent

## Expected Outcomes

### Scenario 1: RAG Improves Quality ✅
- RAG answers are longer and more detailed
- Fewer refinement iterations needed
- URLs are more relevant to the question
- **Action**: Adopt RAG as default

### Scenario 2: No Significant Difference ➡️
- Metrics are similar
- Manual inspection shows equivalent quality
- **Insight**: URL retrieval not the bottleneck (LinuxExpert synthesis is key)
- **Action**: Keep cheaper baseline, save complexity

### Scenario 3: RAG Underperforms ❌
- RAG requires more refinement iterations
- Answers are shorter or less accurate
- **Investigate**: Why does 87.4% content relevance not translate to better answers?
- **Hypothesis**: Content relevance ≠ answer synthesis quality (different skills)

## Integration Points

If RAG proves effective, integrate into:

1. **YAML Generation** (current testing)
   - Replace `SolrExpertAgent` with `SolrExpertRAGAgent`

2. **okp-mcp Fix Loop** (future)
   - Use proven RAG config as starting point
   - Save 3-15 iterations of random parameter search

3. **URL Validation** (future)
   - Replace expensive `URLValidationAgent` with cheap `ContentRelevanceAgent`
   - Save $0.01-0.05 per validation

See `docs/INTEGRATE_RAG_AGENT.md` for implementation details.

## Rollback Plan

If testing reveals issues:

```bash
# Original backup is at:
ls -lh config/extracted_tickets_backup_*.yaml

# RAG extraction is in separate file:
config/extracted_tickets_rag.yaml

# Baseline extraction is unchanged:
config/extracted_tickets.yaml
```

No destructive changes - just comparison testing!

## Cost Analysis

**Current extraction** (SolrExpertAgent):
- Solr queries: Free (HTTP calls)
- LLM calls: LinuxExpert synthesis, AnswerReviewAgent, URLValidationAgent
- Cost: ~$0.02-0.05 per ticket

**RAG extraction** (SolrExpertRAGAgent):
- Solr queries: Free (HTTP calls, slightly more complex params)
- LLM calls: Same as baseline
- Cost: **Identical**

**Savings potential** (future):
- If RAG reduces refinement iterations: -30% LLM calls
- If URLValidationAgent replaced: -$0.01-0.05 per ticket
- Total: **30-40% cost reduction** with same/better quality

## References

- **Findings**: `docs/RETRIEVAL_OPTIMIZATION_FINDINGS.md`
- **Integration Guide**: `docs/INTEGRATE_RAG_AGENT.md`
- **Comparison Results**: `docs/RETRIEVAL_COMPARISON_SUMMARY.md`
- **Presentation**: `docs/HEAL_SLIDES_OUTLINE.md` (Slides 5-7)

## Questions?

- **Q: Why not just use RAG everywhere?**
  - A: Need to validate it actually improves answer quality, not just retrieval metrics

- **Q: What if metrics are close?**
  - A: Manual spot-check is critical - metrics are proxies, human judgment is ground truth

- **Q: Can we A/B test on production?**
  - A: Not yet - need stable baseline first. Test offline, then gradual rollout.

- **Q: What about the existing multi-agent system?**
  - A: Don't touch! LinuxExpert + SolrExpert + AnswerReviewAgent works perfectly.
    We're just swapping Solr retrieval implementation (same interface).
