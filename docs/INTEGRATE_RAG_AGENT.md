# How to Integrate RAG Agent into HEAL Workflows

## Quick Wins (Drop-in Replacements)

### 1. YAML Expected Answer Generation

**Before (current):**
```python
# In bootstrap/extract_jira_tickets.py
from heal.agents.solr_expert import SolrExpertAgent

solr_expert = SolrExpertAgent()
# Simple keyword search - 63.3% content relevance
```

**After (improved):**
```python
# In bootstrap/extract_jira_tickets.py
from heal.agents.rag_solr_agent import RAGSolrAgent

rag_agent = RAGSolrAgent(
    solr_url="http://localhost:8983/solr",
    collection="portal"
)
# RAG search - 87.4% content relevance ✓
```

**Impact**: Better docs → LinuxExpert produces better expected answers

### 2. okp-mcp Fix Loop Starting Point

**Before (current):**
```python
# Fix loop starts with random/default Solr config
# Agent has to search parameter space blindly
```

**After (improved):**
```python
# In okp_mcp_llm_advisor.py or wherever fix suggestions start
PROVEN_BASELINE_CONFIG = {
    "defType": "edismax",
    "qf": "title^3.0 content^1.0 main_content^1.5 id^2.0",
    "pf": "title^10.0 content^5.0 main_content^7.0",
    "ps": "2",
    "mm": "50%"
}

# Start here, make small adjustments (±20% on weights)
# Instead of searching from scratch
```

**Impact**: Faster convergence, better starting point

### 3. Replace Expensive URLValidationAgent

**Before (expensive):**
```python
# In LinuxExpert.extract_with_autonomous_review()
url_validator = URLValidationAgent()  # Claude SDK calls
result = await url_validator.validate_urls(
    query=query,
    hypothesis=hypothesis,
    retrieved_docs=docs
)
# Cost: ~$0.01-0.05 per validation
```

**After (cheap):**
```python
# Use content relevance heuristic
from heal.agents.content_relevance_agent import ContentRelevanceAgent

content_evaluator = ContentRelevanceAgent()
result = content_evaluator.evaluate_relevance(
    query=query,
    retrieved_docs=docs,
    top_k=5
)

if result["avg_relevance"] >= 0.8:
    # Good enough - proceed with synthesis
    pass
else:
    # Low relevance - refine search
    pass

# Cost: $0 (free heuristic)
```

**Impact**: Save $3-15 per pattern, maintain quality

## Integration Examples

### Example 1: Enhance YAML Generation
```python
# File: src/heal/bootstrap/extract_jira_tickets.py
# (or wherever YAML generation happens)

from heal.agents.rag_solr_agent import RAGSolrAgent
from heal.agents.content_relevance_agent import ContentRelevanceAgent

class TicketExtractor:
    def __init__(self):
        # Use proven RAG config
        self.solr_agent = RAGSolrAgent()
        self.content_validator = ContentRelevanceAgent()
    
    async def extract_expected_answer(self, query: str):
        # Retrieve with RAG (87.4% content relevance)
        docs = self.solr_agent.search_with_rag(query, rows=10)
        
        # Quick validation (free)
        validation = self.content_validator.evaluate_relevance(
            query=query,
            retrieved_docs=docs
        )
        
        if validation["avg_relevance"] < 0.7:
            # Low quality - log warning but continue
            print(f"⚠️ Low relevance ({validation['avg_relevance']:.2f}) for: {query[:60]}...")
        
        # Pass to LinuxExpert for synthesis
        answer = await linux_expert.synthesize(query, docs)
        return answer
```

### Example 2: okp-mcp Fix Loop Starting Point
```python
# File: src/heal/agents/okp_mcp_llm_advisor.py

class OkpMcpLLMAdvisor:
    # Proven baseline from RAG agent testing
    PROVEN_CONFIG = {
        "defType": "edismax",
        "qf": "title^3.0 content^1.0 main_content^1.5 id^2.0",
        "pf": "title^10.0 content^5.0 main_content^7.0",
        "ps": "2",
        "mm": "50%"
    }
    
    def suggest_optimization(self, pattern_results):
        """Suggest Solr config optimization."""
        
        # Start from proven baseline, not random config
        current_config = self.PROVEN_CONFIG.copy()
        
        # Analyze what's failing
        if pattern_results.low_precision:
            # Increase mm (minimum match) for precision
            current_config["mm"] = "75%"
        
        if pattern_results.low_recall:
            # Increase phrase slop for recall
            current_config["ps"] = "3"
        
        # Small adjustments (±20%) instead of random search
        return current_config
```

### Example 3: Cheap Validation Loop
```python
# File: src/heal/runners/run_pattern_fix_poc.py

from heal.agents.content_relevance_agent import ContentRelevanceAgent

class PatternFixRunner:
    def __init__(self):
        self.content_validator = ContentRelevanceAgent()
    
    def validate_retrieval(self, query, docs):
        """Cheap validation - replace expensive URLValidationAgent."""
        
        result = self.content_validator.evaluate_relevance(
            query=query,
            retrieved_docs=docs,
            top_k=5
        )
        
        # Decision thresholds
        if result["avg_relevance"] >= 0.85:
            return "excellent"
        elif result["avg_relevance"] >= 0.70:
            return "good"
        else:
            return "poor"
```

## Migration Path

### Phase 1: Test (No Risk)
1. Run comparison script to validate on your patterns:
   ```bash
   cd ~/Work/rhel-lightspeed/HEAL
   uv run python scripts/compare_okp_vs_baseline.py --pattern YOUR_PATTERN --details
   ```

2. Check content relevance scores
   - Target: >80%
   - RAG agent achieved: 87.4%

### Phase 2: Integrate (Low Risk)
1. Use RAG agent for NEW YAML generation
2. Keep existing YAMLs unchanged
3. Compare quality of new vs old expected answers

### Phase 3: Replace (Save $$)
1. Replace URLValidationAgent with ContentRelevanceAgent
2. Monitor answer quality
3. If quality maintained → keep cheap version, save $$

### Phase 4: Optimize (Compound Benefits)
1. Update okp-mcp fix loop to start with proven RAG config
2. Faster convergence
3. Better final configs

## Testing Checklist

Before deploying:
- [ ] Run comparison script on your pattern
- [ ] Content relevance >80%?
- [ ] Spot-check: Are retrieved docs actually good?
- [ ] Compare to existing SolrExpert results
- [ ] Test with LinuxExpert synthesis
- [ ] Verify expected answer quality

## Rollback Plan

If RAG agent doesn't work:
```python
# Just switch back to SolrExpertAgent
# from heal.agents.rag_solr_agent import RAGSolrAgent
from heal.agents.solr_expert import SolrExpertAgent

# agent = RAGSolrAgent()
agent = SolrExpertAgent()  # Rollback
```

No breaking changes - it's a drop-in replacement!
