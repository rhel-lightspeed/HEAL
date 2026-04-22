# HEAL Agents Analysis - Redundancies and Recommendations

## Agent Inventory

### Core Expert Agents (Basic Building Blocks)
| Agent | Size | Purpose | Used By |
|-------|------|---------|---------|
| `solr_expert.py` | 11K | Simple Solr HTTP search with smart params | Bootstrap, validation, multi-agent |
| `linux_expert.py` | 29K | RHEL/Linux expertise for JIRA ticket analysis | okp_mcp_agent, bootstrap |
| `url_validation_agent.py` | 7.7K | Validates if retrieved URLs answer the question | All agents, fix loops |
| `answer_review_agent.py` | 6.6K | Reviews extracted answers for quality | Bootstrap refinement |

### Solr Optimization Agents (REDUNDANCY AREA)
| Agent | Size | Purpose | Architecture | Used By |
|-------|------|---------|--------------|---------|
| `okp_mcp_llm_advisor.py` | 65K | ⚠️ Suggests Solr config/boost/prompt changes | **Single LLM** with smart routing (Haiku/Sonnet/Opus) | okp_mcp_agent.py |
| `solr_multi_agent.py` | 25K | ⚠️ Suggests Solr config improvements | **3 LLM agents**: Theory → Code → Synthesizer | run_pattern_fix_poc.py, fix_agent_debugger.py |

**FINDING 1:** These two do the SAME JOB with different approaches:
- Both analyze ticket metrics
- Both suggest Solr parameter improvements
- Both output suggested code changes

### Solr Diagnostic Agents (REDUNDANCY AREA)
| Agent | Size | Purpose | Used By |
|-------|------|---------|---------|
| `okp_solr_checker.py` | 7.6K | ⚠️ Check if specific URLs/docs exist in Solr | okp_mcp_agent diagnostics |
| `okp_solr_config_analyzer.py` | 18K | ⚠️ Parse config, get explain output, analyze ranking | okp_mcp_agent, LLM advisor |

**FINDING 2:** These overlap in functionality:
- `okp_solr_checker`: Focused on document existence checks
- `okp_solr_config_analyzer`: Focused on explain output parsing and ranking analysis
- Both query Solr for diagnostics
- **Could be merged** into `solr_diagnostics.py`

### Orchestrator Agents
| Agent | Size | Purpose |
|-------|------|---------|
| `okp_mcp_agent.py` | 267K | Main autonomous fix agent (diagnosis → optimization → validation) |
| `okp_mcp_pattern_agent.py` | 17K | Pattern-based batch fixing (finds patterns across tickets) |

---

## Redundancy Analysis

### 🔴 CRITICAL: Solr Optimization Duplication

**Problem:** Two different agents doing the same job:

#### `okp_mcp_llm_advisor.py` (65K)
```python
class OkpMcpLLMAdvisor:
    async def suggest_solr_config_changes(metrics: MetricSummary) -> SolrConfigSuggestion
    async def suggest_boost_query_changes(metrics: MetricSummary) -> SolrConfigSuggestion
    async def suggest_prompt_changes(metrics: MetricSummary) -> PromptSuggestion
    
    # Smart routing by complexity
    async def classify_problem_complexity() -> str  # SIMPLE/MEDIUM/COMPLEX
    # Routes to Haiku/Sonnet/Opus based on complexity
```

#### `solr_multi_agent.py` (25K)
```python
class SolrMultiAgentSystem:
    async def get_optimized_suggestion(ticket_data: TicketData) -> SynthesizedSuggestion
    
    # Multi-agent deliberation
    async def _get_solr_theory_advice() -> SolrTheoryAdvice
    async def _get_okp_mcp_code_analysis() -> OkpMcpCodeAnalysis  
    async def _synthesize_suggestion() -> SynthesizedSuggestion
```

**Current Usage:**
- `okp_mcp_agent.py` uses `OkpMcpLLMAdvisor`
- `run_pattern_fix_poc.py` uses `SolrMultiAgentSystem`

**Recommendation:**

**Option A: Keep Both (Current State)**
- **Pro**: Different architectures for comparison (single-LLM vs multi-agent)
- **Pro**: Can A/B test which works better
- **Con**: Code duplication, maintenance burden

**Option B: Merge into One Unified Agent**
```python
class SolrOptimizationAgent:
    """Unified Solr optimization with configurable strategy."""
    
    async def suggest_improvements(
        ticket_data: TicketData,
        strategy: str = "multi-agent"  # or "smart-routing"
    ) -> SuggestionResult:
        if strategy == "multi-agent":
            return await self._multi_agent_approach()
        else:
            return await self._smart_routing_approach()
```
- **Pro**: Single interface, easier to maintain
- **Pro**: Can switch strategies easily
- **Con**: More complex internal logic

**Option C: Deprecate One**
- **If multi-agent works better**: Deprecate `okp_mcp_llm_advisor.py`
- **If smart routing works better**: Deprecate `solr_multi_agent.py`
- **Pro**: Simpler codebase
- **Con**: Lose alternative approach

**My Recommendation:** **Option A** for now (keep both) BUT:
1. Rename for clarity:
   - `okp_mcp_llm_advisor.py` → `solr_smart_routing_agent.py`
   - `solr_multi_agent.py` → `solr_deliberation_agent.py`
2. Add clear docstrings explaining the difference
3. Document which to use when in `AGENTS.md`
4. Run comparison tests to determine which performs better
5. Eventually deprecate the weaker one

---

### 🟡 MODERATE: Solr Diagnostics Overlap

**Problem:** Two agents for Solr diagnostics with overlapping functionality:

#### `okp_solr_checker.py` (7.6K)
```python
class SolrDocumentChecker:
    def check_document_exists(url: str) -> Dict
    def suggest_urls_for_query(query: str) -> List[Dict]
    def check_all_expected_urls(expected_urls: List[str]) -> Dict
    def get_missing_urls(expected_urls: List[str]) -> List[str]
```
**Focus:** Document availability checks

#### `okp_solr_config_analyzer.py` (18K)  
```python
class SolrConfigAnalyzer:
    def parse_current_config() -> Dict
    def get_explain_output(query: str, doc_id: str) -> str
    def analyze_ranking_problems(explain: str) -> Dict
    def search_for_answer_content(query: str, expected_answer: str) -> Dict
```
**Focus:** Config parsing and explain analysis

**Overlap:**
- Both query Solr for diagnostics
- Both analyze search quality issues
- Both used by `okp_mcp_agent` for diagnosis

**Recommendation:** **Merge into single `solr_diagnostics.py`**

```python
class SolrDiagnostics:
    """Unified Solr diagnostic tools."""
    
    # Document checks (from okp_solr_checker)
    def check_document_exists(url: str) -> Dict
    def check_all_expected_urls(expected_urls: List[str]) -> Dict
    def get_missing_urls(expected_urls: List[str]) -> List[str]
    
    # Config analysis (from okp_solr_config_analyzer)
    def parse_current_config() -> Dict
    def get_explain_output(query: str, doc_id: str) -> str
    def analyze_ranking_problems(explain: str) -> Dict
    
    # Combined diagnostics
    def run_full_diagnostics(query: str, expected_urls: List[str]) -> DiagnosticReport
```

**Migration Plan:**
1. Create new `solr_diagnostics.py` with unified interface
2. Update `okp_mcp_agent.py` to use new unified agent
3. Deprecate old agents with warnings
4. Remove after one release cycle

---

## Final Recommendations

### Immediate Actions

1. **Rename for clarity** (no code changes):
   ```bash
   mv okp_mcp_llm_advisor.py solr_smart_routing_agent.py
   mv solr_multi_agent.py solr_deliberation_agent.py
   ```

2. **Add comparison script** to test both Solr optimization approaches:
   ```bash
   scripts/compare_solr_optimization_agents.py
   ```

3. **Merge diagnostics** (refactor):
   ```bash
   # Combine
   okp_solr_checker.py + okp_solr_config_analyzer.py 
   # Into
   solr_diagnostics.py
   ```

4. **Update `AGENTS.md`** with:
   - Clear purpose of each agent
   - When to use which
   - Migration guide for merged agents

### Long-term Strategy

**Test and Deprecate:**
1. Run A/B test: smart-routing vs deliberation for Solr optimization
2. Measure: quality improvement, token cost, latency
3. Keep the winner, deprecate the loser
4. Document the decision

**Target State:**
```
agents/
├── answer_review_agent.py      # Answer quality validation
├── linux_expert.py              # RHEL/Linux expertise
├── solr_expert.py               # Simple Solr HTTP search
├── solr_optimization_agent.py   # WINNER of smart-routing vs deliberation
├── solr_diagnostics.py          # Merged diagnostics (checker + analyzer)
├── url_validation_agent.py      # URL answer validation
├── okp_mcp_agent.py             # Main orchestrator
└── okp_mcp_pattern_agent.py     # Pattern discovery
```

**From 11 files → 8 files** with clearer separation of concerns.

---

## No Redundancy Found

These agents are distinct with no overlap:
- ✅ `answer_review_agent.py` - Unique purpose (answer quality)
- ✅ `linux_expert.py` - Unique domain (Linux/RHEL)
- ✅ `solr_expert.py` - Simple building block
- ✅ `url_validation_agent.py` - Unique purpose (URL validation)
- ✅ `okp_mcp_agent.py` - Main orchestrator
- ✅ `okp_mcp_pattern_agent.py` - Pattern discovery
