# Pattern Database Integration Guide

This guide shows how to enhance the existing `iteration_history` (single-ticket learning) with cross-ticket pattern learning.

## Current State: Single-Ticket Learning

```python
# okp_mcp_agent.py - fast_retrieval_loop()
iteration_history = []  # ⚠️ Lost after ticket completes

for iteration in range(max_iterations):
    # Pass history to LLM
    metrics.iteration_history = iteration_history
    
    # Get suggestion (learns from THIS ticket's attempts)
    suggestion = llm_advisor.suggest_boost_query_changes(metrics)
    
    # Test and record
    iteration_history.append({
        'iteration': iteration,
        'change': suggestion.suggested_change,
        'improved': new_metrics.url_f1 > old_metrics.url_f1,
    })
```

**Problem**: Next ticket starts from scratch, wastes LLM calls re-discovering same patterns.

## Enhanced: Cross-Ticket Pattern Learning

```python
from heal.core.fix_pattern_database import FixPatternDatabase, FixPattern

class OkpMcpAgent:
    def __init__(self, ...):
        ...
        # Add pattern database
        self.pattern_db = FixPatternDatabase()
    
    def fast_retrieval_loop(self, ticket_id, query, expected_urls, ...):
        iteration_history = []
        
        # STEP 1: Check pattern database BEFORE calling LLM
        print("🔍 Checking pattern database for similar past fixes...")
        
        # Get baseline
        baseline = self.diagnose_retrieval_only(ticket_id, iteration=0)
        
        # Extract explain pattern
        explain_pattern = self._extract_explain_pattern(baseline)
        
        # Find similar past fixes
        similar_pattern = self.pattern_db.find_similar_patterns(
            query=query,
            url_f1=baseline.url_f1,
            explain_pattern=explain_pattern,
            min_similarity=0.7,
        )
        
        reused_from = None
        if similar_pattern:
            if similar_pattern.success_rate_when_reused > 0.7:
                # High confidence - try this fix first!
                print(f"✨ Found high-confidence pattern from {similar_pattern.ticket_id}")
                print(f"   Fix: {similar_pattern.specific_change}")
                print(f"   Success rate when reused: {similar_pattern.success_rate_when_reused:.0%}")
                
                # Try the known good fix first
                first_suggestion = type('Suggestion', (), {
                    'suggested_change': similar_pattern.specific_change,
                    'file_path': similar_pattern.file_path,
                    'reasoning': f"Reusing successful pattern from {similar_pattern.ticket_id}",
                    'confidence': 'high',
                })()
                
                reused_from = similar_pattern.ticket_id
            else:
                # Medium confidence - inform LLM
                print(f"💡 Found similar pattern from {similar_pattern.ticket_id}")
                print(f"   Will inform LLM advisor")
                
                # Add to metrics so LLM can consider it
                baseline.similar_past_fix = {
                    'ticket_id': similar_pattern.ticket_id,
                    'fix': similar_pattern.specific_change,
                    'improvement': similar_pattern.improvement,
                }
        else:
            print("📭 No similar patterns found, LLM will suggest new approach")
            first_suggestion = None
        
        # STEP 2: Optimization loop (with or without pattern hint)
        for iteration in range(1, max_iterations + 1):
            # Use pattern DB suggestion on first iteration if available
            if iteration == 1 and first_suggestion:
                suggestion = first_suggestion
            else:
                # Normal LLM suggestion with history
                metrics = self._prepare_metrics(baseline, iteration_history)
                suggestion = await self.llm_advisor.suggest_boost_query_changes(metrics)
            
            # Apply fix
            self.apply_code_change(suggestion)
            self.restart_okp_mcp()
            
            # Test
            new_result = self.diagnose_retrieval_only(ticket_id, iteration=iteration)
            
            improvement = new_result.url_f1 - baseline.url_f1
            
            # Record in iteration history
            iteration_record = {
                'iteration': iteration,
                'change': suggestion.suggested_change,
                'url_f1_before': baseline.url_f1,
                'url_f1_after': new_result.url_f1,
                'improvement': improvement,
                'improved': improvement > 0.1,
                'reused_from': reused_from if iteration == 1 else None,
            }
            iteration_history.append(iteration_record)
            
            # Check if fixed
            if new_result.url_f1 > 0.5:
                print(f"✅ Fixed! URL F1: {baseline.url_f1:.2f} → {new_result.url_f1:.2f}")
                
                # STEP 3: Record successful pattern for future use
                pattern = FixPattern(
                    # Problem signature
                    problem_type=self._classify_problem_type(baseline),
                    url_f1_before=baseline.url_f1,
                    query_keywords=query.lower().split()[:10],
                    explain_pattern=explain_pattern,
                    
                    # Solution
                    fix_type=self._classify_fix_type(suggestion),
                    specific_change=suggestion.suggested_change,
                    file_path=suggestion.file_path,
                    
                    # Outcome
                    url_f1_after=new_result.url_f1,
                    improvement=improvement,
                    success=True,
                    
                    # Metadata
                    ticket_id=ticket_id,
                    timestamp=datetime.now().isoformat(),
                    iterations_to_fix=iteration,
                    cost=self._estimate_cost(iteration),
                )
                
                self.pattern_db.record_fix(pattern, reused_from=reused_from)
                print(f"💾 Pattern saved for future tickets")
                
                return True
        
        # STEP 4: Record failed attempts too (learn what NOT to do)
        if iteration_history:
            for record in iteration_history:
                if not record['improved']:
                    pattern = FixPattern(
                        problem_type=self._classify_problem_type(baseline),
                        url_f1_before=baseline.url_f1,
                        query_keywords=query.lower().split()[:10],
                        explain_pattern=explain_pattern,
                        fix_type=self._classify_fix_type(record['change']),
                        specific_change=record['change'],
                        file_path='src/okp_mcp/solr.py',
                        url_f1_after=record['url_f1_after'],
                        improvement=record['improvement'],
                        success=False,  # ⚠️ Failed fix
                        ticket_id=ticket_id,
                        timestamp=datetime.now().isoformat(),
                        iterations_to_fix=iteration,
                        cost=0.01,
                    )
                    self.pattern_db.record_fix(pattern)
        
        return False
    
    def _extract_explain_pattern(self, result) -> str:
        """Extract a pattern signature from Solr explain output.
        
        Examples:
            "title_weight_low" - Expected docs have query in title but rank low
            "missing_keywords" - Expected docs not retrieved at all
            "phrase_match_weak" - Expected docs retrieved but outranked
        """
        if not result.solr_explain:
            return "unknown"
        
        # Simple heuristic pattern extraction
        # In practice, could use LLM to classify the explain output
        
        if result.url_f1 == 0.0:
            return "missing_documents"
        
        # Check if expected docs are in results but ranked low
        expected_in_results = False
        for expected_url in result.expected_urls or []:
            if any(expected_url in doc.get('url', '') for doc in result.solr_explain.get('docs', [])):
                expected_in_results = True
                break
        
        if expected_in_results:
            # Docs found but ranked too low
            return "ranking_issue"
        else:
            # Docs not in top results
            return "retrieval_failure"
    
    def _classify_problem_type(self, result) -> str:
        """Classify the type of problem for pattern matching."""
        if result.url_f1 == 0.0:
            return "zero_retrieval"
        elif result.url_f1 < 0.3:
            return "poor_retrieval"
        elif result.context_relevance and result.context_relevance < 0.7:
            return "low_relevance"
        else:
            return "ranking_issue"
    
    def _classify_fix_type(self, suggestion) -> str:
        """Classify the type of fix for pattern categorization."""
        change = suggestion.suggested_change.lower() if isinstance(suggestion, object) else str(suggestion).lower()
        
        if "title" in change and ("^" in change or "boost" in change):
            return "boost_title"
        elif "qf" in change or "field weight" in change:
            return "adjust_field_weights"
        elif "pf" in change or "phrase" in change:
            return "increase_phrase_boost"
        elif "mm" in change or "minimum match" in change:
            return "adjust_minimum_match"
        elif "keyword" in change or "_BOOST_KEYWORDS" in change:
            return "add_boost_keywords"
        elif "snippet" in change or "hl." in change:
            return "adjust_highlighting"
        else:
            return "other"
    
    def _estimate_cost(self, iterations: int) -> float:
        """Estimate LLM API cost for this fix."""
        # Rough estimate: $0.01 per suggestion call
        return iterations * 0.01
```

## Usage Example

```python
# Pattern 1 discovered
ticket_1 = "RSPEED-1001: grub bootloader uefi"
# After 3 iterations, found fix: increase title^5 → title^7
# Records: FixPattern(problem="missing_documents", fix="boost_title", success=True)

# Pattern 1 reused!
ticket_2 = "RSPEED-1002: grub rescue mode"  
# Similar query → pattern DB returns high-confidence match
# Tries title^7 on first iteration → works! ✅
# Updates pattern: times_reused=1, success_rate_when_reused=100%

# Pattern confirmed
ticket_3 = "RSPEED-1003: grub configuration"
# Again matches pattern → tries title^7 → works! ✅
# Updates pattern: times_reused=2, success_rate_when_reused=100%

# Different problem
ticket_4 = "RSPEED-1004: selinux policy"
# No similar pattern → LLM suggests new fix
# Discovers new pattern for SELinux queries...
```

## Benefits

1. **Faster Fixes**: High-confidence patterns work on first iteration (saves 2-3 LLM calls)
2. **Cost Savings**: Reusing patterns saves ~$0.02-0.03 per ticket
3. **Learning Across Tickets**: System gets smarter over time
4. **Transparency**: Can export learning reports showing what patterns work

## Reports

Generate learning reports to see what the system has learned:

```python
pattern_db = FixPatternDatabase()

# View statistics
stats = pattern_db.get_statistics()
print(f"Total patterns: {stats['total_patterns']}")
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Most common fix: {stats['most_common_fixes'][0]}")

# Export detailed report
pattern_db.export_learning_report(Path(".diagnostics/learning_report.md"))
```

## Next Steps

1. ✅ Add `FixPatternDatabase` to `OkpMcpAgent.__init__`
2. ✅ Integrate pattern lookup before LLM calls
3. ✅ Record successful/failed patterns
4. 📊 Run on 20-30 tickets to build initial pattern database
5. 📈 Benchmark: Compare with/without pattern DB
6. 🔬 Research paper: "Learning to Fix RAG Systems"
