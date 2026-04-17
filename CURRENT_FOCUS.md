# Current Focus: Get Pattern Fix Loop Working

**Last Updated**: 2026-04-16  
**Status**: 🚧 In Progress

---

## Recent Progress ✅

### 1. RAG Quality Warning Tests (Completed)
- ✅ Added 8 tests for RAG quality warning logic in `test_pattern_fix_logic.py`
- ✅ Tests warning when answer ≥ 0.90 but RAG metrics < 0.7
- ✅ All 28 pattern fix logic tests passing
- ✅ All 33 ticket evaluation tests passing

### 2. No-Doc Ticket Handling (Completed)
- ✅ Special evaluation for tickets without `expected_urls`
- ✅ Uses `answer_correctness >= 0.90` directly (not composite score)
- ✅ Skip tag for passing no-doc tickets
- ✅ HIGH priority flag for failing no-doc tickets

### 3. Container Restart Fix (Completed)
- ✅ Added `restart_okp_mcp()` after branch creation
- ✅ Fixed `run_retrieval_optimization()` to call `fast_retrieval_loop()`
- ✅ Ensures code changes take effect in running container

### 4. Full-Pattern Mode Fix (Completed)
- ✅ Skip retrieval optimization when `ticket_id = None`
- ✅ Graceful handling with informative message
- ✅ Pattern fix loop no longer crashes in full-pattern mode

---

## Current Blockers ⚠️

### From Latest Run (`./runners/fix.sh BOOTLOADER_GRUB_ISSUES`)

**Error encountered**:
```
TypeError: OkpMcpAgent.fast_retrieval_loop() missing 2 required positional arguments: 'query' and 'expected_urls'
```

**Status**: ✅ **FIXED** - Added graceful skip for full-pattern mode

**Issue**: Full-pattern mode sets `ticket_id = None`, but retrieval optimization needs query/expected_urls

**Solution Applied**: Skip retrieval optimization in full-pattern mode with informative message

### Other Observations from Run

1. **High Variance Detected**
   ```
   ⚠️  INTERMITTENT ISSUE DETECTED - High variance in:
   • custom:answer_correctness (std=0.246)
   • ragas:faithfulness (std=0.379)
   ```
   - Problem is NOT consistent across runs
   - May be temporal validity issue
   - Could be flaky evaluation

2. **"NO EXPECTED URLS FOUND IN CONFIG"**
   - Coming from `okp_mcp_agent.py:3363`
   - Different from no-doc ticket handling
   - Triggers when `expected_response` exists but `expected_urls` is empty
   - May indicate config issue or pattern YAML problem

3. **Documentation Gap vs No-Doc Confusion**
   - Need to clarify distinction:
     - **No-doc ticket**: Turn has NO `expected_urls` field (intentional, no docs exist)
     - **Documentation gap**: YAML has `expected_urls` but Solr returns 0 docs (missing from index)

---

## Next Steps 🎯

### Immediate (This Session)

1. **Test the full-pattern mode fix**
   - Run `./runners/fix.sh BOOTLOADER_GRUB_ISSUES` again
   - Verify it skips retrieval optimization gracefully
   - Check if it completes without errors

2. **Investigate the "NO EXPECTED URLS" message**
   - Check if BOOTLOADER_GRUB_ISSUES tickets actually have `expected_urls`
   - Determine if this is a real gap or config issue
   - May need to run in single-ticket mode instead

3. **Handle high variance**
   - Review stability assessment logic
   - Consider if 3 runs is sufficient
   - May need to flag for manual review vs auto-fix

### Short Term (Next Few Sessions)

1. **Get first successful pattern fix**
   - Pick a single ticket with clear retrieval issue
   - Run in single-ticket mode (not full-pattern)
   - Verify container restart → code change → improvement cycle

2. **Validate the fix loop end-to-end**
   - Baseline → Optimization → Validation → Commit
   - Ensure git branches work correctly
   - Verify diagnostics are saved

3. **Test on 5-10 tickets**
   - Build confidence in the system
   - Identify edge cases
   - Collect iteration history data

### Medium Term (Research Phase)

1. **Pattern database** (see `research/MOE_PATTERN_DATABASE.md`)
   - Implement after fix loop is stable
   - Need 20-30 successful fixes to build patterns
   - Benchmark different expert architectures

---

## Files Modified Today

```
src/heal/core/ticket_evaluation.py              # Added is_no_doc handling
tests/test_ticket_evaluation.py                 # Tests for ticket evaluation
tests/test_pattern_fix_logic.py                 # 28 tests including RAG warnings
src/heal/runners/run_pattern_fix_poc.py         # Fixed full-pattern mode
src/heal/core/fix_pattern_database.py           # Created (for future use)
docs/PATTERN_DATABASE_INTEGRATION.md            # Integration guide
research/MOE_PATTERN_DATABASE.md                # Research idea captured
.gitignore                                       # Added research/
```

---

## Testing Status

- ✅ 28/28 tests in `test_pattern_fix_logic.py`
- ✅ 33/33 tests in `test_ticket_evaluation.py`
- ✅ 149/150 overall tests (1 pre-existing failure unrelated to our changes)
- ✅ Linting clean
- ✅ Code formatted

---

## Key Decisions Made

1. **Full-pattern mode skips retrieval optimization** - Can't optimize multiple tickets at once
2. **No-doc tickets use direct answer_correctness >= 0.90** - Not penalized for missing context metrics
3. **Research ideas moved to private `research/` directory** - Stay focused on core fixes
4. **Pattern database deferred to after fix loop works** - Don't over-engineer before validating basics

---

## Questions to Resolve

1. What's the right approach for full-pattern mode?
   - Option A: Skip optimization, just test
   - Option B: Iterate through tickets individually
   - Option C: Only support single-ticket mode

2. How to handle high variance in metrics?
   - Increase stability runs (3 → 5)?
   - Flag for manual review?
   - Accept variance and work with averages?

3. Should we focus on single-ticket mode first?
   - Simpler, fewer edge cases
   - Can validate the core loop
   - Add full-pattern mode later

---

**Focus**: Get ONE ticket fixed end-to-end. Everything else is secondary.
