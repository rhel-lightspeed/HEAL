# HEAL Demo: The Complete Story
**Autonomous Multi-Agent RAG Fixing with Human-in-the-Loop Safety**

*Updated: April 2026 - Now with interactive review, URL validation, and Jira integration*

---

## 🎯 The Elevator Pitch (30 seconds)

**HEAL is a fully autonomous multi-agent system that:**
1. Extracts quality test cases from JIRA tickets (100% success vs 21% manual)
2. Discovers patterns across failures (fix 10-15 tickets at once)
3. Generates fixes with evaluation-driven iteration
4. **Validates URLs before synthesis** (catches wrong docs early)
5. **Gives you final approval** before committing changes
6. **Automates Jira updates & PR creation** (opt-in, dry-run mode)

**Result:** 60-100x faster than manual, production-ready quality, complete audit trail.

---

## 🏗️ The Architecture: Five Specialized Agents

```
┌─────────────────────────────────────────────────────────────────┐
│                    HEAL Multi-Agent System                      │
└─────────────────────────────────────────────────────────────────┘
          ↓                    ↓                    ↓
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Linux Expert    │  │   Solr Expert    │  │  Review Agent    │
│  15+ yrs RHEL    │  │  Doc Search +    │  │  Quality Gate    │
│  Forms answers   │  │  Verification    │  │  Score ≥ 0.7     │
└──────────────────┘  └──────────────────┘  └──────────────────┘
          ↓                    ↓                    ↓
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ URL Validation   │  │ Pattern Discovery│  │  Fix Agent       │
│ Checks docs      │  │ Clusters         │  │  Interactive     │
│ BEFORE synthesis │  │ failures         │  │  Review          │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

### NEW: URL Validation Agent
**The Problem:** LinuxExpert was synthesizing answers from WRONG docs
- Query: "How to UPDATE GRUB?"
- Retrieved: "How to REINSTALL GRUB" ❌
- Result: Correct format, wrong content

**The Solution:** Validate URLs BEFORE synthesis
- Search Solr with query → get candidate URLs
- URLValidationAgent checks: "Do these docs actually answer the question?"
- If NO → retry search with better queries (max 2 attempts)
- If YES → proceed to synthesis with VALIDATED docs

**Impact:** Reduces answer refinement cycles, catches wrong docs early, saves tokens

### NEW: Interactive Review
**Every code change** gets human approval:
1. Agent shows reasoning & proposed change
2. You approve/reject BEFORE edit
3. Change applied → git diff shown
4. Final approve/reject prompt
5. If rejected → `git restore` (instant revert)
6. If approved → test runs, commits only if passing

**Safety:** Two approval checkpoints, easy rollback, stays on fix branch (never merges to main)

---

## 📊 Live Demo Flow (30 minutes)

### Part 1: Bootstrap Pipeline (15 min)

#### Stage 1: Extract Tickets with URL Validation
```bash
uv run python src/heal/bootstrap/extract_jira_tickets.py \
    --tickets RSPEED-1723,RSPEED-1724,RSPEED-1725 \
    --force-reextract
```

**Watch the logs:**
```
[RSPEED-1723] Searching Solr and validating URLs...
  Query: How to update GRUB in RHEL 9?
  Found: 5 docs from Solr
  
[URL Validation] Checking if docs answer the query...
  Validation score: 0.35
  Passes: False
  ❌ Validation failed:
     - Doc 'How to reinstall GRUB' is about reinstall, not UPDATE
  
  🔧 Retrying search with suggested queries...
  New query: "RHEL 9 update GRUB command grub2-mkconfig"
  Found: 4 docs from Solr
  
[URL Validation] Retry validation...
  Validation score: 0.88
  Passes: True
  ✅ URLs validated on attempt 2

[Linux Expert] Synthesizing answer from VALIDATED docs...
  ✅ High-quality answer on first try (because docs are correct!)
```

**Key Point:** URL validation caught the wrong docs BEFORE wasting tokens on synthesis!

#### Stage 2: Pattern Discovery
```bash
uv run python scripts/validate_yaml_urls.py \
    --pattern BOOTLOADER_GRUB_ISSUES \
    --auto-fix \
    --dry-run
```

**Show the output:**
```
Validation Summary: BOOTLOADER_GRUB_ISSUES
  ✅ Passed (unchanged): 1
  📝 Updated: 2 (would update in-place)
  
URL Changes:
  RSPEED-1723: updated
    Old URLs: ['solutions/3486741', ...]  # Reinstall docs
    New URLs: ['solutions/1521', ...]     # Update docs
```

**Key Point:** Can fix pattern YAMLs in-place without full re-extraction!

---

### Part 2: Pattern Fixing with Interactive Review (15 min)

```bash
./runners/fix.sh BOOTLOADER_GRUB_ISSUES
# Note: Interactive review is DEFAULT ON
```

**Watch the interactive flow:**

```
================================================================================
🤖 AGENT REASONING
================================================================================
Context: Iteration 1/10

File: src/okp_mcp/solr.py

Reasoning: The tickets show that UPDATE GRUB queries are retrieving 
REINSTALL GRUB documentation. This suggests the query weights need 
adjustment to prefer "update" over "reinstall" terms.

Suggested Change: Boost "update" and "grub2-mkconfig" terms in qf parameter

Expected Improvement: URL F1 should improve from 0.32 to 0.70+

Confidence: HIGH (0.92)

Code Snippet:
  "qf": "text^1.0 title^2.0 grub-update^5.0 grub2-mkconfig^3.0"

================================================================================
Proceed with applying this change? (y/n): y ← YOU APPROVE HERE

📝 Applying LLM suggestion to solr.py...
✅ File solr.py modified successfully

================================================================================
📊 GIT DIFF
================================================================================
diff --git a/src/okp_mcp/solr.py b/src/okp_mcp/solr.py
--- a/src/okp_mcp/solr.py
+++ b/src/okp_mcp/solr.py
-    "qf": "text^1.0 title^2.0"
+    "qf": "text^1.0 title^2.0 grub-update^5.0 grub2-mkconfig^3.0"

Does this diff look correct?
  y - Approve and TEST (will only commit if test passes)
  n - Revert changes
Choice (y/n): y ← YOU APPROVE AGAIN

✅ Change approved - will test and commit only if test passes

Running evaluation...
  url_f1: 0.78 ✅ (improved from 0.32!)
  answer_correctness: 0.92 ✅
  
✅ Test passed! Committing change...
```

**Key Points:**
- **Two approval points:** Before edit + after seeing diff
- **Easy rollback:** Type 'n' to revert instantly
- **Test-before-commit:** Only commits if metrics improve
- **Git isolation:** Stays on fix branch, never merges to main

---

### Part 3: Jira Integration & PR Creation (5 min - DEMO ONLY)

```bash
# Preview Jira comments (NO POSTING)
./runners/fix.sh BOOTLOADER_GRUB_ISSUES --dry-run-integrations
```

**Show the preview:**
```
================================================================================
JIRA INTEGRATION: Updating Tickets (DRY RUN)
================================================================================
   Pattern: BOOTLOADER_GRUB_ISSUES
   Tickets: 3
   Dry Run: True

[DRY RUN] Would post to RSPEED-1723
  Preview: ## 🤖 Automated Pattern Fix: Bootloader Grub Issues
  **Status:** ✅ Fix Applied | Branch: `fix/pattern-bootloader-grub-issues`
  
  📊 Results Summary:
  | Metric | Before | After | Change |
  |--------|--------|-------|--------|
  | URL F1 | 0.32 | 0.78 | +0.46 ✅ |
  | Answer Correctness | 0.30 | 0.92 | +0.62 ✅ |
  ...

Full preview saved to: .diagnostics/BOOTLOADER_GRUB_ISSUES/JIRA_COMMENTS_PREVIEW.md
To post these comments, re-run with: --enable-jira
```

**Review the full comment:**
```bash
cat .diagnostics/BOOTLOADER_GRUB_ISSUES/JIRA_COMMENTS_PREVIEW.md
```

**Key Points:**
- **Default: OFF** - No accidental Jira posts
- **Dry-run preview** - See exactly what would be posted
- **Comprehensive** - Metrics, reasoning, warnings, next steps
- **Per-ticket** - Each ticket gets full context

---

## 🎨 The Beautiful Complexity

### The Autonomous Quality Loop
```
┌─────────────────────────────────────────────────────────────┐
│  1. LinuxExpert forms hypothesis                            │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  2. SolrExpert searches docs                                │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  3. ✨ URLValidationAgent validates docs ✨                 │
│     - Pass: Continue                                        │
│     - Fail: Retry search (max 2 attempts)                   │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  4. LinuxExpert synthesizes from VALIDATED docs             │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  5. AnswerReviewAgent checks quality                        │
│     - Score ≥ 0.7: PASS                                     │
│     - Score < 0.7: Refine (up to 3x)                        │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
              ✅ Production-ready Q&A
```

### The Fix Loop with Human-in-the-Loop
```
┌─────────────────────────────────────────────────────────────┐
│  1. Baseline evaluation (identify problem)                  │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  2. Multi-agent diagnosis (Solr + Code experts)             │
│     → Synthesized suggestion with reasoning                 │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  3. ✨ Human approval: Proceed? (y/n) ✨                    │
└─────────┬───────────────────────────────────┬───────────────┘
          ↓ YES                               ↓ NO
┌─────────────────────┐              ┌───────────────────────┐
│  Apply change       │              │  Skip iteration       │
└──────────┬──────────┘              └───────────────────────┘
           ↓
┌─────────────────────────────────────────────────────────────┐
│  4. Show git diff                                           │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  5. ✨ Human approval: Looks good? (y/n) ✨                 │
└─────────┬───────────────────────────────────┬───────────────┘
          ↓ YES                               ↓ NO
┌─────────────────────┐              ┌───────────────────────┐
│  Test + commit      │              │  git restore (revert) │
└──────────┬──────────┘              └───────────────────────┘
           ↓
┌─────────────────────────────────────────────────────────────┐
│  6. Evaluation (did it improve?)                            │
│     → Iterate until stable or max iterations                │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
              ✅ Improved retrieval on fix branch
```

---

## 🚀 Key Flags & Options

### Bootstrap Flags
```bash
# Re-extract with URL validation (updates pattern YAML in-place)
uv run python scripts/validate_yaml_urls.py \
    --pattern BOOTLOADER_GRUB_ISSUES \
    --auto-fix              # Search for better URLs if validation fails
    --dry-run               # Preview changes without saving
```

### Fix Loop Flags
```bash
./runners/fix.sh PATTERN_ID [OPTIONS]

# Interactive Review (DEFAULT: ON)
# No flag needed - you'll be prompted for approval

# YOLO mode (skip review, auto-approve)
./runners/fix.sh PATTERN_ID --yolo

# Jira Integration (DEFAULT: OFF)
./runners/fix.sh PATTERN_ID --enable-jira          # Actually post
./runners/fix.sh PATTERN_ID --dry-run-integrations # Preview only

# PR Creation (DEFAULT: OFF)
./runners/fix.sh PATTERN_ID --create-pr            # Auto-create PR

# A/B Testing
./runners/fix.sh PATTERN_ID --include-judge-reasoning  # Test judge critique impact
```

### Common Workflows
```bash
# Safe exploration (recommended first run)
./runners/fix.sh PATTERN_ID --dry-run-integrations

# Production run (post to Jira, create PR)
./runners/fix.sh PATTERN_ID --enable-jira --create-pr

# Fast automated run (skip all prompts)
./runners/fix.sh PATTERN_ID --yolo --enable-jira --create-pr
```

---

## 📈 Results & Impact

### Extraction Pipeline
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Success Rate | 21% | **100%** | 4.8x |
| Time to Extract | 2-4 hours | 10-15 min | 10-20x faster |
| URL Accuracy | Unknown | Validated | ✅ Verified |
| Answer Quality | Unverified | Score ≥ 0.7 | ✅ Production-ready |

### Fix Pipeline
| Metric | Manual | HEAL | Improvement |
|--------|--------|------|-------------|
| Answer Refinement Cycles | N/A | Reduced ~30% | ✅ URL validation |
| Code Review | Manual | Interactive | ✅ Human-in-loop |
| Jira Updates | Manual | Automated | ✅ Dry-run preview |
| Git Safety | Risk of merge | Fix branch only | ✅ PR-based |

---

## 🎤 Demo Script for Presentation

### Opening (2 min)
"I'm going to show you something that transforms RAG fixing from days of manual work into hours of autonomous operation - with human oversight at the critical points."

### The Problem (3 min)
Show JIRA board with 68 tickets. Explain:
- Each ticket = wrong answer to user
- Manual extraction: 21% success, hallucinations
- Manual fixing: slow, doesn't scale
- **NEW problem discovered:** Even when extraction succeeds, we were synthesizing from WRONG docs!

### The Solution Architecture (5 min)
Walk through the agent diagram:
- Five specialized agents working together
- **NEW: URLValidationAgent** - catches wrong docs before synthesis
- **NEW: Interactive review** - human approval at critical points
- Autonomous quality loop with human oversight

### Live Demo Part 1: Bootstrap (10 min)
Run URL validation script:
```bash
uv run python scripts/validate_yaml_urls.py \
    --pattern BOOTLOADER_GRUB_ISSUES \
    --auto-fix \
    --dry-run
```

Narrate:
- "Watch it search Solr..."
- "See it detect wrong docs..."
- "Now it retries with better queries..."
- "Validation passes! URLs are correct before synthesis"

### Live Demo Part 2: Interactive Fix (10 min)
Run fix loop:
```bash
./runners/fix.sh BOOTLOADER_GRUB_ISSUES
```

Pause at each approval point:
- "Here's where YOU review the agent's reasoning"
- "Here's where YOU see the actual code change"
- "Type 'y' to approve, 'n' to revert instantly"

### Live Demo Part 3: Jira Preview (3 min)
Show dry-run:
```bash
./runners/fix.sh BOOTLOADER_GRUB_ISSUES --dry-run-integrations
cat .diagnostics/BOOTLOADER_GRUB_ISSUES/JIRA_COMMENTS_PREVIEW.md
```

Point out:
- "This is what WOULD be posted to Jira"
- "You review it first"
- "Only post when ready with --enable-jira"

### Closing (2 min)
"HEAL gives you the best of both worlds:
- Autonomous operation (100x faster than manual)
- Human oversight (you approve the critical decisions)
- Production quality (validated docs, tested code)
- Complete safety (dry-run mode, git isolation)"

**Total time:** 35 minutes + Q&A

---

## 💡 Key Messages

### What Makes HEAL Different
1. **Not just automation** - It's autonomous AGENTS with human oversight
2. **Not just extraction** - End-to-end: JIRA → pattern → fix → PR
3. **Not just fast** - VALIDATED quality at every step
4. **Not just smart** - SAFE with interactive review

### The Innovation Stack
- ✅ Multi-agent collaboration (5 specialized agents)
- ✅ Autonomous quality loops (100% success on valid tickets)
- ✅ URL validation (catches wrong docs early)
- ✅ Interactive review (human-in-the-loop safety)
- ✅ Jira integration (with dry-run preview)
- ✅ Git safety (fix branches, never auto-merge)

### The Business Value
- **Speed:** 60-100x faster than manual
- **Quality:** Production-ready, validated
- **Safety:** Human approval, easy rollback
- **Scale:** Pattern-based (fix 10-15 tickets at once)
- **Audit:** Complete trail (URLs, metrics, reasoning)

---

## 📚 Resources for Audience

**Try it yourself:**
```bash
git clone [repo]
./scripts/demo_heal_workflow.sh --quick
```

**Documentation:**
- Architecture: `docs/DESIGN_INTENT_AND_INTEGRATION.md`
- Bootstrap guide: `docs/BOOTSTRAP_GUIDE.md`
- Demo plan: `docs/DEMO_PLAN.md`
- One-pager: `docs/HEAL_ONE_PAGER.md`

**Contact:**
- GitHub: [Coming Soon]
- Questions: See README

---

*This is the complete story of HEAL: Autonomous intelligence + Human oversight = Production-ready RAG fixes at scale.*
