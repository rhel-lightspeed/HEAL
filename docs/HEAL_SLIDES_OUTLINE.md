# HEAL Presentation Slides Outline
**For use with Google Slides, PowerPoint, or Keynote**

---

## Slide 1: Title
**HEAL: Autonomous Multi-Agent RAG Fixing**
*With Human-in-the-Loop Safety*

[Your Name]
[Date]

---

## Slide 2: The Problem
**RAG Debugging Doesn't Scale**

- 68 JIRA tickets for incorrect AI answers
- Manual extraction: 21% success (hallucinations)
- 2-4 hours per ticket, requires SME expertise
- No way to find patterns
- **Even successful extractions used WRONG docs!**

[Image: Screenshot of JIRA board]

---

## Slide 3: The Vision
**What if we could...**

✅ Extract quality test cases **automatically** (100% success)  
✅ **Validate docs** before synthesis (catch wrong content early)  
✅ Discover patterns across failures (fix 10-15 tickets at once)  
✅ Generate fixes with **human approval** (safe iteration)  
✅ Automate Jira/PR updates (with dry-run preview)

**60-100x faster than manual, production-ready quality**

---

## Slide 4: Meet the Team - 5 Specialized Agents

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Linux Expert │  │ Solr Expert  │  │ Review Agent │
│ 15+ yrs RHEL │  │ Doc Search   │  │ Quality Gate │
└──────────────┘  └──────────────┘  └──────────────┘
       ↓                 ↓                  ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ URL Validator│  │  Pattern     │  │  Fix Agent   │
│ Pre-synthesis│  │  Discovery   │  │  Interactive │
└──────────────┘  └──────────────┘  └──────────────┘
```

**Each agent has a specialized role**

---

## Slide 5: The Research-Based Approach

**WE DON'T JUST BUILD - WE MEASURE, LEARN, PIVOT**

**Hypothesis:**
- Need LLM-based URL validation ($0.01-0.05/query)
- Multi-agent validation for quality

**Experiment:**
- Compared 3 retrieval strategies on BOOTLOADER pattern
- Measured: URL F1 (exact match) vs Content Relevance (semantic)

**Data-Driven Findings:**
- RAG agent: **87.4% content relevance** (cheap, no LLM!)
- Simple agent: 63.3% content relevance
- URL F1 low (6.7%) BUT semantically correct docs retrieved!

**Pivot Decision:**
- Can replace expensive validation with cheap heuristic
- Save $3-15 per pattern, maintain quality
- **Research validates the cheaper approach works**

[Graph showing content relevance vs cost]

---

## Slide 6: Discovery - The "Expected URLs" Problem

**WHAT THE DATA TAUGHT US:**

Original metric: **URL F1** (did we retrieve URLs 1, 2, 3, 4, 5?)
- RAG agent: 6.7% URL F1 ❌
- Simple agent: 4.4% URL F1 ❌
- **Both look like failures!**

**BUT** checking content relevance revealed:
- RAG agent: **87.4% keyword overlap** ✓
- Retrieved DIFFERENT but VALID docs!

**THE INSIGHT:**
Expected URLs aren't exhaustive - there are MANY valid answers!

**NEW METRIC:**
- ❌ Don't optimize for exact URL matches
- ✅ DO optimize for semantic relevance + spot-check answer quality

**This discovery changed our optimization strategy!**

[Side-by-side comparison of metrics]

---

## Slide 7: How We Benchmark & Iterate

**COMPARISON FRAMEWORK:**

```bash
# Compare retrieval strategies with data
uv run python scripts/compare_okp_vs_baseline.py \
    --pattern BOOTLOADER_GRUB_ISSUES --details
```

**What we test:**
1. **Simple agent**: Basic keyword search (baseline)
2. **RAG agent**: edismax + field boosting (enhanced)  
3. **okp-mcp**: Multi-agent validation (expensive)

**Metrics tracked:**
- URL F1 (exact match)
- Content relevance (semantic)
- Iteration improvements
- Cost per query

**Iteration 1 finding:** QueryParser reformulation made results WORSE
- Original query: 26.7% URL F1 ✓
- Reformulated: 0% URL F1 ✗
- **Pivot:** Don't replace query, AUGMENT with technical term boosts

**This is continuous research, not one-time success**

[Terminal output showing comparison]

---

## Slide 8: The Innovation - Interactive Review

**TWO APPROVAL CHECKPOINTS:**

```
1. Agent proposes change
   └─> "Proceed? (y/n)" ← YOU APPROVE

2. Change applied, git diff shown
   └─> "Looks good? (y/n)" ← YOU APPROVE AGAIN

If NO → git restore (instant revert)
If YES → test runs, commits only if passing
```

**Safety:** Easy rollback, test-before-commit, fix branch only

[Screenshot of approval prompts]

---

## Slide 9: Live Demo - Retrieval Comparison

**WATCH THE RESEARCH IN ACTION:**

```bash
uv run python scripts/compare_okp_vs_baseline.py \
    --pattern BOOTLOADER_GRUB_ISSUES --details
```

**What you'll see:**
1. Simple vs RAG agent head-to-head comparison
2. Iteration-by-iteration improvement tracking
3. **RAG agent: 87.4% content relevance** ✓
4. QueryParser reformulation (iteration 1: WORSE!)
5. Data-driven decision: Pivot strategy mid-test

**THE POINT:**
Not just "it works" - **WHY it works, with data to prove it**

[Terminal output showing iteration details]

---

## Slide 10: Live Demo - Interactive Fix

**WATCH THE APPROVAL FLOW:**

```bash
./runners/fix.sh BOOTLOADER_GRUB_ISSUES
```

**What you'll see:**
1. Agent shows reasoning
2. **Approval #1:** "Proceed?"
3. Change applied
4. Git diff shown
5. **Approval #2:** "Looks good?"
6. Test runs
7. Commits if passing

[Video/GIF of interactive session]

---

## Slide 9: Live Demo - Jira Preview

**DRY-RUN MODE:**

```bash
./runners/fix.sh PATTERN_ID --dry-run-integrations
```

**What you'll see:**
- Comprehensive comment preview
- Metrics (before/after)
- Model reasoning
- Warnings
- Next steps for reviewers

**NO accidental posting!**

[Screenshot of preview comment]

---

## Slide 10: The Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Success Rate** | 21% | **100%** | 4.8x |
| **Time** | 2-4 hrs | 10-15 min | 10-20x |
| **URL Accuracy** | Unknown | **Validated** | ✅ |
| **Review** | Manual | **Interactive** | ✅ |
| **Jira Updates** | Manual | **Automated** | ✅ |

**Real deployment: 68 tickets, 100% success on valid RHEL questions**

---

## Slide 11: The Autonomous Quality Loop

```
1. LinuxExpert forms hypothesis
           ↓
2. SolrExpert searches docs
           ↓
3. ✨ URLValidationAgent validates ✨
   (NEW: catches wrong docs!)
           ↓
4. LinuxExpert synthesizes from VALIDATED docs
           ↓
5. AnswerReviewAgent checks quality
   Score ≥ 0.7: Pass
   Score < 0.7: Refine (up to 3x)
           ↓
   Production-ready Q&A (100% success)
```

**No human intervention needed** (but you can intervene!)

---

## Slide 12: The Fix Loop with Human Oversight

```
1. Baseline evaluation (identify problem)
           ↓
2. Multi-agent diagnosis (propose fix)
           ↓
3. ✨ Human approval: Proceed? (y/n) ✨
           ↓
4. Apply change
           ↓
5. Show git diff
           ↓
6. ✨ Human approval: Looks good? (y/n) ✨
           ↓
7. Test + commit (only if passing)
           ↓
8. Evaluate improvement
           ↓
   Iterate until stable
```

**Best of both worlds:** Autonomous + oversight

---

## Slide 13: Safety Features

**GIT ISOLATION:**
- Creates fix branch (never merges to main)
- You create PR when ready
- Easy rollback at any point

**DRY-RUN MODE:**
- Preview Jira comments
- Preview PR content
- Test integrations safely

**INTERACTIVE REVIEW:**
- Two approval checkpoints
- Type 'n' to revert instantly
- Test before commit

**YOLO MODE (OPTIONAL):**
- `--yolo` flag skips prompts
- For automation/trusted scenarios

---

## Slide 14: Command Reference

**BOOTSTRAP:**
```bash
# Validate/fix URLs in pattern YAML
scripts/validate_yaml_urls.py \
    --pattern BOOTLOADER_GRUB_ISSUES \
    --auto-fix --dry-run
```

**FIX LOOP:**
```bash
# Interactive review (DEFAULT)
./runners/fix.sh PATTERN_ID

# Dry-run Jira preview
./runners/fix.sh PATTERN_ID --dry-run-integrations

# Production (post to Jira, create PR)
./runners/fix.sh PATTERN_ID --enable-jira --create-pr

# YOLO mode (skip prompts)
./runners/fix.sh PATTERN_ID --yolo
```

---

## Slide 15: Real-World Impact

**EXTRACTION PIPELINE:**
- 42 RHEL tickets extracted (100% success)
- 26 meta-tickets filtered (38% noise)
- 8 jailbreak attempts blocked
- 1-1.5 hours vs 100+ hours manual

**FIX PIPELINE:**
- ~30% fewer answer refinement cycles (URL validation)
- Human approval prevents mistakes
- Dry-run preview builds confidence
- Complete audit trail

**BUSINESS VALUE:**
- Faster resolution (hours vs days)
- Production quality (validated)
- Pattern-based scaling (10-15 tickets/fix)
- Risk reduction (jailbreak protection, regression tests)

---

## Slide 16: Product-Agnostic Architecture

**HEAL works for ANY domain with documentation:**

| Component | RHEL | Your Product |
|-----------|------|--------------|
| Expert Agent | Linux Expert | **Your Expert** |
| Search Backend | Solr (OKP) | **Your Docs API** |
| Review Guidelines | RHEL-specific | **Configure** |
| Pattern Discovery | No changes | **Works as-is** |

**Use cases:**
- OpenShift, Kubernetes, enterprise software
- Medical/legal knowledge bases
- Any RAG application with authoritative docs

---

## Slide 17: What Makes HEAL Different

**NOT just automation:**
→ Autonomous AGENTS with human oversight

**NOT just extraction:**
→ End-to-end: JIRA → pattern → fix → PR

**NOT just fast:**
→ VALIDATED quality at every step

**NOT just smart:**
→ SAFE with interactive review

---

## Slide 18: Key Innovations

1. **Multi-Agent Collaboration**
   - 5 specialized agents working together
   - Each agent has expertise and responsibility

2. **URL Validation**
   - Catches wrong docs BEFORE synthesis
   - Reduces wasted tokens and refinement cycles

3. **Autonomous Quality Loops**
   - 100% extraction success on valid tickets
   - Iterative refinement until production-ready

4. **Human-in-the-Loop Safety**
   - Two approval checkpoints
   - Easy rollback, test-before-commit

5. **Integration with Guardrails**
   - Dry-run preview mode
   - Git isolation (no auto-merge)

---

## Slide 19: Try It Yourself

**QUICK START:**
```bash
git clone [repo]
uv sync --extra dev
./scripts/demo_heal_workflow.sh --quick
```

**DOCUMENTATION:**
- Architecture: `docs/DESIGN_INTENT_AND_INTEGRATION.md`
- Bootstrap: `docs/BOOTSTRAP_GUIDE.md`
- Demo script: `docs/HEAL_DEMO_2026.md`
- One-pager: `docs/HEAL_ONE_PAGER.md`

**CONTACT:**
- GitHub: [Coming Soon]
- Questions: See README

---

## Slide 20: Q&A

**COMMON QUESTIONS:**

**Q: How do you ensure answer quality?**
A: Three layers - Solr verification, URL validation, autonomous review loop

**Q: What about security?**
A: Scope check blocks jailbreaks, dry-run mode for safe testing

**Q: Can humans intervene?**
A: Yes! Interactive review at two checkpoints, easy rollback

**Q: What if it makes a mistake?**
A: Type 'n' to revert, git isolation prevents bad merges

**Q: How long does it take?**
A: 1-1.5 hours for 68 tickets vs 100+ hours manual

---

## Slide 21: The Vision

**TODAY:**
RHEL Lightspeed RAG fixing

**TOMORROW:**
Any RAG application with documentation

**FUTURE:**
Self-healing AI systems that diagnose and fix their own failures

---

## Slide 22: Thank You!

**HEAL: Autonomous Intelligence + Human Oversight**
= Production-Ready RAG Fixes at Scale

**Resources:**
- Documentation: [Link]
- Demo: [Link]
- Contact: [Email]

**Questions?**

---

## Appendix Slides (Backup)

### Technical Deep Dive: The Autonomous Quality Loop

[Detailed flowchart with code snippets]

### Search Intelligence System

[Database schema, query logging, learning over time]

### Multi-Repository PR Coordination

[Future roadmap for cross-repo changes]

### Cost Analysis

[Token usage, API costs, optimization opportunities]

---

## Presenter Notes

**SLIDE 1-3:** Set the context (2-3 min)
- Start with the pain point
- Build tension with manual approach failures
- Introduce HEAL as the solution

**SLIDE 4-6:** Explain the innovation (5 min)
- Walk through each agent
- Focus on URL validation (new!)
- Emphasize interactive review

**SLIDE 7-9:** LIVE DEMO (15 min)
- Run validate_yaml_urls.py (show wrong docs caught)
- Run fix.sh (show approval flow)
- Run dry-run (show Jira preview)
- **Pause for reactions between demos**

**SLIDE 10-13:** Show the results (5 min)
- Metrics table
- Quality loops diagram
- Safety features

**SLIDE 14-16:** Practical application (3 min)
- Command reference (they can screenshot this!)
- Real-world impact
- Product-agnostic architecture

**SLIDE 17-20:** Wrap up (3 min)
- What makes it different
- Key innovations
- How to try it
- Q&A

**TOTAL:** ~35 minutes + Q&A buffer

---

*This outline gives you complete flexibility to create slides in any tool while maintaining the compelling narrative flow.*
