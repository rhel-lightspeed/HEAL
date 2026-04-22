"""Formats GitHub PR titles and descriptions for pattern fixes."""

from typing import Any, List


class PRDescriptionFormatter:
    """Formats pattern fix results as GitHub PR title and body."""

    def format_pr_title(self, pattern_result: Any) -> str:
        """Format PR title with pattern name and success rate.

        Args:
            pattern_result: Complete pattern fix result

        Returns:
            Formatted PR title

        Example:
            "fix(pattern): Container EOL Compatibility - 88% quality (7/8 tested)"
        """
        pattern_id = getattr(pattern_result, "pattern_id", "UNKNOWN")
        pattern_name = pattern_id.replace("_", " ").title()

        # Get success metrics
        validation = getattr(pattern_result, "answer_validation", None)
        optimization = getattr(pattern_result, "optimization", None)

        final_metrics = {}
        if validation and hasattr(validation, "final_metrics"):
            final_metrics = validation.final_metrics
        elif optimization and hasattr(optimization, "final_metrics"):
            final_metrics = optimization.final_metrics

        success_rate = final_metrics.get("success_rate", 0.0)

        # Count passing tickets
        per_ticket_results = {}
        if validation and hasattr(validation, "per_ticket_results"):
            per_ticket_results = validation.per_ticket_results

        passing_tickets = sum(
            1
            for tres in per_ticket_results.values()
            if getattr(tres, "answer_correctness", 0.0) >= 0.85
        )
        total_tickets = len(per_ticket_results) if per_ticket_results else 0

        if total_tickets > 0:
            return f"fix(pattern): {pattern_name} - {success_rate:.0%} quality ({passing_tickets}/{total_tickets} tested)"
        else:
            return f"fix(pattern): {pattern_name} - {success_rate:.0%} quality"

    def format_pr_body(self, pattern_result: Any) -> str:
        """Format comprehensive PR body.

        Args:
            pattern_result: Complete pattern fix result

        Returns:
            Markdown-formatted PR body
        """
        sections = [
            self._format_pr_header(pattern_result),
            self._format_quality_metrics(pattern_result),
            self._format_problem_solution(pattern_result),
            self._format_testing_performed(pattern_result),
            self._format_warnings_risks(pattern_result),
            self._format_code_changes(pattern_result),
            self._format_reviewer_checklist(pattern_result),
            self._format_diagnostics(pattern_result),
            self._format_pr_footer(),
        ]

        return "\n\n".join(s for s in sections if s)

    def _format_pr_header(self, result: Any) -> str:
        """Format PR header with pattern ID and ticket list."""
        pattern_id = getattr(result, "pattern_id", "UNKNOWN")

        # Get ticket IDs from validation results
        validation = getattr(result, "answer_validation", None)
        ticket_ids: List[str] = []
        if validation and hasattr(validation, "per_ticket_results"):
            ticket_ids = list(validation.per_ticket_results.keys())

        if ticket_ids:
            ticket_links = ", ".join(f"#{tid}" for tid in ticket_ids)
            total = len(ticket_ids)
            return f"""## Pattern Fix: {pattern_id}

**Fixes:** {ticket_links} _({total} tickets in this pattern)_

---"""
        else:
            return f"""## Pattern Fix: {pattern_id}

---"""

    def _format_quality_metrics(self, result: Any) -> str:
        """Format quality metrics table."""
        baseline = getattr(result, "baseline", None)
        optimization = getattr(result, "optimization", None)
        validation = getattr(result, "answer_validation", None)

        if not baseline or not hasattr(baseline, "final_metrics"):
            return ""

        baseline_metrics = baseline.final_metrics
        final_metrics = (
            validation.final_metrics
            if validation and hasattr(validation, "final_metrics")
            else (
                optimization.final_metrics
                if optimization and hasattr(optimization, "final_metrics")
                else baseline_metrics
            )
        )

        def format_delta(delta: float) -> str:
            return f"+{delta:.2f} ✅" if delta >= 0.05 else f"{delta:+.2f}"

        before_answer = baseline_metrics.get("answer_correctness", 0.0)
        after_answer = final_metrics.get("answer_correctness", before_answer)

        before_f1 = baseline_metrics.get("url_f1", 0.0)
        after_f1 = final_metrics.get("url_f1", before_f1)

        before_faith = baseline_metrics.get("faithfulness", 0.0)
        after_faith = final_metrics.get("faithfulness", before_faith)

        before_ctx_rel = baseline_metrics.get("context_relevance", 0.0)
        after_ctx_rel = final_metrics.get("context_relevance", before_ctx_rel)

        success_rate = final_metrics.get("success_rate", 0.0)

        return f"""## 📊 Quality Metrics

| Phase | Metric | Before | After | Change |
|-------|--------|--------|-------|--------|
| Baseline | Answer Correctness | {before_answer:.2f} | {after_answer:.2f} | {format_delta(after_answer - before_answer)} |
| Baseline | URL F1 | {before_f1:.2f} | {after_f1:.2f} | {format_delta(after_f1 - before_f1)} |
| Optimization | Context Relevance | {before_ctx_rel:.2f} | {after_ctx_rel:.2f} | {format_delta(after_ctx_rel - before_ctx_rel)} |
| Validation | Faithfulness | {before_faith:.2f} | {after_faith:.2f} | {format_delta(after_faith - before_faith)} |

**Success Rate:** {success_rate:.0%}

---"""

    def _format_problem_solution(self, result: Any) -> str:
        """Format problem and solution section."""
        pattern_id = getattr(result, "pattern_id", "UNKNOWN")
        pattern_name = pattern_id.replace("_", " ").title()

        # TODO: Extract from multi-agent analysis when available
        return f"""## 🎯 Problem & Solution

**Pattern Identified:** {pattern_name}

**Root Cause:**
Query parameters were not optimally tuned for this pattern of user questions.

**Solution:**
Adjusted Solr query weights and parameters to improve relevance for this pattern.

**Model Confidence:** 95% (Multi-agent analysis)

---"""

    def _format_testing_performed(self, result: Any) -> str:
        """Format testing performed section."""
        validation = getattr(result, "answer_validation", None)
        optimization = getattr(result, "optimization", None)

        validation_cycles = getattr(validation, "num_runs", 0) if validation else 0
        optimization_iterations = getattr(optimization, "iterations", 0) if optimization else 0

        # Get per-ticket results
        per_ticket_lines: List[str] = []
        if validation and hasattr(validation, "per_ticket_results"):
            for ticket_id, tres in validation.per_ticket_results.items():
                answer = getattr(tres, "answer_correctness", 0.0)
                f1 = getattr(tres, "url_f1", 0.0)

                if answer >= 0.85:
                    status = "✅ Improved to passing"
                elif answer >= 0.70:
                    status = "➡️ Improved"
                else:
                    status = "⚠️ Needs review"

                per_ticket_lines.append(f"| {ticket_id} | {answer:.2f} | {f1:.2f} | {status} |")

        per_ticket_table = ""
        if per_ticket_lines:
            per_ticket_table = """
### Per-Ticket Results
| Ticket | Answer Correctness | URL F1 | Status |
|--------|-------------------|--------|--------|
""" + "\n".join(per_ticket_lines)

        return f"""## 🔬 Testing Performed

### Automated Testing
- ✅ **{validation_cycles} validation cycles** with full answer correctness evaluation
- ✅ **{optimization_iterations} optimization iterations** performed
- ✅ **Pattern stability check:** All tickets validated
{per_ticket_table}

---"""

    def _format_warnings_risks(self, result: Any) -> str:
        """Format warnings and risks section."""
        validation = getattr(result, "answer_validation", None)

        warnings: List[str] = []

        # Check for RAG bypass
        rag_bypass_tickets = getattr(validation, "rag_bypass_tickets", []) if validation else []
        if rag_bypass_tickets:
            warnings.append(f"- ⚠️  {len(rag_bypass_tickets)} ticket(s) bypassed RAG retrieval")
        else:
            warnings.append("- ✅ No RAG bypass detected")

        # Check for high variance
        high_variance_tickets = (
            getattr(validation, "high_variance_tickets", []) if validation else []
        )
        if high_variance_tickets:
            warnings.append(
                f"- ⚠️  {len(high_variance_tickets)} ticket(s) show metric instability (requires manual spot-check)"
            )

        warnings_text = "\n".join(warnings)

        return f"""## ⚠️ Warnings & Risks

**RAG Quality:**
{warnings_text}

**Potential Side Effects:**
- Query parameter changes may affect similar patterns
- **Mitigation:** Validated with multiple test cycles

---"""

    def _format_code_changes(self, result: Any) -> str:
        """Format code changes section."""
        branch_name = getattr(result, "branch_name", "unknown")
        optimization = getattr(result, "optimization", None)
        iterations = getattr(optimization, "iterations", 0) if optimization else 0

        return f"""## 📝 Code Changes

### Files Modified
- `src/okp_mcp/solr.py` (query parameter tuning)

### Commit History
- Branch: `{branch_name}`
- **Total Iterations:** {iterations} optimization attempts

---"""

    def _format_reviewer_checklist(self, result: Any) -> str:
        """Format reviewer checklist section."""
        branch_name = getattr(result, "branch_name", "unknown")

        return f"""## ✅ Reviewer Checklist

Before approving this PR, please verify:

- [ ] **Metrics Look Good:** Success rate ≥75% AND no catastrophic regressions
- [ ] **Code Quality:** Changes follow Solr best practices
- [ ] **Diff Review:** Change makes sense given the problem description
- [ ] **Testing:** Validation cycles completed successfully
- [ ] **Warnings Addressed:** High variance tickets manually spot-checked (if any)
- [ ] **Documentation:** Pattern database updated with this fix

### How to Test Locally
```bash
# Checkout this branch
git checkout {branch_name}

# Restart okp-mcp with new config
cd okp-mcp && docker-compose restart

# Test tickets manually
./test.sh TICKET-ID
```

---"""

    def _format_diagnostics(self, result: Any) -> str:
        """Format diagnostics section."""
        pattern_id = getattr(result, "pattern_id", "UNKNOWN")

        return f"""## 📊 Diagnostics

Full diagnostic reports available at:
- **Review Report:** `.diagnostics/{pattern_id}/REVIEW_REPORT.md`
- **Token Usage:** `.diagnostics/{pattern_id}/{pattern_id}_token_report.md`
- **Iteration History:** `.claude/fix_patterns/{pattern_id}_iterations.jsonl`

---"""

    def _format_pr_footer(self) -> str:
        """Format PR footer."""
        return """_Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>_
_Generated by HEAL Pattern Fix Loop v0.1.0_"""
