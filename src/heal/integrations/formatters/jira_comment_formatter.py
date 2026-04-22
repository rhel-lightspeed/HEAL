"""Formats Jira comments for pattern fix results."""

from typing import Any, Dict, List


class JiraCommentFormatter:
    """Formats pattern fix results as Jira-compatible Markdown."""

    def format_pattern_comment(
        self,
        pattern_result: Any,
        pattern_id: str,
        all_tickets: List[str],
        current_ticket: str,
    ) -> str:
        """Format comprehensive Jira comment for a pattern fix.

        Args:
            pattern_result: Complete pattern fix result
            pattern_id: Pattern identifier
            all_tickets: All ticket IDs in this pattern
            current_ticket: The ticket this comment is being posted to

        Returns:
            Markdown-formatted comment
        """
        # Extract data from result
        baseline = getattr(pattern_result, "baseline", None)
        optimization = getattr(pattern_result, "optimization", None)
        validation = getattr(pattern_result, "answer_validation", None)

        # Build comment sections
        sections = [
            self._format_header(pattern_id, pattern_result),
            self._format_metrics_table(baseline, optimization, validation),
            self._format_problem_solution(pattern_result),
            self._format_ticket_list(all_tickets, current_ticket, validation),
            self._format_warnings(validation),
            self._format_code_changes(pattern_result),
            self._format_next_steps(pattern_result),
            self._format_footer(pattern_result),
        ]

        return "\n\n".join(s for s in sections if s)

    def _format_header(self, pattern_id: str, result: Any) -> str:
        """Format comment header with status and links."""
        status_emoji = "✅" if getattr(result, "success", False) else "⚠️"
        status_text = "Fix Applied" if getattr(result, "success", False) else "Partial Fix"

        # Convert pattern_id to human-readable name
        pattern_name = pattern_id.replace("_", " ").title()

        branch_name = getattr(result, "branch_name", "unknown")

        return f"""## 🤖 Automated Pattern Fix: {pattern_name}

**Status:** {status_emoji} {status_text} | Branch: `{branch_name}` | [📊 Full Diagnostics](.diagnostics/{pattern_id}/REVIEW_REPORT.md)

---"""

    def _format_metrics_table(self, baseline: Any, optimization: Any, validation: Any) -> str:
        """Format before/after metrics table."""
        if not baseline or not hasattr(baseline, "final_metrics") or not baseline.final_metrics:
            return ""

        def format_delta(delta: float) -> str:
            if delta >= 0.10:
                return f"+{delta:.2f} ✅"
            elif delta >= 0.05:
                return f"+{delta:.2f} ➡️"
            elif delta >= -0.05:
                return f"{delta:+.2f} ➡️"
            else:
                return f"{delta:+.2f} ❌"

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

        before_answer = baseline_metrics.get("answer_correctness", 0.0)
        after_answer = final_metrics.get("answer_correctness", before_answer)

        before_f1 = baseline_metrics.get("url_f1", 0.0)
        after_f1 = final_metrics.get("url_f1", before_f1)

        before_faith = baseline_metrics.get("faithfulness", 0.0)
        after_faith = final_metrics.get("faithfulness", before_faith)

        before_ctx_rel = baseline_metrics.get("context_relevance", 0.0)
        after_ctx_rel = final_metrics.get("context_relevance", before_ctx_rel)

        success_rate = final_metrics.get("success_rate", 0.0)

        return f"""### 📊 Results Summary

| Metric | Before | After | Change | Status |
|--------|--------|-------|--------|--------|
| Answer Correctness | {before_answer:.2f} | {after_answer:.2f} | {format_delta(after_answer - before_answer)} |
| URL F1 | {before_f1:.2f} | {after_f1:.2f} | {format_delta(after_f1 - before_f1)} |
| Faithfulness | {before_faith:.2f} | {after_faith:.2f} | {format_delta(after_faith - before_faith)} |
| Context Relevance | {before_ctx_rel:.2f} | {after_ctx_rel:.2f} | {format_delta(after_ctx_rel - before_ctx_rel)} |

**Success Rate:** {success_rate:.0%}

---"""

    def _format_problem_solution(self, result: Any) -> str:
        """Format problem description and solution."""
        pattern_id = getattr(result, "pattern_id", "Unknown")
        pattern_name = pattern_id.replace("_", " ").title()

        solution_summary = self._extract_solution_summary(result)
        model_reasoning = self._extract_model_reasoning(result)

        return f"""### 🎯 What Was Fixed

**Pattern:** {pattern_name}

**Solution Applied:**
{solution_summary}

**Model Reasoning:**
{model_reasoning}

---"""

    def _extract_solution_summary(self, result: Any) -> str:
        """Extract solution summary from optimization phase."""
        optimization = getattr(result, "optimization", None)
        if (
            not optimization
            or not hasattr(optimization, "final_metrics")
            or not optimization.final_metrics
        ):
            return "Changes applied to improve retrieval quality."

        # TODO: Extract from multi-agent SynthesizedSuggestion when available
        return "Adjusted Solr query parameters to improve relevance."

    def _extract_model_reasoning(self, result: Any) -> str:
        """Extract model reasoning from optimization phase."""
        # TODO: Extract from multi-agent SynthesizedSuggestion.reasoning when available
        return "> Multi-agent analysis recommended targeted improvements to query weighting."

    def _format_ticket_list(
        self,
        all_tickets: List[str],
        current_ticket: str,
        validation: Any,
    ) -> str:
        """Format list of all tickets in pattern with results."""
        if len(all_tickets) <= 1:
            return ""  # Skip if only one ticket

        lines = ["### 🎫 Tickets Fixed in This Pattern\n"]

        # Try to get per-ticket results from validation
        per_ticket_results: Dict[str, Any] = {}
        if validation and hasattr(validation, "per_ticket_results"):
            per_ticket_results = validation.per_ticket_results

        for ticket_id in all_tickets:
            is_current = ticket_id == current_ticket
            marker = "**This ticket** " if is_current else ""

            # Get per-ticket metrics if available
            if ticket_id in per_ticket_results:
                tres = per_ticket_results[ticket_id]
                answer = getattr(tres, "answer_correctness", 0.0)
                f1 = getattr(tres, "url_f1", 0.0)

                if answer >= 0.85:
                    status = f"{answer:.2f} ✅"
                elif answer >= 0.70:
                    status = f"{answer:.2f} ➡️"
                else:
                    status = f"{answer:.2f} ⚠️"

                lines.append(f"- {marker}({ticket_id}): Answer {status}, F1 {f1:.2f}")
            else:
                lines.append(f"- {marker}({ticket_id}): Results pending")

        lines.append("\n---")
        return "\n".join(lines)

    def _format_warnings(self, validation: Any) -> str:
        """Format warnings section (RAG quality, high variance, etc.)."""
        if not validation:
            return ""

        warnings: List[str] = []

        # Check for RAG bypass tickets
        rag_bypass_tickets = getattr(validation, "rag_bypass_tickets", [])
        if rag_bypass_tickets:
            warnings.append("**RAG Quality:**")
            warnings.append(f"- ⚠️  {len(rag_bypass_tickets)} ticket(s) bypassed retrieval")
            for ticket_id in rag_bypass_tickets:
                warnings.append(f"  - {ticket_id}")
            warnings.append("")

        # Check for high variance tickets
        high_variance_tickets = getattr(validation, "high_variance_tickets", [])
        if high_variance_tickets:
            if not warnings:
                warnings.append("**RAG Quality:**")
                warnings.append("- ✅ No RAG bypass detected")
                warnings.append("")
            warnings.append("**Metric Stability:**")
            for ticket_id in high_variance_tickets:
                warnings.append(f"- ⚠️  {ticket_id} shows high variance across validation runs")
                warnings.append("  - **Recommendation:** Manual verification suggested")
            warnings.append("")

        if not warnings:
            return """### ⚠️ Warnings & Considerations

**RAG Quality:**
- ✅ No issues detected

**Testing Performed:**
- ✅ Full validation with answer correctness evaluation

---"""

        warnings.insert(0, "### ⚠️ Warnings & Considerations\n")
        warnings.append("**Testing Performed:**")
        warnings.append("- ✅ Full validation with answer correctness evaluation")
        warnings.append("\n---")

        return "\n".join(warnings)

    def _format_code_changes(self, result: Any) -> str:
        """Format code changes section."""
        optimization = getattr(result, "optimization", None)
        iterations = getattr(optimization, "iterations", 0) if optimization else 0
        branch_name = getattr(result, "branch_name", "unknown")

        return f"""### 📝 Code Changes

**File Modified:** `src/okp_mcp/solr.py`

**Commits:** {iterations} iteration(s)
- Branch: `{branch_name}`

---"""

    def _format_next_steps(self, result: Any) -> str:
        """Format next steps for reviewers/QA."""
        return """### ✅ Next Steps

**For Reviewers:**
1. Review PR for code changes (link above)
2. Verify fix quality by spot-checking tickets
3. Approve merge if metrics look good

**For QA:**
1. Test fix manually on this ticket
2. Verify no regressions in related queries
3. Close ticket if fix confirmed working

**For This Ticket:**
- [ ] Code review PR
- [ ] QA verification
- [ ] Close ticket when merged

---"""

    def _format_footer(self, result: Any) -> str:
        """Format footer with metadata."""
        duration_seconds = getattr(result, "duration_seconds", 0)
        duration_min = duration_seconds / 60

        return (
            f"""_Generated by HEAL v0.1.0 | Pattern Fix Loop | Runtime: {duration_min:.1f} min_"""
        )
