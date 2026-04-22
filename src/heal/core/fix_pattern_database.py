"""Pattern database for learning from successful ticket fixes.

Stores fix patterns across tickets so the system can learn what works
for different types of retrieval problems.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class FixIteration(BaseModel):
    """Single iteration in an iterative fix cycle."""

    iteration: int
    cycle: int  # Which outer cycle (1st answer check, 2nd answer check, etc.)
    timestamp: str

    # Suggestion
    suggested_change: str
    reasoning: str
    confidence: float

    # Metrics before this fix
    before_url_f1: float
    before_answer: float
    before_faithfulness: Optional[float] = None

    # Metrics after this fix
    after_url_f1: float
    after_answer: float
    after_faithfulness: Optional[float] = None

    # Deltas
    url_f1_delta: float
    answer_delta: float
    faithfulness_delta: float = 0.0

    # Outcome
    improved: bool
    committed: bool
    git_commit_hash: Optional[str] = None


class FixPattern(BaseModel):
    """A successful fix pattern that can be reused."""

    # Problem signature
    problem_type: str = Field(description="retrieval_failure|ranking_issue|snippet_quality")
    url_f1_before: float = Field(description="URL F1 before fix")
    query_keywords: List[str] = Field(description="Key terms in the query")
    explain_pattern: str = Field(description="Pattern from Solr explain (e.g., 'title_weight_low')")

    # Solution
    fix_type: str = Field(description="boost_title|add_keyword|increase_mm|etc")
    specific_change: str = Field(description="Exact code change made")
    file_path: str = Field(description="File that was edited")

    # Outcome
    url_f1_after: float = Field(description="URL F1 after fix")
    improvement: float = Field(description="Improvement delta")
    answer_correctness_after: Optional[float] = None
    success: bool = Field(description="Did this fix solve the problem?")

    # Metadata
    ticket_id: str
    timestamp: str
    iterations_to_fix: int = Field(description="How many iterations to find this fix")
    cost: float = Field(description="LLM API cost in USD")

    # Learning data
    judge_reasoning: Optional[str] = None
    times_reused: int = Field(
        default=0, description="How many times this pattern helped other tickets"
    )
    success_rate_when_reused: float = Field(
        default=0.0, description="Success rate when applied to similar tickets"
    )

    # Iterative improvement history
    iterations: List[FixIteration] = Field(
        default_factory=list, description="All iterations that led to this fix"
    )


class FixPatternDatabase:
    """Database of successful fix patterns for reuse across tickets.

    Usage:
        # During fix loop:
        db = FixPatternDatabase()

        # Try to find similar past fix
        similar = db.find_similar_patterns(
            query="grub bootloader uefi",
            url_f1=0.0,
            explain_pattern="title_weight_low"
        )

        if similar and similar.success_rate_when_reused > 0.7:
            # High confidence match - try this fix first
            suggestion = similar.specific_change
        else:
            # Ask LLM for new suggestion
            suggestion = llm_advisor.suggest(metrics)

        # After testing:
        db.record_fix(
            pattern=FixPattern(...),
            reused_from=similar.ticket_id if similar else None
        )
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize pattern database.

        Args:
            db_path: Path to store patterns (default: .claude/fix_patterns.jsonl)
        """
        if db_path is None:
            db_path = Path(".claude/fix_patterns/patterns.jsonl")

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Load patterns into memory for fast lookup
        self.patterns: List[FixPattern] = self._load_patterns()

    def _load_patterns(self) -> List[FixPattern]:
        """Load all patterns from disk."""
        if not self.db_path.exists():
            return []

        patterns = []
        with open(self.db_path) as f:
            for line in f:
                if line.strip():
                    patterns.append(FixPattern.model_validate_json(line))

        return patterns

    def record_fix(
        self,
        pattern: FixPattern,
        reused_from: Optional[str] = None,
    ) -> None:
        """Record a successful (or failed) fix pattern.

        Args:
            pattern: The fix pattern to record
            reused_from: Ticket ID if this was reused from a similar pattern
        """
        # If reused, update success stats on original pattern
        if reused_from:
            for p in self.patterns:
                if p.ticket_id == reused_from:
                    p.times_reused += 1
                    # Update success rate with running average
                    old_rate = p.success_rate_when_reused
                    n = p.times_reused
                    new_success = 1.0 if pattern.success else 0.0
                    p.success_rate_when_reused = (old_rate * (n - 1) + new_success) / n
                    break

        # Append new pattern
        self.patterns.append(pattern)

        # Save to disk (append)
        with open(self.db_path, "a") as f:
            f.write(pattern.model_dump_json() + "\n")

    def find_similar_patterns(
        self,
        query: str,
        url_f1: float,
        explain_pattern: Optional[str] = None,
        min_similarity: float = 0.7,
        top_k: int = 5,
    ) -> Optional[FixPattern]:
        """Find similar past fixes that might work for this problem.

        Args:
            query: Current query string
            url_f1: Current URL F1 score
            explain_pattern: Pattern extracted from Solr explain
            min_similarity: Minimum similarity threshold (0-1)
            top_k: Return top K most similar patterns

        Returns:
            Best matching pattern, or None if no good match
        """
        if not self.patterns:
            return None

        # Extract keywords from query
        query_keywords = set(query.lower().split())

        # Score each pattern
        scored_patterns = []
        for pattern in self.patterns:
            # Only consider successful patterns
            if not pattern.success:
                continue

            # Only consider patterns that actually improved things
            if pattern.improvement < 0.1:
                continue

            # Calculate similarity
            score = self._calculate_similarity(
                query_keywords=query_keywords,
                url_f1=url_f1,
                explain_pattern=explain_pattern,
                candidate=pattern,
            )

            if score >= min_similarity:
                scored_patterns.append((score, pattern))

        if not scored_patterns:
            return None

        # Sort by score descending
        scored_patterns.sort(reverse=True, key=lambda x: x[0])

        # Return best match
        best_score, best_pattern = scored_patterns[0]

        # Boost confidence if this pattern has been successfully reused
        if best_pattern.times_reused > 0 and best_pattern.success_rate_when_reused > 0.7:
            print(f"🎯 Found high-confidence pattern from {best_pattern.ticket_id}:")
            print(
                f"   Reused {best_pattern.times_reused} times with {best_pattern.success_rate_when_reused:.0%} success"
            )
        else:
            print(
                f"💡 Found similar pattern from {best_pattern.ticket_id} (similarity: {best_score:.2f})"
            )

        return best_pattern

    def _calculate_similarity(
        self,
        query_keywords: set,
        url_f1: float,
        explain_pattern: Optional[str],
        candidate: FixPattern,
    ) -> float:
        """Calculate similarity between current problem and candidate pattern.

        Returns:
            Similarity score between 0.0 and 1.0
        """
        score = 0.0
        weights = []

        # 1. Query keyword overlap (40% weight)
        candidate_keywords = set(candidate.query_keywords)
        overlap = len(query_keywords & candidate_keywords)
        total = len(query_keywords | candidate_keywords)
        keyword_sim = overlap / total if total > 0 else 0.0
        score += keyword_sim * 0.4
        weights.append(0.4)

        # 2. URL F1 similarity (30% weight)
        # Problems with similar F1 scores likely need similar fixes
        f1_diff = abs(url_f1 - candidate.url_f1_before)
        f1_sim = max(0, 1 - f1_diff)  # 0.0 diff = 1.0 sim, 1.0 diff = 0.0 sim
        score += f1_sim * 0.3
        weights.append(0.3)

        # 3. Explain pattern match (30% weight)
        if explain_pattern and candidate.explain_pattern:
            pattern_match = 1.0 if explain_pattern == candidate.explain_pattern else 0.0
            score += pattern_match * 0.3
            weights.append(0.3)

        # Normalize by total weight used
        total_weight = sum(weights)
        return score / total_weight if total_weight > 0 else 0.0

    def get_statistics(self) -> Dict:
        """Get statistics about the pattern database.

        Returns:
            Dict with stats (total patterns, success rate, most common fixes, etc.)
        """
        if not self.patterns:
            return {"total_patterns": 0}

        successful = [p for p in self.patterns if p.success]
        reused = [p for p in self.patterns if p.times_reused > 0]

        # Find most common fix types
        fix_type_counts = {}
        for p in successful:
            fix_type_counts[p.fix_type] = fix_type_counts.get(p.fix_type, 0) + 1

        most_common = sorted(fix_type_counts.items(), key=lambda x: x[1], reverse=True)

        return {
            "total_patterns": len(self.patterns),
            "successful_patterns": len(successful),
            "success_rate": len(successful) / len(self.patterns),
            "patterns_reused": len(reused),
            "most_common_fixes": most_common[:5],
            "avg_improvement": (
                sum(p.improvement for p in successful) / len(successful) if successful else 0
            ),
            "total_cost_saved": sum(p.cost for p in reused),  # Cost of patterns that were reused
        }

    def export_learning_report(self, output_path: Path) -> None:
        """Export a human-readable report of learned patterns.

        Args:
            output_path: Where to write the markdown report
        """
        stats = self.get_statistics()

        lines = [
            "# Fix Pattern Learning Report",
            "",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Summary Statistics",
            "",
            f"- Total patterns recorded: {stats['total_patterns']}",
            f"- Successful patterns: {stats['successful_patterns']} ({stats['success_rate']:.1%})",
            f"- Patterns reused: {stats['patterns_reused']}",
            f"- Average improvement: {stats['avg_improvement']:.3f}",
            "",
            "## Most Common Fixes",
            "",
        ]

        for fix_type, count in stats["most_common_fixes"]:
            lines.append(f"- **{fix_type}**: {count} times")

        lines.extend(
            [
                "",
                "## High-Confidence Patterns (Reused Successfully)",
                "",
            ]
        )

        high_conf = [
            p for p in self.patterns if p.times_reused >= 2 and p.success_rate_when_reused >= 0.7
        ]
        high_conf.sort(key=lambda x: x.times_reused, reverse=True)

        for pattern in high_conf:
            lines.extend(
                [
                    f"### {pattern.ticket_id}",
                    "",
                    f"**Problem**: {pattern.problem_type} (F1: {pattern.url_f1_before:.2f})",
                    f"**Fix**: {pattern.specific_change}",
                    f"**Outcome**: F1 improved to {pattern.url_f1_after:.2f} (+{pattern.improvement:.2f})",
                    f"**Reused**: {pattern.times_reused} times with {pattern.success_rate_when_reused:.0%} success",
                    "",
                ]
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines))
        print(f"📊 Learning report exported to: {output_path}")

    def record_iteration(
        self,
        pattern_id: str,
        iteration: int,
        cycle: int,
        suggested_change: str,
        reasoning: str,
        confidence: float,
        before_metrics: Dict[str, float],
        after_metrics: Dict[str, float],
        committed: bool,
        git_commit_hash: Optional[str] = None,
    ) -> FixIteration:
        """Record a single iteration in the fix cycle.

        This builds up the iterative history showing how fixes evolved.

        Args:
            pattern_id: Pattern being fixed
            iteration: Iteration number within cycle
            cycle: Outer cycle number (1st answer check, 2nd, etc.)
            suggested_change: What was changed
            reasoning: Why it was suggested
            confidence: Confidence score
            before_metrics: Metrics before fix
            after_metrics: Metrics after fix
            committed: Whether fix was committed
            git_commit_hash: Git commit if committed

        Returns:
            FixIteration record
        """
        url_f1_delta = after_metrics.get("url_f1", 0) - before_metrics.get("url_f1", 0)
        answer_delta = after_metrics.get("answer", 0) - before_metrics.get("answer", 0)
        faith_delta = (
            after_metrics.get("faithfulness", 0) - before_metrics.get("faithfulness", 0)
            if before_metrics.get("faithfulness") and after_metrics.get("faithfulness")
            else 0.0
        )

        improved = answer_delta > 0.02  # 2% threshold

        iter_record = FixIteration(
            iteration=iteration,
            cycle=cycle,
            timestamp=datetime.now().isoformat(),
            suggested_change=suggested_change,
            reasoning=reasoning,
            confidence=confidence,
            before_url_f1=before_metrics.get("url_f1", 0),
            before_answer=before_metrics.get("answer", 0),
            before_faithfulness=before_metrics.get("faithfulness"),
            after_url_f1=after_metrics.get("url_f1", 0),
            after_answer=after_metrics.get("answer", 0),
            after_faithfulness=after_metrics.get("faithfulness"),
            url_f1_delta=url_f1_delta,
            answer_delta=answer_delta,
            faithfulness_delta=faith_delta,
            improved=improved,
            committed=committed,
            git_commit_hash=git_commit_hash,
        )

        # Store in pattern-specific iterations file
        iter_file = self.db_path.parent / f"{pattern_id}_iterations.jsonl"
        with open(iter_file, "a") as f:
            f.write(iter_record.model_dump_json() + "\n")

        return iter_record

    def get_iteration_context(self, pattern_id: str) -> str:
        """Get formatted context of all iterations for multi-agent.

        This provides the agent with complete history of what was tried
        and what worked, enabling cumulative improvements.

        Args:
            pattern_id: Pattern being fixed

        Returns:
            Formatted context string for agent prompt
        """
        iter_file = self.db_path.parent / f"{pattern_id}_iterations.jsonl"

        if not iter_file.exists():
            return (
                "No prior iterations for this pattern.\n\nThis is the first optimization attempt."
            )

        # Load iterations
        iterations = []
        with open(iter_file) as f:
            for line in f:
                if line.strip():
                    iterations.append(FixIteration.model_validate_json(line))

        if not iterations:
            return "No prior iterations recorded."

        # Build context
        lines = []
        lines.append(f"PRIOR FIX ATTEMPTS FOR {pattern_id}")
        lines.append("=" * 80)
        lines.append("")

        # Group by cycle
        cycles: Dict[int, List[FixIteration]] = {}
        for it in iterations:
            if it.cycle not in cycles:
                cycles[it.cycle] = []
            cycles[it.cycle].append(it)

        # Show cumulative progress
        baseline_answer = iterations[0].before_answer if iterations else 0.0
        best_answer = max((it.after_answer for it in iterations), default=baseline_answer)
        total_improvement = best_answer - baseline_answer

        lines.append(f"Baseline Answer Correctness: {baseline_answer:.2f}")
        lines.append(f"Current Best: {best_answer:.2f} (Δ {total_improvement:+.2f})")
        lines.append(f"Total Iterations: {len(iterations)}")
        lines.append(f"Committed Fixes: {sum(1 for it in iterations if it.committed)}")
        lines.append("")

        # Show each cycle
        for cycle_num in sorted(cycles.keys()):
            cycle_its = cycles[cycle_num]
            lines.append(f"CYCLE {cycle_num}:")
            lines.append("-" * 40)

            for it in cycle_its:
                status = (
                    "✅ COMMITTED"
                    if it.committed
                    else "❌ REVERTED" if not it.improved else "⏸️  PENDING"
                )
                lines.append(f"  Iteration {it.iteration}:")
                lines.append(f"    Change: {it.suggested_change}")
                lines.append(
                    f"    Answer: {it.before_answer:.2f} → {it.after_answer:.2f} (Δ {it.answer_delta:+.2f})"
                )
                lines.append(
                    f"    URL F1: {it.before_url_f1:.2f} → {it.after_url_f1:.2f} (Δ {it.url_f1_delta:+.2f})"
                )
                lines.append(f"    Status: {status}")
                if it.committed and it.git_commit_hash:
                    lines.append(f"    Git: {it.git_commit_hash[:8]}")
                lines.append("")

        # Provide guidance for next iteration
        lines.append("=" * 80)
        lines.append("GUIDANCE FOR NEXT FIX:")
        lines.append("")

        # What worked
        successful = [it for it in iterations if it.committed]
        if successful:
            lines.append("✅ Successful changes so far:")
            for it in successful[-3:]:  # Last 3 successes
                lines.append(
                    f"  • {it.suggested_change} → +{it.answer_delta:.2f} answer improvement"
                )
            lines.append("")
            lines.append("→ BUILD ON THESE: Refine and extend what's working!")
        else:
            lines.append("⚠️  No successful commits yet.")
            lines.append("→ Consider a different approach")

        lines.append("")

        # What didn't work
        failed = [it for it in iterations if not it.improved]
        if failed:
            lines.append("❌ Approaches that didn't help:")
            for it in failed[-2:]:  # Last 2 failures
                lines.append(f"  • {it.suggested_change}")
            lines.append("")
            lines.append("→ AVOID repeating these patterns")

        return "\n".join(lines)
