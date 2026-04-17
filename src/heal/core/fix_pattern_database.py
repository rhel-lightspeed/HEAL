"""Pattern database for learning from successful ticket fixes.

Stores fix patterns across tickets so the system can learn what works
for different types of retrieval problems.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


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
    times_reused: int = Field(default=0, description="How many times this pattern helped other tickets")
    success_rate_when_reused: float = Field(default=0.0, description="Success rate when applied to similar tickets")


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
            print(f"   Reused {best_pattern.times_reused} times with {best_pattern.success_rate_when_reused:.0%} success")
        else:
            print(f"💡 Found similar pattern from {best_pattern.ticket_id} (similarity: {best_score:.2f})")

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
            "avg_improvement": sum(p.improvement for p in successful) / len(successful) if successful else 0,
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
            p
            for p in self.patterns
            if p.times_reused >= 2 and p.success_rate_when_reused >= 0.7
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
