"""Evaluation classes for ticket and pattern assessment.

This module provides structured classes for tracking evaluation results
across individual tickets and patterns, with support for baseline comparison
and multi-run stability analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TicketEvaluation:
    """Evaluation results for a single ticket across multiple runs.

    Attributes:
        ticket_id: JIRA ticket identifier (e.g., "RSPEED-1724")
        runs: List of metric dictionaries, one per evaluation run
        baseline: Optional baseline evaluation for comparison
        is_no_doc: If True, ticket has no expected_urls (evaluated on answer only)
    """

    ticket_id: str
    runs: List[Dict[str, float]] = field(default_factory=list)
    baseline: Optional["TicketEvaluation"] = None
    is_no_doc: bool = False

    @property
    def num_runs(self) -> int:
        """Number of evaluation runs completed."""
        return len(self.runs)

    @property
    def mean_answer_correctness(self) -> float:
        """Average answer_correctness score across all runs."""
        scores = [r.get("answer_correctness", 0.0) for r in self.runs]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def mean_context_relevance(self) -> float:
        """Average context_relevance score across all runs."""
        scores = [r.get("context_relevance", 0.0) for r in self.runs]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def mean_context_precision(self) -> float:
        """Average context_precision score across all runs."""
        scores = [r.get("context_precision", 0.0) for r in self.runs]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def mean_url_f1(self) -> float:
        """Average URL F1 score across all runs."""
        scores = [r.get("url_f1", 0.0) for r in self.runs]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def composite_score(self) -> float:
        """Composite score using weighted metrics.

        Current weights:
        - answer_correctness: 80%
        - context_relevance: 15%
        - context_precision: 5%
        """
        return (
            self.mean_answer_correctness * 0.80
            + self.mean_context_relevance * 0.15
            + self.mean_context_precision * 0.05
        )

    @property
    def variance(self) -> float:
        """Variance in answer_correctness scores across runs."""
        if len(self.runs) < 2:
            return 0.0

        scores = [r.get("answer_correctness", 0.0) for r in self.runs]
        mean = sum(scores) / len(scores)
        return sum((s - mean) ** 2 for s in scores) / len(scores)

    @property
    def status(self) -> str:
        """Classify ticket status based on scores and variance.

        Returns:
            One of: STABLE_PASSING, CONSISTENTLY_FAILING, IMPROVING,
                   REGRESSING, ERRATIC
        """
        if not self.runs:
            return "NO_DATA"

        mean_score = self.mean_answer_correctness
        var = self.variance

        # High variance = erratic
        if var > 0.05:
            return "ERRATIC"

        # Stable passing
        if mean_score >= 0.90:
            return "STABLE_PASSING"

        # Consistently failing
        if mean_score < 0.50:
            return "CONSISTENTLY_FAILING"

        # Compare to baseline if available
        if self.baseline:
            improvement = self.improvement_over_baseline()
            if improvement > 0.10:
                return "IMPROVING"
            if improvement < -0.10:
                return "REGRESSING"

        # Default: in-progress
        return "IN_PROGRESS"

    def improvement_over_baseline(self) -> float:
        """Calculate improvement delta vs baseline.

        Returns:
            Delta in answer_correctness score (positive = improvement)
            Returns 0.0 if no baseline available
        """
        if not self.baseline:
            return 0.0

        return self.mean_answer_correctness - self.baseline.mean_answer_correctness

    def passes(self, threshold: float = 0.80) -> bool:
        """Check if ticket passes.

        For no-doc tickets: Uses answer_correctness directly (threshold 0.90)
        For regular tickets: Uses composite score (default threshold 0.80)

        Args:
            threshold: Minimum composite score to pass (default: 0.80)
                      Ignored for no-doc tickets (always uses 0.90 on answer)

        Returns:
            True if ticket passes criteria
        """
        if self.is_no_doc:
            # No-doc tickets: Use answer_correctness directly with 0.90 threshold
            # Don't penalize for missing context metrics
            return self.mean_answer_correctness >= 0.90
        else:
            # Regular tickets: Use composite score
            return self.composite_score >= threshold

    def to_dict(self) -> Dict:
        """Export to dictionary for serialization."""
        return {
            "ticket_id": self.ticket_id,
            "runs": self.runs,
            "mean_answer_correctness": self.mean_answer_correctness,
            "mean_context_relevance": self.mean_context_relevance,
            "mean_context_precision": self.mean_context_precision,
            "mean_url_f1": self.mean_url_f1,
            "composite_score": self.composite_score,
            "variance": self.variance,
            "status": self.status,
        }


@dataclass
class PatternEvaluation:
    """Evaluation results for a pattern across all tickets.

    Attributes:
        pattern_id: Pattern identifier (e.g., "BOOTLOADER_GRUB_ISSUES")
        tickets: Dictionary mapping ticket_id -> TicketEvaluation
        baseline: Optional baseline pattern evaluation for comparison
    """

    pattern_id: str
    tickets: Dict[str, TicketEvaluation] = field(default_factory=dict)
    baseline: Optional["PatternEvaluation"] = None

    @property
    def num_tickets(self) -> int:
        """Total number of tickets in pattern."""
        return len(self.tickets)

    @property
    def passing_tickets(self) -> List[str]:
        """List of ticket IDs that pass composite threshold."""
        return [tid for tid, ticket_eval in self.tickets.items() if ticket_eval.passes()]

    @property
    def failing_tickets(self) -> List[str]:
        """List of ticket IDs that fail composite threshold."""
        return [tid for tid, ticket_eval in self.tickets.items() if not ticket_eval.passes()]

    @property
    def success_rate(self) -> float:
        """Percentage of tickets passing (0.0 to 1.0)."""
        if self.num_tickets == 0:
            return 0.0
        return len(self.passing_tickets) / self.num_tickets

    @property
    def mean_composite_score(self) -> float:
        """Average composite score across all tickets."""
        if not self.tickets:
            return 0.0

        scores = [ticket.composite_score for ticket in self.tickets.values()]
        return sum(scores) / len(scores)

    def passes(self, criteria: str = "majority") -> bool:
        """Check if pattern passes using specified criteria.

        Args:
            criteria: Pass criteria, one of:
                - "majority": >50% of tickets pass (default)
                - "all": 100% of tickets pass
                - "any": At least one ticket passes
                - "average": Average composite score >= 0.80

        Returns:
            True if pattern passes according to criteria
        """
        if criteria == "majority":
            return self.success_rate > 0.5
        elif criteria == "all":
            return self.success_rate == 1.0
        elif criteria == "any":
            return len(self.passing_tickets) > 0
        elif criteria == "average":
            return self.mean_composite_score >= 0.80
        else:
            raise ValueError(f"Unknown criteria: {criteria}")

    def get_ticket_by_status(self, status: str) -> List[str]:
        """Get list of ticket IDs with given status.

        Args:
            status: One of STABLE_PASSING, CONSISTENTLY_FAILING, IMPROVING,
                   REGRESSING, ERRATIC, IN_PROGRESS, NO_DATA

        Returns:
            List of ticket IDs matching status
        """
        return [tid for tid, ticket_eval in self.tickets.items() if ticket_eval.status == status]

    def to_dict(self) -> Dict:
        """Export to dictionary for serialization."""
        return {
            "pattern_id": self.pattern_id,
            "num_tickets": self.num_tickets,
            "passing_tickets": self.passing_tickets,
            "failing_tickets": self.failing_tickets,
            "success_rate": self.success_rate,
            "mean_composite_score": self.mean_composite_score,
            "tickets": {tid: ticket.to_dict() for tid, ticket in self.tickets.items()},
        }
