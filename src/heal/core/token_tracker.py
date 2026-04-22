"""Token usage tracking for pattern-based learning efficiency analysis.

This module tracks token usage across the fix loop to prove the thesis:
"Pattern-based learning achieves higher quality with lower token costs."

Usage:
    tracker = TokenTracker()

    # Track LLM call
    with tracker.track_call("multi_agent_solr_expert"):
        result = llm.query(prompt)

    # Get summary
    summary = tracker.get_summary()
    print(f"Total tokens: {summary['total_tokens']}")
    print(f"Total cost: ${summary['total_cost_usd']:.4f}")
    print(f"Cost per quality point: ${summary['cost_per_quality_point']:.4f}")
"""

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

# Token costs per model (as of 2025-01)
# Source: https://www.anthropic.com/pricing
TOKEN_COSTS = {
    "claude-opus-4": {"input": 15.00 / 1_000_000, "output": 75.00 / 1_000_000},
    "claude-sonnet-4": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-sonnet-3.5": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-haiku-3.5": {"input": 0.80 / 1_000_000, "output": 4.00 / 1_000_000},
    # Gemini costs (for comparison)
    "gemini-1.5-pro": {"input": 1.25 / 1_000_000, "output": 5.00 / 1_000_000},
    "gemini-1.5-flash": {"input": 0.075 / 1_000_000, "output": 0.30 / 1_000_000},
}


class LLMCall(BaseModel):
    """Record of a single LLM API call."""

    timestamp: str
    call_type: str  # "multi_agent", "ragas_answer", "ragas_faithfulness", etc.
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    duration_seconds: float
    iteration: Optional[int] = None
    cycle: Optional[int] = None
    used_pattern_context: bool = False  # Did this call use pattern DB context?


class IterationTokenSummary(BaseModel):
    """Token summary for a single iteration."""

    iteration: int
    cycle: int

    # Token breakdown
    multi_agent_tokens: int = 0
    ragas_tokens: int = 0
    other_tokens: int = 0
    total_tokens: int = 0

    # Cost breakdown
    multi_agent_cost: float = 0.0
    ragas_cost: float = 0.0
    other_cost: float = 0.0
    total_cost: float = 0.0

    # Quality metrics
    before_answer_correctness: float = 0.0
    after_answer_correctness: float = 0.0
    answer_improvement: float = 0.0

    # Efficiency
    cost_per_quality_point: float = 0.0  # cost / improvement
    tokens_per_quality_point: float = 0.0

    # Learning
    used_pattern_context: bool = False
    context_length_tokens: int = 0  # How many tokens was the pattern context?


class TokenTracker:
    """Tracks token usage across pattern fix loop for efficiency analysis."""

    _instance: Optional["TokenTracker"] = None

    def __init__(self, pattern_id: str, output_dir: Optional[Path] = None):
        """Initialize token tracker.

        Args:
            pattern_id: Pattern being optimized
            output_dir: Directory to save token reports (default: .diagnostics/{pattern_id})
        """
        self.pattern_id = pattern_id
        self.output_dir = output_dir or Path(f".diagnostics/{pattern_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.calls: List[LLMCall] = []
        self.current_iteration: Optional[int] = None
        self.current_cycle: Optional[int] = None
        self._call_start_time: Optional[float] = None

        # Baseline tracking (for comparison)
        self.baseline_answer_correctness: float = 0.0

        # Set as singleton instance
        TokenTracker._instance = self

    @classmethod
    def get_instance(cls) -> Optional["TokenTracker"]:
        """Get current tracker instance (if active).

        Returns:
            Active TokenTracker instance or None if not initialized
        """
        return cls._instance

    def set_iteration(self, iteration: int, cycle: int):
        """Set current iteration/cycle for tracking."""
        self.current_iteration = iteration
        self.current_cycle = cycle

    def set_baseline(self, answer_correctness: float):
        """Set baseline answer_correctness for comparison."""
        self.baseline_answer_correctness = answer_correctness

    @contextmanager
    def track_call(
        self,
        call_type: str,
        model: str,
        used_pattern_context: bool = False,
    ):
        """Context manager to track a single LLM call.

        Usage:
            with tracker.track_call("multi_agent_solr", model="claude-sonnet-4"):
                result = llm.query(prompt)
                # Manually record tokens after call
                tracker.record_tokens(input_tokens=1000, output_tokens=500)
        """
        self._call_start_time = time.time()
        self._current_call_type = call_type
        self._current_model = model
        self._current_used_context = used_pattern_context

        try:
            yield self
        finally:
            # Context manager completed - if record_tokens wasn't called,
            # the call object will be created when record_tokens is called
            pass

    def record_tokens(
        self,
        input_tokens: int,
        output_tokens: int,
        call_type: Optional[str] = None,
        model: Optional[str] = None,
        used_pattern_context: Optional[bool] = None,
    ):
        """Record token usage for an LLM call.

        Can be called within track_call() context or standalone.
        """
        # Use context manager values if available
        call_type = call_type or getattr(self, "_current_call_type", "unknown")
        model = model or getattr(self, "_current_model", "claude-sonnet-4")
        used_pattern_context = used_pattern_context or getattr(self, "_current_used_context", False)

        total_tokens = input_tokens + output_tokens

        # Calculate cost
        if model in TOKEN_COSTS:
            cost = (
                input_tokens * TOKEN_COSTS[model]["input"]
                + output_tokens * TOKEN_COSTS[model]["output"]
            )
        else:
            # Default to Sonnet pricing
            cost = (
                input_tokens * TOKEN_COSTS["claude-sonnet-4"]["input"]
                + output_tokens * TOKEN_COSTS["claude-sonnet-4"]["output"]
            )

        # Calculate duration
        duration = 0.0
        if self._call_start_time is not None:
            duration = time.time() - self._call_start_time
            self._call_start_time = None

        # Create call record
        call = LLMCall(
            timestamp=datetime.now().isoformat(),
            call_type=call_type,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            duration_seconds=duration,
            iteration=self.current_iteration,
            cycle=self.current_cycle,
            used_pattern_context=used_pattern_context,
        )

        self.calls.append(call)

        # Save incrementally to JSONL
        calls_file = self.output_dir / f"{self.pattern_id}_token_calls.jsonl"
        with open(calls_file, "a") as f:
            f.write(call.model_dump_json() + "\n")

    def record_iteration_summary(
        self,
        iteration: int,
        cycle: int,
        before_answer: float,
        after_answer: float,
        used_pattern_context: bool,
        context_length_tokens: int = 0,
    ):
        """Record token summary for an iteration.

        This creates an aggregate view of all calls in this iteration.
        """
        # Get all calls for this iteration
        iter_calls = [c for c in self.calls if c.iteration == iteration and c.cycle == cycle]

        if not iter_calls:
            return

        # Breakdown by call type
        multi_agent_calls = [c for c in iter_calls if "multi_agent" in c.call_type.lower()]
        ragas_calls = [c for c in iter_calls if "ragas" in c.call_type.lower()]
        other_calls = [c for c in iter_calls if c not in multi_agent_calls and c not in ragas_calls]

        multi_agent_tokens = sum(c.total_tokens for c in multi_agent_calls)
        ragas_tokens = sum(c.total_tokens for c in ragas_calls)
        other_tokens = sum(c.total_tokens for c in other_calls)
        total_tokens = sum(c.total_tokens for c in iter_calls)

        multi_agent_cost = sum(c.cost_usd for c in multi_agent_calls)
        ragas_cost = sum(c.cost_usd for c in ragas_calls)
        other_cost = sum(c.cost_usd for c in other_calls)
        total_cost = sum(c.cost_usd for c in iter_calls)

        # Calculate efficiency
        improvement = after_answer - before_answer
        cost_per_quality = total_cost / improvement if improvement > 0 else 0.0
        tokens_per_quality = total_tokens / improvement if improvement > 0 else 0.0

        summary = IterationTokenSummary(
            iteration=iteration,
            cycle=cycle,
            multi_agent_tokens=multi_agent_tokens,
            ragas_tokens=ragas_tokens,
            other_tokens=other_tokens,
            total_tokens=total_tokens,
            multi_agent_cost=multi_agent_cost,
            ragas_cost=ragas_cost,
            other_cost=other_cost,
            total_cost=total_cost,
            before_answer_correctness=before_answer,
            after_answer_correctness=after_answer,
            answer_improvement=improvement,
            cost_per_quality_point=cost_per_quality,
            tokens_per_quality_point=tokens_per_quality,
            used_pattern_context=used_pattern_context,
            context_length_tokens=context_length_tokens,
        )

        # Save to JSONL
        summary_file = self.output_dir / f"{self.pattern_id}_token_summaries.jsonl"
        with open(summary_file, "a") as f:
            f.write(summary.model_dump_json() + "\n")

    def get_summary(self) -> Dict:
        """Get overall token usage summary.

        Returns:
            Dict with aggregated statistics
        """
        if not self.calls:
            return {
                "total_calls": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "answer_improvement": 0.0,
            }

        total_tokens = sum(c.total_tokens for c in self.calls)
        total_cost = sum(c.cost_usd for c in self.calls)

        # Breakdown by call type
        multi_agent_calls = [c for c in self.calls if "multi_agent" in c.call_type.lower()]
        ragas_calls = [c for c in self.calls if "ragas" in c.call_type.lower()]

        multi_agent_tokens = sum(c.total_tokens for c in multi_agent_calls)
        multi_agent_cost = sum(c.cost_usd for c in multi_agent_calls)

        ragas_tokens = sum(c.total_tokens for c in ragas_calls)
        ragas_cost = sum(c.cost_usd for c in ragas_calls)

        # Breakdown by model
        by_model: Dict[str, Dict[str, float]] = {}
        for call in self.calls:
            if call.model not in by_model:
                by_model[call.model] = {
                    "tokens": 0,
                    "cost": 0.0,
                    "calls": 0,
                }
            by_model[call.model]["tokens"] += call.total_tokens
            by_model[call.model]["cost"] += call.cost_usd
            by_model[call.model]["calls"] += 1

        # Get final answer correctness
        final_answer = 0.0
        if self.calls:
            # Find most recent iteration summary
            summary_file = self.output_dir / f"{self.pattern_id}_token_summaries.jsonl"
            if summary_file.exists():
                with open(summary_file) as f:
                    for line in f:
                        summary = json.loads(line)
                        final_answer = summary["after_answer_correctness"]

        answer_improvement = final_answer - self.baseline_answer_correctness

        # Calculate efficiency
        cost_per_quality = total_cost / answer_improvement if answer_improvement > 0 else 0.0
        tokens_per_quality = total_tokens / answer_improvement if answer_improvement > 0 else 0.0

        return {
            "total_calls": len(self.calls),
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "multi_agent_tokens": multi_agent_tokens,
            "multi_agent_cost_usd": multi_agent_cost,
            "ragas_tokens": ragas_tokens,
            "ragas_cost_usd": ragas_cost,
            "by_model": by_model,  # Per-model breakdown
            "baseline_answer": self.baseline_answer_correctness,
            "final_answer": final_answer,
            "answer_improvement": answer_improvement,
            "cost_per_quality_point": cost_per_quality,
            "tokens_per_quality_point": tokens_per_quality,
            "efficiency_ratio": answer_improvement / total_cost if total_cost > 0 else 0.0,
        }

    def generate_report(self, output_file: Optional[Path] = None) -> str:
        """Generate markdown report of token usage and efficiency.

        Args:
            output_file: Optional file to save report to

        Returns:
            Markdown report as string
        """
        summary = self.get_summary()

        report = f"""# Token Usage & Efficiency Report

## Pattern: {self.pattern_id}

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total LLM Calls** | {summary['total_calls']:,} |
| **Total Tokens** | {summary['total_tokens']:,} |
| **Total Cost** | ${summary['total_cost_usd']:.4f} USD |
| **Baseline Answer** | {summary['baseline_answer']:.2f} |
| **Final Answer** | {summary['final_answer']:.2f} |
| **Improvement** | +{summary['answer_improvement']:.2f} |
| **Cost per Quality Point** | ${summary['cost_per_quality_point']:.4f} |
| **Tokens per Quality Point** | {summary['tokens_per_quality_point']:,.0f} |
| **Efficiency Ratio** | {summary['efficiency_ratio']:.4f} (improvement/cost) |

---

## Token Breakdown

### By Component

| Component | Tokens | Cost | % of Total |
|-----------|--------|------|------------|
| Multi-Agent System | {summary['multi_agent_tokens']:,} | ${summary['multi_agent_cost_usd']:.4f} | {(summary['multi_agent_tokens'] / summary['total_tokens'] * 100) if summary['total_tokens'] > 0 else 0:.1f}% |
| RAGAS Metrics | {summary['ragas_tokens']:,} | ${summary['ragas_cost_usd']:.4f} | {(summary['ragas_tokens'] / summary['total_tokens'] * 100) if summary['total_tokens'] > 0 else 0:.1f}% |
| Other | {summary['total_tokens'] - summary['multi_agent_tokens'] - summary['ragas_tokens']:,} | ${summary['total_cost_usd'] - summary['multi_agent_cost_usd'] - summary['ragas_cost_usd']:.4f} | {((summary['total_tokens'] - summary['multi_agent_tokens'] - summary['ragas_tokens']) / summary['total_tokens'] * 100) if summary['total_tokens'] > 0 else 0:.1f}% |

### By Model

| Model | Calls | Tokens | Cost | % of Total |
|-------|-------|--------|------|------------|
"""

        # Add per-model rows
        for model, stats in sorted(
            summary["by_model"].items(), key=lambda x: x[1]["cost"], reverse=True
        ):
            pct = (
                (stats["tokens"] / summary["total_tokens"] * 100)
                if summary["total_tokens"] > 0
                else 0
            )
            report += f"| {model} | {stats['calls']} | {stats['tokens']:,} | ${stats['cost']:.4f} | {pct:.1f}% |\n"

        report += """
---

## Thesis Validation

**Hypothesis**: Pattern-based learning achieves higher quality with lower token costs.

### Efficiency Metrics

- **Quality achieved**: {summary['final_answer']:.2f} answer_correctness
- **Tokens invested**: {summary['total_tokens']:,}
- **Efficiency ratio**: {summary['efficiency_ratio']:.4f} quality points per dollar

**Interpretation**:
- Each quality point improvement cost ${summary['cost_per_quality_point']:.4f}
- Each quality point required {summary['tokens_per_quality_point']:,.0f} tokens

### Pattern-Based Learning Impact

"""

        # Load iteration summaries to analyze pattern context usage
        summary_file = self.output_dir / f"{self.pattern_id}_token_summaries.jsonl"
        if summary_file.exists():
            with_context = []
            without_context = []

            with open(summary_file) as f:
                for line in f:
                    iter_summary = json.loads(line)
                    if iter_summary["used_pattern_context"]:
                        with_context.append(iter_summary)
                    else:
                        without_context.append(iter_summary)

            if with_context and without_context:
                # Compare efficiency
                avg_with = sum(s["tokens_per_quality_point"] for s in with_context) / len(
                    with_context
                )
                avg_without = sum(s["tokens_per_quality_point"] for s in without_context) / len(
                    without_context
                )

                improvement_pct = (
                    ((avg_without - avg_with) / avg_without * 100) if avg_without > 0 else 0
                )

                report += f"""
**With Pattern Context**:
- Iterations: {len(with_context)}
- Avg tokens per quality point: {avg_with:,.0f}

**Without Pattern Context**:
- Iterations: {len(without_context)}
- Avg tokens per quality point: {avg_without:,.0f}

**Pattern Learning Efficiency Gain**: {improvement_pct:+.1f}%

{"✅ **THESIS VALIDATED**: Pattern-based learning is more efficient!" if improvement_pct > 0 else "⚠️  Pattern context did not improve efficiency in this run"}
"""

        report += """
---

## Detailed Call Log

See: `.diagnostics/{pattern_id}/{pattern_id}_token_calls.jsonl`

## Iteration Summaries

See: `.diagnostics/{pattern_id}/{pattern_id}_token_summaries.jsonl`
"""

        if output_file:
            output_file.write_text(report)
            print(f"📄 Token report saved: {output_file}")

        return report
