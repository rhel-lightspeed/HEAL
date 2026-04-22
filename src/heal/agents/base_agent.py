"""Base class for all HEAL agents with automatic token tracking and model escalation.

Provides common functionality for all agents:
- Model tier management (simple/medium/complex)
- Automatic complexity classification
- Token tracking per model
- LLM calling infrastructure with Claude Agent SDK
- Credential handling (swaps GOOGLE_APPLICATION_CREDENTIALS for ADC)
"""

import logging
import os
import re
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from claude_agent_sdk import query as claude_query, ClaudeAgentOptions

    CLAUDE_SDK_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CLAUDE_SDK_AVAILABLE = False
    claude_query = None
    ClaudeAgentOptions = None

from heal.core.token_tracker import TokenTracker

logger = logging.getLogger(__name__)


@dataclass
class ModelTierConfig:
    """Configuration for model tiers.

    Instead of hardcoding model names, this allows runtime configuration
    of which models to use for each complexity tier.
    """

    simple: str = "claude-haiku-4-5"  # Fast, cheap (classification, simple problems)
    medium: str = "claude-sonnet-4-6"  # Default, balanced (most work)
    complex: str = "claude-opus-4-6"  # Slow, expensive (hard problems)

    def __post_init__(self):
        """Validate model names."""
        valid_prefixes = ["claude-haiku", "claude-sonnet", "claude-opus"]
        for tier, model in [
            ("simple", self.simple),
            ("medium", self.medium),
            ("complex", self.complex),
        ]:
            if not any(model.startswith(prefix) for prefix in valid_prefixes):
                raise ValueError(
                    f"Invalid model for {tier} tier: {model}. "
                    f"Must start with one of: {valid_prefixes}"
                )


@dataclass
class AgentResponse:
    """Response from agent with usage stats extracted from ResultMessage."""

    content: str
    """The actual text response from the LLM"""

    input_tokens: int
    """Input tokens (including cache hits)"""

    output_tokens: int
    """Output tokens generated"""

    total_tokens: int
    """Total tokens (input + output)"""

    cost_usd: float
    """Estimated cost in USD"""

    model: str
    """Model used"""

    duration_ms: float
    """API call duration in milliseconds"""

    cache_read_tokens: int = 0
    """Tokens read from cache (subset of input_tokens)"""

    cache_creation_tokens: int = 0
    """Tokens written to cache"""


@dataclass
class TicketMetrics:
    """Minimal ticket metrics for complexity classification.

    Used by base class to classify problem complexity without knowing
    the full MetricSummary/TicketData structures from subclasses.
    """

    ticket_id: str
    query: str
    url_f1: float
    mrr: float
    answer_correctness: Optional[float] = None
    faithfulness: Optional[float] = None


class BaseAgent(ABC):
    """Base class for all HEAL agents with token tracking and model escalation.

    Provides:
    - Model tier management (no hardcoded model names)
    - Automatic complexity classification (SIMPLE/MEDIUM/COMPLEX)
    - Token tracking per model (auto-extracts from ResultMessage)
    - Common LLM calling infrastructure
    - Claude Agent SDK setup with credential handling

    All agents should inherit from this to get automatic cost optimization.
    """

    def __init__(
        self,
        model_tiers: Optional[ModelTierConfig] = None,
        use_tiered_routing: bool = True,
        default_model: Optional[str] = None,
    ):
        """Initialize base agent.

        Args:
            model_tiers: Model configuration for each tier (simple/medium/complex)
            use_tiered_routing: Enable automatic model selection by complexity
            default_model: Override model for all tiers (disables routing)
        """
        if not CLAUDE_SDK_AVAILABLE:
            raise ImportError(
                "claude-agent-sdk not available. " "Install with: uv pip install claude-agent-sdk"
            )

        self.use_tiered_routing = use_tiered_routing

        # Initialize model tiers
        if model_tiers is None:
            model_tiers = ModelTierConfig()
        self.model_tiers = model_tiers

        # Override for testing or consistency
        if default_model:
            self.model_tiers.simple = default_model
            self.model_tiers.medium = default_model
            self.model_tiers.complex = default_model
            self.use_tiered_routing = False
            logger.info(f"Using fixed model: {default_model} (routing disabled)")

        logger.info(
            f"Initialized {self.__class__.__name__} with model tiers:\n"
            f"  Simple:  {self.model_tiers.simple}\n"
            f"  Medium:  {self.model_tiers.medium}\n"
            f"  Complex: {self.model_tiers.complex}\n"
            f"  Tiered routing: {self.use_tiered_routing}"
        )

    async def classify_complexity(
        self,
        tickets: List[TicketMetrics],
        additional_context: Optional[str] = None,
    ) -> str:
        """Classify problem complexity: SIMPLE, MEDIUM, or COMPLEX.

        Uses cheap model (simple tier) for fast classification.

        Args:
            tickets: List of tickets to analyze
            additional_context: Optional context (e.g., Solr explain, error logs)

        Returns:
            "SIMPLE", "MEDIUM", or "COMPLEX"
        """
        # Build classification prompt
        avg_url_f1 = sum(t.url_f1 for t in tickets) / len(tickets)
        avg_mrr = sum(t.mrr for t in tickets) / len(tickets)

        answer_scores = [t.answer_correctness for t in tickets if t.answer_correctness is not None]
        avg_answer = sum(answer_scores) / len(answer_scores) if answer_scores else None

        system_prompt = """You are a quick diagnostic classifier for technical problems.

Classify problem complexity as SIMPLE, MEDIUM, or COMPLEX.

SIMPLE problems (use Haiku):
- Small metric gaps (<0.2 from threshold)
- Single obvious issue
- Clear pattern across tickets

MEDIUM problems (use Sonnet):
- Moderate metric gaps (0.2-0.5)
- Multiple interacting factors
- Requires balanced trade-offs

COMPLEX problems (use Opus):
- Large metric gaps (>0.5)
- Deep architectural issues
- Contradictory evidence
- Requires nuanced reasoning

Respond with ONLY one word: SIMPLE, MEDIUM, or COMPLEX"""

        user_prompt = f"""Analyze this problem:

Tickets: {len(tickets)}
Average URL F1: {avg_url_f1:.2f} (threshold: 0.7)
Average MRR: {avg_mrr:.2f}
"""

        if avg_answer:
            user_prompt += f"Average Answer Correctness: {avg_answer:.2f} (threshold: 0.75)\n"

        if additional_context:
            user_prompt += f"\nAdditional Context:\n{additional_context[:500]}...\n"

        user_prompt += "\nClassify as: SIMPLE, MEDIUM, or COMPLEX"

        # Call with simple model
        response = await self.query_claude(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.model_tiers.simple,
            call_type="classify_complexity",
        )

        # Parse response
        complexity = response.content.strip().upper()
        if complexity not in ["SIMPLE", "MEDIUM", "COMPLEX"]:
            logger.warning(f"Invalid complexity: {complexity}, defaulting to MEDIUM")
            complexity = "MEDIUM"

        logger.info(f"Classified complexity: {complexity}")
        return complexity

    def get_model_for_complexity(self, complexity: str) -> str:
        """Get appropriate model for given complexity.

        Args:
            complexity: "SIMPLE", "MEDIUM", or "COMPLEX"

        Returns:
            Model name from configured tiers
        """
        if not self.use_tiered_routing:
            return self.model_tiers.medium

        complexity_map = {
            "SIMPLE": self.model_tiers.simple,
            "MEDIUM": self.model_tiers.medium,
            "COMPLEX": self.model_tiers.complex,
        }

        return complexity_map.get(complexity, self.model_tiers.medium)

    async def query_claude(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        call_type: str = "agent_query",
        max_turns: int = 1,
        permission_mode: str = "auto",
        used_pattern_context: bool = False,
    ) -> AgentResponse:
        """Query Claude with automatic token tracking.

        Extracts usage stats from ResultMessage and auto-tracks to TokenTracker.

        Args:
            system_prompt: System instructions
            user_prompt: User query
            model: Model to use (defaults to medium tier)
            call_type: Type of call for tracking (e.g., "answer_review", "linux_expert_eval")
            max_turns: Maximum conversation turns
            permission_mode: Permission mode for Claude SDK
            used_pattern_context: Whether pattern DB context was used

        Returns:
            AgentResponse with content and usage stats
        """
        model = model or self.model_tiers.medium

        # CRITICAL: Temporarily unset GOOGLE_APPLICATION_CREDENTIALS
        # This is set for Gemini evaluations but conflicts with Claude ADC
        saved_google_creds = os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

        try:
            # Combine prompts for Agent SDK
            full_prompt = f"""{system_prompt}

USER REQUEST:
{user_prompt}"""

            options = ClaudeAgentOptions(
                model=model,
                max_turns=max_turns,
                permission_mode=permission_mode,
            )

            # Query Claude
            final_message = None
            response_text = ""

            async for message in claude_query(prompt=full_prompt, options=options):
                final_message = message

                # Extract text from content blocks (for streaming responses)
                if hasattr(message, "content"):
                    for block in message.content:
                        if hasattr(block, "text"):
                            response_text += block.text

            # Extract usage stats from ResultMessage
            if final_message is None:
                raise RuntimeError("No response received from Claude SDK")

            # ResultMessage has these fields:
            # - total_cost_usd
            # - usage: dict with input_tokens, output_tokens, cache_*_input_tokens
            # - duration_ms

            usage = getattr(final_message, "usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            cache_read = usage.get("cache_read_input_tokens", 0)
            cache_creation = usage.get("cache_creation_input_tokens", 0)

            total_tokens = input_tokens + output_tokens
            cost_usd = getattr(final_message, "total_cost_usd", 0.0)
            duration_ms = getattr(final_message, "duration_ms", 0)

            # Track tokens if tracker is active
            tracker = TokenTracker.get_instance()
            if tracker:
                tracker.record_tokens(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    call_type=call_type,
                    model=model,
                    used_pattern_context=used_pattern_context,
                )

            # Log usage
            logger.debug(
                f"{call_type}: {input_tokens} in + {output_tokens} out = "
                f"{total_tokens} tokens (${cost_usd:.4f}) [{model}]"
            )

            return AgentResponse(
                content=response_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                model=model,
                duration_ms=duration_ms,
                cache_read_tokens=cache_read,
                cache_creation_tokens=cache_creation,
            )

        finally:
            # Restore GOOGLE_APPLICATION_CREDENTIALS
            if saved_google_creds:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = saved_google_creds
