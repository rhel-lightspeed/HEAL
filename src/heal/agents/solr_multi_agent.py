"""Multi-Agent Solr Optimization System.

Uses specialized agents for better Solr configuration suggestions:
- Solr Expert: Deep knowledge of Solr/Lucene theory
- OKP-MCP Code Expert: Understands actual implementation
- Synthesizer: Combines theory + reality into practical suggestions
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from heal.agents.base_agent import (
    BaseAgent,
    ModelTierConfig,
    TicketMetrics as BaseTicketMetrics,
)

try:
    from claude_agent_sdk import query as claude_query, ClaudeAgentOptions

    CLAUDE_SDK_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CLAUDE_SDK_AVAILABLE = False
    claude_query = None
    ClaudeAgentOptions = None

try:
    from heal.core.token_tracker import TokenTracker

    TOKEN_TRACKER_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TOKEN_TRACKER_AVAILABLE = False
    TokenTracker = None

logger = logging.getLogger(__name__)


@dataclass
class TicketData:
    """Data for a single ticket in the pattern."""

    ticket_id: str
    query: str
    expected_urls: List[str]
    retrieved_urls: List[str]
    metrics: Dict[str, float]  # url_f1, mrr, etc.
    solr_explain: Optional[str] = None

    # NEW: Diagnostic data for deeper analysis
    expected_response: Optional[str] = None  # Ground truth answer
    actual_llm_response: Optional[str] = None  # What the LLM actually generated
    judge_reasoning: Optional[Dict[str, str]] = (
        None  # Judge's critique (answer_correctness_reason, faithfulness_reason, etc.)
    )


@dataclass
class SolrTheoryAdvice:
    """Advice from Solr theory expert."""

    problem_analysis: str
    ideal_config: Dict[str, Any]
    reasoning: str
    relevant_docs: List[str]


@dataclass
class OkpMcpCodeAnalysis:
    """Analysis from OKP-MCP code expert."""

    current_implementation: str
    constraints: List[str]
    bugs_found: List[str]
    relevant_code_sections: Dict[str, str]  # file → code snippet
    warnings: List[str]


@dataclass
class SynthesizedSuggestion:
    """Final suggestion combining theory + implementation."""

    suggested_change: str
    file_path: str
    old_code: str
    new_code: str
    reasoning: str
    confidence: float  # 0.0-1.0
    risks: List[str]


class SolrMultiAgentSystem(BaseAgent):
    """Multi-agent system for Solr optimization.

    Inherits from BaseSolrOptimizer to get:
    - ModelTierConfig for configurable model tiers
    - classify_complexity() for smart routing
    - get_model_for_complexity() for tier selection
    - _call_llm() common Claude Agent SDK logic
    """

    def __init__(
        self,
        okp_mcp_root: Path,
        model_tiers: Optional[ModelTierConfig] = None,
        use_tiered_routing: bool = True,
        default_model: Optional[str] = None,
        include_judge_reasoning: bool = False,
        # Backward compatibility
        model: Optional[str] = None,
    ):
        """Initialize multi-agent system.

        Args:
            okp_mcp_root: Path to okp-mcp repository
            model_tiers: Model configuration for each tier (simple/medium/complex)
            use_tiered_routing: Enable automatic model selection by complexity
            default_model: Override model for all tiers (disables routing)
            include_judge_reasoning: Include LLM judge's critique in diagnostics (default: False, for A/B testing)
            model: Deprecated - use default_model instead (backward compatibility)
        """
        # Backward compatibility: if old 'model' param used, map to default_model
        if model is not None and default_model is None:
            default_model = model
            use_tiered_routing = False
            logger.warning(
                f"'model' parameter is deprecated, use 'default_model' instead. "
                f"Using default_model={default_model}"
            )

        # Initialize base class with model tier management
        super().__init__(
            model_tiers=model_tiers,
            use_tiered_routing=use_tiered_routing,
            default_model=default_model,
        )

        # Store okp-mcp root for Solr config analysis
        self.okp_mcp_root = okp_mcp_root
        self.include_judge_reasoning = include_judge_reasoning

        logger.info(f"Initialized multi-agent system")
        logger.info(f"Judge reasoning in diagnostics: {include_judge_reasoning}")

    async def _call_llm(
        self, model: str, system_prompt: str, user_prompt: str, call_type: str = "multi_agent"
    ) -> str:
        """Call Claude via Agent SDK with token tracking.

        Overrides base class to add token tracking for HEAL's cost analysis.

        Args:
            model: Model to use for this call
            system_prompt: System instructions for the agent
            user_prompt: User query for the agent
            call_type: Type of call for token tracking (e.g., "multi_agent_solr_expert")

        Returns:
            Raw text response from Claude
        """
        import tempfile
        from pathlib import Path

        # CRITICAL: Temporarily unset GOOGLE_APPLICATION_CREDENTIALS
        # This is set for Gemini evaluations but conflicts with Claude ADC
        saved_google_creds = os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

        try:
            # Combine system and user prompts
            full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"

            # Create debug log file
            from heal.core.config import HEALConfig

            log_file = HEALConfig.get_log_dir() / "solr_multi_agent_debug.log"
            with open(log_file, "a") as log:
                log.write(f"\n{'='*80}\n")
                log.write(f"_call_llm() - model: {model}\n")
                log.write(
                    f"GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')} (should be None)\n"
                )
                log.write(f"Prompt length: {len(full_prompt)} chars\n")
                log.write(f"{'='*80}\n")

                # Use temp directory to avoid CLAUDE.md interference
                # Using allowed_tools=[] prevents tool usage and minimizes MCP overhead
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Use Claude Agent SDK with NO tools (just LLM response)
                    options = ClaudeAgentOptions(
                        model=model,
                        allowed_tools=[],  # Disable all tools - just get text response
                        permission_mode="auto",
                        max_turns=1,
                        debug_stderr=log,  # Write debug output to log file
                        cwd=tmpdir,  # Use temp directory (avoids loading project context)
                    )

                    # Collect response text and track tokens
                    response_text = ""
                    input_tokens_estimate = (
                        len(full_prompt) // 4
                    )  # Rough estimate: 4 chars per token

                    # Get token tracker if available
                    tracker = None
                    if TOKEN_TRACKER_AVAILABLE:
                        tracker = TokenTracker.get_instance()

                    try:
                        async for message in claude_query(prompt=full_prompt, options=options):
                            if hasattr(message, "content"):
                                for block in message.content:
                                    if hasattr(block, "text"):
                                        response_text += block.text
                                        log.write(f"\n[Response block]: {block.text[:200]}...\n")

                            # Try to get actual token usage from message (if available)
                            if hasattr(message, "usage") and tracker:
                                input_tokens_estimate = message.usage.input_tokens

                    except Exception as e:
                        logger.error(f"Claude Agent SDK error: {e}")
                        logger.error(f"Error type: {type(e).__name__}")
                        logger.error(f"See debug log: {log_file}")
                        log.write(f"\n❌ ERROR: {e}\n")
                        raise RuntimeError(
                            f"Failed to get LLM response via Claude Agent SDK: {e}"
                        ) from e

                    # Record token usage
                    if tracker and response_text:
                        output_tokens_estimate = len(response_text) // 4  # Rough estimate
                        tracker.record_tokens(
                            input_tokens=input_tokens_estimate,
                            output_tokens=output_tokens_estimate,
                            call_type=call_type,
                            model=model,
                        )

            if not response_text:
                raise RuntimeError("Claude Agent SDK returned empty response")

            return response_text

        finally:
            # Restore original GOOGLE_APPLICATION_CREDENTIALS for Gemini evaluations
            if saved_google_creds:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = saved_google_creds

    async def get_optimized_suggestion(
        self,
        pattern_id: str,
        failing_tickets: List[TicketData],
        iteration_context: Optional[str] = None,
    ) -> SynthesizedSuggestion:
        """Get optimized Solr suggestion for a PATTERN (all failing tickets together).

        Args:
            pattern_id: Pattern identifier (e.g., "BOOTLOADER_GRUB_ISSUES")
            failing_tickets: List of all failing tickets in the pattern
            iteration_context: Optional context from prior iterations (from pattern DB)

        Returns:
            Synthesized suggestion that should help ALL tickets in the pattern
        """
        logger.info(f"Starting multi-agent optimization for pattern: {pattern_id}")
        logger.info(f"  Analyzing {len(failing_tickets)} failing tickets together")

        # Classify complexity and select appropriate model tier
        if self.use_tiered_routing:
            # Convert to BaseTicketMetrics for classification
            ticket_metrics = [
                BaseTicketMetrics(
                    ticket_id=t.ticket_id,
                    query=t.query,
                    url_f1=t.metrics.get("url_f1", 0.0),
                    mrr=t.metrics.get("mrr", 0.0),
                    answer_correctness=t.metrics.get("answer_correctness"),
                    faithfulness=t.metrics.get("faithfulness"),
                )
                for t in failing_tickets
            ]

            # Pass solr_explain as additional context
            solr_context = None
            if failing_tickets and failing_tickets[0].solr_explain:
                solr_context = f"Solr explain output:\n{failing_tickets[0].solr_explain}"

            complexity = await self.classify_complexity(
                tickets=ticket_metrics,
                additional_context=solr_context,
            )
            model_to_use = self.get_model_for_complexity(complexity)
            logger.info(f"  Pattern complexity: {complexity} → using {model_to_use}")
        else:
            model_to_use = self.model_tiers.medium

        # Phase 1: Solr Expert analyzes the PATTERN from theory perspective
        logger.info("Phase 1: Consulting Solr Expert (pattern analysis)...")
        solr_advice = await self._get_solr_theory_advice(
            pattern_id, failing_tickets, model=model_to_use
        )

        # Phase 2: OKP-MCP Code Expert reads actual implementation
        logger.info("Phase 2: Consulting OKP-MCP Code Expert...")
        code_analysis = await self._get_okp_mcp_code_analysis(
            pattern_id, solr_advice, model=model_to_use
        )

        # Phase 3: Synthesizer combines both to create practical suggestion
        logger.info("Phase 3: Synthesizing practical suggestion...")
        suggestion = await self._synthesize_suggestion(
            pattern_id,
            failing_tickets,
            solr_advice,
            code_analysis,
            iteration_context,
            model=model_to_use,
        )

        return suggestion

    async def _get_solr_theory_advice(
        self,
        pattern_id: str,
        failing_tickets: List[TicketData],
        model: str,
    ) -> SolrTheoryAdvice:
        """Get advice from Solr theory expert analyzing the PATTERN.

        Args:
            pattern_id: Pattern identifier
            failing_tickets: All failing tickets in the pattern
            model: Model to use for this analysis

        Returns:
            Solr theory advice addressing the common root cause
        """
        system_prompt = """You are a world-class expert in Apache Solr and Lucene search technology.

You have deep knowledge of:
- Solr edismax query parser and all its parameters
- BM25 and TF-IDF scoring algorithms
- Query analysis, tokenization, and stopword handling
- Field boosting strategies (qf, pf, pf2, pf3)
- Minimum match (mm) configuration patterns
- Phrase slop (ps) tuning for scattered terms
- Highlighting and snippet extraction
- Re-ranking strategies

Your task: Analyze a PATTERN of failing search queries (multiple tickets with a common root cause).

Find the COMMON PROBLEM across all tickets and recommend ONE Solr configuration change that will help ALL of them.

DO NOT optimize for individual tickets - find the pattern-level issue.
DO NOT worry about implementation constraints - focus on what SHOULD work in theory.
The code expert will handle implementation details.

Return your analysis as JSON:
{
  "problem_analysis": "Common root cause across all tickets in this pattern",
  "ideal_config": {
    "mm": "recommended mm value",
    "qf": "recommended field weights",
    "pf": "recommended phrase boosting",
    "other_params": "any other relevant params"
  },
  "reasoning": "Why this configuration addresses the pattern-level problem",
  "relevant_docs": ["list of relevant Solr concepts/docs"]
}
"""

        # Build user prompt with ALL tickets
        tickets_description = []
        for ticket in failing_tickets:
            desc = f"""
**Ticket {ticket.ticket_id}:**
  Query: {ticket.query}
  Expected URLs: {', '.join(ticket.expected_urls[:3])}{'...' if len(ticket.expected_urls) > 3 else ''}
  Retrieved URLs: {', '.join(ticket.retrieved_urls[:3]) if ticket.retrieved_urls else '(none)'}
  Metrics: F1={ticket.metrics.get('url_f1', 0):.2f}, MRR={ticket.metrics.get('mrr', 0):.2f}
"""

            # Add expected answer and actual LLM response if available
            if ticket.expected_response:
                desc += f"  Expected Answer: {ticket.expected_response[:200]}{'...' if len(ticket.expected_response) > 200 else ''}\n"

            if ticket.actual_llm_response:
                desc += f"  Actual LLM Response: {ticket.actual_llm_response[:200]}{'...' if len(ticket.actual_llm_response) > 200 else ''}\n"

            # Conditionally include judge reasoning (for A/B testing)
            if self.include_judge_reasoning and ticket.judge_reasoning:
                if ticket.judge_reasoning.get("answer_correctness_reason"):
                    desc += f"  Judge Critique: {ticket.judge_reasoning['answer_correctness_reason'][:150]}...\n"

            tickets_description.append(desc)

        user_prompt = f"""Analyze this PATTERN of failing search queries:

**Pattern ID:** {pattern_id}
**Number of Failing Tickets:** {len(failing_tickets)}

{''.join(tickets_description)}

**Your Task:**
Look across ALL these tickets and identify the COMMON ROOT CAUSE. What single Solr configuration change would improve retrieval for the entire pattern?

Based on Solr/Lucene theory, what configuration would IDEALLY help ALL these tickets?

Return JSON only."""

        # Call Claude Agent SDK (track as Solr Expert)
        response_text = await self._call_llm(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            call_type="multi_agent_solr_expert",
        )

        # Parse JSON from response with error handling
        import re

        json_match = re.search(r"```json\s*(\{.+?\})\s*```", response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
        elif "{" in response_text:
            # Try to extract JSON without code blocks
            start = response_text.index("{")
            end = response_text.rindex("}") + 1
            response_text = response_text[start:end]

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Solr Expert JSON response: {e}")
            logger.error(f"Response text: {response_text[:500]}...")
            # Return minimal fallback result
            return SolrTheoryAdvice(
                problem_analysis="Failed to parse LLM response",
                ideal_config={},
                reasoning="JSON parsing error - using fallback",
                relevant_docs=[],
            )

        return SolrTheoryAdvice(
            problem_analysis=result.get("problem_analysis", ""),
            ideal_config=result.get("ideal_config", {}),
            reasoning=result.get("reasoning", ""),
            relevant_docs=result.get("relevant_docs", []),
        )

    async def _get_okp_mcp_code_analysis(
        self,
        pattern_id: str,
        solr_advice: SolrTheoryAdvice,
        model: str,
    ) -> OkpMcpCodeAnalysis:
        """Get analysis from OKP-MCP code expert.

        Args:
            pattern_id: Pattern identifier
            solr_advice: Advice from Solr theory expert
            model: Model to use for this analysis

        Returns:
            OKP-MCP code analysis
        """
        system_prompt = """You are an expert code analyst specializing in the okp-mcp codebase.

Your task: Read the actual okp-mcp implementation and analyze how it currently works.

You have access to read files from the okp-mcp repository.

Focus on:
1. How Solr queries are built (query preprocessing, parameter setting)
2. BM25 re-ranking implementation
3. Highlighting/snippet extraction logic
4. Any special handling for specific query patterns
5. Bugs or edge cases in the implementation
6. Constraints that limit what changes are possible

Return your analysis as JSON:
{
  "current_implementation": "Description of how it currently works",
  "constraints": ["List of constraints that limit changes"],
  "bugs_found": ["Any bugs or issues you found"],
  "relevant_code_sections": {
    "file_path": "relevant code snippet with line numbers"
  },
  "warnings": ["Things to watch out for when making changes"]
}
"""

        # Read the main Solr file
        solr_file = self.okp_mcp_root / "src" / "okp_mcp" / "solr.py"
        if solr_file.exists():
            with open(solr_file) as f:
                solr_code = f.read()
        else:
            solr_code = "(File not found)"

        user_prompt = f"""Analyze the okp-mcp Solr implementation for this pattern:

**Pattern ID:** {pattern_id}

**Solr Theory Expert's Analysis:**
Problem: {solr_advice.problem_analysis}

**Solr Theory Expert's Ideal Config:**
```json
{json.dumps(solr_advice.ideal_config, indent=2)}
```

**Solr Theory Expert's Reasoning:**
{solr_advice.reasoning}

**Your Task:**
1. Read src/okp_mcp/solr.py (provided below)
2. Understand how Solr queries are currently built
3. Identify constraints, bugs, or special handling
4. Determine if the theory expert's ideal config can be implemented
5. Flag any issues or conflicts

**src/okp_mcp/solr.py:**
```python
{solr_code}
```

Analyze this code and return JSON with your findings.

Return JSON only."""

        # Call Claude Agent SDK (track as Code Expert)
        response_text = await self._call_llm(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            call_type="multi_agent_code_expert",
        )

        # Parse JSON from response with better error handling
        import re

        json_match = re.search(r"```json\s*(\{.+?\})\s*```", response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
        elif "{" in response_text:
            start = response_text.index("{")
            end = response_text.rindex("}") + 1
            response_text = response_text[start:end]

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Code Expert JSON response: {e}")
            logger.error(f"Response text: {response_text[:500]}...")
            # Return minimal fallback result
            return OkpMcpCodeAnalysis(
                current_implementation="Failed to parse LLM response",
                constraints=["JSON parsing error - using fallback"],
                bugs_found=[],
                relevant_code_sections={},
                warnings=["Code Expert response parsing failed"],
            )

        return OkpMcpCodeAnalysis(
            current_implementation=result.get("current_implementation", ""),
            constraints=result.get("constraints", []),
            bugs_found=result.get("bugs_found", []),
            relevant_code_sections=result.get("relevant_code_sections", {}),
            warnings=result.get("warnings", []),
        )

    async def _synthesize_suggestion(
        self,
        pattern_id: str,
        failing_tickets: List[TicketData],
        solr_advice: SolrTheoryAdvice,
        code_analysis: OkpMcpCodeAnalysis,
        iteration_context: Optional[str] = None,
        model: str = None,
    ) -> SynthesizedSuggestion:
        """Synthesize practical suggestion from theory + code analysis.

        Args:
            pattern_id: Pattern identifier
            failing_tickets: All failing tickets in the pattern
            solr_advice: Advice from Solr expert
            code_analysis: Analysis from code expert
            iteration_context: Optional context from prior iterations
            model: Model to use for this analysis

        Returns:
            Synthesized practical suggestion for the entire pattern
        """
        system_prompt = """You are a senior software engineer who synthesizes theoretical advice with practical implementation.

IMPORTANT: You are in READ-ONLY analysis mode. Do NOT edit any files or execute any commands.
Your job is to analyze the inputs and suggest a code change by returning JSON ONLY.

CRITICAL - Incremental Improvement Philosophy:
- If prior fix attempts exist, BUILD ON what worked rather than starting over
- NEVER suggest reverting improvements - only refine and extend successful changes
- If a previous fix improved URL F1 but answer correctness is still low, ask: "What's the next bottleneck?"
- Think cumulatively: each fix adds to the previous ones like building blocks
- Example: If Fix A improved retrieval, suggest Fix B that improves answer quality using those better docs

Your task: Recommend a PRACTICAL code change that:
1. Incorporates Solr theory best practices
2. Works within okp-mcp implementation constraints
3. Fixes any bugs identified
4. BUILDS ON prior successful fixes (if any exist)
5. Has high confidence of improving metrics FOR THE ENTIRE PATTERN

Return ONLY a JSON object with this structure (no file edits, no tool calls):
{
  "suggested_change": "Brief description of the recommended change",
  "file_path": "src/okp_mcp/solr.py",
  "old_code": "Snippet of existing code that should be changed",
  "new_code": "Recommended new code to replace it",
  "reasoning": "Why this change will help ALL tickets in the pattern",
  "confidence": 0.85,
  "risks": ["Potential risks or side effects"]
}

The old_code and new_code fields are DESCRIPTIVE SUGGESTIONS ONLY - they will NOT be applied automatically.
Just provide your analysis and recommendation in JSON format.
"""

        # Summarize pattern metrics
        avg_f1 = sum(t.metrics.get("url_f1", 0) for t in failing_tickets) / len(failing_tickets)
        avg_mrr = sum(t.metrics.get("mrr", 0) for t in failing_tickets) / len(failing_tickets)

        # Build user prompt with optional iteration context
        iteration_section = ""
        if iteration_context:
            iteration_section = f"""
**PRIOR FIX ATTEMPTS:**
{iteration_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""

        user_prompt = f"""Synthesize a practical Solr config change for this PATTERN:

**Pattern ID:** {pattern_id}
**Number of Failing Tickets:** {len(failing_tickets)}

**Pattern-Level Metrics:**
- Avg F1: {avg_f1:.2f}
- Avg MRR: {avg_mrr:.2f}

{iteration_section}**Solr Theory Expert Says:**
Problem (across all tickets): {solr_advice.problem_analysis}

Ideal Config:
```json
{json.dumps(solr_advice.ideal_config, indent=2)}
```

Reasoning: {solr_advice.reasoning}

**Code Expert Says:**
Current Implementation: {code_analysis.current_implementation}

Constraints:
{chr(10).join(f'  - {c}' for c in code_analysis.constraints)}

Bugs Found:
{chr(10).join(f'  - {b}' for b in code_analysis.bugs_found) if code_analysis.bugs_found else '  (none)'}

Warnings:
{chr(10).join(f'  - {w}' for w in code_analysis.warnings)}

**Your Task:**
Recommend a PRACTICAL code change that:
1. Addresses the PATTERN-LEVEL root cause
2. Applies Solr theory where possible
3. Respects okp-mcp constraints
4. Fixes bugs if found
5. Is likely to improve F1/MRR for ALL {len(failing_tickets)} tickets

IMPORTANT: Return ONLY JSON with your suggestion. Do NOT edit files or use tools.
The JSON should include old_code and new_code fields describing your recommended change.
These are suggestions that will be reviewed - they will NOT be applied automatically."""

        # Call Claude Agent SDK (track as Synthesizer)
        response_text = await self._call_llm(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            call_type="multi_agent_synthesizer",
        )

        # Parse JSON from response with error handling
        import re

        json_match = re.search(r"```json\s*(\{.+?\})\s*```", response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
        elif "{" in response_text:
            start = response_text.index("{")
            end = response_text.rindex("}") + 1
            response_text = response_text[start:end]

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Synthesizer JSON response: {e}")
            logger.error(f"Response text: {response_text[:500]}...")
            # Return minimal fallback result
            return SynthesizedSuggestion(
                suggested_change="Failed to parse LLM response",
                file_path="unknown",
                old_code="",
                new_code="",
                reasoning="JSON parsing error - using fallback",
                confidence=0.0,
                risks=["Synthesizer response parsing failed"],
            )

        return SynthesizedSuggestion(
            suggested_change=result.get("suggested_change", ""),
            file_path=result.get("file_path", ""),
            old_code=result.get("old_code", ""),
            new_code=result.get("new_code", ""),
            reasoning=result.get("reasoning", ""),
            confidence=result.get("confidence", 0.7),
            risks=result.get("risks", []),
        )

    async def suggest_improvements(
        self,
        tickets: List[TicketData],
        iteration_context: Optional[str] = None,
        pattern_id: str = "UNKNOWN",
        **kwargs,
    ) -> SynthesizedSuggestion:
        """Generate Solr optimization suggestions (implements BaseSolrOptimizer interface).

        Args:
            tickets: List of ticket data to analyze
            iteration_context: Optional context from previous iterations
            pattern_id: Pattern identifier for logging
            **kwargs: Additional arguments

        Returns:
            Synthesized suggestion for improving the pattern
        """
        return await self.get_optimized_suggestion(
            pattern_id=pattern_id,
            failing_tickets=tickets,
            iteration_context=iteration_context,
        )
