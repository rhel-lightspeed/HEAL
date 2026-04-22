#!/usr/bin/env python3
"""LLM-powered advisor for okp-mcp boost query suggestions.

Uses Claude Agent SDK to analyze metrics and suggest code changes.
Supports tiered model routing to optimize costs.
"""

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage
from pydantic import BaseModel, Field

from heal.agents.base_agent import (
    BaseAgent,
    ModelTierConfig,
    TicketMetrics,
)


class SolrConfigSuggestion(BaseModel):
    """Structured suggestion for Solr eDismax configuration changes.

    This Pydantic model represents an AI-generated suggestion for improving
    Solr document retrieval by modifying search configuration parameters.
    Used by OkpMcpLLMAdvisor to communicate changes to okp-mcp's Solr search.

    The advisor analyzes evaluation metrics (URL F1, MRR, context relevance) and
    Solr explain output to suggest targeted configuration changes that should
    improve document retrieval for specific queries.

    Configuration Parameters That Can Be Suggested:
        - Field weights (qf): "title^4.0 main_content^2.0" - which fields to search
        - Phrase boosts (pf/pf2/pf3): "title^8.0" - boost exact phrase matches
        - Phrase slop (ps/ps2/ps3): Allow word proximity matching
        - Minimum match (mm): "2<-1 5<60%" - how many query terms must match
        - Boost/demote multipliers: Dynamic score adjustments
        - Boost/demote keyword lists: Keywords that trigger score changes

    Usage Flow:
        1. OkpMcpAgent runs evaluation, gets poor retrieval metrics
        2. Calls OkpMcpLLMAdvisor.suggest_boost_query_changes()
        3. LLM analyzes metrics + Solr explain output
        4. Returns SolrConfigSuggestion with specific change to try
        5. Agent applies change to okp-mcp/src/okp_mcp/solr.py
        6. Restarts service and re-tests

    Example:
        suggestion = SolrConfigSuggestion(
            reasoning="Expected docs have 'uefi' in title, but title weight is low",
            file_path="src/okp_mcp/solr.py",
            suggested_change="Increase title boost from 4.0 to 6.0",
            code_snippet='"qf": "title^6.0 main_content^2.0",',
            expected_improvement="URL F1 should increase from 0.2 to >0.5",
            confidence="high"
        )

    Attributes:
        reasoning: Why this change is needed (based on metrics/explain output)
        file_path: Path to file to edit (usually 'src/okp_mcp/solr.py')
        suggested_change: Human-readable description of the change
        code_snippet: Exact Python code after the change (REQUIRED for Edit tool)
        expected_improvement: What metrics should improve and by how much
        confidence: "high", "medium", or "low" - LLM's confidence in suggestion
    """

    reasoning: str = Field(
        description="Why these changes are needed based on the metrics and Solr explain output"
    )
    file_path: str = Field(
        description="Relative path to file to edit (typically 'src/okp_mcp/solr.py')"
    )
    suggested_change: str = Field(
        description="Specific config change to make (e.g., 'Increase title phrase boost from pf: title^8 to title^12' or 'Add \"compatibility matrix\" to _EXTRACTION_BOOST_KEYWORDS')"
    )
    code_snippet: str = Field(
        description="REQUIRED: Python code snippet showing the exact line after the change. Must include the full line with correct syntax. Example: '\"mm\": \"2<-1 5<60%\",' or 'multiplier *= 3.0  # boost'"
    )
    expected_improvement: str = Field(
        description="What metrics should improve (e.g., 'URL F1 should increase from 0.33 to >0.7, expected docs should rank in top 3')"
    )
    confidence: str = Field(description="Confidence level: high, medium, or low")


# Backward compatibility alias
BoostQuerySuggestion = SolrConfigSuggestion


class JudgeDisagreement(BaseModel):
    """Report of disagreement between Gemini judge and Claude advisor.

    When Claude (advisor) disagrees with Gemini's (judge) assessment, this captures
    the disagreement for human review. Especially valuable for flaky evaluations where
    scores bounce around - if judge/advisor also disagree, the test itself may be unreliable.

    Why This Matters:
        - Quality control on judge: Catch when Gemini makes mistakes
        - Test data validation: Flag bad expected_response values
        - Borderline cases: When score is 0.7-0.8, disagreement reveals ambiguity
        - Skip optimization: Don't waste iterations fixing non-problems
        - Flaky test detection: Unstable scores + disagreement = unreliable evaluation

    Usage Flow:
        1. OkpMcpAgent gets evaluation with judge reasoning
        2. Calls OkpMcpLLMAdvisor.check_judge_agreement()
        3. Claude reviews judge's assessment against actual response
        4. If disagreement detected, returns JudgeDisagreement
        5. Agent flags for human review instead of auto-fixing
        6. Logged in diagnostic output for review

    Example:
        disagreement = JudgeDisagreement(
            metric="answer_correctness",
            judge_score=0.75,
            judge_reasoning="Missing the EOL date for RHEL 6",
            advisor_assessment="I see the EOL date clearly stated: 'November 30, 2020'",
            severity="high",
            recommendation="Expected response may be unclear. Needs SME review."
        )

    Attributes:
        metric: Which metric has the disagreement (e.g., "answer_correctness")
        judge_score: Score given by Gemini judge (0.0-1.0)
        judge_reasoning: Gemini's explanation for the score
        advisor_assessment: Claude's independent analysis of the response
        severity: "high" (clear error), "medium" (ambiguous), "low" (minor difference)
        recommendation: What should be done (review expected_response, tune judge, etc.)
    """

    metric: str = Field(description="Metric name (e.g., 'answer_correctness')")
    judge_score: float = Field(description="Score from Gemini judge (0.0-1.0)")
    judge_reasoning: str = Field(description="Judge's explanation for the score")
    advisor_assessment: str = Field(description="Claude's independent analysis")
    severity: str = Field(description="Severity: high, medium, or low")
    recommendation: str = Field(
        description="What to do next (human review, fix expected_response, etc.)"
    )


class PromptSuggestion(BaseModel):
    """Structured suggestion for system prompt modifications.

    This Pydantic model represents an AI-generated suggestion for improving
    answer quality by modifying the system prompt used by the LLM under test.
    Used by OkpMcpLLMAdvisor when retrieval is good but answers are incorrect.

    When to Use:
        - Good retrieval (URL F1 > 0.5, context relevance > 0.7)
        - Poor answer quality (answer_correctness < 0.90, keywords missing)
        - LLM is not properly using the retrieved context

    Common Prompt Issues Addressed:
        - LLM ignoring context → Add "ONLY use provided documentation"
        - Keywords missing → Add "Include specific terms: X, Y, Z"
        - Wrong tone/format → Add output formatting instructions
        - Hallucination → Add "Do not make assumptions beyond the documentation"
        - Context not grounded → Add "Quote directly from provided context"

    Usage Flow:
        1. OkpMcpAgent detects answer problem (good retrieval, bad answer)
        2. Calls OkpMcpLLMAdvisor.suggest_prompt_changes()
        3. LLM analyzes metrics + actual vs expected response
        4. Returns PromptSuggestion with specific prompt modification
        5. Agent applies change to okp-mcp system prompt
        6. Restarts service and re-tests answer quality

    Example:
        suggestion = PromptSuggestion(
            reasoning="Answer missing 'deprecated' keyword despite context containing it",
            suggested_change="Add instruction: 'When describing removed features, "
                           "explicitly use the word deprecated or removed'",
            expected_improvement="keywords_score should increase from 0.5 to >0.8",
            confidence="medium"
        )

    Attributes:
        reasoning: Why prompt changes are needed (based on metrics and response analysis)
        suggested_change: Specific modification to make to the system prompt
        expected_improvement: What metrics should improve (keywords, answer_correctness, etc.)
        confidence: "high", "medium", or "low" - LLM's confidence in suggestion
    """

    reasoning: str = Field(description="Why prompt changes are needed based on the metrics")
    suggested_change: str = Field(description="Specific prompt modification to make")
    expected_improvement: str = Field(description="What metrics should improve")
    confidence: str = Field(description="Confidence level: high, medium, or low")


@dataclass
class MetricSummary:
    """Comprehensive evaluation metrics package for LLM-powered analysis.

    This dataclass aggregates all metrics, ground truth, and diagnostic data needed
    for OkpMcpLLMAdvisor to analyze retrieval and answer quality issues. It serves
    as the input to the LLM advisor's suggestion methods.

    The summary is converted to a human-readable prompt context via to_prompt_context()
    and passed to Claude along with Solr explain output to generate targeted fixes.

    Metric Categories:
        1. Retrieval Metrics: How well documents are retrieved
           - url_f1: F1 score for expected URL retrieval (0.0-1.0, threshold: 0.7)
           - mrr: Mean Reciprocal Rank (0.0-1.0, threshold: 0.5)
           - context_relevance: Are retrieved docs relevant? (Ragas, threshold: 0.7)
           - context_precision: What % of retrieved docs useful? (Ragas, threshold: 0.7)

        2. Answer Quality Metrics: How good is the LLM's answer
           - answer_correctness: Factual accuracy vs expected (custom, threshold: 0.90)
           - faithfulness: Answer grounded in context? (Ragas, threshold: 0.8)
           - response_relevancy: Answer addresses question? (Ragas, threshold: 0.8)
           - keywords_score: Required keywords present? (custom, threshold: 0.7)
           - forbidden_claims_score: No forbidden claims? (custom, threshold: 1.0)

        3. Ground Truth: Expected values for comparison
           - expected_response: What the answer should say (from SME)
           - expected_keywords: Required terms that must appear
           - expected_urls: Which docs should be retrieved
           - forbidden_claims: Statements that must NOT appear

        4. Diagnostic Data: For root cause analysis
           - solr_explain: Solr's scoring explanation for each doc
           - solr_config_snapshot: Current search config parameters
           - ranking_analysis: Why expected docs ranked poorly
           - iteration_history: Previous fix attempts and results

    Usage Flow:
        1. OkpMcpAgent runs evaluation, gets metrics
        2. Creates MetricSummary with all data
        3. Passes to OkpMcpLLMAdvisor.suggest_boost_query_changes() or suggest_prompt_changes()
        4. LLM analyzes metrics + diagnostics
        5. Returns SolrConfigSuggestion or PromptSuggestion
        6. Agent applies fix and re-tests

    Example:
        metrics = MetricSummary(
            ticket_id="RSPEED-2482",
            query="Can I run RHEL 6 containers on RHEL 9?",
            url_f1=0.0,  # No expected docs retrieved!
            mrr=0.2,
            context_relevance=0.0,
            answer_correctness=0.3,
            rag_used=True,
            docs_retrieved=True,
            num_docs=5,
            expected_urls=["docs/rhel9/container-compatibility.html"],
            retrieved_urls=["docs/rhel8/general-info.html"],
            solr_explain={...},  # Why wrong docs ranked higher
            solr_config_snapshot={"qf": "title^4.0 main_content^2.0", ...}
        )
        suggestion = await advisor.suggest_boost_query_changes(metrics)

    Attributes:
        ticket_id: JIRA ticket ID (e.g., "RSPEED-2482")
        query: User's question/search query
        url_f1: F1 score for URL retrieval (None if not measured)
        mrr: Mean Reciprocal Rank (None if not measured)
        context_relevance: Ragas context relevance score (None if not measured)
        context_precision: Ragas context precision score (None if not measured)
        keywords_score: Custom keywords metric (None if not measured)
        forbidden_claims_score: Custom forbidden claims metric (None if not measured)
        faithfulness: Ragas faithfulness score (None if not measured)
        answer_correctness: Custom answer correctness score (None if not measured)
        response_relevancy: Ragas response relevancy score (None if not measured)
        rag_used: Was RAG/search tool called?
        docs_retrieved: Were any documents retrieved?
        num_docs: Number of documents retrieved
        answer_correctness_reason: Judge's explanation for answer_correctness score
        faithfulness_reason: Judge's explanation for faithfulness score
        response_relevancy_reason: Judge's explanation for response_relevancy score
        context_relevance_reason: Judge's explanation for context_relevance score
        context_precision_reason: Judge's explanation for context_precision score
        response: Actual LLM answer (None if not generated)
        expected_response: Expected answer from SME (None if not provided)
        expected_keywords: Required keywords (None if not specified)
        expected_urls: Expected URLs to retrieve (None if not provided)
        forbidden_claims: Forbidden statements (None if not specified)
        retrieved_urls: Actually retrieved URLs (None if RAG not used)
        contexts: Retrieved document texts (None if not available)
        iteration_history: Previous fix attempts (None on first iteration)
        solr_explain: Solr explain output (None if not available)
        solr_config_summary: Full Solr config text (None if not loaded)
        solr_config_snapshot: Structured Solr config (None if not extracted)
        ranking_analysis: Why expected docs ranked poorly (None if not analyzed)
    """

    ticket_id: str
    query: str
    url_f1: Optional[float]
    mrr: Optional[float]
    context_relevance: Optional[float]
    context_precision: Optional[float]
    keywords_score: Optional[float]
    forbidden_claims_score: Optional[float]
    faithfulness: Optional[float]
    answer_correctness: Optional[float]
    response_relevancy: Optional[float]
    rag_used: bool
    docs_retrieved: bool
    num_docs: int

    # LLM Judge reasoning (detailed explanations from evaluation)
    answer_correctness_reason: Optional[str] = None
    faithfulness_reason: Optional[str] = None
    response_relevancy_reason: Optional[str] = None
    context_relevance_reason: Optional[str] = None
    context_precision_reason: Optional[str] = None

    # Ground truth / expected values
    response: Optional[str] = None  # Actual LLM answer
    expected_response: Optional[str] = None  # What answer should say
    expected_keywords: Optional[list] = None  # Which keywords should be present
    expected_urls: Optional[list] = None  # Which URLs should be retrieved
    forbidden_claims: Optional[list] = None  # What should NOT be in answer
    retrieved_urls: Optional[list] = None  # Which URLs were actually retrieved
    contexts: Optional[str] = None  # Retrieved document contexts

    # Iteration history for learning from previous attempts
    iteration_history: Optional[list] = None  # List of previous iteration attempts

    # Solr explain output and configuration analysis
    solr_explain: Optional[dict] = (
        None  # Solr explain output showing why docs ranked the way they did
    )
    solr_config_summary: Optional[str] = None  # Current Solr config summary (full text)
    solr_config_snapshot: Optional[dict] = (
        None  # Structured Solr config (faster, replaces file reads)
    )
    ranking_analysis: Optional[dict] = None  # Analysis of why expected docs didn't rank well

    def to_prompt_context(self) -> str:
        """Convert metrics to human-readable context for LLM."""
        lines = [
            f"Ticket: {self.ticket_id}",
            f"Query: {self.query}",
            "",
            "RAG Status:",
            f"  - RAG Used: {self.rag_used}",
            f"  - Docs Retrieved: {self.docs_retrieved}",
            f"  - Num Docs: {self.num_docs}",
            "",
            "Retrieval Metrics:",
        ]

        if self.url_f1 is not None:
            lines.append(f"  - URL F1: {self.url_f1:.2f} (threshold: 0.7)")
        if self.mrr is not None:
            lines.append(f"  - MRR: {self.mrr:.2f} (threshold: 0.5)")
        if self.context_relevance is not None:
            lines.append(f"  - Context Relevance: {self.context_relevance:.2f} (threshold: 0.7)")
            if self.context_relevance_reason:
                lines.append(f"    Judge: {self.context_relevance_reason[:200]}")
        if self.context_precision is not None:
            lines.append(f"  - Context Precision: {self.context_precision:.2f} (threshold: 0.7)")
            if self.context_precision_reason:
                lines.append(f"    Judge: {self.context_precision_reason[:200]}")

        lines.append("")
        lines.append("Answer Metrics:")

        if self.faithfulness is not None:
            lines.append(f"  - Faithfulness: {self.faithfulness:.2f} (threshold: 0.8)")
            if self.faithfulness_reason:
                lines.append(f"    Judge: {self.faithfulness_reason[:200]}")
        if self.answer_correctness is not None:
            lines.append(f"  - Answer Correctness: {self.answer_correctness:.2f} (threshold: 0.90)")
            if self.answer_correctness_reason:
                lines.append(f"    Judge: {self.answer_correctness_reason[:300]}")
        if self.response_relevancy is not None:
            lines.append(f"  - Response Relevancy: {self.response_relevancy:.2f} (threshold: 0.8)")
            if self.response_relevancy_reason:
                lines.append(f"    Judge: {self.response_relevancy_reason[:200]}")
        if self.keywords_score is not None:
            lines.append(f"  - Keywords: {self.keywords_score:.2f} (threshold: 0.7)")
        if self.forbidden_claims_score is not None:
            lines.append(
                f"  - Forbidden Claims: {self.forbidden_claims_score:.2f} (threshold: 1.0)"
            )

        # Add ground truth / expected values for better context
        if self.expected_urls:
            lines.append("")
            lines.append("Expected URLs (should be retrieved):")
            for url in self.expected_urls:
                lines.append(f"  - {url}")

        if self.retrieved_urls:
            lines.append("")
            lines.append("Retrieved URLs (actual):")
            for url in self.retrieved_urls:
                lines.append(f"  - {url}")

        if self.expected_keywords:
            lines.append("")
            lines.append("Expected Keywords (should be in answer):")
            for keyword_set in self.expected_keywords:
                if isinstance(keyword_set, list):
                    lines.append(f"  - {' OR '.join(keyword_set)}")
                else:
                    lines.append(f"  - {keyword_set}")

        if self.forbidden_claims:
            lines.append("")
            lines.append("Forbidden Claims (must NOT be in answer):")
            for claim in self.forbidden_claims:
                lines.append(f"  - {claim}")

        if self.expected_response and isinstance(self.expected_response, str):
            lines.append("")
            lines.append("Expected Answer Guidance:")
            # Truncate if too long
            expected = (
                self.expected_response[:500] + "..."
                if len(self.expected_response) > 500
                else self.expected_response
            )
            lines.append(f"  {expected}")

        if self.response and isinstance(self.response, str):
            lines.append("")
            lines.append("Actual LLM Response:")
            # Truncate if too long
            response = self.response[:500] + "..." if len(self.response) > 500 else self.response
            lines.append(f"  {response}")

        if self.contexts:
            lines.append("")
            lines.append("Retrieved Contexts (compact):")
            # Truncate contexts to 200 chars to save tokens
            contexts = (
                str(self.contexts)[:200] + "..."
                if len(str(self.contexts)) > 200
                else str(self.contexts)
            )
            lines.append(f"  {contexts}")

        # Add Solr configuration - use snapshot if available (faster, more focused)
        if self.solr_config_snapshot:
            lines.append("")
            lines.append("=" * 40)
            lines.append("SOLR CONFIGURATION (CURRENT):")
            lines.append("=" * 40)
            snap = self.solr_config_snapshot
            lines.append("\nDocument Ranking Parameters:")
            lines.append(f"  mm (minimum match): {snap['solr_params'].get('mm')}")
            lines.append(f"  qf (query fields): {snap['solr_params'].get('qf')}")
            lines.append(f"  pf (phrase fields): {snap['solr_params'].get('pf')}")
            lines.append(f"  pf2 (bigram boost): {snap['solr_params'].get('pf2')}")
            lines.append(f"  pf3 (trigram boost): {snap['solr_params'].get('pf3')}")
            lines.append(f"  boost_multiplier: {snap['solr_params'].get('boost_multiplier')}x")
            lines.append(f"  demote_multiplier: {snap['solr_params'].get('demote_multiplier')}x")

            lines.append("\nHighlighting Parameters (snippet selection):")
            if "highlighting_params" in snap:
                lines.append(
                    f"  hl.snippets: {snap['highlighting_params'].get('hl.snippets')} (how many snippets per doc)"
                )
                lines.append(
                    f"  hl.fragsize: {snap['highlighting_params'].get('hl.fragsize')} (chars per snippet)"
                )
                lines.append(
                    f"  hl.score.k1: {snap['highlighting_params'].get('hl.score.k1')} (BM25 term saturation)"
                )
                lines.append(
                    f"  hl.score.b: {snap['highlighting_params'].get('hl.score.b')} (BM25 length normalization)"
                )
                lines.append(
                    f"  hl.score.pivot: {snap['highlighting_params'].get('hl.score.pivot')} (BM25 avg snippet length)"
                )

            lines.append(f"\nBoost Keywords: {snap['boost_keywords_count']} total")
            lines.append(f"  Sample (first 30): {', '.join(snap['boost_keywords_sample'])}")
            lines.append(f"\nDemote Keywords: {snap['demote_keywords_count']} total")
            if snap["demote_keywords_sample"]:
                lines.append(f"  Sample: {', '.join(snap['demote_keywords_sample'])}")
            lines.append("\nFile Locations:")
            for key, loc in snap["file_locations"].items():
                lines.append(f"  {key}: {loc}")
        elif self.solr_config_summary:
            # Fallback to full config summary if snapshot not available
            lines.append("")
            lines.append("=" * 40)
            lines.append("SOLR CONFIGURATION (CURRENT):")
            lines.append("=" * 40)
            lines.append(self.solr_config_summary)

        if self.ranking_analysis:
            lines.append("")
            lines.append("=" * 40)
            lines.append("RANKING ANALYSIS (compact):")
            lines.append("=" * 40)
            lines.append(
                f"Expected: {self.ranking_analysis.get('expected_count', 0)}, Retrieved: {self.ranking_analysis.get('retrieved_count', 0)}"
            )

            # Show only top 3 missing docs to save tokens
            if self.ranking_analysis.get("missing_docs"):
                lines.append("")
                lines.append("Top 3 Missing Expected Docs:")
                for doc in self.ranking_analysis["missing_docs"][:3]:
                    lines.append(
                        f"  - {doc['url']} (rank: {doc['rank']}, score: {doc.get('score', 'N/A')})"
                    )

            # Suggestions are valuable, keep all
            if self.ranking_analysis.get("suggestions"):
                lines.append("")
                lines.append("Suggestions:")
                for suggestion in self.ranking_analysis["suggestions"]:
                    lines.append(f"  - {suggestion}")

        if self.solr_explain:
            lines.append("")
            lines.append("=" * 40)
            lines.append("SOLR EXPLAIN (Top 2 docs, compact):")
            lines.append("=" * 40)
            # Show explain for top 2 docs only, truncate to 200 chars
            docs = self.solr_explain.get("docs", [])[:2]
            explain_data = self.solr_explain.get("explain", {})
            for i, doc in enumerate(docs, 1):
                doc_id = doc.get("id", "")
                lines.append(f"\n{i}. {doc.get('title', '')} (score: {doc.get('score', 0):.2f})")
                lines.append(f"   URL: {doc.get('url', '')}")
                # Truncate explain to 200 chars to save tokens
                explain_text = explain_data.get(doc_id, "")
                if len(explain_text) > 200:
                    explain_text = explain_text[:200] + "..."
                if explain_text:
                    lines.append(f"   Explain: {explain_text}")

        # Add iteration history for learning from previous attempts
        # Use compact format to save tokens (~50 tokens/iteration vs 200+)
        if self.iteration_history:
            lines.append("")
            lines.append("=" * 40)
            lines.append("PREVIOUS ATTEMPTS (compact):")
            lines.append("=" * 40)

            # Table header
            lines.append("Iter | Change | Metric Δ | Overlap | Result")
            lines.append("-" * 60)

            for record in self.iteration_history:
                iter_num = record.get("iteration", "?")

                # Truncate change description to ~30 chars
                change = record.get("change", "N/A")
                if len(change) > 30:
                    change = change[:27] + "..."

                # Metric delta
                if "metric_before" in record and "metric_after" in record:
                    delta = record["metric_after"] - record["metric_before"]
                    metric_str = f"{delta:+.2f}"
                else:
                    metric_str = "N/A"

                # URL overlap (compact indicator)
                if (
                    "metrics" in record
                    and record["metrics"].get("url_overlap_with_previous") is not None
                ):
                    overlap = record["metrics"]["url_overlap_with_previous"]
                    overlap_str = f"{overlap:.2f}"
                else:
                    overlap_str = "N/A"

                # Result (compact)
                improved = record.get("improved", False)
                result = "✓" if improved else "✗"

                lines.append(
                    f"{iter_num:4} | {change:30} | {metric_str:8} | {overlap_str:7} | {result}"
                )

                # Only add details for interesting cases (to save tokens)
                # Show query augmentation if detected
                if "solr_query_inspection" in record and record["solr_query_inspection"]:
                    sqr = record["solr_query_inspection"]
                    if sqr.get("injected_terms"):
                        lines.append(
                            f"     ⚠️  Query augmented: +{len(sqr['injected_terms'])} terms"
                        )

            lines.append("")
            lines.append("⚠️  Code reset to original. Learn from patterns above.")
            lines.append(
                "   Don't repeat failed approaches. Low overlap + no improvement = made it worse."
            )

        return "\n".join(lines)


class OkpMcpLLMAdvisor(BaseAgent):
    """AI-powered advisor for automatically diagnosing and fixing okp-mcp RAG issues.

    This class uses Claude (via Claude Agent SDK) to analyze evaluation metrics and
    suggest targeted code changes to improve document retrieval and answer quality.
    Inherits from BaseSolrOptimizer for tiered model routing.

    Core Capabilities:
        1. Diagnose Problems: Analyzes metrics to identify retrieval vs. answer issues
        2. Suggest Fixes: Generates specific code changes for Solr config or prompts
        3. Learn from History: Uses iteration_history to avoid repeating failed approaches
        4. Route Smartly: Uses Haiku for classification, Sonnet for most work, Opus for hard problems

    Problem Types Addressed:
        - Poor Document Retrieval: Wrong docs retrieved → Suggest Solr config changes
        - Good Retrieval, Bad Answer: Right docs but LLM ignores them → Suggest prompt changes
        - Complex Multi-faceted Issues: Automatically escalates to more capable model

    Tiered Model Routing (from BaseSolrOptimizer):
        - Simple (Haiku): Fast problem classification (SIMPLE/MEDIUM/COMPLEX)
        - Medium (Sonnet): Default for most suggestions, good balance of cost/quality
        - Complex (Opus): Escalation for ambiguous or multi-faceted problems

        Cost comparison (approximate):
        - Haiku: $0.25/$1.25 per 1M tokens (input/output)
        - Sonnet: $3/$15 per 1M tokens
        - Opus: $15/$75 per 1M tokens

        The advisor automatically classifies problem complexity and routes to the
        appropriate model, falling back to cheaper models on failure.

    Usage Example - Solr Config Optimization:
        # Initialize advisor
        advisor = OkpMcpLLMAdvisor(
            okp_mcp_root=Path("~/Work/okp-mcp"),
            model_tiers=ModelTierConfig(
                simple="claude-haiku-4-5",
                medium="claude-sonnet-4-6",
                complex="claude-opus-4-7",
            ),
        )

        # Create metrics summary from evaluation
        metrics = MetricSummary(
            ticket_id="RSPEED-2482",
            query="Can I run RHEL 6 containers on RHEL 9?",
            url_f1=0.0,  # Poor retrieval!
            context_relevance=0.0,
            expected_urls=["docs/rhel9/container-compatibility.html"],
            retrieved_urls=["docs/rhel8/general-info.html"],
            solr_explain={...}
        )

        # Get suggestion (automatically routes to appropriate model)
        suggestion = await advisor.suggest_boost_query_changes(metrics)
        # Returns: SolrConfigSuggestion with specific Solr config change

        # Agent applies the change and re-tests

    Usage Example - Prompt Optimization:
        metrics = MetricSummary(
            ticket_id="RSPEED-2003",
            query="Is DHCP deprecated in RHEL 10?",
            url_f1=0.8,  # Good retrieval!
            context_relevance=0.9,
            answer_correctness=0.5,  # But bad answer
            keywords_score=0.3,  # Missing "deprecated" keyword
            expected_keywords=["deprecated", "RHEL 10"],
            response="DHCP is still available in RHEL 10",
            expected_response="ISC DHCP is deprecated in RHEL 10..."
        )

        suggestion = await advisor.suggest_prompt_changes(metrics)
        # Returns: PromptSuggestion to add keyword instructions

    Integration with OkpMcpAgent:
        The advisor is designed to work with OkpMcpAgent's fix loop:
        1. Agent runs evaluation → gets metrics
        2. Agent calls advisor.suggest_* → gets suggestion
        3. Agent applies code change using Edit tool
        4. Agent restarts okp-mcp and re-tests
        5. Repeat until fixed or max iterations reached

    Attributes:
        okp_mcp_root: Path to okp-mcp repository for code editing
        model_tiers: ModelTierConfig with simple/medium/complex model names
        use_tiered_routing: Whether to enable smart model routing

    Methods:
        suggest_boost_query_changes: Analyze retrieval issues, suggest Solr config fixes
        suggest_prompt_changes: Analyze answer issues, suggest system prompt fixes
        classify_problem_complexity: Classify as SIMPLE/MEDIUM/COMPLEX for routing (inherited)
    """

    def __init__(
        self,
        okp_mcp_root: Optional[Path] = None,
        model_tiers: Optional[ModelTierConfig] = None,
        use_tiered_routing: bool = True,
        default_model: Optional[str] = None,
    ):
        """Initialize LLM advisor with Claude Agent SDK.

        Args:
            okp_mcp_root: Path to okp-mcp repository for code context (default: ~/Work/okp-mcp)
            model_tiers: Model configuration for each tier (simple/medium/complex)
            use_tiered_routing: Enable smart model routing (default: True)
            default_model: Override model for all tiers (disables routing)
        """
        # Initialize base class with model tier management
        super().__init__(
            model_tiers=model_tiers,
            use_tiered_routing=use_tiered_routing,
            default_model=default_model,
        )

        # Store okp-mcp root for Solr config analysis
        from heal.core.config import HEALConfig

        self.okp_mcp_root = okp_mcp_root or HEALConfig.get_okp_mcp_root()

        if not self.okp_mcp_root:
            raise ValueError(
                "OKP-MCP repository not found. Set OKP_MCP_ROOT environment variable "
                "or place okp-mcp repository adjacent to HEAL (../okp-mcp)."
            )

        print("✅ Initialized LLM Advisor with Claude Agent SDK")

    async def _call_with_structured_output(
        self, model: str, system_prompt: str, user_prompt: str, output_schema: dict
    ) -> dict:
        """Call Claude with structured output via JSON parsing.

        Args:
            model: Model to use
            system_prompt: System prompt
            user_prompt: User prompt
            output_schema: JSON schema for output

        Returns:
            Parsed JSON response matching schema
        """
        import json
        import re

        # Combine system and user prompts for Agent SDK
        full_prompt = f"""{system_prompt}

USER REQUEST:
{user_prompt}

CRITICAL - YOU MUST COMPLETE THESE TWO STEPS:

STEP 1: Make the code change (use Edit tool)
STEP 2: Provide a JSON summary (MANDATORY - your response MUST end with this JSON)

The JSON summary must match this exact schema:
{json.dumps(output_schema, indent=2)}

Format the JSON exactly like this (this is REQUIRED, not optional):
```json
{{
  "reasoning": "...",
  "file_path": "...",
  "suggested_change": "...",
  "expected_improvement": "...",
  "confidence": "high|medium|low"
}}
```

IMPORTANT: Your response MUST end with the JSON block above. Do not skip this step."""

        response_text = ""

        from pathlib import Path
        import os

        # CRITICAL: Temporarily unset GOOGLE_APPLICATION_CREDENTIALS
        # The .env file sets this for Gemini, but it conflicts with Claude CLI's ADC
        saved_google_creds = os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

        print(f"🔍 DEBUG: _call_with_structured_output called with model={model}")
        print(f"🔍 DEBUG: okp_mcp_root={self.okp_mcp_root}")
        print(f"🔍 DEBUG: Prompt length: {len(full_prompt)} chars")

        try:
            # Log debug output to file
            from heal.core.config import HEALConfig

            log_file = HEALConfig.get_log_dir() / "claude_sdk_debug.log"
            print(f"🔍 DEBUG: Opening log file: {log_file}")
            print(f"🔍 DEBUG: Claude will edit files in: {self.okp_mcp_root}")
            print(f"🔍 DEBUG: Working directory exists: {Path(self.okp_mcp_root).exists()}")
            with open(log_file, "a") as log:
                log.write(f"\n{'='*80}\n")
                log.write(f"suggest_boost_query_changes() - {model}\n")
                log.write(f"okp_mcp_root (cwd): {self.okp_mcp_root}\n")
                log.write(f"okp_mcp_root exists: {Path(self.okp_mcp_root).exists()}\n")
                log.write(
                    f"GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')} (should be None)\n"
                )
                log.write(f"{'='*80}\n")

                # Use okp-mcp directory so Claude can edit files
                try:
                    async for message in query(
                        prompt=full_prompt,
                        options=ClaudeAgentOptions(
                            model=model,
                            allowed_tools=["Read", "Edit", "Glob", "Grep"],  # Enable file editing
                            permission_mode="acceptEdits",  # Auto-approve edits
                            max_turns=20,  # Increased from 10 - Claude needs turns for: Read, analyze, Edit, provide JSON
                            debug_stderr=log,  # Write to log file
                            cwd=str(self.okp_mcp_root),  # Work in okp-mcp repo
                        ),
                    ):
                        if isinstance(message, AssistantMessage):
                            for block in message.content:
                                if hasattr(block, "text"):
                                    response_text += block.text
                                    log.write(f"\n[Assistant text block]: {block.text[:200]}...\n")
                                    log.flush()  # Ensure it's written
                                elif hasattr(block, "type"):
                                    log.write(f"\n[Assistant block type]: {block.type}\n")
                                    log.flush()
                except Exception as e:
                    log.write(f"\n❌ EXCEPTION during query loop: {type(e).__name__}: {e}\n")
                    log.flush()
                    raise

                log.write(
                    f"\n📝 Response collection complete. Total length: {len(response_text)} chars\n"
                )
                log.write(f"\n{'='*80}\n")
                log.write("FULL RESPONSE TEXT:\n")
                log.write(f"{'='*80}\n")
                log.write(
                    response_text[-2000:] if len(response_text) > 2000 else response_text
                )  # Last 2000 chars
                log.write(f"\n{'='*80}\n")
                log.flush()

                # Check if any files were modified by Claude
                import subprocess

                git_status = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=str(self.okp_mcp_root),
                    capture_output=True,
                    text=True,
                ).stdout
                log.write(f"\n📝 Git status after Claude edits:\n{git_status}\n")
                log.flush()

                # Extract JSON from response (handle code blocks)
                json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    log.write("\n✅ Found JSON in code block\n")
                else:
                    # Try to find raw JSON object
                    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        log.write("\n✅ Found raw JSON object\n")
                    else:
                        log.write("\n❌ NO JSON FOUND IN RESPONSE\n")
                        log.write(f"Response text ({len(response_text)} chars):\n{response_text}\n")
                        raise ValueError(
                            f"No JSON found in response. Response length: {len(response_text)} chars. First 500 chars: {response_text[:500]}"
                        )

                # Parse and return
                try:
                    result = json.loads(json_str)
                    log.write(f"\n✅ Successfully parsed JSON: {list(result.keys())}\n")
                    return result
                except json.JSONDecodeError as e:
                    log.write(f"\n❌ JSON parse error: {e}\n")
                    log.write(f"Attempted to parse: {json_str[:500]}\n")
                    raise
        finally:
            # Restore original GOOGLE_APPLICATION_CREDENTIALS
            if saved_google_creds:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = saved_google_creds

    async def check_judge_agreement(
        self, metrics: MetricSummary, focus_metric: str = "answer_correctness"
    ) -> Optional[JudgeDisagreement]:
        """Check if Claude (advisor) agrees with Gemini's (judge) assessment.

        This provides quality control on the judge and helps identify:
        - Judge errors or hallucinations
        - Ambiguous/unclear expected responses
        - Unreliable evaluations (especially for flaky tests)
        - Cases where auto-fix should be skipped

        Args:
            metrics: Evaluation metrics with judge reasoning
            focus_metric: Which metric to check (default: answer_correctness)

        Returns:
            JudgeDisagreement if disagreement detected, None if in agreement or insufficient data
        """
        # Extract judge's assessment for the focus metric
        judge_score = None
        judge_reasoning = None

        if focus_metric == "answer_correctness":
            judge_score = metrics.answer_correctness
            judge_reasoning = metrics.answer_correctness_reason
        elif focus_metric == "faithfulness":
            judge_score = metrics.faithfulness
            judge_reasoning = metrics.faithfulness_reason
        elif focus_metric == "response_relevancy":
            judge_score = metrics.response_relevancy
            judge_reasoning = metrics.response_relevancy_reason
        elif focus_metric == "context_relevance":
            judge_score = metrics.context_relevance
            judge_reasoning = metrics.context_relevance_reason
        elif focus_metric == "context_precision":
            judge_score = metrics.context_precision
            judge_reasoning = metrics.context_precision_reason

        # Need all three to analyze: score, reasoning, and actual response
        if judge_score is None or not judge_reasoning or not metrics.response:
            return None

        # Only check for scores that are borderline or failing (< 0.9)
        # If score >= 0.9, generally trust the judge
        if judge_score >= 0.9:
            return None

        # Build prompt for Claude to review judge's assessment
        review_prompt = f"""You are reviewing an evaluation made by another LLM (Gemini) to check if you agree with its assessment.

**Evaluation Context:**

Metric: {focus_metric}
Judge Score: {judge_score:.2f}
Threshold: 0.90 (passing)

**Query:**
{metrics.query}

**Expected Response (Ground Truth):**
{metrics.expected_response if metrics.expected_response else "Not provided"}

**Actual LLM Response:**
{metrics.response[:1000]}

**Judge's Reasoning:**
{judge_reasoning}

---

**Your Task:**

Independently assess whether you agree with the judge's score and reasoning. Consider:

1. Does the actual response adequately address the query?
2. Is the judge's criticism valid and specific?
3. If expected response is provided, does actual response align with it?
4. Are there facts or details the judge missed or misinterpreted?

If you DISAGREE with the judge's assessment, respond with a JudgeDisagreement object.
If you AGREE (even if score is borderline), respond with: "AGREEMENT: <brief reason>"

Be critical and thorough. False positives (flagging agreement as disagreement) are okay,
but we want to catch cases where the judge is clearly wrong or the evaluation is ambiguous.
"""

        try:
            # Save and remove GOOGLE_APPLICATION_CREDENTIALS to avoid conflicts
            saved_google_creds = os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

            try:
                # Use Sonnet for this analysis (good balance of cost/quality)
                result = await self._call_with_structured_output(
                    prompt=review_prompt,
                    output_model=JudgeDisagreement,
                    model_preference="sonnet",
                    allow_text_fallback=True,  # Allow "AGREEMENT" text response
                )

                # If we got a JudgeDisagreement object, return it
                if isinstance(result, JudgeDisagreement):
                    return result

                # If we got a text response saying "AGREEMENT", return None
                return None

            finally:
                # Restore original GOOGLE_APPLICATION_CREDENTIALS
                if saved_google_creds:
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = saved_google_creds

        except Exception as e:
            # If analysis fails, don't block the workflow - just log and continue
            print(f"⚠️  Judge agreement check failed: {e}")
            return None

    async def suggest_solr_config_changes(
        self, metrics: MetricSummary, auto_escalate: bool = True
    ) -> SolrConfigSuggestion:
        """Suggest Solr configuration improvements based on metrics and explain output.

        Args:
            metrics: Evaluation metrics summary with Solr explain output
            auto_escalate: If True and problem is COMPLEX, use more expensive model

        Returns:
            Structured suggestion for Solr config changes
        """
        print(f"🔍 DEBUG: suggest_solr_config_changes called for ticket {metrics.ticket_id}")
        print(f"🔍 DEBUG: auto_escalate={auto_escalate}, medium_model={self.model_tiers.medium}")

        # Classify complexity if tiered models enabled
        model_to_use = self.model_tiers.medium
        if self.use_tiered_routing and auto_escalate:
            # Convert MetricSummary to base class format for classification
            # TicketMetrics already imported from base_agent at top

            ticket_metrics = TicketMetrics(
                ticket_id=metrics.ticket_id,
                query=metrics.query,
                url_f1=metrics.url_f1 or 0.0,
                mrr=metrics.mrr or 0.0,
                answer_correctness=metrics.answer_correctness,
                faithfulness=metrics.faithfulness,
            )

            complexity = await self.classify_complexity(
                tickets=[ticket_metrics],
                solr_explain=str(metrics.solr_explain) if metrics.solr_explain else None,
            )
            print(f"  Problem complexity: {complexity}")

            model_to_use = self.get_model_for_complexity(complexity)
            print(f"  Using model: {model_to_use}")

        system_prompt = """You are an expert in Solr/Lucene search optimization with deep knowledge of edismax query parser.

Your task is to analyze evaluation metrics from an okp-mcp RAG system and suggest specific
configuration changes to improve document retrieval.

=== CURRENT SOLR CONFIGURATION ===

okp-mcp uses Solr with edismax query parser. Configuration is in src/okp_mcp/solr.py (lines 95-151):

QUERY FIELD WEIGHTS (qf):
  "title^5 main_content heading_h1^3 heading_h2 portal_synopsis allTitle^3 content^2 all_content^1"

  - title^5: Matches in title get 5x weight (most important)
  - heading_h1^3: H1 headings get 3x weight
  - main_content: Body content (baseline 1x)
  - allTitle^3, content^2: Additional title/content fields

  TUNING: Increase title^ if expected docs have query in title but rank low.
          Increase main_content if body text matches are important.

PHRASE BOOSTING (pf, pf2, pf3):
  pf: "main_content^5 title^8"    - Exact phrase boost
  pf2: "main_content^3 title^5"   - Bigram phrase boost
  pf3: "main_content^1 title^2"   - Trigram phrase boost

  When query terms appear as exact phrase, these boost the score.

  TUNING: Increase pf: title^8 → title^12 if exact title matches rank too low.
          Decrease if forcing exact phrases hurts recall.

PHRASE SLOP (ps, ps2, ps3):
  ps: "3"   - Terms can be 3 positions apart for phrase boost
  ps2: "2"  - Bigrams can be 2 positions apart
  ps3: "5"  - Trigrams can be 5 positions apart

  TUNING: Increase if query terms scattered across doc (e.g., ps: 5).
          Decrease for stricter phrase matching.

MINIMUM MATCH (mm):
  "2<-1 5<75%"

  Meaning:
  - 1-2 terms: all must match (-1 means all)
  - 5+ terms: at least 75% must match

  TUNING: Increase "75%" → "90%" if getting too many irrelevant results (stricter).
          Decrease "75%" → "60%" if missing expected docs (more lenient).

HIGHLIGHTING PARAMS (snippet selection):
  hl.snippets: "6"      - How many snippets to extract per document (sent to LLM as context)
  hl.fragsize: "600"    - Target size in characters per snippet
  hl.score.k1: "1.0"    - BM25 term saturation for snippet scoring
  hl.score.b: "0.65"    - BM25 length normalization for snippet scoring
  hl.score.pivot: "200" - BM25 average snippet length for normalization

  TUNING: Increase snippets (6 → 8) if answer is in doc but LLM missing context.
          Increase fragsize (600 → 800) if snippets are too short to answer question.
          Increase k1 (1.0 → 1.2) to favor snippets with repeated query terms.
          Adjust b to control whether shorter/longer snippets are preferred.

=== BM25 RE-RANKING MULTIPLIERS ===

After Solr returns results, okp-mcp re-ranks with BM25 and applies multipliers (lines 301-357):

BOOST MULTIPLIER (default 2.0x):
  Applied to paragraphs containing these keywords:
  ["deprecated", "removed", "no longer", "not available", "end of life",
   "unsupported", "required", "must", "warning", "important", "recommended",
   "cockpit", "virsh", "cockpit-machines", "life cycle", "full support",
   "maintenance support", "extended life"]

  TUNING: Increase 2.0x → 3.0x if critical info (deprecations) ranks too low.
          Add new keywords if specific terms should be boosted.

DEMOTE MULTIPLIER (default 0.05x):
  Applied to paragraphs about RHV when query has no RHV intent:
  ["red hat virtualization", "rhv", "rhev", "red hat hyperconverged"]

  TUNING: Decrease 0.05x → 0.01x to more aggressively demote.
          Add new demote patterns for other unwanted content.

=== WHAT YOU CAN CHANGE ===

You can edit src/okp_mcp/solr.py to modify:

1. FIELD WEIGHTS (qf):
   Example: "title^5" → "title^7" to make title matches more important

2. PHRASE BOOSTS (pf, pf2, pf3):
   Example: "title^8" → "title^12" for stronger exact phrase matching in titles

3. PHRASE SLOP (ps, ps2, ps3):
   Example: "ps": "3" → "ps": "5" to allow more flexibility

4. MINIMUM MATCH (mm):
   Example: "2<-1 5<75%" → "2<-1 5<90%" for stricter matching

5. HIGHLIGHTING PARAMS (hl.snippets, hl.fragsize, hl.score.*):
   Example: "hl.snippets": "6" → "hl.snippets": "8" for more context per doc
   Example: "hl.fragsize": "600" → "hl.fragsize": "800" for longer snippets
   Example: "hl.score.k1": "1.0" → "hl.score.k1": "1.2" for stronger term matching

6. BOOST/DEMOTE MULTIPLIERS (lines 308-313):
   Example: multiplier *= 2.0 → multiplier *= 3.0

7. BOOST/DEMOTE KEYWORDS (lines 248-278):
   Example: Add "compatibility matrix" to _EXTRACTION_BOOST_KEYWORDS

=== COMMON PATTERNS ===

PROBLEM: URL F1 = 0.0 (completely wrong docs)
→ Expected docs have different documentKind or special keywords
→ FIX: Add keywords to _EXTRACTION_BOOST_KEYWORDS or adjust field weights

PROBLEM: Low MRR (< 0.5) (right docs exist but ranked too low)
→ Expected docs being outranked by less relevant docs
→ FIX: Increase phrase boost (pf: title^8 → title^12) or field weights

PROBLEM: Missing expected docs with query in title
→ Title matches not weighted enough
→ FIX: Increase "title^5" → "title^7" or "pf: title^8" → "title^12"

PROBLEM: Too many irrelevant results
→ Minimum match too lenient
→ FIX: Tighten mm from "5<75%" to "5<90%"

PROBLEM: Missing docs where query terms are scattered
→ Phrase slop too strict
→ FIX: Increase ps: "3" → "5"

PROBLEM: Right docs retrieved but answer incorrect/incomplete
→ LLM not getting enough context from snippets
→ FIX: Increase hl.snippets: "6" → "8" or hl.fragsize: "600" → "800"

PROBLEM: Answer missing key details that ARE in the document
→ Important sections not being highlighted/sent to LLM
→ FIX: Tune hl.score.k1 (1.0 → 1.2) to favor snippets with more query term matches

=== GUIDELINES ===

1. Make ONE specific change per iteration
2. Be SPECIFIC: exact line number, exact value, exact reasoning
3. Start with conservative changes (2x boost, not 10x)
4. Explain WHY based on metrics (not just guessing)
5. Predict WHAT should improve (specific metrics)
6. Provide confidence level: high (clear pattern), medium (likely), low (experimental)

Always suggest the SIMPLEST fix first. Don't over-engineer."""

        # Adjust instructions based on whether we have a config snapshot
        if metrics.solr_config_snapshot:
            # Config is already in the prompt - don't need to Read files
            step1_instructions = """STEP 1 - ANALYZE:
1. Review the Solr explain output above to understand WHY docs ranked the way they did
2. Review the ranking analysis to see which expected docs are missing/ranked low
3. Review the CURRENT SOLR CONFIGURATION shown above (parameters already provided)
4. Determine ONE specific config change in src/okp_mcp/solr.py that will improve retrieval
   - You can change field weights (qf), phrase boosts (pf/pf2/pf3), mm, or keyword lists
   - Config snapshot shows current values and file locations"""
        else:
            # Need to use Read tool to examine files
            step1_instructions = """STEP 1 - ANALYZE:
1. Review the Solr explain output above to understand WHY docs ranked the way they did
2. Review the ranking analysis to see which expected docs are missing/ranked low
3. Review the CURRENT SOLR CONFIGURATION shown above
4. Determine ONE specific config change in src/okp_mcp/solr.py that will improve retrieval
   - You can change field weights (qf), phrase boosts (pf/pf2/pf3), mm, or keyword lists"""

        user_prompt = f"""Analyze these evaluation metrics and Solr explain output to suggest a config change:

{metrics.to_prompt_context()}

Problem context:
- This is okp-mcp, a RAG system for Red Hat documentation
- Query: "{metrics.query}"
- Current state: {self._diagnosis_text(metrics)}

YOU MUST COMPLETE BOTH STEPS BELOW:

{step1_instructions}

STEP 2 - PROVIDE JSON (MANDATORY):
You MUST provide a JSON response with these fields:
- reasoning: Why this change is needed based on the metrics
- file_path: "src/okp_mcp/solr.py"
- suggested_change: Brief description of the change
- code_snippet: REQUIRED - The exact line of code AFTER the change. Include full syntax.
  Examples:
    \"mm\": \"2<-1 5<60%\",
    multiplier *= 3.0  # boost
    \"rhel 6\",  # Add to _EXTRACTION_BOOST_KEYWORDS
- expected_improvement: What metrics should improve
- confidence: high, medium, or low

Be concrete: which parameter, which line, which value, why based on explain output."""

        # Try with selected model, fall back to medium model if it fails
        try:
            result = await self._call_with_structured_output(
                model=model_to_use,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=SolrConfigSuggestion.model_json_schema(),
            )
        except Exception as e:
            error_msg = str(e)
            # Check if this is a Claude Agent SDK failure
            if (
                "Command failed with exit code 1" in error_msg
                and model_to_use != self.model_tiers.medium
            ):
                print(f"⚠️  {model_to_use} failed with: {error_msg}")
                print(f"  Falling back to {self.model_tiers.medium}...")
                # Retry with medium model
                result = await self._call_with_structured_output(
                    model=self.model_tiers.medium,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_schema=SolrConfigSuggestion.model_json_schema(),
                )
            else:
                # Some other error, re-raise
                raise

        return SolrConfigSuggestion(**result)

    # Backward compatibility alias
    async def suggest_boost_query_changes(
        self, metrics: MetricSummary, auto_escalate: bool = True
    ) -> SolrConfigSuggestion:
        """Backward compatibility alias for suggest_solr_config_changes."""
        return await self.suggest_solr_config_changes(metrics, auto_escalate)

    async def suggest_prompt_changes(
        self, metrics: MetricSummary, auto_escalate: bool = True
    ) -> PromptSuggestion:
        """Suggest system prompt improvements based on metrics.

        Args:
            metrics: Evaluation metrics summary
            auto_escalate: If True and problem is COMPLEX, use more expensive model

        Returns:
            Structured suggestion for prompt changes
        """
        # Classify complexity if tiered models enabled
        model_to_use = self.model_tiers.medium
        if self.use_tiered_routing and auto_escalate:
            # Convert MetricSummary to base class format for classification
            # TicketMetrics already imported from base_agent at top

            ticket_metrics = TicketMetrics(
                ticket_id=metrics.ticket_id,
                query=metrics.query,
                url_f1=metrics.url_f1 or 0.0,
                mrr=metrics.mrr or 0.0,
                answer_correctness=metrics.answer_correctness,
                faithfulness=metrics.faithfulness,
            )

            complexity = await self.classify_complexity(
                tickets=[ticket_metrics],
                solr_explain=str(metrics.solr_explain) if metrics.solr_explain else None,
            )
            print(f"  Problem complexity: {complexity}")

            model_to_use = self.get_model_for_complexity(complexity)
            print(f"  Using model: {model_to_use}")

        system_prompt = """You are an expert in LLM prompt engineering for RAG systems.

Your task is to analyze evaluation metrics and suggest system prompt improvements
to help the LLM better utilize retrieved documents.

Common issues:
- Keywords missing despite good retrieval → LLM ignoring context, needs stronger instruction
- Hallucination despite context → Need explicit "only use provided context" instruction
- Wrong tone/format → Need specific output formatting instructions

When suggesting changes:
1. Be SPECIFIC about what to add/modify in the prompt
2. Explain WHY based on the metrics
3. Predict WHAT should improve
4. Provide confidence level

Always suggest minimal changes first (add one instruction, not rewrite entire prompt)."""

        user_prompt = f"""Analyze these evaluation metrics and suggest system prompt improvements:

{metrics.to_prompt_context()}

Problem context:
- This is okp-mcp, a RAG system for Red Hat documentation
- Query: "{metrics.query}"
- Current state: {self._diagnosis_text(metrics)}

YOU MUST COMPLETE BOTH STEPS:

STEP 1: Suggest ONE specific system prompt change to improve answer quality
        Be concrete: what to add/modify, why

STEP 2: Provide a JSON summary (MANDATORY - your response MUST end with this)"""

        # Try with selected model, fall back to medium model if it fails
        try:
            result = await self._call_with_structured_output(
                model=model_to_use,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=PromptSuggestion.model_json_schema(),
            )
        except Exception as e:
            error_msg = str(e)
            # Check if this is a Claude Agent SDK failure
            if (
                "Command failed with exit code 1" in error_msg
                and model_to_use != self.model_tiers.medium
            ):
                print(f"⚠️  {model_to_use} failed with: {error_msg}")
                print(f"  Falling back to {self.model_tiers.medium}...")
                # Retry with medium model
                result = await self._call_with_structured_output(
                    model=self.model_tiers.medium,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_schema=PromptSuggestion.model_json_schema(),
                )
            else:
                # Some other error, re-raise
                raise

        return PromptSuggestion(**result)

    def _diagnosis_text(self, metrics: MetricSummary) -> str:
        """Generate diagnosis text for LLM context."""
        if not metrics.rag_used:
            return "RAG NOT USED - LLM answered from general knowledge"

        if metrics.rag_used and not metrics.docs_retrieved:
            return "RAG CALLED BUT NO DOCUMENTS RETRIEVED"

        if metrics.url_f1 is not None and metrics.url_f1 < 0.7:
            if metrics.url_f1 == 0.0:
                return "RETRIEVAL PROBLEM - Wrong documents retrieved (none of expected docs)"
            return f"RETRIEVAL PROBLEM - Some expected docs missing (F1={metrics.url_f1:.2f})"

        if (
            metrics.url_f1 is not None
            and metrics.url_f1 >= 0.7
            and metrics.keywords_score is not None
            and metrics.keywords_score < 0.7
        ):
            return "ANSWER PROBLEM - Right docs retrieved but keywords missing"

        return "Metrics look good overall"

    async def suggest_improvements(
        self,
        tickets: List[MetricSummary],
        iteration_context: Optional[str] = None,
        **kwargs,
    ) -> SolrConfigSuggestion:
        """Generate Solr optimization suggestions (implements BaseSolrOptimizer interface).

        Args:
            tickets: List of metric summaries to analyze (typically single ticket)
            iteration_context: Optional context from previous iterations
            **kwargs: Additional arguments

        Returns:
            Solr config suggestion for improving the ticket(s)
        """
        # LLM advisor typically works on single tickets
        if not tickets:
            raise ValueError("No tickets provided for analysis")

        # For now, analyze the first ticket
        # Future: Could aggregate metrics across multiple tickets
        return await self.suggest_solr_config_changes(tickets[0])


if __name__ == "__main__":
    import sys

    # Example usage
    print("=" * 80)
    print("OKP-MCP LLM Advisor - Test Mode")
    print("=" * 80)
    print("\nUsing Claude Agent SDK")
    print("Authentication: Uses Claude Code's existing auth")
    print()

    try:
        advisor = OkpMcpLLMAdvisor(
            use_tiered_routing=True,  # Uses default model tiers
        )
    except Exception as e:
        print(f"❌ Error initializing advisor: {e}")
        sys.exit(1)

    # Test with sample metrics from RSPEED-2482
    metrics = MetricSummary(
        ticket_id="RSPEED-2482",
        query="Can I run a RHEL 6 container on RHEL 9?",
        url_f1=0.0,
        mrr=0.2,
        context_relevance=0.0,
        context_precision=0.7,
        keywords_score=1.0,
        forbidden_claims_score=1.0,
        faithfulness=0.6,
        answer_correctness=0.80,
        response_relevancy=0.7,
        rag_used=True,
        docs_retrieved=True,
        num_docs=5,
    )

    print("=" * 80)
    print("BOOST QUERY SUGGESTION")
    print("=" * 80)
    print("\nCalling Claude via Agent SDK...")

    async def test():
        try:
            suggestion = await advisor.suggest_boost_query_changes(metrics)
            print("\n✅ Success!")
            print(f"\nReasoning: {suggestion.reasoning}")
            print(f"\nFile: {suggestion.file_path}")
            print(f"\nChange: {suggestion.suggested_change}")
            if suggestion.code_snippet:
                print(f"\nCode:\n{suggestion.code_snippet}")
            print(f"\nExpected Improvement: {suggestion.expected_improvement}")
            print(f"\nConfidence: {suggestion.confidence}")
        except Exception as e:
            print(f"\n❌ Error getting suggestion: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    asyncio.run(test())
