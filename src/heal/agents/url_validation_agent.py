"""URL Validation Agent - verifies retrieved URLs actually answer the question.

Validates that documentation URLs from SolrExpert are relevant to the query
BEFORE LinuxExpert synthesizes an answer from them. This reduces wasted tokens
on synthesis from wrong docs and improves answer quality.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

try:
    from claude_agent_sdk import query as claude_query, ClaudeAgentOptions

    CLAUDE_SDK_AVAILABLE = True
except ModuleNotFoundError:
    CLAUDE_SDK_AVAILABLE = False
    claude_query = None
    ClaudeAgentOptions = None

logger = logging.getLogger(__name__)


@dataclass
class URLValidationResult:
    """Result of URL validation check."""

    passes: bool
    score: float  # 0.0-1.0
    issues: list[str]
    suggested_search_queries: list[str] = None  # Better queries if URLs don't match

    def __post_init__(self):
        """Initialize empty list if None."""
        if self.suggested_search_queries is None:
            self.suggested_search_queries = []


@dataclass
class URLValidationAgent:
    """Validates retrieved documentation URLs answer the query.

    Uses Claude Agent SDK to check if Solr-retrieved docs are relevant
    before LinuxExpert synthesizes an answer from them.
    """

    model: str = "claude-sonnet-4-5@20250929"

    async def validate_urls(
        self,
        query: str,
        hypothesis: str,
        retrieved_docs: list[dict[str, Any]],
    ) -> URLValidationResult:
        """Validate that retrieved URLs can answer the query.

        Args:
            query: User's technical question
            hypothesis: LinuxExpert's hypothesis about the answer
            retrieved_docs: Documents from SolrExpert with url, title, content

        Returns:
            URLValidationResult with pass/fail, issues, suggested refinements
        """
        if not CLAUDE_SDK_AVAILABLE:
            raise RuntimeError(
                "claude-agent-sdk is not installed. "
                "Install it with: uv pip install claude-agent-sdk"
            )

        if not retrieved_docs:
            return URLValidationResult(
                passes=False,
                score=0.0,
                issues=["No documents retrieved"],
                suggested_search_queries=[query],
            )

        logger.info(
            f"\n[URL Validation Agent] Validating {len(retrieved_docs)} URLs for: {query[:60]}..."
        )

        validation_result = await self._validate_with_claude(query, hypothesis, retrieved_docs)

        logger.info(f"  Score: {validation_result['score']:.2f}")
        logger.info(f"  Passes: {validation_result['passes']}")
        if validation_result["issues"]:
            logger.info(f"  Issues: {len(validation_result['issues'])}")

        return URLValidationResult(
            passes=validation_result["passes"],
            score=validation_result["score"],
            issues=validation_result["issues"],
            suggested_search_queries=validation_result.get("suggested_search_queries", []),
        )

    async def _validate_with_claude(
        self, query: str, hypothesis: str, retrieved_docs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Validate URLs using Claude Agent SDK.

        Args:
            query: User's question
            hypothesis: Expected answer hypothesis
            retrieved_docs: Retrieved documents to validate

        Returns:
            Dict with passes, score, issues, suggested_search_queries
        """
        system_prompt = """You are a Documentation Relevance Validator for RHEL technical support.

Your task: Verify that retrieved documentation URLs can actually answer the user's technical question BEFORE we waste tokens synthesizing an answer from wrong docs.

VALIDATION CRITERIA:

**URL Relevance (Primary):**
- Do the docs address the SPECIFIC question asked?
- Are they about the RIGHT topic (e.g., "update GRUB" vs "reinstall GRUB")?
- Do they cover the RIGHT RHEL versions mentioned in the query?
- Do they contain actionable information (commands, procedures, config examples)?

**Content Quality (Secondary):**
- Is the content complete enough to form an answer?
- Are there step-by-step procedures or factual details?
- Is it official RHEL documentation (not community forums)?

**Common Failure Patterns:**
- ❌ Query asks about "update" but docs are about "install/reinstall"
- ❌ Query asks about RHEL 9 but docs are RHEL 7-specific
- ❌ Query asks "how to configure X" but docs only explain "what is X"
- ❌ Query asks about feature support but docs are about troubleshooting
- ❌ Generic overview docs when specific procedures are needed
- ❌ Docs about different but related topics (e.g., bootloader vs kernel)

**What to check:**
1. Read the doc titles and first ~500 words of content
2. Does the main topic match the query? (not just keyword overlap)
3. Do the docs provide information that answers the question?
4. If multiple docs, do at least 2-3 directly address the query?
5. Are there any docs that are clearly OFF-TOPIC?

**Scoring:**
- 1.0 = Perfect - multiple docs directly answer the question
- 0.8-0.9 = Good - most docs relevant, some may be tangential
- 0.6-0.7 = Mixed - some relevant docs but also irrelevant ones
- 0.4-0.5 = Poor - mostly irrelevant, might have 1 tangentially related doc
- 0.0-0.3 = Fail - no docs answer the question, wrong topic

**Passes if score >= 0.7**

If validation fails, suggest better search queries that might find the RIGHT docs.

Return JSON:
{
  "passes": true/false,
  "score": 0.0-1.0,
  "issues": ["specific issue 1", "specific issue 2"],
  "suggested_search_queries": ["better query 1", "better query 2"]
}

Example issues:
- "Doc 'How to reinstall GRUB' is about reinstall, but query asks about UPDATE"
- "Doc is RHEL 7-specific but query asks about RHEL 9"
- "All docs are conceptual overviews, but query needs step-by-step procedure"
"""

        # Build doc preview (titles + first 500 chars)
        doc_previews = []
        for i, doc in enumerate(retrieved_docs[:5], 1):
            title = doc.get("title", "Untitled")
            url = doc.get("url", "unknown")
            content = doc.get("content", "")[:500]
            doc_previews.append(f"**Doc {i}: {title}**\nURL: {url}\nPreview: {content}...")

        docs_text = "\n\n".join(doc_previews)

        user_prompt = f"""Validate these retrieved URLs:

**Query:**
{query}

**Expected Answer Hypothesis:**
{hypothesis[:300]}...

**Retrieved Documentation ({len(retrieved_docs)} docs):**
{docs_text}

Do these docs actually answer the query? Or are they about a different topic?

Return your validation as JSON only."""

        full_prompt = f"""{system_prompt}

---

{user_prompt}"""

        # Temporarily unset GOOGLE_APPLICATION_CREDENTIALS for Claude SDK
        saved_google_creds = os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

        try:
            # Use Claude Agent SDK
            options = ClaudeAgentOptions(
                model=self.model,
                max_turns=1,
            )

            response_text = ""
            async for message in claude_query(prompt=full_prompt, options=options):
                if hasattr(message, "content"):
                    for block in message.content:
                        if hasattr(block, "text"):
                            response_text += block.text

            # Parse JSON from response
            json_match = re.search(r"```json\s*(\{.+?\})\s*```", response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

            return json.loads(response_text)

        finally:
            # Restore GOOGLE_APPLICATION_CREDENTIALS for Gemini
            if saved_google_creds:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = saved_google_creds
