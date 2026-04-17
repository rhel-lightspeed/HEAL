"""Answer Review Agent - validates extracted answers against production quality guidelines.

Reviews expected_response fields to ensure they match production system prompt style.
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
class ReviewResult:
    """Result of answer quality review."""

    passes: bool
    score: float  # 0.0-1.0
    issues: list[str]
    suggested_fix: str = ""


@dataclass
class AnswerReviewAgent:
    """Reviews extracted answers against production quality guidelines.

    Uses Claude Agent SDK to check if answers match production system prompt style.
    """

    model: str = "claude-sonnet-4-5@20250929"

    async def review_answer(
        self,
        query: str,
        expected_response: str,
        sources: list[str],
    ) -> ReviewResult:
        """Review answer quality against production guidelines.

        Args:
            query: User's technical question
            expected_response: Extracted answer to review
            sources: Source URLs used in answer

        Returns:
            ReviewResult with pass/fail, issues, and suggested fixes
        """
        if not CLAUDE_SDK_AVAILABLE:
            raise RuntimeError(
                "claude-agent-sdk is not installed. "
                "Install it with: uv pip install claude-agent-sdk"
            )

        if not expected_response or not expected_response.strip():
            return ReviewResult(
                passes=False,
                score=0.0,
                issues=["Empty expected_response"],
                suggested_fix="",
            )

        logger.info(f"\n[Review Agent] Reviewing answer for: {query[:60]}...")

        review_result = await self._review_with_claude(query, expected_response, sources)

        logger.info(f"  Score: {review_result['score']:.2f}")
        logger.info(f"  Passes: {review_result['passes']}")
        if review_result["issues"]:
            logger.info(f"  Issues: {len(review_result['issues'])}")

        return ReviewResult(
            passes=review_result["passes"],
            score=review_result["score"],
            issues=review_result["issues"],
            suggested_fix=review_result.get("suggested_fix", ""),
        )

    async def _review_with_claude(
        self, query: str, expected_response: str, sources: list[str]
    ) -> dict[str, Any]:
        """Review answer using Claude Agent SDK.

        Args:
            query: User's question
            expected_response: Answer to review
            sources: Source URLs

        Returns:
            Dict with passes, score, issues, suggested_fix
        """
        system_prompt = """You are a Quality Reviewer for RHEL technical answers.

Your task: Review an extracted answer against production quality guidelines used in RHEL Lightspeed.

PRODUCTION QUALITY GUIDELINES:

**Response Length:**
- Be concise. Avoid verbose explanations, repetition, or unnecessary preambles.
- For "how to" questions: include actual commands, prerequisites, and procedure steps. NEVER say "refer to the documentation for detailed steps" when you have the steps.
- For factual questions: aim for clear, focused answers under 500 words.

**Content Rules:**
- **NEVER say**: "based on the documentation", "according to the docs", or similar phrases that distance you from the answer
- **Command format**: Omit `$` in commands (e.g., `getenforce` not `$ getenforce`)
- **Completeness**: When providing commands, include ALL required parameters (omitting them changes behavior)
- **Deprecation**: If feature is deprecated, lead with deprecation status and recommend replacement
- **Unsupported**: State "Unsupported" clearly when applicable - don't suggest workarounds
- **Version-specific**: Include RHEL version numbers when relevant (e.g., "RHEL 9 uses firewalld")
- **Markdown formatting**: Use code blocks for commands, bold for emphasis

**What to check:**
1. Is the answer concise (under 500 words for factual questions)?
2. Does it say "based on the documentation" or similar distancing phrases?
3. For "how to" questions: Does it include actual steps or say "refer to documentation"?
4. Do commands include all required parameters?
5. Do commands have `$` prompts that should be removed?
6. Is it properly formatted with markdown?
7. Is it actually answering the question asked?

Return JSON:
{
  "passes": true/false,
  "score": 0.0-1.0,
  "issues": ["specific issue 1", "specific issue 2"],
  "suggested_fix": "revised answer if needed (optional)"
}

Score:
- 1.0 = Perfect, meets all guidelines
- 0.8-0.9 = Minor issues (e.g., verbose but otherwise good)
- 0.6-0.7 = Some issues (e.g., missing parameters, says "based on docs")
- 0.4-0.5 = Major issues (e.g., says "refer to documentation", incomplete)
- 0.0-0.3 = Fails (e.g., empty, wrong answer, meta-commentary)

Passes if score >= 0.7
"""

        user_prompt = f"""Review this extracted answer:

**Question:**
{query}

**Answer to Review:**
{expected_response}

**Sources Used:**
{chr(10).join(f'- {url}' for url in sources) if sources else '(no sources)'}

Evaluate the answer against production quality guidelines.

Return your review as JSON only."""

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
