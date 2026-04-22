"""Integration tests for agent model tier usage.

Verifies that agents use the correct model tiers in real scenarios:
- LinuxExpertAgent uses Haiku for scope checks, Sonnet for synthesis
- AnswerReviewAgent uses Sonnet for reviews
- Token tracking works end-to-end
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from heal.agents.linux_expert import LinuxExpertAgent
from heal.agents.answer_review_agent import AnswerReviewAgent
from heal.core.token_tracker import TokenTracker


@pytest.fixture
def mock_claude_response():
    """Mock a Claude SDK ResultMessage."""

    def _make_response(text_content: str, model: str, tokens: int = 1000):
        mock = MagicMock()
        mock.usage = {
            "input_tokens": tokens,
            "output_tokens": tokens // 2,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        mock.total_cost_usd = 0.01
        mock.duration_ms = 1000

        content_block = MagicMock()
        content_block.text = text_content
        mock.content = [content_block]

        return mock

    return _make_response


@pytest.mark.asyncio
class TestLinuxExpertModelUsage:
    """Test LinuxExpertAgent uses correct models."""

    async def test_scope_check_uses_haiku(self, mock_claude_response):
        """Test scope check uses Haiku (simple tier)."""
        agent = LinuxExpertAgent()

        # Mock response saying ticket is in scope
        scope_response = mock_claude_response(
            '```json\n{"in_scope": true, "reasoning": "Valid RHEL question"}\n```',
            "claude-haiku-4-5",
            tokens=500,
        )

        with patch("heal.agents.base_agent.claude_query") as mock_query:

            async def mock_gen(*args, **kwargs):
                yield scope_response

            mock_query.return_value = mock_gen()

            result = await agent._check_rhel_scope(
                key="RSPEED-1",
                summary="How do I configure SELinux?",
                description="I need help with SELinux",
            )

        # Verify scope check called with Haiku
        mock_query.assert_called_once()
        call_kwargs = mock_query.call_args.kwargs
        assert call_kwargs["options"].model == "claude-haiku-4-5"

        assert result["in_scope"] is True

    async def test_form_hypothesis_uses_sonnet(self, mock_claude_response):
        """Test hypothesis formation uses Sonnet (medium tier)."""
        agent = LinuxExpertAgent()

        hypothesis_response = mock_claude_response(
            """```json
{
  "query": "How do I configure SELinux?",
  "hypothesis": "Use setenforce and edit /etc/selinux/config",
  "verification_queries": [
    {"query": "setenforce command", "context": "SELinux modes", "expected_doc_type": "documentation"}
  ]
}
```""",
            "claude-sonnet-4-6",
            tokens=2000,
        )

        with patch("heal.agents.base_agent.claude_query") as mock_query:

            async def mock_gen(*args, **kwargs):
                yield hypothesis_response

            mock_query.return_value = mock_gen()

            result = await agent._form_hypothesis(
                key="RSPEED-1",
                summary="How do I configure SELinux?",
                description="Need to set SELinux to enforcing mode",
            )

        # Verify hypothesis used Sonnet (default medium tier)
        mock_query.assert_called_once()
        call_kwargs = mock_query.call_args.kwargs
        assert call_kwargs["options"].model == "claude-sonnet-4-6"

        assert result["query"] == "How do I configure SELinux?"

    async def test_evaluate_answer_uses_sonnet(self, mock_claude_response):
        """Test answer evaluation uses Sonnet (medium tier)."""
        agent = LinuxExpertAgent()

        eval_response = mock_claude_response(
            """```json
{
  "correctness": 0.85,
  "completeness": 0.80,
  "faithfulness": 0.90,
  "overall_score": 0.85,
  "notes": "Good technical accuracy"
}
```""",
            "claude-sonnet-4-6",
            tokens=1500,
        )

        with patch("heal.agents.base_agent.claude_query") as mock_query:

            async def mock_gen(*args, **kwargs):
                yield eval_response

            mock_query.return_value = mock_gen()

            result = await agent.evaluate_answer(
                query="How do I configure SELinux?",
                answer="Use setenforce 1 to enable enforcing mode...",
                contexts=["SELinux documentation..."],
            )

        # Verify evaluation used Sonnet
        mock_query.assert_called_once()
        call_kwargs = mock_query.call_args.kwargs
        assert call_kwargs["options"].model == "claude-sonnet-4-6"

        assert result["overall_score"] == 0.85


@pytest.mark.asyncio
class TestAnswerReviewerModelUsage:
    """Test AnswerReviewAgent uses correct models."""

    async def test_review_uses_sonnet(self, mock_claude_response):
        """Test answer review uses Sonnet (medium tier)."""
        agent = AnswerReviewAgent()

        review_response = mock_claude_response(
            """```json
{
  "passes": true,
  "score": 0.90,
  "issues": [],
  "suggested_fix": ""
}
```""",
            "claude-sonnet-4-6",
            tokens=1200,
        )

        with patch("heal.agents.base_agent.claude_query") as mock_query:

            async def mock_gen(*args, **kwargs):
                yield review_response

            mock_query.return_value = mock_gen()

            result = await agent.review_answer(
                query="How do I configure SELinux?",
                expected_response="Use setenforce 1...",
                sources=["https://access.redhat.com/docs/..."],
            )

        # Verify review used Sonnet
        mock_query.assert_called_once()
        call_kwargs = mock_query.call_args.kwargs
        assert call_kwargs["options"].model == "claude-sonnet-4-6"

        assert result.passes is True
        assert result.score == 0.90


@pytest.mark.asyncio
class TestEndToEndTokenTracking:
    """Test token tracking works end-to-end across agents."""

    async def test_multi_agent_token_tracking(self, mock_claude_response, tmp_path):
        """Test tokens tracked across multiple agent calls."""
        # Initialize tracker
        tracker = TokenTracker(pattern_id="test_pattern", output_dir=tmp_path)

        linux_agent = LinuxExpertAgent()
        review_agent = AnswerReviewAgent()

        # Create mock responses for each call in sequence
        scope_response = mock_claude_response(
            '{"in_scope": true, "reasoning": "Valid"}', "claude-haiku-4-5", 500
        )
        eval_response = mock_claude_response(
            '{"correctness": 0.8, "completeness": 0.8, "faithfulness": 0.8, "overall_score": 0.8, "notes": "Good"}',
            "claude-sonnet-4-6",
            1500,
        )
        review_response = mock_claude_response(
            '{"passes": true, "score": 0.9, "issues": [], "suggested_fix": ""}',
            "claude-sonnet-4-6",
            1200,
        )

        # Mock with side_effect to return different responses for each call
        async def mock_scope_gen(*args, **kwargs):
            yield scope_response

        async def mock_eval_gen(*args, **kwargs):
            yield eval_response

        async def mock_review_gen(*args, **kwargs):
            yield review_response

        with patch("heal.agents.base_agent.claude_query") as mock_query:
            mock_query.side_effect = [
                mock_scope_gen(),  # First call: scope check
                mock_eval_gen(),  # Second call: evaluate
                mock_review_gen(),  # Third call: review
            ]

            # Call agents in sequence
            await linux_agent._check_rhel_scope("RSPEED-1", "Question", "Description")
            await linux_agent.evaluate_answer("Query", "Answer", ["Context"])
            await review_agent.review_answer("Query", "Answer", ["source"])

        # Verify all 3 calls tracked
        assert len(tracker.calls) == 3

        # Verify model breakdown
        summary = tracker.get_summary()
        assert "claude-haiku-4-5" in summary["by_model"]
        assert "claude-sonnet-4-6" in summary["by_model"]

        # Haiku: 1 call
        assert summary["by_model"]["claude-haiku-4-5"]["calls"] == 1

        # Sonnet: 2 calls (evaluate + review)
        assert summary["by_model"]["claude-sonnet-4-6"]["calls"] == 2


@pytest.mark.asyncio
class TestModelTierDefaults:
    """Test default model tier configuration."""

    def test_linux_expert_defaults(self):
        """Test LinuxExpertAgent has correct default tiers."""
        agent = LinuxExpertAgent()

        assert agent.model_tiers.simple == "claude-haiku-4-5"
        assert agent.model_tiers.medium == "claude-sonnet-4-6"
        assert agent.model_tiers.complex == "claude-opus-4-6"
        assert agent.use_tiered_routing is True

    def test_answer_reviewer_defaults(self):
        """Test AnswerReviewAgent has correct default tiers."""
        agent = AnswerReviewAgent()

        assert agent.model_tiers.simple == "claude-haiku-4-5"
        assert agent.model_tiers.medium == "claude-sonnet-4-6"
        assert agent.model_tiers.complex == "claude-opus-4-6"
        assert agent.use_tiered_routing is True

    def test_explicit_default_model_disables_routing(self):
        """Test setting default_model disables tiered routing."""
        agent = LinuxExpertAgent(default_model="claude-sonnet-4-6")

        # All tiers become the same
        assert agent.model_tiers.simple == "claude-sonnet-4-6"
        assert agent.model_tiers.medium == "claude-sonnet-4-6"
        assert agent.model_tiers.complex == "claude-sonnet-4-6"

        # Routing disabled
        assert agent.use_tiered_routing is False


@pytest.mark.asyncio
class TestTokenLogging:
    """Test token usage is logged correctly."""

    async def test_token_costs_logged(self, mock_claude_response, caplog):
        """Test token costs appear in logs."""
        import logging

        caplog.set_level(logging.INFO)

        agent = LinuxExpertAgent()

        eval_response = mock_claude_response(
            '{"correctness": 0.8, "completeness": 0.8, "faithfulness": 0.8, "overall_score": 0.8, "notes": "Good"}',
            "claude-sonnet-4-6",
            tokens=2000,
        )

        with patch("heal.agents.base_agent.claude_query") as mock_query:

            async def mock_gen(*args, **kwargs):
                yield eval_response

            mock_query.return_value = mock_gen()

            await agent.evaluate_answer("Query", "Answer", ["Context"])

        # Check logs contain token info
        assert "tokens" in caplog.text.lower() or "cost" in caplog.text.lower()
