"""Tests for BaseAgent token tracking and model escalation.

Verifies:
- Token tracking captures usage from Claude SDK
- Model escalation (Haiku → Sonnet → Opus) works correctly
- Token costs are calculated properly
- TokenTracker integration works
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from heal.agents.base_agent import BaseAgent, ModelTierConfig, AgentResponse, TicketMetrics
from heal.core.token_tracker import TokenTracker


class DummyAgent(BaseAgent):
    """Test implementation of BaseAgent for testing purposes."""

    async def test_query(self, prompt: str) -> AgentResponse:
        """Simple test query."""
        return await self.query_claude(
            system_prompt="You are a test assistant.",
            user_prompt=prompt,
            call_type="test_query",
        )


@pytest.fixture
def model_tiers():
    """Default model tiers."""
    return ModelTierConfig(
        simple="claude-haiku-4-5",
        medium="claude-sonnet-4-6",
        complex="claude-opus-4-6",
    )


@pytest.fixture
def mock_result_message():
    """Mock ResultMessage from Claude SDK."""
    mock = MagicMock()
    mock.usage = {
        "input_tokens": 1000,
        "output_tokens": 500,
        "cache_read_input_tokens": 200,
        "cache_creation_input_tokens": 100,
    }
    mock.total_cost_usd = 0.0456
    mock.duration_ms = 2340

    # Mock content blocks
    content_block = MagicMock()
    content_block.text = "Test response from Claude"
    mock.content = [content_block]

    return mock


@pytest.mark.asyncio
class TestModelTierConfiguration:
    """Test model tier configuration."""

    def test_default_tiers(self, model_tiers):
        """Test default model tiers are set correctly."""
        assert model_tiers.simple == "claude-haiku-4-5"
        assert model_tiers.medium == "claude-sonnet-4-6"
        assert model_tiers.complex == "claude-opus-4-6"

    def test_custom_tiers(self):
        """Test custom model tier configuration."""
        custom = ModelTierConfig(
            simple="claude-haiku-4-5-20251001",
            medium="claude-sonnet-4-5@20250929",
            complex="claude-opus-4-7",
        )
        assert custom.simple == "claude-haiku-4-5-20251001"
        assert custom.medium == "claude-sonnet-4-5@20250929"
        assert custom.complex == "claude-opus-4-7"

    def test_invalid_model_raises(self):
        """Test invalid model names raise ValueError."""
        with pytest.raises(ValueError, match="Invalid model"):
            ModelTierConfig(
                simple="gpt-4",  # Not a Claude model
                medium="claude-sonnet-4-6",
                complex="claude-opus-4-6",
            )

    def test_agent_initialization_with_tiers(self, model_tiers):
        """Test agent initializes with model tiers."""
        agent = DummyAgent(model_tiers=model_tiers, use_tiered_routing=True)

        assert agent.model_tiers.simple == "claude-haiku-4-5"
        assert agent.model_tiers.medium == "claude-sonnet-4-6"
        assert agent.model_tiers.complex == "claude-opus-4-6"
        assert agent.use_tiered_routing is True

    def test_default_model_disables_routing(self):
        """Test default_model disables tiered routing."""
        agent = DummyAgent(default_model="claude-sonnet-4-6")

        assert agent.model_tiers.simple == "claude-sonnet-4-6"
        assert agent.model_tiers.medium == "claude-sonnet-4-6"
        assert agent.model_tiers.complex == "claude-sonnet-4-6"
        assert agent.use_tiered_routing is False


@pytest.mark.asyncio
class TestTokenTracking:
    """Test token tracking from Claude SDK."""

    async def test_token_extraction_from_result_message(self, mock_result_message):
        """Test tokens are correctly extracted from ResultMessage."""
        agent = DummyAgent()

        with patch("heal.agents.base_agent.claude_query") as mock_query:
            # Mock async generator returning our mock message
            async def mock_generator(*args, **kwargs):
                yield mock_result_message

            mock_query.return_value = mock_generator()

            response = await agent.test_query("Test prompt")

        # Verify AgentResponse has correct values
        assert response.input_tokens == 1000
        assert response.output_tokens == 500
        assert response.total_tokens == 1500
        assert response.cost_usd == 0.0456
        assert response.duration_ms == 2340
        assert response.cache_read_tokens == 200
        assert response.cache_creation_tokens == 100
        assert response.content == "Test response from Claude"
        assert response.model == "claude-sonnet-4-6"  # Default medium tier

    async def test_token_tracker_integration(self, mock_result_message, tmp_path):
        """Test tokens are recorded to TokenTracker."""
        # Initialize TokenTracker
        tracker = TokenTracker(pattern_id="test_pattern", output_dir=tmp_path)

        agent = DummyAgent()

        with patch("heal.agents.base_agent.claude_query") as mock_query:

            async def mock_generator(*args, **kwargs):
                yield mock_result_message

            mock_query.return_value = mock_generator()

            response = await agent.test_query("Test prompt")

        # Verify TokenTracker recorded the call
        assert len(tracker.calls) == 1

        call = tracker.calls[0]
        assert call.call_type == "test_query"
        assert call.model == "claude-sonnet-4-6"
        assert call.input_tokens == 1000
        assert call.output_tokens == 500
        assert call.total_tokens == 1500
        # TokenTracker recalculates cost from TOKEN_COSTS, not ResultMessage
        # For Sonnet: 1000 * 0.000003 + 500 * 0.000015 = 0.0105
        assert call.cost_usd == pytest.approx(0.0105)

    async def test_multiple_calls_tracked_separately(self, mock_result_message, tmp_path):
        """Test multiple LLM calls are tracked separately."""
        tracker = TokenTracker(pattern_id="test_pattern", output_dir=tmp_path)

        agent = DummyAgent()

        with patch("heal.agents.base_agent.claude_query") as mock_query:

            async def mock_generator(*args, **kwargs):
                yield mock_result_message

            # Return new generator for each call
            mock_query.side_effect = lambda *args, **kwargs: mock_generator()

            # Make 3 calls
            await agent.test_query("Call 1")
            await agent.test_query("Call 2")
            await agent.test_query("Call 3")

        # Verify all 3 calls tracked
        assert len(tracker.calls) == 3

        # Verify cumulative totals
        summary = tracker.get_summary()
        assert summary["total_calls"] == 3
        assert summary["total_tokens"] == 1500 * 3
        # TokenTracker recalculates cost: 3 * (1000*0.000003 + 500*0.000015) = 3 * 0.0105 = 0.0315
        assert summary["total_cost_usd"] == pytest.approx(0.0105 * 3)


@pytest.mark.asyncio
class TestModelEscalation:
    """Test model escalation based on complexity."""

    async def test_haiku_for_simple_tasks(self, mock_result_message):
        """Test Haiku is used for simple classification tasks."""
        agent = DummyAgent(use_tiered_routing=True)

        with patch("heal.agents.base_agent.claude_query") as mock_query:

            async def mock_generator(*args, **kwargs):
                yield mock_result_message

            mock_query.return_value = mock_generator()

            # Explicitly request simple tier
            response = await agent.query_claude(
                system_prompt="Classify this.",
                user_prompt="Is this a RHEL question?",
                model=agent.model_tiers.simple,  # Request Haiku
                call_type="classify",
            )

        # Verify Haiku was used
        assert response.model == "claude-haiku-4-5"

        # Verify correct options passed to Claude SDK
        mock_query.assert_called_once()
        call_kwargs = mock_query.call_args.kwargs
        assert call_kwargs["options"].model == "claude-haiku-4-5"

    async def test_sonnet_for_medium_tasks(self, mock_result_message):
        """Test Sonnet is used for medium complexity tasks."""
        agent = DummyAgent(use_tiered_routing=True)

        with patch("heal.agents.base_agent.claude_query") as mock_query:

            async def mock_generator(*args, **kwargs):
                yield mock_result_message

            mock_query.return_value = mock_generator()

            # Default call (no model specified = medium tier)
            response = await agent.query_claude(
                system_prompt="Generate answer.",
                user_prompt="How do I configure SELinux?",
                call_type="generate",
            )

        # Verify Sonnet was used (default medium tier)
        assert response.model == "claude-sonnet-4-6"

    async def test_opus_for_complex_tasks(self, mock_result_message):
        """Test Opus is used for complex tasks."""
        agent = DummyAgent(use_tiered_routing=True)

        with patch("heal.agents.base_agent.claude_query") as mock_query:

            async def mock_generator(*args, **kwargs):
                yield mock_result_message

            mock_query.return_value = mock_generator()

            # Explicitly request complex tier
            response = await agent.query_claude(
                system_prompt="Deep technical analysis.",
                user_prompt="Complex architectural question...",
                model=agent.model_tiers.complex,  # Request Opus
                call_type="complex_analysis",
            )

        # Verify Opus was used
        assert response.model == "claude-opus-4-6"

        call_kwargs = mock_query.call_args.kwargs
        assert call_kwargs["options"].model == "claude-opus-4-6"

    async def test_classify_complexity(self, mock_result_message):
        """Test complexity classification returns correct tier."""
        agent = DummyAgent(use_tiered_routing=True)

        tickets = [
            TicketMetrics(
                ticket_id="RSPEED-1",
                query="Simple query",
                url_f1=0.85,  # High score, small gap
                mrr=0.90,
                answer_correctness=0.88,
            )
        ]

        with patch("heal.agents.base_agent.claude_query") as mock_query:
            # Mock classification response saying "SIMPLE"
            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            mock_response.content[0].text = "SIMPLE"
            mock_response.usage = {"input_tokens": 100, "output_tokens": 10}
            mock_response.total_cost_usd = 0.001
            mock_response.duration_ms = 500

            async def mock_generator(*args, **kwargs):
                yield mock_response

            mock_query.return_value = mock_generator()

            complexity = await agent.classify_complexity(tickets)

        assert complexity == "SIMPLE"

    async def test_get_model_for_complexity(self):
        """Test correct model is returned for each complexity level."""
        agent = DummyAgent(use_tiered_routing=True)

        assert agent.get_model_for_complexity("SIMPLE") == "claude-haiku-4-5"
        assert agent.get_model_for_complexity("MEDIUM") == "claude-sonnet-4-6"
        assert agent.get_model_for_complexity("COMPLEX") == "claude-opus-4-6"

        # Unknown complexity defaults to medium
        assert agent.get_model_for_complexity("UNKNOWN") == "claude-sonnet-4-6"

    async def test_routing_disabled_uses_medium(self):
        """Test disabled routing always uses medium tier."""
        agent = DummyAgent(use_tiered_routing=False)

        assert agent.get_model_for_complexity("SIMPLE") == "claude-sonnet-4-6"
        assert agent.get_model_for_complexity("MEDIUM") == "claude-sonnet-4-6"
        assert agent.get_model_for_complexity("COMPLEX") == "claude-sonnet-4-6"


@pytest.mark.asyncio
class TestCredentialHandling:
    """Test GOOGLE_APPLICATION_CREDENTIALS handling."""

    async def test_google_creds_removed_during_call(self, mock_result_message, monkeypatch):
        """Test GOOGLE_APPLICATION_CREDENTIALS is temporarily removed."""
        # Set environment variable
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/creds.json")

        agent = DummyAgent()

        creds_during_call = None

        def check_creds(*args, **kwargs):
            nonlocal creds_during_call
            import os

            creds_during_call = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

            # Mock async generator
            async def mock_gen():
                yield mock_result_message

            return mock_gen()

        with patch("heal.agents.base_agent.claude_query", side_effect=check_creds):
            await agent.test_query("Test")

        # Verify creds were removed during call
        assert creds_during_call is None

    async def test_google_creds_restored_after_call(self, mock_result_message, monkeypatch):
        """Test GOOGLE_APPLICATION_CREDENTIALS is restored after call."""
        original_creds = "/path/to/creds.json"
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", original_creds)

        agent = DummyAgent()

        with patch("heal.agents.base_agent.claude_query") as mock_query:

            async def mock_generator(*args, **kwargs):
                yield mock_result_message

            mock_query.return_value = mock_generator()

            await agent.test_query("Test")

        # Verify creds were restored
        import os

        assert os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") == original_creds

    async def test_google_creds_restored_on_exception(self, monkeypatch):
        """Test GOOGLE_APPLICATION_CREDENTIALS is restored even on error."""
        original_creds = "/path/to/creds.json"
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", original_creds)

        agent = DummyAgent()

        with patch("heal.agents.base_agent.claude_query", side_effect=Exception("Test error")):
            with pytest.raises(Exception, match="Test error"):
                await agent.test_query("Test")

        # Verify creds were restored despite error
        import os

        assert os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") == original_creds


@pytest.mark.asyncio
class TestTokenCostCalculation:
    """Test token cost calculation."""

    async def test_cost_matches_result_message(self, mock_result_message):
        """Test cost from ResultMessage is used correctly."""
        agent = DummyAgent()

        with patch("heal.agents.base_agent.claude_query") as mock_query:

            async def mock_generator(*args, **kwargs):
                yield mock_result_message

            mock_query.return_value = mock_generator()

            response = await agent.test_query("Test")

        # Verify cost matches ResultMessage
        assert response.cost_usd == 0.0456

    async def test_token_tracker_cost_aggregation(self, mock_result_message, tmp_path):
        """Test TokenTracker aggregates costs correctly."""
        tracker = TokenTracker(pattern_id="test", output_dir=tmp_path)
        agent = DummyAgent()

        with patch("heal.agents.base_agent.claude_query") as mock_query:

            async def mock_generator(*args, **kwargs):
                yield mock_result_message

            # Return new generator for each call
            mock_query.side_effect = lambda *args, **kwargs: mock_generator()

            # Make 5 calls
            for i in range(5):
                await agent.test_query(f"Call {i}")

        summary = tracker.get_summary()
        # TokenTracker recalculates from TOKEN_COSTS: 5 * (1000*0.000003 + 500*0.000015) = 0.0525
        assert summary["total_cost_usd"] == pytest.approx(0.0105 * 5)


@pytest.mark.asyncio
class TestAgentResponseDataclass:
    """Test AgentResponse dataclass."""

    def test_agent_response_creation(self):
        """Test AgentResponse is created with all fields."""
        response = AgentResponse(
            content="Test response",
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            cost_usd=0.045,
            model="claude-sonnet-4-6",
            duration_ms=2000,
            cache_read_tokens=100,
            cache_creation_tokens=50,
        )

        assert response.content == "Test response"
        assert response.input_tokens == 1000
        assert response.output_tokens == 500
        assert response.total_tokens == 1500
        assert response.cost_usd == 0.045
        assert response.model == "claude-sonnet-4-6"
        assert response.duration_ms == 2000
        assert response.cache_read_tokens == 100
        assert response.cache_creation_tokens == 50
