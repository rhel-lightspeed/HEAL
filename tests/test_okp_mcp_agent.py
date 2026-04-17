"""Tests for OkpMcpAgent check_answer_in_retrieved_docs method."""

import pytest
from claude_agent_sdk import AssistantMessage


@pytest.fixture
def mock_llm_advisor(mocker):
    """Mock LLM advisor for testing."""
    advisor = mocker.MagicMock()
    advisor.model = "claude-sonnet-4-6"
    return advisor


@pytest.fixture
def okp_mcp_agent(mocker, tmp_path, mock_llm_advisor):
    """Create OkpMcpAgent instance with mocked dependencies."""
    # Mock the imports to avoid dependency issues
    mocker.patch("heal.agents.okp_mcp_agent.LLM_ADVISOR_AVAILABLE", True)
    mocker.patch("heal.agents.okp_mcp_agent.SOLR_CHECKER_AVAILABLE", False)

    from heal.agents.okp_mcp_agent import OkpMcpAgent

    # Create agent with minimal setup
    agent = OkpMcpAgent(
        eval_root=tmp_path / "eval",
        okp_mcp_root=tmp_path / "okp-mcp",
        lscore_deploy_root=tmp_path / "lscore-deploy",
        enable_llm_advisor=False,  # We'll set it manually
    )

    # Manually set the mocked advisor
    agent.llm_advisor = mock_llm_advisor

    return agent


class TestCheckAnswerInRetrievedDocs:
    """Tests for check_answer_in_retrieved_docs method."""

    def test_returns_none_when_llm_advisor_not_available(self, okp_mcp_agent):
        """Should return None when LLM advisor is not available."""
        okp_mcp_agent.llm_advisor = None

        result = okp_mcp_agent.check_answer_in_retrieved_docs(
            expected_answer="RHEL 10 is supported",
            retrieved_contexts=["RHEL 10 docs"],
        )

        assert result["contains_answer"] is None
        assert result["confidence"] == 0.0
        assert "not available" in result["explanation"]

    def test_returns_false_when_no_contexts(self, okp_mcp_agent):
        """Should return False when no documents retrieved."""
        result = okp_mcp_agent.check_answer_in_retrieved_docs(
            expected_answer="RHEL 10 is supported",
            retrieved_contexts=[],
        )

        assert result["contains_answer"] is False
        assert result["confidence"] == 1.0
        assert "No documents" in result["explanation"]

    def test_parses_json_response_from_claude(self, okp_mcp_agent, mocker):
        """Should parse JSON response from Claude SDK."""
        # Create mock text block
        mock_text_block = mocker.MagicMock()
        mock_text_block.text = (
            '{"contains_answer": true, "confidence": 0.85, "explanation": "Docs have the info"}'
        )

        # Create mock message that looks like AssistantMessage
        mock_message = mocker.MagicMock(spec=AssistantMessage)
        mock_message.content = [mock_text_block]

        async def mock_query_generator(*args, **kwargs):
            yield mock_message

        mocker.patch("claude_agent_sdk.query", side_effect=mock_query_generator)

        result = okp_mcp_agent.check_answer_in_retrieved_docs(
            expected_answer="Test answer",
            retrieved_contexts=["Test context"],
        )

        assert result["contains_answer"] is True
        assert result["confidence"] == 0.85
        assert "Docs have the info" in result["explanation"]

    def test_extracts_json_from_markdown(self, okp_mcp_agent, mocker):
        """Should extract JSON from markdown code blocks."""
        mock_text_block = mocker.MagicMock()
        mock_text_block.text = '```json\n{"contains_answer": false, "confidence": 0.3, "explanation": "Missing info"}\n```'

        mock_message = mocker.MagicMock(spec=AssistantMessage)
        mock_message.content = [mock_text_block]

        async def mock_query_generator(*args, **kwargs):
            yield mock_message

        mocker.patch("claude_agent_sdk.query", side_effect=mock_query_generator)

        result = okp_mcp_agent.check_answer_in_retrieved_docs(
            expected_answer="Test",
            retrieved_contexts=["Context"],
        )

        assert result["contains_answer"] is False
        assert result["confidence"] == 0.3
        assert "Missing info" in result["explanation"]

    def test_handles_claude_sdk_errors(self, okp_mcp_agent, mocker):
        """Should return error info when Claude SDK fails."""

        async def mock_query_error(*args, **kwargs):
            raise Exception("API connection failed")
            yield  # Make it a generator

        mocker.patch("claude_agent_sdk.query", side_effect=mock_query_error)

        result = okp_mcp_agent.check_answer_in_retrieved_docs(
            expected_answer="Test",
            retrieved_contexts=["Context"],
        )

        assert result["contains_answer"] is None
        assert result["confidence"] == 0.0
        assert "Error" in result["explanation"]
        assert "API connection failed" in result["explanation"]
