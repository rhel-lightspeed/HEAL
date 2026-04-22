"""Tests for URL Validation Agent."""

import pytest


@pytest.mark.asyncio
async def test_url_validation_agent_import():
    """Test URLValidationAgent can be imported."""
    from heal.core import URLValidationAgent

    assert URLValidationAgent is not None


@pytest.mark.asyncio
async def test_url_validation_agent_no_docs():
    """Test validation fails when no docs provided."""
    from heal.core import URLValidationAgent

    validator = URLValidationAgent()

    result = await validator.validate_urls(
        query="How to update GRUB in RHEL 9?",
        hypothesis="Use yum update grub2*",
        retrieved_docs=[],
    )

    assert result.passes is False
    assert result.score == 0.0
    assert "No documents retrieved" in result.issues


@pytest.mark.asyncio
async def test_url_validation_agent_with_docs(mocker):
    """Test validation with mock Claude SDK."""
    from heal.core import URLValidationAgent

    # Mock the Claude SDK
    mock_query = mocker.patch("heal.agents.url_validation_agent.claude_query")

    # Create a mock message with validation result
    mock_message = mocker.MagicMock()
    mock_block = mocker.MagicMock()
    mock_block.text = """```json
{
  "passes": true,
  "score": 0.9,
  "issues": [],
  "suggested_search_queries": []
}
```"""
    mock_message.content = [mock_block]

    # Make the async generator return our mock message
    async def mock_async_generator(*args, **kwargs):
        yield mock_message

    mock_query.return_value = mock_async_generator()

    validator = URLValidationAgent()

    result = await validator.validate_urls(
        query="How to update GRUB in RHEL 9?",
        hypothesis="Use yum update grub2*",
        retrieved_docs=[
            {
                "title": "How to update GRUB",
                "url": "solutions/1521",
                "content": "To update GRUB in RHEL 9, run: yum update grub2*",
            }
        ],
    )

    assert result.passes is True
    assert result.score == 0.9
    assert len(result.issues) == 0


@pytest.mark.asyncio
async def test_url_validation_agent_wrong_docs(mocker):
    """Test validation fails when docs don't match query."""
    from heal.core import URLValidationAgent

    # Mock the Claude SDK
    mock_query = mocker.patch("heal.agents.url_validation_agent.claude_query")

    # Create a mock message with failed validation
    mock_message = mocker.MagicMock()
    mock_block = mocker.MagicMock()
    mock_block.text = """```json
{
  "passes": false,
  "score": 0.3,
  "issues": ["Doc is about reinstall, but query asks about update"],
  "suggested_search_queries": ["RHEL 9 update GRUB command", "grub2-mkconfig RHEL 9"]
}
```"""
    mock_message.content = [mock_block]

    # Make the async generator return our mock message
    async def mock_async_generator(*args, **kwargs):
        yield mock_message

    mock_query.return_value = mock_async_generator()

    validator = URLValidationAgent()

    result = await validator.validate_urls(
        query="How to update GRUB in RHEL 9?",
        hypothesis="Use yum update grub2*",
        retrieved_docs=[
            {
                "title": "How to reinstall GRUB",
                "url": "solutions/3486741",
                "content": "To reinstall GRUB, boot into rescue mode...",
            }
        ],
    )

    assert result.passes is False
    assert result.score == 0.3
    assert len(result.issues) == 1
    assert "reinstall" in result.issues[0].lower()
    assert len(result.suggested_search_queries) == 2
