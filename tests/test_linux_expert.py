#!/usr/bin/env python3
"""Test Linux Expert Agent functionality.

Tests:
- TP-006: Linux Expert - Hypothesis Formation
- TP-007: Linux Expert - Answer Synthesis
"""

import asyncio
import sys

import pytest

from heal.agents.linux_expert import LinuxExpertAgent
from heal.agents.solr_expert import VerificationResult


@pytest.mark.asyncio
async def test_tp006_hypothesis_formation():
    """TP-006: Verify _form_hypothesis() can analyze ticket and return JSON.

    Verifies:
    - Can process JIRA ticket
    - Returns dict with required fields
    - Fields have expected types
    """
    agent = LinuxExpertAgent()

    # Test ticket
    ticket_key = "RSPEED-2482"
    summary = "Incorrect answer: Can I run a RHEL 6 container on RHEL 9?"
    description = (
        "User asked about RHEL 6 container support. CLA said it's supported. This is wrong."
    )

    print("\nCalling _form_hypothesis with:")
    print(f"  Key: {ticket_key}")
    print(f"  Summary: {summary}")
    print(f"  Description: {description}")

    # Call method
    result = await agent._form_hypothesis(ticket_key, summary, description)

    print(f"\nGot result type: {type(result)}")
    print(f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")

    # Validate structure
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "query" in result, f"Missing 'query' field in result: {result.keys()}"
    assert "hypothesis" in result, f"Missing 'hypothesis' field in result: {result.keys()}"
    assert "verification_queries" in result, "Missing 'verification_queries' field"

    # Validate types
    assert isinstance(result["query"], str), f"query should be str, got {type(result['query'])}"
    assert isinstance(result["hypothesis"], str), "hypothesis should be str"
    assert isinstance(result["verification_queries"], list), "verification_queries should be list"

    # Validate content
    assert len(result["query"]) > 10, f"Query too short: {result['query']}"
    assert len(result["hypothesis"]) > 20, "Hypothesis too short"
    assert len(result["verification_queries"]) > 0, "Should have verification queries"

    # Validate verification query structure
    for vq in result["verification_queries"]:
        assert "query" in vq, f"Verification query missing 'query' field: {vq}"
        assert "context" in vq, "Verification query missing 'context' field"
        assert "expected_doc_type" in vq, "Verification query missing 'expected_doc_type' field"

    print("\n✅ TP-006 PASS:")
    print(f"  Query: {result['query']}")
    print(f"  Hypothesis: {result['hypothesis'][:100]}...")
    print(f"  Verification queries: {len(result['verification_queries'])}")


@pytest.mark.asyncio
async def test_tp007_answer_synthesis():
    """TP-007: Verify _synthesize_verified_answer() creates final response.

    Verifies:
    - Can process verification results
    - Returns dict with required fields
    - Handles HIGH/MEDIUM/LOW confidence
    """
    agent = LinuxExpertAgent()

    # Mock hypothesis
    hypothesis = {
        "query": "Can I run a RHEL 6 container on RHEL 9?",
        "hypothesis": "RHEL 6 is EOL and not supported on RHEL 9 hosts.",
        "verification_queries": [
            {
                "query": "RHEL 6 EOL date",
                "context": "Verify when RHEL 6 reached end of life",
                "expected_doc_type": "documentation",
            }
        ],
    }

    # Mock verification result
    verification = VerificationResult(
        found_docs=[
            {
                "title": "RHEL 6 End of Life",
                "url": "https://access.redhat.com/articles/rhel6-eol",
                "content": "RHEL 6 reached end of life on November 30, 2020. No further updates or support.",
            }
        ],
        key_facts=["RHEL 6 reached EOL November 30, 2020"],
        confidence="HIGH",
        source_urls=["https://access.redhat.com/articles/rhel6-eol"],
        reasoning="Found official EOL documentation",
    )

    # Call method
    result = await agent._synthesize_verified_answer(
        "RSPEED-2482",
        "Can I run a RHEL 6 container on RHEL 9?",
        "User asked about support",
        hypothesis,
        verification,
    )

    # Validate structure
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "query" in result, "Missing 'query' field"
    assert "expected_response" in result, "Missing 'expected_response' field"
    assert "confidence" in result, "Missing 'confidence' field"
    assert "reasoning" in result, "Missing 'reasoning' field"
    assert "sources" in result, "Missing 'sources' field"
    assert "inferred" in result, "Missing 'inferred' field"

    # Validate types
    assert isinstance(result["query"], str), "query should be str"
    assert isinstance(result["expected_response"], str), "expected_response should be str"
    assert result["confidence"] in [
        "HIGH",
        "MEDIUM",
        "LOW",
    ], f"Invalid confidence: {result['confidence']}"
    assert isinstance(result["sources"], list), "sources should be list"
    assert isinstance(result["inferred"], bool), "inferred should be bool"

    # Validate content
    assert len(result["expected_response"]) > 20, "Expected response too short"
    assert not result["expected_response"].startswith(
        "TODO:"
    ), "Should not be TODO with HIGH confidence docs"

    print("\n✅ TP-007 PASS:")
    print(f"  Query: {result['query']}")
    print(f"  Expected Response: {result['expected_response'][:100]}...")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Sources: {len(result['sources'])}")
    print(f"  Inferred: {result['inferred']}")


@pytest.mark.asyncio
async def test_extract_with_verification_propagates_hypothesis_failure(mocker):
    """Verify extract_with_verification propagates _form_hypothesis failures.

    Current implementation: If hypothesis formation fails (Claude timeout, invalid JSON),
    the exception propagates to the caller. No error handling in this method.

    This documents current behavior. In production, caller must handle exceptions.
    """
    from heal.agents.solr_expert import SolrExpertAgent

    ticket = {
        "key": "TEST-001",
        "fields": {"summary": "Test ticket", "description": "Test description"},
    }

    solr_expert = mocker.MagicMock(spec=SolrExpertAgent)
    linux_expert = LinuxExpertAgent()

    # Mock _form_hypothesis to fail
    async def mock_hypothesis_failure(*args):
        raise Exception("Claude timeout")

    mocker.patch.object(linux_expert, "_form_hypothesis", side_effect=mock_hypothesis_failure)

    # Current implementation: exception propagates
    with pytest.raises(Exception, match="Claude timeout"):
        await linux_expert.extract_with_verification(ticket, solr_expert)


@pytest.mark.asyncio
async def test_extract_with_verification_propagates_synthesis_failure(mocker):
    """Verify extract_with_verification propagates _synthesize_verified_answer failures.

    Current implementation: If synthesis fails after successful hypothesis+verification,
    the exception propagates. No fallback handling.
    """
    from heal.agents.solr_expert import SolrExpertAgent, VerificationResult

    ticket = {
        "key": "TEST-002",
        "fields": {"summary": "Test ticket 2", "description": "Test description 2"},
    }

    # Mock hypothesis formation to succeed
    mock_hypothesis = {
        "query": "Test query",
        "hypothesis": "Test hypothesis",
        "verification_queries": [
            {"query": "test", "context": "test", "expected_doc_type": "documentation"}
        ],
    }

    async def mock_form_hypothesis(*args):
        return mock_hypothesis

    # Mock synthesis to fail
    async def mock_synthesis_failure(*args):
        raise Exception("Claude synthesis timeout")

    linux_expert = LinuxExpertAgent()
    mocker.patch.object(linux_expert, "_form_hypothesis", side_effect=mock_form_hypothesis)
    mocker.patch.object(
        linux_expert, "_synthesize_verified_answer", side_effect=mock_synthesis_failure
    )

    # Mock Solr expert verification
    mock_verification = VerificationResult(
        found_docs=[], key_facts=[], confidence="LOW", source_urls=[], reasoning="test"
    )

    solr_expert = mocker.MagicMock(spec=SolrExpertAgent)
    solr_expert.search_for_verification = mocker.AsyncMock(return_value=mock_verification)

    # Current implementation: exception propagates
    with pytest.raises(Exception, match="Claude synthesis timeout"):
        await linux_expert.extract_with_verification(ticket, solr_expert)


if __name__ == "__main__":
    print("Running Linux Expert Agent Tests")
    print("=" * 80)

    async def run_tests():
        print("\nTP-006: Hypothesis Formation")
        print("-" * 80)
        try:
            await test_tp006_hypothesis_formation()
        except Exception as e:
            print(f"\n❌ TP-006 FAILED: {e}")
            import traceback

            traceback.print_exc()
            return False

        print("\nTP-007: Answer Synthesis")
        print("-" * 80)
        try:
            await test_tp007_answer_synthesis()
        except Exception as e:
            print(f"\n❌ TP-007 FAILED: {e}")
            import traceback

            traceback.print_exc()
            return False

        return True

    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
