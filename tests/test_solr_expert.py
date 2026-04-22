#!/usr/bin/env python3
"""Tests for Solr Expert Agent error handling and integration.

These tests focus on error scenarios and integration with SearchIntelligenceManager
that aren't covered by the integration tests in test_multi_agent_system.py.

Test Coverage:
- HTTP error handling (timeouts, 500s, malformed JSON)
- Search intelligence integration
- Initialization edge cases
"""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from heal.agents.solr_expert import SolrExpertAgent, VerificationQuery
from heal.core.search_intelligence import SearchIntelligenceManager


class TestSolrErrorHandling:
    """Tests for Solr HTTP error handling."""

    @pytest.mark.asyncio
    async def test_query_solr_handles_connection_timeout(self, mocker):
        """Verify _query_solr handles Solr connection timeouts gracefully.

        If Solr doesn't respond within timeout, should:
        1. Log error
        2. Return empty list
        3. Not crash
        """
        import httpx

        agent = SolrExpertAgent()

        # Mock httpx client to timeout
        mock_client = mocker.MagicMock()
        mock_client.get.side_effect = httpx.TimeoutException("Solr timeout")

        docs = await agent._query_solr(mock_client, "test query", num_results=5)

        assert docs == []

    @pytest.mark.asyncio
    async def test_query_solr_handles_http_500_error(self, mocker):
        """Verify _query_solr handles Solr 500 errors gracefully.

        If Solr returns 500 Internal Server Error, should return empty list.
        """
        import httpx

        agent = SolrExpertAgent()

        # Mock httpx response with 500 error
        mock_request = mocker.MagicMock()
        mock_response = mocker.MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 error", request=mock_request, response=mock_response
        )

        mock_client = mocker.AsyncMock()
        mock_client.get.return_value = mock_response

        docs = await agent._query_solr(mock_client, "test query", num_results=5)

        assert docs == []

    @pytest.mark.asyncio
    async def test_query_solr_handles_malformed_json(self, mocker):
        """Verify _query_solr propagates JSONDecodeError from malformed JSON.

        Current implementation: If Solr returns invalid JSON, the JSONDecodeError
        propagates (not caught by httpx.HTTPError handler).

        This test documents current behavior. In production, this would be caught
        by the calling code.
        """
        agent = SolrExpertAgent()

        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("bad json", "", 0)

        mock_client = mocker.AsyncMock()
        mock_client.get.return_value = mock_response

        # Current implementation: JSONDecodeError propagates
        with pytest.raises(json.JSONDecodeError):
            await agent._query_solr(mock_client, "test query", num_results=5)

    @pytest.mark.asyncio
    async def test_query_solr_handles_missing_response_docs(self, mocker):
        """Verify _query_solr handles Solr response missing 'docs' key.

        If Solr returns valid JSON but wrong structure, should handle gracefully.
        """
        agent = SolrExpertAgent()

        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        # Response missing response.docs structure
        mock_response.json.return_value = {"error": "query failed"}

        mock_client = mocker.AsyncMock()
        mock_client.get.return_value = mock_response

        docs = await agent._query_solr(mock_client, "test query", num_results=5)

        assert docs == []


class TestSearchIntelligenceIntegration:
    """Tests for Solr Expert integration with SearchIntelligenceManager."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        pytest.__version__ < "7.0", reason="Requires pytest 7.0+ for asyncio support"
    )
    async def test_search_for_verification_logs_to_search_intelligence(self, mocker):
        """Verify successful searches are logged to SearchIntelligenceManager.

        When verification finds documents with HIGH confidence, should:
        1. Log search to search intelligence
        2. Record query, topic, docs, confidence
        """
        with TemporaryDirectory() as tmpdir:
            # Create agent with real SearchIntelligenceManager and ticket_key
            search_mgr = SearchIntelligenceManager(db_path=Path(tmpdir) / "search_intelligence")
            agent = SolrExpertAgent(search_intelligence_mgr=search_mgr, ticket_key="RSPEED-100")

            # Mock Solr query
            async def mock_query_solr(client, query, num_results):
                return [{"title": "RHEL 6 EOL", "url": "http://example.com/eol", "score": 0.95}]

            mocker.patch.object(agent, "_query_solr", side_effect=mock_query_solr)

            queries = [
                VerificationQuery(
                    query="RHEL 6 EOL date", context="Verify EOL", expected_doc_type="documentation"
                )
            ]

            _result = await agent.search_for_verification(queries)

            # Verify search was logged
            stats = search_mgr.get_stats()
            assert stats["total_searches"] >= 1
            assert stats["successful_queries"] >= 1

    @pytest.mark.asyncio
    async def test_search_for_verification_no_crash_if_search_intelligence_none(self, mocker):
        """Verify agent works without SearchIntelligenceManager.

        If search_intelligence_mgr is None, should still work (just no logging).
        """
        # Create agent without search intelligence
        agent = SolrExpertAgent(search_intelligence_mgr=None)

        # Mock Solr query
        async def mock_query_solr(client, query, num_results):
            return [{"title": "Test", "url": "http://example.com", "score": 0.8}]

        mocker.patch.object(agent, "_query_solr", side_effect=mock_query_solr)

        queries = [
            VerificationQuery(query="test query", context="test", expected_doc_type="documentation")
        ]

        # Should not crash
        result = await agent.search_for_verification(queries)

        assert result.confidence in ["HIGH", "MEDIUM", "LOW"]


class TestSolrInitialization:
    """Tests for SolrExpertAgent initialization edge cases."""

    def test_solr_expert_uses_env_solr_url(self, monkeypatch):
        """Verify SolrExpertAgent uses SOLR_URL environment variable.

        If SOLR_URL is set, should override default.
        """
        monkeypatch.setenv("SOLR_URL", "http://custom-solr:9999/solr/custom")

        agent = SolrExpertAgent()

        assert agent.solr_url == "http://custom-solr:9999/solr/custom"

    def test_solr_expert_strips_trailing_slash_from_env(self, monkeypatch):
        """Verify agent strips trailing slash from SOLR_URL.

        Ensures consistent URL format regardless of how user sets env var.
        """
        monkeypatch.setenv("SOLR_URL", "http://custom-solr:9999/solr/custom/")

        agent = SolrExpertAgent()

        assert agent.solr_url == "http://custom-solr:9999/solr/custom"
        assert not agent.solr_url.endswith("/")

    def test_solr_expert_handles_search_intelligence_init_failure(self, mocker):
        """Verify agent continues if SearchIntelligenceManager fails to initialize.

        If search intelligence can't be initialized (permission denied, etc.),
        agent should log warning and continue with search_intelligence_mgr=None.
        """

        # Patch the SearchIntelligenceManager class that gets imported inside __post_init__
        # We need to patch it at the point of use, not at import
        def mock_search_mgr_init(*args, **kwargs):
            raise Exception("Permission denied")

        # Patch where it's used (inside solr_expert module after import)
        mocker.patch(
            "heal.core.search_intelligence.SearchIntelligenceManager",
            side_effect=mock_search_mgr_init,
        )

        # Should still create agent, just without search intelligence
        agent = SolrExpertAgent()

        assert agent.search_intelligence_mgr is None
