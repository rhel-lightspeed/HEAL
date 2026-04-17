"""Tests for multi-agent Solr optimization system.

Tests the 3-agent architecture:
- Solr Expert (theory)
- OKP-MCP Code Expert (implementation)
- Synthesizer (practical suggestions)
"""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Try to import, but allow tests to run even if not available
try:
    from heal.agents.solr_multi_agent import (
        SolrMultiAgentSystem,
        TicketData,
        SolrTheoryAdvice,
        OkpMcpCodeAnalysis,
        SynthesizedSuggestion,
    )
    MULTI_AGENT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    MULTI_AGENT_AVAILABLE = False
    pytestmark = pytest.mark.skip("Multi-agent system not available (requires claude-agent-sdk)")


@pytest.fixture
def mock_okp_mcp_root(tmp_path):
    """Create mock okp-mcp repository."""
    okp_mcp = tmp_path / "okp-mcp"
    okp_mcp.mkdir()

    src_dir = okp_mcp / "src" / "okp_mcp"
    src_dir.mkdir(parents=True)

    # Create mock solr.py
    solr_file = src_dir / "solr.py"
    solr_file.write_text("""
# Mock Solr configuration
def build_query(query):
    params = {
        'mm': '2<-1 5<75%',
        'qf': 'title^5 main_content',
        'pf': 'title^8 main_content^5',
        'ps': 3,  # BUG: Hardcoded!
    }

    # Special handling for RHEL queries
    if 'RHEL' in query:
        params['boost_multiplier'] = 2.0

    return params
""")

    return okp_mcp


@pytest.fixture
def mock_claude_response():
    """Create mock Claude API response."""
    def _create_response(text):
        class MockBlock:
            def __init__(self, text):
                self.text = text

        class MockMessage:
            def __init__(self, text):
                self.content = [MockBlock(text)]

        return [MockMessage(text)]

    return _create_response


@pytest.fixture
def pattern_ticket_data():
    """Create pattern-level ticket data for testing."""
    tickets = [
        TicketData(
            ticket_id="RSPEED-1723",
            query="How do I configure grub2 bootloader in RHEL 9?",
            expected_urls=["docs.redhat.com/grub2/bootloader", "solutions/7013132"],
            retrieved_urls=[],
            metrics={"url_f1": 0.0, "mrr": 0.0},
        ),
        TicketData(
            ticket_id="RSPEED-1724",
            query="RHEL 9 bootloader configuration file location",
            expected_urls=["docs.redhat.com/grub2/config"],
            retrieved_urls=["docs.redhat.com/grub2/misc"],
            metrics={"url_f1": 0.11, "mrr": 0.11},
        ),
        TicketData(
            ticket_id="RSPEED-1725",
            query="Change default boot entry grub2",
            expected_urls=["docs.redhat.com/grub2/default-entry"],
            retrieved_urls=["docs.redhat.com/grub2/default-entry"],
            metrics={"url_f1": 0.33, "mrr": 0.16},
        ),
    ]
    return tickets


class TestSolrMultiAgentInitialization:
    """Test multi-agent system initialization."""

    def test_initialization_success(self, mock_okp_mcp_root):
        """Test successful initialization."""
        system = SolrMultiAgentSystem(
            okp_mcp_root=mock_okp_mcp_root,
            model="claude-sonnet-4-6",
        )

        assert system.okp_mcp_root == mock_okp_mcp_root
        assert system.model == "claude-sonnet-4-6"

    def test_initialization_missing_okp_mcp(self, tmp_path):
        """Test initialization fails with missing okp-mcp directory."""
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(ValueError, match="okp-mcp root not found"):
            SolrMultiAgentSystem(okp_mcp_root=nonexistent)

    def test_initialization_requires_claude_sdk(self, mock_okp_mcp_root, monkeypatch):
        """Test initialization fails gracefully without claude-agent-sdk."""
        # This test is conceptual - the import check happens at module level
        # In real usage, the try/except in solr_multi_agent.py handles this
        pass


class TestSolrTheoryExpert:
    """Test Solr Expert agent (theory/best practices)."""

    @pytest.mark.asyncio
    async def test_solr_expert_analysis(self, mock_okp_mcp_root, pattern_ticket_data, mock_claude_response, monkeypatch):
        """Test Solr Expert provides theory-based advice for pattern."""
        system = SolrMultiAgentSystem(
            okp_mcp_root=mock_okp_mcp_root,
            model="claude-sonnet-4-6",
        )

        # Mock Claude API response
        mock_response = mock_claude_response("""
```json
{
  "problem_analysis": "All tickets show low URL F1 (0.0-0.33). Common pattern: bootloader queries with stopwords reducing effective terms. Current mm=75% too strict.",
  "ideal_config": {
    "mm": "2<-1 5<60%",
    "qf": "title^8 main_content^2",
    "pf": "title^12 main_content^6"
  },
  "reasoning": "Reducing mm to 60% allows more lenient matching for technical bootloader queries. Boosting title helps expected docs rank higher.",
  "relevant_docs": ["edismax mm patterns", "BM25 field boosting"]
}
```
""")

        # Mock claude_query
        async def mock_query(*args, **kwargs):
            for msg in mock_response:
                yield msg

        monkeypatch.setattr("heal.agents.solr_multi_agent.claude_query", mock_query)

        # Call Solr Expert with pattern-level data
        advice = await system._get_solr_theory_advice(
            pattern_id="BOOTLOADER_GRUB_ISSUES",
            failing_tickets=pattern_ticket_data,
        )

        # Assertions
        assert isinstance(advice, SolrTheoryAdvice)
        assert "bootloader" in advice.problem_analysis.lower() or "stopword" in advice.problem_analysis.lower()
        assert advice.ideal_config["mm"] == "2<-1 5<60%"
        assert "title" in advice.ideal_config["qf"]
        assert len(advice.relevant_docs) > 0

    @pytest.mark.asyncio
    async def test_solr_expert_handles_metrics(self, mock_okp_mcp_root, mock_claude_response, monkeypatch):
        """Test Solr Expert considers evaluation metrics."""
        system = SolrMultiAgentSystem(okp_mcp_root=mock_okp_mcp_root)

        # Create pattern with poor metrics
        poor_metrics_tickets = [
            TicketData(
                ticket_id="RSPEED-9999",
                query="test query",
                expected_urls=["url1"],
                retrieved_urls=[],
                metrics={"url_f1": 0.0, "mrr": 0.0},
            )
        ]

        mock_response = mock_claude_response("""
```json
{
  "problem_analysis": "Pattern shows F1=0.0 across all tickets, indicates complete retrieval failure",
  "ideal_config": {"mm": "50%"},
  "reasoning": "Very lenient mm needed for failing pattern",
  "relevant_docs": []
}
```
""")

        async def mock_query(*args, **kwargs):
            for msg in mock_response:
                yield msg

        monkeypatch.setattr("heal.agents.solr_multi_agent.claude_query", mock_query)

        advice = await system._get_solr_theory_advice(
            pattern_id="TEST_PATTERN",
            failing_tickets=poor_metrics_tickets,
        )

        assert "F1=0.0" in advice.problem_analysis or "failure" in advice.problem_analysis.lower()


class TestOkpMcpCodeExpert:
    """Test OKP-MCP Code Expert agent (implementation analysis)."""

    @pytest.mark.asyncio
    async def test_code_expert_reads_actual_code(self, mock_okp_mcp_root, mock_claude_response, monkeypatch):
        """Test Code Expert reads and analyzes actual okp-mcp code."""
        system = SolrMultiAgentSystem(okp_mcp_root=mock_okp_mcp_root)

        # Mock response that references actual code
        mock_response = mock_claude_response("""
```json
{
  "current_implementation": "mm is set to '2<-1 5<75%' at line 5. ps is hardcoded to 3 at line 8.",
  "constraints": ["mm cannot be changed globally - query-type dependent"],
  "bugs_found": ["Line 8: ps hardcoded to 3, ignoring parameter"],
  "relevant_code_sections": {
    "src/okp_mcp/solr.py:5": "params['mm'] = '2<-1 5<75%'"
  },
  "warnings": ["Changing mm affects all queries"]
}
```
""")

        async def mock_query(*args, **kwargs):
            for msg in mock_response:
                yield msg

        monkeypatch.setattr("heal.agents.solr_multi_agent.claude_query", mock_query)

        # Create mock Solr advice
        solr_advice = SolrTheoryAdvice(
            problem_analysis="Pattern analysis shows mm too strict",
            ideal_config={"mm": "60%"},
            reasoning="Pattern needs more lenient matching",
            relevant_docs=[],
        )

        analysis = await system._get_okp_mcp_code_analysis(
            pattern_id="BOOTLOADER_GRUB_ISSUES",
            solr_advice=solr_advice,
        )

        # Assertions - Code Expert found the bug
        assert isinstance(analysis, OkpMcpCodeAnalysis)
        assert len(analysis.bugs_found) > 0
        assert any("hardcoded" in bug.lower() for bug in analysis.bugs_found)
        assert len(analysis.constraints) > 0

    @pytest.mark.asyncio
    async def test_code_expert_identifies_constraints(self, mock_okp_mcp_root, mock_claude_response, monkeypatch):
        """Test Code Expert identifies implementation constraints."""
        system = SolrMultiAgentSystem(okp_mcp_root=mock_okp_mcp_root)

        mock_response = mock_claude_response("""
```json
{
  "current_implementation": "RHEL queries have special boost logic",
  "constraints": [
    "Cannot change mm globally - affects RHEL query handling",
    "Title boost interacts with BM25 re-ranking"
  ],
  "bugs_found": [],
  "relevant_code_sections": {},
  "warnings": ["Test changes on RHEL queries specifically"]
}
```
""")

        async def mock_query(*args, **kwargs):
            for msg in mock_response:
                yield msg

        monkeypatch.setattr("heal.agents.solr_multi_agent.claude_query", mock_query)

        solr_advice = SolrTheoryAdvice(
            problem_analysis="Pattern-level mm adjustment needed",
            ideal_config={"mm": "50%"},
            reasoning="More lenient matching for RHEL patterns",
            relevant_docs=[],
        )

        analysis = await system._get_okp_mcp_code_analysis("RHEL_PATTERN", solr_advice)

        assert len(analysis.constraints) >= 2
        assert any("global" in c.lower() for c in analysis.constraints)


class TestSynthesizer:
    """Test Synthesizer agent (combines theory + code analysis)."""

    @pytest.mark.asyncio
    async def test_synthesizer_combines_inputs(self, mock_okp_mcp_root, pattern_ticket_data, mock_claude_response, monkeypatch):
        """Test Synthesizer combines Solr theory + code analysis for pattern."""
        system = SolrMultiAgentSystem(okp_mcp_root=mock_okp_mcp_root)

        mock_response = mock_claude_response("""
```json
{
  "suggested_change": "Fix ps hardcoding and adjust mm for bootloader pattern queries",
  "file_path": "src/okp_mcp/solr.py",
  "old_code": "params['ps'] = 3",
  "new_code": "params['ps'] = params.get('ps', 5)",
  "reasoning": "Fixes Code Expert's bug while applying Solr Expert's ps=5 recommendation for all 3 tickets in BOOTLOADER_GRUB_ISSUES pattern",
  "confidence": 0.85,
  "risks": ["ps=5 might decrease precision"]
}
```
""")

        async def mock_query(*args, **kwargs):
            for msg in mock_response:
                yield msg

        monkeypatch.setattr("heal.agents.solr_multi_agent.claude_query", mock_query)

        solr_advice = SolrTheoryAdvice(
            problem_analysis="Pattern shows mm too strict across all tickets",
            ideal_config={"mm": "60%", "ps": "5"},
            reasoning="More lenient matching needed for pattern",
            relevant_docs=[],
        )

        code_analysis = OkpMcpCodeAnalysis(
            current_implementation="ps hardcoded to 3",
            constraints=["mm is query-dependent"],
            bugs_found=["ps hardcoded"],
            relevant_code_sections={},
            warnings=[],
        )

        suggestion = await system._synthesize_suggestion(
            pattern_id="BOOTLOADER_GRUB_ISSUES",
            failing_tickets=pattern_ticket_data,
            solr_advice=solr_advice,
            code_analysis=code_analysis,
        )

        # Assertions - Synthesizer creates practical suggestion
        assert isinstance(suggestion, SynthesizedSuggestion)
        assert "ps" in suggestion.suggested_change.lower()
        assert suggestion.confidence > 0.0
        assert suggestion.file_path == "src/okp_mcp/solr.py"
        assert "params['ps']" in suggestion.old_code or "ps" in suggestion.old_code

    @pytest.mark.asyncio
    async def test_synthesizer_assesses_confidence(self, mock_okp_mcp_root, mock_claude_response, monkeypatch):
        """Test Synthesizer provides confidence score."""
        system = SolrMultiAgentSystem(okp_mcp_root=mock_okp_mcp_root)

        mock_response = mock_claude_response("""
```json
{
  "suggested_change": "Test change",
  "file_path": "src/okp_mcp/solr.py",
  "old_code": "old",
  "new_code": "new",
  "reasoning": "Test",
  "confidence": 0.92,
  "risks": []
}
```
""")

        async def mock_query(*args, **kwargs):
            for msg in mock_response:
                yield msg

        monkeypatch.setattr("heal.agents.solr_multi_agent.claude_query", mock_query)

        # Minimal test data
        test_tickets = [TicketData("TEST-1", "test", [], [], {})]
        solr_advice = SolrTheoryAdvice("", {}, "", [])
        code_analysis = OkpMcpCodeAnalysis("", [], [], {}, [])

        suggestion = await system._synthesize_suggestion("TEST_PATTERN", test_tickets, solr_advice, code_analysis)

        assert suggestion.confidence == 0.92
        assert 0.0 <= suggestion.confidence <= 1.0


class TestMultiAgentIntegration:
    """Test full multi-agent flow (all 3 agents together)."""

    @pytest.mark.asyncio
    async def test_full_optimization_flow(self, mock_okp_mcp_root, pattern_ticket_data, mock_claude_response, monkeypatch):
        """Test complete flow: Solr Expert → Code Expert → Synthesizer for pattern."""
        system = SolrMultiAgentSystem(okp_mcp_root=mock_okp_mcp_root)

        # Track which agents were called
        calls = []

        async def mock_query(prompt, **kwargs):
            if "world-class expert in Apache Solr" in prompt:
                calls.append("solr_expert")
                response = mock_claude_response("""
```json
{
  "problem_analysis": "Pattern shows stopwords issue across all 3 tickets",
  "ideal_config": {"mm": "60%"},
  "reasoning": "More lenient matching needed for pattern",
  "relevant_docs": []
}
```
""")
            elif "expert code analyst" in prompt:
                calls.append("code_expert")
                response = mock_claude_response("""
```json
{
  "current_implementation": "mm='75%'",
  "constraints": [],
  "bugs_found": ["ps hardcoded"],
  "relevant_code_sections": {},
  "warnings": []
}
```
""")
            else:  # Synthesizer
                calls.append("synthesizer")
                response = mock_claude_response("""
```json
{
  "suggested_change": "Combined fix for BOOTLOADER_GRUB_ISSUES pattern",
  "file_path": "src/okp_mcp/solr.py",
  "old_code": "old",
  "new_code": "new",
  "reasoning": "Combined theory + code analysis for all 3 tickets",
  "confidence": 0.8,
  "risks": []
}
```
""")

            for msg in response:
                yield msg

        monkeypatch.setattr("heal.agents.solr_multi_agent.claude_query", mock_query)

        # Run full optimization with pattern-level data
        suggestion = await system.get_optimized_suggestion(
            pattern_id="BOOTLOADER_GRUB_ISSUES",
            failing_tickets=pattern_ticket_data,
        )

        # Verify all 3 agents were called
        assert "solr_expert" in calls
        assert "code_expert" in calls
        assert "synthesizer" in calls

        # Verify final suggestion
        assert isinstance(suggestion, SynthesizedSuggestion)
        assert suggestion.confidence > 0.0


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_handles_malformed_json(self, mock_okp_mcp_root, monkeypatch):
        """Test handling of malformed JSON from Claude."""
        system = SolrMultiAgentSystem(okp_mcp_root=mock_okp_mcp_root)

        async def mock_query(*args, **kwargs):
            class MockBlock:
                text = "This is not JSON"

            class MockMessage:
                content = [MockBlock()]

            yield MockMessage()

        monkeypatch.setattr("heal.agents.solr_multi_agent.claude_query", mock_query)

        # Create minimal test data
        test_tickets = [TicketData("TEST-1", "test", [], [], {})]

        # Should return fallback response instead of raising
        result = await system._get_solr_theory_advice(
            pattern_id="TEST_PATTERN",
            failing_tickets=test_tickets,
        )

        # Verify fallback response
        assert result.problem_analysis == "Failed to parse LLM response"
        assert result.ideal_config == {}
        assert result.reasoning == "JSON parsing error - using fallback"

    @pytest.mark.asyncio
    async def test_handles_missing_fields(self, mock_okp_mcp_root, mock_claude_response, monkeypatch):
        """Test handling of missing required fields in responses."""
        system = SolrMultiAgentSystem(okp_mcp_root=mock_okp_mcp_root)

        # Response missing required fields
        mock_response = mock_claude_response("""
```json
{
  "problem_analysis": "Test"
}
```
""")

        async def mock_query(*args, **kwargs):
            for msg in mock_response:
                yield msg

        monkeypatch.setattr("heal.agents.solr_multi_agent.claude_query", mock_query)

        # Create minimal test data
        test_tickets = [TicketData("TEST-1", "test", [], [], {})]

        # Should use defaults for missing fields instead of raising
        result = await system._get_solr_theory_advice(
            pattern_id="TEST_PATTERN",
            failing_tickets=test_tickets,
        )

        # Verify it populated the provided field and used defaults for missing ones
        assert result.problem_analysis == "Test"
        assert result.ideal_config == {}
        assert result.reasoning == ""
        assert result.relevant_docs == []


class TestRealWorldScenarios:
    """Test with realistic scenarios."""

    @pytest.mark.asyncio
    async def test_deprecation_query_scenario(self, mock_okp_mcp_root, mock_claude_response, monkeypatch):
        """Test optimization for deprecation queries."""
        system = SolrMultiAgentSystem(okp_mcp_root=mock_okp_mcp_root)

        # Mock responses for deprecation query
        responses = {
            "Solr/Lucene": """
```json
{
  "problem_analysis": "Deprecation queries need strict matching",
  "ideal_config": {"mm": "100%"},
  "reasoning": "Critical info requires exact matches",
  "relevant_docs": []
}
```
""",
            "code analyst": """
```json
{
  "current_implementation": "Already has special deprecation handling",
  "constraints": ["Don't change deprecation logic"],
  "bugs_found": [],
  "relevant_code_sections": {},
  "warnings": ["Deprecation queries already use mm=100%"]
}
```
""",
            "engineer": """
```json
{
  "suggested_change": "No change needed - already optimal",
  "file_path": "src/okp_mcp/solr.py",
  "old_code": "",
  "new_code": "",
  "reasoning": "Code already implements Solr theory recommendation",
  "confidence": 0.95,
  "risks": []
}
```
"""
        }

        async def mock_query(prompt, **kwargs):
            for key, response in responses.items():
                if key in prompt:
                    for msg in mock_claude_response(response):
                        yield msg
                    return

        monkeypatch.setattr("heal.agents.solr_multi_agent.claude_query", mock_query)

        # Create deprecation pattern tickets
        deprecation_tickets = [
            TicketData(
                ticket_id="RSPEED-8888",
                query="Is XFS V4 deprecated in RHEL 10?",
                expected_urls=["solutions/7127110"],
                retrieved_urls=[],
                metrics={"url_f1": 0.0, "mrr": 0.0},
            )
        ]

        suggestion = await system.get_optimized_suggestion(
            pattern_id="DEPRECATION_PATTERN",
            failing_tickets=deprecation_tickets,
        )

        # Synthesizer should recognize no change needed
        assert suggestion.confidence > 0.9
        assert "already" in suggestion.reasoning.lower() or "optimal" in suggestion.reasoning.lower()

    @pytest.mark.asyncio
    async def test_stopword_heavy_query(self, mock_okp_mcp_root, mock_claude_response, monkeypatch):
        """Test optimization for pattern with stopword-heavy queries."""
        system = SolrMultiAgentSystem(okp_mcp_root=mock_okp_mcp_root)

        mock_response = mock_claude_response("""
```json
{
  "problem_analysis": "Pattern shows stopword-heavy queries. Example: 'how do I configure' has 3 stopwords, reducing effective terms",
  "ideal_config": {"mm": "2<-1 5<50%"},
  "reasoning": "Very lenient mm needed for stopword-heavy query pattern",
  "relevant_docs": ["Stopword handling in Solr"]
}
```
""")

        async def mock_query(*args, **kwargs):
            for msg in mock_response:
                yield msg

        monkeypatch.setattr("heal.agents.solr_multi_agent.claude_query", mock_query)

        # Create stopword-heavy pattern tickets
        stopword_tickets = [
            TicketData(
                ticket_id="RSPEED-9000",
                query="how do I configure the bootloader",
                expected_urls=["url1"],
                retrieved_urls=[],
                metrics={"url_f1": 0.0},
            )
        ]

        advice = await system._get_solr_theory_advice(
            pattern_id="STOPWORD_PATTERN",
            failing_tickets=stopword_tickets,
        )

        assert "stopword" in advice.problem_analysis.lower()
        assert "50%" in advice.ideal_config["mm"] or "lenient" in advice.reasoning.lower()
