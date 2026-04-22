"""Test that all core modules can be imported."""


def test_import_heal():
    """Test basic heal package import."""
    import heal

    assert heal is not None


def test_import_okp_mcp_agent():
    """Test OkpMcpAgent import."""
    from heal.agents.okp_mcp_agent import OkpMcpAgent

    assert OkpMcpAgent is not None


def test_import_okp_mcp_llm_advisor():
    """Test OkpMcpLLMAdvisor import."""
    from heal.agents.okp_mcp_llm_advisor import OkpMcpLLMAdvisor

    assert OkpMcpLLMAdvisor is not None


def test_import_okp_mcp_pattern_agent():
    """Test OkpMcpPatternAgent import."""
    from heal.agents.okp_mcp_pattern_agent import OkpMcpPatternAgent

    assert OkpMcpPatternAgent is not None


def test_import_linux_expert():
    """Test LinuxExpertAgent import from new location."""
    from heal.agents.linux_expert import LinuxExpertAgent

    assert LinuxExpertAgent is not None


def test_import_linux_expert_backward_compat():
    """Test LinuxExpertAgent import from old location (backward compatibility)."""
    from heal.core import LinuxExpertAgent

    assert LinuxExpertAgent is not None


def test_import_solr_expert():
    """Test SolrExpertAgent import from new location."""
    from heal.agents.solr_expert import SolrExpertAgent

    assert SolrExpertAgent is not None


def test_import_solr_expert_backward_compat():
    """Test SolrExpertAgent import from old location (backward compatibility)."""
    from heal.core import SolrExpertAgent

    assert SolrExpertAgent is not None


def test_import_pattern_discovery():
    """Test pattern discovery module import."""
    from heal.core.pattern_discovery import PatternDiscoveryAgent

    assert PatternDiscoveryAgent is not None


def test_import_search_intelligence():
    """Test SearchIntelligenceManager import."""
    from heal.core.search_intelligence import SearchIntelligenceManager

    assert SearchIntelligenceManager is not None
