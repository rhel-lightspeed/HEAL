# Multi-Agent Solr Optimization Tests

## Overview

Comprehensive pytest test suite for the 3-agent Solr optimization system:
- **Solr Expert**: Theory and best practices
- **OKP-MCP Code Expert**: Implementation analysis  
- **Synthesizer**: Combines theory + reality

## Test Coverage

✅ **14 tests total - all passing**

### Test Classes

1. **TestSolrMultiAgentInitialization** (3 tests)
   - Successful initialization
   - Missing okp-mcp directory handling
   - Claude SDK requirement check

2. **TestSolrTheoryExpert** (2 tests)
   - Solr Expert provides theory-based advice
   - Handles evaluation metrics

3. **TestOkpMcpCodeExpert** (2 tests)
   - Reads actual okp-mcp code
   - Identifies implementation constraints

4. **TestSynthesizer** (2 tests)
   - Combines theory + code analysis
   - Provides confidence scores

5. **TestMultiAgentIntegration** (1 test)
   - Full 3-agent pipeline flow

6. **TestErrorHandling** (2 tests)
   - Malformed JSON responses
   - Missing required fields

7. **TestRealWorldScenarios** (2 tests)
   - Deprecation query optimization
   - Stopword-heavy query handling

## Running Tests

```bash
cd /home/emackey/Work/rhel-lightspeed/HEAL

# Run all multi-agent tests
.venv/bin/pytest tests/test_solr_multi_agent.py -v

# Run specific test class
.venv/bin/pytest tests/test_solr_multi_agent.py::TestSolrTheoryExpert -v

# Run with verbose output
.venv/bin/pytest tests/test_solr_multi_agent.py -xvs
```

## Prerequisites

- `pytest` and `pytest-asyncio` installed in venv (already configured)
- `claude-agent-sdk` installed
- Python 3.14+ (HEAL venv)

## Key Implementation Details

### Fixtures

- `mock_okp_mcp_root`: Creates temporary okp-mcp repository structure
- `mock_claude_response`: Creates mock Claude API responses

### Mocking Strategy

Uses `monkeypatch.setattr` to mock `claude_query` from claude-agent-sdk:

```python
async def mock_query(prompt, **kwargs):
    if "world-class expert in Apache Solr" in prompt:
        # Return Solr Expert response
    elif "expert code analyst" in prompt:
        # Return Code Expert response
    else:
        # Return Synthesizer response
```

### Test Philosophy

- **Unit tests**: Each agent tested individually
- **Integration tests**: Full 3-agent pipeline
- **Error handling**: Malformed responses, missing fields
- **Real-world scenarios**: Practical query types

## CI/CD Integration

Tests can be run as part of CI/CD pipeline:

```bash
.venv/bin/pytest tests/test_solr_multi_agent.py --junit-xml=test-results.xml
```

## Related Files

- Implementation: `src/heal/agents/solr_multi_agent.py`
- Documentation: `docs/MULTI_AGENT_SOLR_OPTIMIZATION.md`
- Integration: `src/heal/runners/run_pattern_fix_poc.py`
