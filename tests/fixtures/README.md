# Test Fixtures

This directory contains real evaluation results captured from actual runs, used for testing evaluation logic without hitting LLMs.

## Structure

```
fixtures/
├── README.md (this file)
├── bootloader_grub_pattern/
│   ├── run_001_results.json  # Raw evaluation results
│   ├── run_002_results.json
│   ├── run_003_results.json
│   └── metadata.json          # Pattern info
└── extract_fixtures.py        # Script to extract fixtures from runs
```

## How to create fixtures

### Extract from existing evaluation run:

```bash
python tests/fixtures/extract_fixtures.py \
    --input /path/to/okp_mcp_full_output/suite_*/run_001/evaluation_*_detailed.csv \
    --output tests/fixtures/my_pattern/run_001_results.json \
    --pattern-id MY_PATTERN
```

### Or manually copy from recent run:

```bash
# Find recent run
ls -lt okp_mcp_full_output/suite_*/run_*/evaluation_*_detailed.csv | head -1

# Copy to fixtures
cp /path/to/detailed.csv tests/fixtures/my_pattern/run_001.csv
```

## Using fixtures in tests

```python
from tests.fixtures import load_fixture

def test_pattern_evaluation_display(mocker):
    """Test per-ticket display using real data."""
    # Load fixture data (no LLM calls!)
    fixture = load_fixture("bootloader_grub_pattern/run_001_results.json")
    
    # Mock parse_results_per_ticket to return fixture data
    mocker.patch.object(
        OkpMcpAgent, 
        "parse_results_per_ticket",
        return_value=fixture["per_ticket_results"]
    )
    
    # Test the logic
    agent = OkpMcpAgent(...)
    result = agent.diagnose(ticket_id=None, runs=3)
    
    # Assertions on display/routing logic
    assert isinstance(result, PatternEvaluationResult)
    assert len(result.per_ticket_results) == 3
    # etc...
```

## Benefits

✅ **Fast**: No LLM calls, tests run in milliseconds
✅ **Deterministic**: Same input → same output every time
✅ **Edge cases**: Can test rare scenarios (high variance, RAG bypass, etc.)
✅ **Debugging**: Iterate on logic without waiting for evaluations
✅ **CI/CD**: Tests don't require API keys or running containers

## When to update fixtures

- After fixing evaluation logic bugs (capture the corrected data)
- When adding new metrics (capture new metric structure)
- When testing edge cases (manually craft or find real examples)
- Quarterly (keep fixtures representative of current data)
