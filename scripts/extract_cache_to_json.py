#!/usr/bin/env python3
"""Extract okp-mcp cache to simple JSON format (no litellm needed to read it later)."""

import json
import pickle
import sqlite3
import sys
from pathlib import Path

# Add eval path for unpickling
EVAL_ROOT = Path("/home/emackey/Work/lightspeed-core/lightspeed-evaluation")
sys.path.insert(0, str(EVAL_ROOT / "src"))

cache_db = EVAL_ROOT / ".caches" / "mcp_direct_cache" / "cache.db"

if not cache_db.exists():
    print(f"Cache not found: {cache_db}")
    sys.exit(1)

conn = sqlite3.connect(cache_db)
cursor = conn.cursor()
cursor.execute("SELECT value FROM Cache")

results = []
for (value,) in cursor.fetchall():
    try:
        data = pickle.loads(value)

        # Extract query from tool_calls
        if hasattr(data, 'tool_calls') and data.tool_calls:
            first_tool = data.tool_calls[0][0] if isinstance(data.tool_calls[0], list) else data.tool_calls[0]
            query = first_tool.get('arguments', {}).get('query', '')

            # Extract contexts (retrieved docs)
            contexts = data.contexts if hasattr(data, 'contexts') else []

            results.append({
                'query': query,
                'contexts': contexts,
                'conversation_id': data.conversation_id if hasattr(data, 'conversation_id') else 'unknown',
            })

    except Exception as e:
        print(f"Error loading cache entry: {e}")

conn.close()

# Write to JSON
output_file = Path(__file__).parent / "okp_mcp_cached_results.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Extracted {len(results)} cached results to {output_file}")
