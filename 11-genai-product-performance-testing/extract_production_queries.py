#!/usr/bin/env python3
"""
Extract actual production queries from Phoenix traces.
Get examples of queries that DID and did NOT use web search.
"""
import os
import httpx
import json
from collections import defaultdict

PHOENIX_URL = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "https://localhost:6006")
PHOENIX_PROJECT_NAME = os.environ.get("PHOENIX_PROJECT_NAME", "default")
PHOENIX_API_KEY = os.environ.get("PHOENIX_API_KEY", "")

headers = {}
if PHOENIX_API_KEY:
    headers["Authorization"] = f"Bearer {PHOENIX_API_KEY}"

print("=" * 80)
print("EXTRACTING PRODUCTION QUERIES FROM PHOENIX")
print("=" * 80)
print(f"Phoenix URL: {PHOENIX_URL}")
print(f"Project: {PHOENIX_PROJECT_NAME}")
print("=" * 80)

# Fetch recent spans (last 2000 to get good sample)
print("\nFetching recent spans...")
all_spans = []
cursor = None

for page in range(1, 21):  # Get 20 pages = 2000 spans
    url = f"{PHOENIX_URL}/v1/projects/{PHOENIX_PROJECT_NAME}/spans"
    if cursor:
        url += f"?cursor={cursor}"

    response = httpx.get(url, headers=headers, verify=False, timeout=60.0)
    data = response.json()
    spans = data.get('data', [])
    all_spans.extend(spans)

    print(f"  Page {page}: {len(spans)} spans (total: {len(all_spans)})")

    cursor = data.get('next_cursor')
    if not cursor or len(spans) == 0:
        break

print(f"\nFetched {len(all_spans)} total spans")

# Group by trace_id
traces = defaultdict(list)
for span in all_spans:
    trace_id = span.get('context', {}).get('trace_id', '')
    if trace_id:
        traces[trace_id].append(span)

print(f"Found {len(traces)} unique traces\n")

# Classify traces and extract queries
web_search_queries = []
no_web_search_queries = []

for trace_id, span_list in traces.items():
    # Find generator span to get the query
    generator_span = None
    has_web_search = False

    for span in span_list:
        if span.get('name') == 'OpenAIWebSearchGenerator.run':
            generator_span = span
        if span.get('name') == 'web_search_call':
            has_web_search = True

    if not generator_span:
        continue

    # Try to extract the user query from span attributes
    attrs = generator_span.get('attributes', {})

    # Look for user query in various attribute keys
    query = None
    for key in ['input.value', 'llm.input_messages', 'messages', 'input']:
        if key in attrs:
            value = attrs[key]
            # Try to parse as JSON if it's a string
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    value = parsed
                except:
                    pass

            # Extract user message
            if isinstance(value, list):
                for msg in value:
                    if isinstance(msg, dict) and msg.get('role') == 'user':
                        query = msg.get('content', '')
                        break
            elif isinstance(value, str):
                query = value

            if query:
                break

    if query and len(query) > 20:  # Only keep substantial queries
        if has_web_search:
            web_search_queries.append({
                'query': query[:500],  # Truncate long queries
                'trace_id': trace_id,
                'used_web_search': True
            })
        else:
            no_web_search_queries.append({
                'query': query[:500],
                'trace_id': trace_id,
                'used_web_search': False
            })

print("=" * 80)
print("EXTRACTION RESULTS")
print("=" * 80)
print(f"\nQueries that USED web search: {len(web_search_queries)}")
print(f"Queries that did NOT use web search: {len(no_web_search_queries)}")

# Show samples
if web_search_queries:
    print("\n" + "=" * 80)
    print("SAMPLE: Queries that USED web search")
    print("=" * 80)
    for i, item in enumerate(web_search_queries[:5], 1):
        print(f"\n{i}. {item['query'][:200]}...")

if no_web_search_queries:
    print("\n" + "=" * 80)
    print("SAMPLE: Queries that did NOT use web search")
    print("=" * 80)
    for i, item in enumerate(no_web_search_queries[:5], 1):
        print(f"\n{i}. {item['query'][:200]}...")

# Save to file
output = {
    'web_search_queries': web_search_queries[:20],  # Save top 20 of each
    'no_web_search_queries': no_web_search_queries[:20],
    'total_web_search': len(web_search_queries),
    'total_no_web_search': len(no_web_search_queries),
    'baseline_rate': len(web_search_queries) / (len(web_search_queries) + len(no_web_search_queries)) if (len(web_search_queries) + len(no_web_search_queries)) > 0 else 0
}

with open('production_queries.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n" + "=" * 80)
print(f"✅ Saved to: production_queries.json")
print(f"   Baseline web search rate: {output['baseline_rate']*100:.1f}%")
print("=" * 80)
