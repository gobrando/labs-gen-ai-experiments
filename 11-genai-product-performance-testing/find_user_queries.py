#!/usr/bin/env python3
"""
Find actual user queries by examining full traces
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

print("Fetching spans to find user queries...")

# Fetch first 500 spans
all_spans = []
for page in range(1, 6):
    url = f"{PHOENIX_URL}/v1/projects/{PHOENIX_PROJECT_NAME}/spans"
    if page > 1:
        url += f"?cursor={cursor}"

    response = httpx.get(url, headers=headers, verify=False, timeout=60.0)
    data = response.json()
    spans = data.get('data', [])
    all_spans.extend(spans)
    cursor = data.get('next_cursor')
    if not cursor:
        break

print(f"Fetched {len(all_spans)} spans")

# Group by trace
traces = defaultdict(list)
for span in all_spans:
    trace_id = span.get('context', {}).get('trace_id', '')
    if trace_id:
        traces[trace_id].append(span)

print(f"Found {len(traces)} traces\n")

# Find a trace with web search
sample_trace = None
for trace_id, span_list in traces.items():
    has_generator = any(s.get('name') == 'OpenAIWebSearchGenerator.run' for s in span_list)
    has_search = any(s.get('name') == 'web_search_call' for s in span_list)

    if has_generator and has_search:
        sample_trace = (trace_id, span_list)
        break

if not sample_trace:
    # Find any trace with generator
    for trace_id, span_list in traces.items():
        if any(s.get('name') == 'OpenAIWebSearchGenerator.run' for s in span_list):
            sample_trace = (trace_id, span_list)
            break

if sample_trace:
    trace_id, span_list = sample_trace
    print(f"{'='*80}")
    print(f"EXAMINING TRACE: {trace_id}")
    print(f"{'='*80}")
    print(f"\nSpans in this trace ({len(span_list)}):")

    for i, span in enumerate(span_list, 1):
        span_name = span.get('name', 'UNNAMED')
        print(f"\n{i}. {span_name}")

        attrs = span.get('attributes', {})

        # Look for user query in various places
        user_query_found = False

        # Check for query parameter
        if 'query' in attrs:
            print(f"   ✅ FOUND 'query': {str(attrs['query'])[:200]}")
            user_query_found = True

        # Check for user_request
        if 'user_request' in attrs:
            print(f"   ✅ FOUND 'user_request': {str(attrs['user_request'])[:200]}")
            user_query_found = True

        # Check for input.value (might be JSON)
        if 'input.value' in attrs:
            input_val = attrs['input.value']
            if isinstance(input_val, str) and len(input_val) < 500:
                try:
                    parsed = json.loads(input_val)
                    if 'query' in str(parsed):
                        print(f"   ✅ FOUND in 'input.value': {str(input_val)[:200]}")
                        user_query_found = True
                except:
                    pass

        # Check for messages with user role
        for key in attrs:
            if 'message' in key.lower() and 'user' in str(attrs.get(key, '')).lower():
                print(f"   🔍 Possible user message in '{key}': {str(attrs[key])[:100]}")

        # Show first few attribute keys
        if not user_query_found:
            attr_keys = list(attrs.keys())[:5]
            if attr_keys:
                print(f"   Attributes: {', '.join(attr_keys)}")

    # Try to extract actual user query
    print(f"\n{'='*80}")
    print("LOOKING FOR USER QUERY")
    print(f"{'='*80}")

    for span in span_list:
        attrs = span.get('attributes', {})

        # Check all attributes for 'query' or 'user_request'
        for key, value in attrs.items():
            if 'query' in key.lower() and not key.startswith('llm.'):
                print(f"\nFound in span '{span.get('name')}':")
                print(f"  Key: {key}")
                print(f"  Value: {str(value)[:300]}")
            elif 'request' in key.lower() and 'user' in key.lower():
                print(f"\nFound in span '{span.get('name')}':")
                print(f"  Key: {key}")
                print(f"  Value: {str(value)[:300]}")
else:
    print("❌ No traces found with OpenAIWebSearchGenerator.run")
