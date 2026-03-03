#!/usr/bin/env python3
"""Debug script to see what spans exist and their dates"""
import os
import httpx
from datetime import datetime
from collections import Counter

PHOENIX_URL = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "https://localhost:6006")
PHOENIX_PROJECT_NAME = os.environ.get("PHOENIX_PROJECT_NAME", "default")
PHOENIX_API_KEY = os.environ.get("PHOENIX_API_KEY", "")

headers = {}
if PHOENIX_API_KEY:
    headers["Authorization"] = f"Bearer {PHOENIX_API_KEY}"

print(f"Fetching spans from {PHOENIX_URL}/v1/projects/{PHOENIX_PROJECT_NAME}/spans")

url = f"{PHOENIX_URL}/v1/projects/{PHOENIX_PROJECT_NAME}/spans"
response = httpx.get(url, headers=headers, verify=False, timeout=60.0)
data = response.json()
spans = data.get('data', [])

print(f"\n✅ Got {len(spans)} spans in first page\n")

# Analyze span names
span_names = [s.get('name', 'UNNAMED') for s in spans]
name_counts = Counter(span_names)

print("=" * 80)
print("SPAN NAMES (top 20):")
print("=" * 80)
for name, count in name_counts.most_common(20):
    print(f"  {count:4d}x {name}")

# Find OpenAIWebSearchGenerator and web_search_call spans
generator_spans = [s for s in spans if s.get('name') == 'OpenAIWebSearchGenerator.run']
search_call_spans = [s for s in spans if s.get('name') == 'web_search_call']

print(f"\n" + "=" * 80)
print(f"WEB SEARCH SPANS:")
print(f"=" * 80)
print(f"  OpenAIWebSearchGenerator.run spans: {len(generator_spans)}")
print(f"  web_search_call spans: {len(search_call_spans)}")

# Show timestamps
if spans:
    print(f"\n" + "=" * 80)
    print("SPAN TIMESTAMPS (first 10):")
    print("=" * 80)
    for i, span in enumerate(spans[:10], 1):
        name = span.get('name', 'UNNAMED')
        start_time = span.get('start_time', 'NO_TIME')
        print(f"  {i}. [{start_time}] {name}")

# Show a few web_search_call spans if they exist
if search_call_spans:
    print(f"\n" + "=" * 80)
    print(f"WEB_SEARCH_CALL EXAMPLES:")
    print("=" * 80)
    for i, span in enumerate(search_call_spans[:5], 1):
        attrs = span.get('attributes', {})
        query = attrs.get('query') or attrs.get('tool.parameters.query', 'NO_QUERY')
        action_type = attrs.get('action_type') or attrs.get('tool.parameters.action_type', 'NO_ACTION')
        print(f"\n  {i}. Action: {action_type}")
        print(f"     Query: {str(query)[:100]}")
        print(f"     Time: {span.get('start_time')}")
