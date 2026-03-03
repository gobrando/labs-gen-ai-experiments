#!/usr/bin/env python3
"""
Find all span names that might be related to web search
"""
import os
import httpx
from collections import Counter

PHOENIX_URL = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "https://localhost:6006")
PHOENIX_PROJECT_NAME = os.environ.get("PHOENIX_PROJECT_NAME", "default")
PHOENIX_API_KEY = os.environ.get("PHOENIX_API_KEY", "")

headers = {}
if PHOENIX_API_KEY:
    headers["Authorization"] = f"Bearer {PHOENIX_API_KEY}"

print("Fetching ALL spans to find web search related names...")

all_spans = []
cursor = None
page = 0

while True:
    page += 1
    url = f"{PHOENIX_URL}/v1/projects/{PHOENIX_PROJECT_NAME}/spans"
    if cursor:
        url += f"?cursor={cursor}"

    response = httpx.get(url, headers=headers, verify=False, timeout=60.0)
    data = response.json()
    spans = data.get('data', [])
    all_spans.extend(spans)

    if page % 10 == 0:
        print(f"  Fetched {len(all_spans)} spans so far...")

    cursor = data.get('next_cursor')
    if not cursor or len(spans) == 0:
        break

print(f"\n✅ Total spans: {len(all_spans)}\n")

# Get all unique span names
span_names = [s.get('name', 'UNNAMED') for s in all_spans]
name_counts = Counter(span_names)

print("=" * 80)
print("ALL SPAN NAMES (looking for web search related):")
print("=" * 80)

for name, count in sorted(name_counts.items()):
    # Highlight names that might be web search related
    if any(keyword in name.lower() for keyword in ['web', 'search', 'tool', 'function', 'call']):
        print(f"  🔍 {count:4d}x {name}")
    else:
        print(f"     {count:4d}x {name}")

print("\n" + "=" * 80)
print("SPANS WITH 'SEARCH' OR 'WEB' IN NAME:")
print("=" * 80)

search_related = {name: count for name, count in name_counts.items()
                  if 'search' in name.lower() or 'web' in name.lower()}

if search_related:
    for name, count in search_related.items():
        print(f"  {count:4d}x {name}")

    # Show a sample of these spans
    print("\n" + "=" * 80)
    print("SAMPLE SPANS (first match for each type):")
    print("=" * 80)

    for name in search_related.keys():
        matching_spans = [s for s in all_spans if s.get('name') == name]
        if matching_spans:
            sample = matching_spans[0]
            print(f"\nSpan: {name}")
            print(f"  Attributes: {sample.get('attributes', {}).keys()}")
            attrs = sample.get('attributes', {})
            for key, value in list(attrs.items())[:10]:  # Show first 10 attributes
                print(f"    {key}: {str(value)[:100]}")
else:
    print("  ❌ No spans with 'search' or 'web' in the name!")

# Check for spans with web_search in attributes
print("\n" + "=" * 80)
print("CHECKING SPAN ATTRIBUTES FOR WEB_SEARCH:")
print("=" * 80)

spans_with_websearch_attrs = []
for span in all_spans[:1000]:  # Check first 1000
    attrs = span.get('attributes', {})
    attrs_str = str(attrs).lower()
    if 'web_search' in attrs_str or 'websearch' in attrs_str:
        spans_with_websearch_attrs.append(span)

if spans_with_websearch_attrs:
    print(f"  ✅ Found {len(spans_with_websearch_attrs)} spans with 'web_search' in attributes")
    sample = spans_with_websearch_attrs[0]
    print(f"\n  Sample span name: {sample.get('name')}")
    print(f"  Sample attributes:")
    for key, value in list(sample.get('attributes', {}).items())[:15]:
        print(f"    {key}: {str(value)[:100]}")
else:
    print("  ❌ No spans with 'web_search' in attributes (checked first 1000)")
