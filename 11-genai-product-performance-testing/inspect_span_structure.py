#!/usr/bin/env python3
"""
Inspect actual span structure to find where queries are stored
"""
import os
import httpx
import json

PHOENIX_URL = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "https://localhost:6006")
PHOENIX_PROJECT_NAME = os.environ.get("PHOENIX_PROJECT_NAME", "default")
PHOENIX_API_KEY = os.environ.get("PHOENIX_API_KEY", "")

headers = {}
if PHOENIX_API_KEY:
    headers["Authorization"] = f"Bearer {PHOENIX_API_KEY}"

print("Fetching sample spans to inspect structure...")

# Fetch just first page
url = f"{PHOENIX_URL}/v1/projects/{PHOENIX_PROJECT_NAME}/spans"
response = httpx.get(url, headers=headers, verify=False, timeout=60.0)
data = response.json()
spans = data.get('data', [])

print(f"\nFetched {len(spans)} spans")

# Find a generator span
generator_spans = [s for s in spans if s.get('name') == 'OpenAIWebSearchGenerator.run']

if generator_spans:
    print(f"\n{'='*80}")
    print("FOUND OpenAIWebSearchGenerator.run SPAN")
    print('='*80)
    sample = generator_spans[0]

    print(f"\nFull span structure:")
    print(json.dumps(sample, indent=2)[:5000])  # First 5000 chars

    print(f"\n{'='*80}")
    print("ATTRIBUTES KEYS:")
    print('='*80)
    attrs = sample.get('attributes', {})
    for key in sorted(attrs.keys()):
        value = attrs[key]
        value_str = str(value)[:200]  # Truncate long values
        print(f"  {key}: {value_str}")
else:
    print("\n❌ No OpenAIWebSearchGenerator.run spans found in first 100 spans")
    print("\nShowing all span names found:")
    span_names = set(s.get('name', 'UNNAMED') for s in spans)
    for name in sorted(span_names):
        count = sum(1 for s in spans if s.get('name') == name)
        print(f"  {count:3d}x {name}")
