#!/usr/bin/env python3
"""
Debug: Examine actual span structure from Phoenix
"""
import httpx
import json

PHOENIX_URL = "https://phoenix.referral-pilot-dev.navateam.com:6006"
PHOENIX_PROJECT_NAME = "pilot-prod"
PHOENIX_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJBcGlLZXk6MTMifQ.GA8Jh3OWNAAWd1hfHPUn4APm-TVMLClH_aeOYNQzm0Y"

headers = {"Authorization": f"Bearer {PHOENIX_API_KEY}"}

print("Fetching a few spans to examine structure...")
url = f"{PHOENIX_URL}/v1/projects/{PHOENIX_PROJECT_NAME}/spans"
response = httpx.get(url, headers=headers, verify=False, timeout=60.0)
data = response.json()
spans = data.get('data', [])

if spans:
    print(f"\n{'='*80}")
    print(f"Examining first span")
    print(f"{'='*80}")

    first_span = spans[0]
    print(f"\nSpan keys: {list(first_span.keys())}")
    print(f"\nSpan name: {first_span.get('name', 'N/A')}")
    print(f"Span kind: {first_span.get('span_kind', 'N/A')}")

    attrs = first_span.get('attributes', {})
    print(f"\nAttributes keys (first 50): {list(attrs.keys())[:50]}")

    # Look for any attribute containing "input" or "message"
    print(f"\n{'='*80}")
    print("Attributes containing 'input' or 'message' or 'prompt':")
    print(f"{'='*80}")
    for key in attrs.keys():
        if 'input' in key.lower() or 'message' in key.lower() or 'prompt' in key.lower():
            value = attrs[key]
            value_str = str(value)[:500] if value else "None"
            print(f"\n{key}:")
            print(f"  Type: {type(value)}")
            print(f"  Value preview: {value_str}")

    # Save full first span for inspection
    with open('debug_first_span.json', 'w') as f:
        json.dump(first_span, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print("Saved full first span to: debug_first_span.json")
    print(f"{'='*80}")
else:
    print("No spans found!")
