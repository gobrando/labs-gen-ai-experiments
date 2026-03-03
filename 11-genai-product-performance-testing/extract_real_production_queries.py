#!/usr/bin/env python3
"""
Extract ACTUAL production queries from generate_referrals_rag--centraltx spans
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

print("="*80)
print("EXTRACTING REAL PRODUCTION QUERIES")
print("="*80)
print(f"Phoenix URL: {PHOENIX_URL}")
print(f"Project: {PHOENIX_PROJECT_NAME}")
print("="*80)

# Fetch ALL spans
print("\nFetching all spans...")
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
    print(f"  Page {page}: {len(spans)} spans (total: {len(all_spans)})")

    cursor = data.get('next_cursor')
    if not cursor or len(spans) == 0:
        break

print(f"\n✅ Total spans fetched: {len(all_spans)}\n")

# Group by trace_id
traces = defaultdict(list)
for span in all_spans:
    trace_id = span.get('context', {}).get('trace_id', '')
    if trace_id:
        traces[trace_id].append(span)

print(f"📊 Total unique traces: {len(traces)}\n")

# Extract queries from generate_referrals_rag--centraltx spans
# and categorize by whether trace used web search
web_search_queries = []
no_web_search_queries = []

for trace_id, span_list in traces.items():
    # Find the main referral span
    referral_span = None
    has_web_search = False

    for span in span_list:
        if span.get('name') == 'generate_referrals_rag--centraltx':
            referral_span = span
        if span.get('name') == 'web_search_call':
            has_web_search = True

    if not referral_span:
        continue

    # Extract query from input.value
    attrs = referral_span.get('attributes', {})
    query = attrs.get('input.value', '')

    if query and len(query) > 10:  # Only keep substantial queries
        query_data = {
            'query': query.strip(),
            'trace_id': trace_id,
            'used_web_search': has_web_search
        }

        if has_web_search:
            web_search_queries.append(query_data)
        else:
            no_web_search_queries.append(query_data)

print("="*80)
print("EXTRACTION RESULTS")
print("="*80)
print(f"\nQueries that USED web search: {len(web_search_queries)}")
print(f"Queries that did NOT use web search: {len(no_web_search_queries)}")

total = len(web_search_queries) + len(no_web_search_queries)
if total > 0:
    baseline_rate = len(web_search_queries) / total
    print(f"Baseline web search rate: {baseline_rate*100:.1f}%")

# Show samples
if web_search_queries:
    print("\n" + "="*80)
    print("SAMPLE: Queries that USED web search")
    print("="*80)
    for i, item in enumerate(web_search_queries[:10], 1):
        print(f"\n{i}. {item['query'][:300]}")

if no_web_search_queries:
    print("\n" + "="*80)
    print("SAMPLE: Queries that did NOT use web search")
    print("="*80)
    for i, item in enumerate(no_web_search_queries[:10], 1):
        print(f"\n{i}. {item['query'][:300]}")

# Save to file
output = {
    'web_search_queries': web_search_queries[:50],  # Save top 50 of each
    'no_web_search_queries': no_web_search_queries[:50],
    'total_web_search': len(web_search_queries),
    'total_no_web_search': len(no_web_search_queries),
    'baseline_rate': baseline_rate if total > 0 else 0,
    'query_format': 'category-based',
    'note': 'Production uses category-based queries like "Include resources that support the following categories: X, Y, Z"'
}

with open('real_production_queries.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n" + "="*80)
print(f"✅ Saved to: real_production_queries.json")
if total > 0:
    print(f"   Baseline web search rate: {baseline_rate*100:.1f}%")
    print(f"   Total queries extracted: {total}")
print("="*80)
