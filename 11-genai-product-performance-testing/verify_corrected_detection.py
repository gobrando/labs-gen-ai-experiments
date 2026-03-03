#!/usr/bin/env python3
"""
Quick verification that corrected web search detection works
Uses 5 real production queries to verify before re-running full tests
"""
import os
import json
from openai import OpenAI
from datetime import datetime

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Load sample production queries
with open('sample_production_queries.json', 'r') as f:
    data = json.load(f)

# Take 3 queries that had web search and 2 that didn't in production
test_queries = (
    data['web_search_queries'][:3] +  # 3 with web search
    data['no_web_search_queries'][:2]  # 2 without
)

SYSTEM_PROMPT = """You are an AI assistant helping social services case workers in Central Texas find resources for their clients.

Based on the client's request, search for and identify relevant social service resources.

Current date: February 20, 2026

Important guidelines:
- ALWAYS use web search for current, time-sensitive information
- ALWAYS search when the query mentions specific programs, locations, or current availability
- Use your knowledge only for general guidance, but search for specific resources
"""

print("="*80)
print("VERIFICATION: Corrected Web Search Detection")
print("="*80)
print(f"Testing with 5 production queries")
print(f"Expected: Mix of web search and no web search\n")

results = []

for i, query_data in enumerate(test_queries, 1):
    query = query_data['query']
    expected_web_search = query_data['used_web_search']

    try:
        response = client.responses.create(
            model="gpt-5.1",
            reasoning={"effort": "none"},
            temperature=0.9,
            tools=[{"type": "web_search"}],
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ]
        )

        # CORRECTED DETECTION CODE
        detected_web_search = False
        web_search_queries = []

        if response.output:
            for item in response.output:
                if hasattr(item, 'type') and item.type == 'web_search_call':
                    detected_web_search = True
                    if hasattr(item, 'action') and hasattr(item.action, 'queries'):
                        web_search_queries = item.action.queries
                    break

        match = "✅" if detected_web_search == expected_web_search else "❌"
        search_status = "YES" if detected_web_search else "NO"
        expected_status = "YES" if expected_web_search else "NO"

        print(f"{i}. {match} {search_status} (expected: {expected_status}) | {query[:60]}")
        if detected_web_search and web_search_queries:
            print(f"   Queries: {web_search_queries}")

        results.append({
            'query': query,
            'expected_web_search': expected_web_search,
            'detected_web_search': detected_web_search,
            'match': detected_web_search == expected_web_search,
            'web_search_queries': web_search_queries
        })

    except Exception as e:
        print(f"{i}. ❌ ERROR | {query[:60]}")
        print(f"   Error: {e}")
        results.append({
            'query': query,
            'error': str(e)
        })

print("\n" + "="*80)
print("VERIFICATION RESULTS")
print("="*80)

matches = sum(1 for r in results if r.get('match', False))
total = len([r for r in results if 'error' not in r])

print(f"Detection accuracy: {matches}/{total} queries matched production behavior")
print(f"Detected web search in: {sum(1 for r in results if r.get('detected_web_search', False))}/{total} queries")

if matches == total:
    print("\n✅ VERIFICATION PASSED - Detection code is working correctly!")
    print("   Ready to re-run full temperature and reasoning tests")
else:
    print("\n⚠️  VERIFICATION ISSUES - Detection may still need adjustment")
    for i, r in enumerate(results, 1):
        if not r.get('match', False) and 'error' not in r:
            print(f"   Query {i}: Expected {r['expected_web_search']}, got {r['detected_web_search']}")

# Save results
with open('verification_results.json', 'w') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'accuracy': f"{matches}/{total}",
        'results': results
    }, f, indent=2)

print(f"\n✅ Results saved to: verification_results.json")
