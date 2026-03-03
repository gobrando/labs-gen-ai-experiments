#!/usr/bin/env python3
"""
Production-Matching Configuration Test

Uses EXACT production configuration:
- Model: gpt-5-mini (not gpt-5.1)
- Reasoning effort: "low" (not "none")
- No temperature parameter (incompatible with reasoning)
"""
import os
import json
import time
from openai import OpenAI

# Load real production queries
with open('sample_production_queries.json', 'r') as f:
    data = json.load(f)

# Use ALL queries for better statistical significance
all_queries = data['web_search_queries'] + data['no_web_search_queries']
print(f"Loaded {len(all_queries)} real production queries")
print(f"Production baseline: {data['baseline_rate']*100:.1f}% web search\n")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# System prompt with web search guidance (similar to production)
SYSTEM_PROMPT = """You are helping social services case workers in Central Texas find resources for their clients.

Based on the client's request, identify relevant social service resources.

Use web search when:
- You need current program availability
- You need up-to-date contact information
- You need to verify if a program still exists
- The query asks about specific current programs or schedules

Do NOT use web search when:
- The query is about general resource categories
- You have sufficient knowledge about well-established resources
- The information doesn't change frequently
"""

print("="*80)
print("PRODUCTION-MATCHING CONFIGURATION TEST")
print("="*80)
print("Using EXACT production config:")
print("  - Model: gpt-5-mini")
print("  - Reasoning effort: low")
print("  - No temperature parameter")
print("="*80)

web_search_count = 0
total_calls = 0
latencies = []
match_count = 0

for i, query_data in enumerate(all_queries, 1):
    query = query_data['query']
    expected_web_search = query_data['used_web_search']

    try:
        start = time.time()

        # EXACT production configuration from pipeline_wrapper.py line 141
        response = client.responses.create(
            model="gpt-5-mini",
            reasoning={"effort": "low"},
            tools=[{"type": "web_search"}],
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ]
        )

        latency = time.time() - start
        latencies.append(latency)
        total_calls += 1

        # Check if web search was used
        used_web_search = False
        if response.output:
            for item in response.output:
                if hasattr(item, 'type') and item.type == 'web_search_call':
                    used_web_search = True
                    web_search_count += 1
                    break

        # Check if behavior matches production
        if used_web_search == expected_web_search:
            match_count += 1
            match = "✅"
        else:
            match = "⚠️ "

        status = "WEB" if used_web_search else " NO"
        expected = "WEB" if expected_web_search else " NO"

        print(f"  {i:2d}. {match} {status} (prod:{expected}) | {latency:6.2f}s | {query[:50]}")

        time.sleep(0.5)  # Rate limiting

    except Exception as e:
        print(f"  {i:2d}. ❌ ERROR: {str(e)[:80]}")
        continue

# Calculate stats
web_search_rate = (web_search_count / total_calls * 100) if total_calls > 0 else 0
avg_latency = sum(latencies) / len(latencies) if latencies else 0
baseline_diff = web_search_rate - (data['baseline_rate'] * 100)
match_rate = (match_count / total_calls * 100) if total_calls > 0 else 0

print(f"\n{'='*80}")
print("RESULTS")
print(f"{'='*80}")
print(f"  Web search rate: {web_search_rate:.1f}% ({web_search_count}/{total_calls})")
print(f"  Production baseline: {data['baseline_rate']*100:.1f}%")
print(f"  Difference: {baseline_diff:+.1f} percentage points")
print(f"  Match rate with production: {match_rate:.1f}%")
print(f"  Avg latency: {avg_latency:.2f}s")

# Analysis
print(f"\n{'='*80}")
print("ANALYSIS")
print(f"{'='*80}")

if abs(baseline_diff) < 10:
    print(f"✅ GOOD: Test web search rate ({web_search_rate:.1f}%) is within 10pp of production ({data['baseline_rate']*100:.1f}%)")
    print(f"   This configuration closely matches production behavior!")
else:
    print(f"⚠️  MISMATCH: Test web search rate ({web_search_rate:.1f}%) differs by {abs(baseline_diff):.1f}pp from production")
    print(f"   Possible reasons:")
    print(f"   - Production prompt from Phoenix may have different/more explicit guidance")
    print(f"   - Model behavior differences between test and production environments")

# Save results
output = {
    'production_baseline': data['baseline_rate'] * 100,
    'test_web_search_rate': web_search_rate,
    'difference': baseline_diff,
    'match_rate': match_rate,
    'total_queries': total_calls,
    'avg_latency': avg_latency,
    'config': {
        'model': 'gpt-5-mini',
        'reasoning_effort': 'low',
        'temperature': None
    }
}

with open('production_matching_test_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n✅ Results saved to: production_matching_test_results.json")
print("="*80)
