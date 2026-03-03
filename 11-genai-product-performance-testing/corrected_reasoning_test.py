#!/usr/bin/env python3
"""
Corrected Reasoning Level vs Web Search Test - Using Real Production Queries
Tests if reasoning effort affects web search invocation rate using actual production queries
"""
import os
import json
import time
from openai import OpenAI

# Load real production queries
with open('sample_production_queries.json', 'r') as f:
    data = json.load(f)

# Combine all queries
all_queries = data['web_search_queries'] + data['no_web_search_queries']
print(f"Loaded {len(all_queries)} real production queries")
print(f"Production baseline web search rate: {data['baseline_rate']*100:.1f}%\n")

# Use subset for cost efficiency
QUERIES = [q['query'] for q in all_queries[:30]]
print(f"Testing with {len(QUERIES)} queries\n")

REASONING_LEVELS = ["none", "low"]

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# System prompt from production
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
print("CORRECTED REASONING LEVEL vs WEB SEARCH TEST")
print("="*80)
print("Testing with REAL production queries")
print(f"Production baseline: {data['baseline_rate']*100:.1f}% web search rate\n")

results = {}

for reasoning in REASONING_LEVELS:
    print(f"\n{'='*80}")
    print(f"TESTING REASONING: {reasoning}")
    print(f"{'='*80}")

    web_search_count = 0
    total_calls = 0
    latencies = []

    # Note: temperature only works with reasoning="none"
    api_params = {
        "model": "gpt-5.1",
        "reasoning": {"effort": reasoning},
        "tools": [{"type": "web_search"}],
    }

    # Only add temperature for reasoning="none"
    if reasoning == "none":
        api_params["temperature"] = 0.9  # Match production setting

    for i, query in enumerate(QUERIES, 1):
        try:
            start = time.time()

            response = client.responses.create(
                **api_params,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query}
                ]
            )

            latency = time.time() - start
            latencies.append(latency)
            total_calls += 1

            # Check if web search was used - CORRECTED DETECTION
            used_web_search = False
            if response.output:
                for item in response.output:
                    if hasattr(item, 'type') and item.type == 'web_search_call':
                        used_web_search = True
                        web_search_count += 1
                        break

            status = "🌐 WEB" if used_web_search else "   NO"
            print(f"  {i:2d}. {status} | {latency:6.2f}s | {query[:60]}")

            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"  {i:2d}. ERROR: {str(e)[:100]}")
            continue

    # Calculate stats
    web_search_rate = (web_search_count / total_calls * 100) if total_calls > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    results[reasoning] = {
        'web_search_count': web_search_count,
        'total_calls': total_calls,
        'web_search_rate': web_search_rate,
        'avg_latency': avg_latency
    }

    print(f"\n  Web search rate: {web_search_rate:.1f}% ({web_search_count}/{total_calls})")
    print(f"  Avg latency: {avg_latency:.2f}s")

# Final summary
print(f"\n{'='*80}")
print("CORRECTED TEST RESULTS SUMMARY")
print(f"{'='*80}")
print(f"\nProduction baseline: {data['baseline_rate']*100:.1f}% web search rate")
print(f"Total queries tested per reasoning level: {len(QUERIES)}")
print(f"\nResults by reasoning level:\n")

for reasoning in REASONING_LEVELS:
    r = results[reasoning]
    rate_diff = r['web_search_rate'] - (data['baseline_rate'] * 100)
    print(f"  Reasoning '{reasoning}':")
    print(f"    Web search rate: {r['web_search_rate']:5.1f}% ({r['web_search_count']}/{r['total_calls']})")
    print(f"    Difference from baseline: {rate_diff:+.1f} percentage points")
    print(f"    Avg latency: {r['avg_latency']:.2f}s")
    print()

# Compare latency impact
if "none" in results and "low" in results:
    latency_increase = results["low"]["avg_latency"] - results["none"]["avg_latency"]
    latency_percent = (latency_increase / results["none"]["avg_latency"]) * 100 if results["none"]["avg_latency"] > 0 else 0
    print(f"Latency impact of reasoning='low':")
    print(f"  +{latency_increase:.2f}s ({latency_percent:+.1f}%)\n")

# Save results
output = {
    'production_baseline_rate': data['baseline_rate'] * 100,
    'test_query_count': len(QUERIES),
    'results': results,
    'conclusion': 'Reasoning level effect on web search invocation rate using real production queries'
}

with open('corrected_reasoning_test_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"✅ Results saved to: corrected_reasoning_test_results.json")
print("="*80)
