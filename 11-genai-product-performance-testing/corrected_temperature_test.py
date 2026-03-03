#!/usr/bin/env python3
"""
Corrected Temperature vs Web Search Test - Using Real Production Queries
Tests if temperature affects web search invocation rate using actual production queries
"""
import os
import json
import time
from openai import OpenAI

# Load real production queries
with open('sample_production_queries.json', 'r') as f:
    data = json.load(f)

# Combine all queries (both web search and no web search)
all_queries = data['web_search_queries'] + data['no_web_search_queries']
print(f"Loaded {len(all_queries)} real production queries")
print(f"Production baseline web search rate: {data['baseline_rate']*100:.1f}%\n")

# Use subset for cost efficiency
QUERIES = [q['query'] for q in all_queries[:30]]  # Use 30 queries
print(f"Testing with {len(QUERIES)} queries\n")

TEMPERATURES = [0.0, 0.5, 1.0]

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# System prompt from production (simplified for testing)
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
print("CORRECTED TEMPERATURE vs WEB SEARCH TEST")
print("="*80)
print("Testing with REAL production queries")
print(f"Production baseline: {data['baseline_rate']*100:.1f}% web search rate\n")

results = {}

for temp in TEMPERATURES:
    print(f"\n{'='*80}")
    print(f"TESTING TEMPERATURE: {temp}")
    print(f"{'='*80}")

    web_search_count = 0
    total_calls = 0
    latencies = []

    for i, query in enumerate(QUERIES, 1):
        try:
            start = time.time()

            response = client.responses.create(
                model="gpt-5.1",
                reasoning={"effort": "none"},
                temperature=temp,
                tools=[{"type": "web_search"}],
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
            print(f"  {i:2d}. {status} | {latency:5.2f}s | {query[:60]}")

            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"  {i:2d}. ERROR: {str(e)[:100]}")
            continue

    # Calculate stats
    web_search_rate = (web_search_count / total_calls * 100) if total_calls > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    results[temp] = {
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
print(f"Total queries tested per temperature: {len(QUERIES)}")
print(f"\nResults by temperature:\n")

for temp in TEMPERATURES:
    r = results[temp]
    rate_diff = r['web_search_rate'] - (data['baseline_rate'] * 100)
    print(f"  Temperature {temp:.1f}:")
    print(f"    Web search rate: {r['web_search_rate']:5.1f}% ({r['web_search_count']}/{r['total_calls']})")
    print(f"    Difference from baseline: {rate_diff:+.1f} percentage points")
    print(f"    Avg latency: {r['avg_latency']:.2f}s")
    print()

# Save results
output = {
    'production_baseline_rate': data['baseline_rate'] * 100,
    'test_query_count': len(QUERIES),
    'results': results,
    'conclusion': 'Temperature effect on web search invocation rate using real production queries'
}

with open('corrected_temperature_test_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"✅ Results saved to: corrected_temperature_test_results.json")
print("="*80)
