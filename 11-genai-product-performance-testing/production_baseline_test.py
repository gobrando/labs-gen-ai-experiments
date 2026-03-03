#!/usr/bin/env python3
"""
Production Baseline Test - Establish true baseline before testing parameter effects

First establish baseline with minimal prompt, then test temperature/reasoning effects
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
print(f"Production baseline from Phoenix: {data['baseline_rate']*100:.1f}% web search\n")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# MINIMAL system prompt - let model decide naturally
MINIMAL_PROMPT = "You are an AI assistant helping social services case workers in Central Texas find resources for their clients."

def run_test(queries, config_name, temperature=None, reasoning="none"):
    """Run test with specific configuration"""
    print(f"\n{'='*80}")
    print(f"TESTING: {config_name}")
    print(f"{'='*80}")

    web_search_count = 0
    total_calls = 0
    latencies = []

    # Build API params
    api_params = {
        "model": "gpt-5.1",
        "reasoning": {"effort": reasoning},
        "tools": [{"type": "web_search"}],
    }

    # Only add temperature for reasoning="none"
    if reasoning == "none" and temperature is not None:
        api_params["temperature"] = temperature

    for i, query_data in enumerate(queries, 1):
        query = query_data['query']
        expected_web_search = query_data['used_web_search']

        try:
            start = time.time()

            response = client.responses.create(
                **api_params,
                input=[
                    {"role": "system", "content": MINIMAL_PROMPT},
                    {"role": "user", "content": query}
                ]
            )

            latency = time.time() - start
            latencies.append(latency)
            total_calls += 1

            # Corrected web search detection
            used_web_search = False
            if response.output:
                for item in response.output:
                    if hasattr(item, 'type') and item.type == 'web_search_call':
                        used_web_search = True
                        web_search_count += 1
                        break

            # Show if behavior matches production expectation
            match = "✅" if used_web_search == expected_web_search else "⚠️"
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

    print(f"\n  Results:")
    print(f"    Web search rate: {web_search_rate:.1f}% ({web_search_count}/{total_calls})")
    print(f"    Difference from prod baseline: {baseline_diff:+.1f} percentage points")
    print(f"    Avg latency: {avg_latency:.2f}s")

    return {
        'web_search_count': web_search_count,
        'total_calls': total_calls,
        'web_search_rate': web_search_rate,
        'avg_latency': avg_latency,
        'baseline_diff': baseline_diff
    }

print("="*80)
print("PRODUCTION BASELINE VERIFICATION TEST")
print("="*80)
print("Goal: Establish proper baseline with minimal prompt, then test parameter effects\n")

results = {}

# Step 1: Establish baseline with default settings (temperature=0.9, reasoning=none)
print("\n" + "="*80)
print("STEP 1: ESTABLISH BASELINE")
print("="*80)
print("Using: temperature=0.9, reasoning='none', minimal system prompt")
results['baseline'] = run_test(all_queries, "BASELINE (temp=0.9, reasoning=none)",
                               temperature=0.9, reasoning="none")

# Step 2: Test temperature variations
print("\n" + "="*80)
print("STEP 2: TEST TEMPERATURE VARIATIONS")
print("="*80)

results['temp_0.0'] = run_test(all_queries, "Temperature 0.0",
                               temperature=0.0, reasoning="none")

results['temp_1.0'] = run_test(all_queries, "Temperature 1.0",
                               temperature=1.0, reasoning="none")

# Step 3: Test reasoning variation
print("\n" + "="*80)
print("STEP 3: TEST REASONING VARIATION")
print("="*80)

results['reasoning_low'] = run_test(all_queries, "Reasoning 'low' (no temperature)",
                                   temperature=None, reasoning="low")

# Final Summary
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"\nProduction baseline from Phoenix: {data['baseline_rate']*100:.1f}%")
print(f"Total queries tested: {len(all_queries)}\n")

print("Configuration                      | Web Search Rate | Diff from Prod | Avg Latency")
print("-" * 85)

for key, result in results.items():
    config_name = {
        'baseline': 'BASELINE (temp=0.9, r=none)',
        'temp_0.0': 'Temperature 0.0',
        'temp_1.0': 'Temperature 1.0',
        'reasoning_low': 'Reasoning low (no temp)'
    }[key]

    print(f"{config_name:34s} | {result['web_search_rate']:14.1f}% | {result['baseline_diff']:+13.1f}pp | {result['avg_latency']:10.2f}s")

# Analysis
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

baseline_rate = results['baseline']['web_search_rate']

# Temperature effect
temp_0_diff = results['temp_0.0']['web_search_rate'] - baseline_rate
temp_1_diff = results['temp_1.0']['web_search_rate'] - baseline_rate

print(f"\nTemperature Effect on Web Search Rate:")
print(f"  temp=0.0 vs baseline: {temp_0_diff:+.1f} percentage points")
print(f"  temp=1.0 vs baseline: {temp_1_diff:+.1f} percentage points")

if abs(temp_0_diff) < 5 and abs(temp_1_diff) < 5:
    print(f"  ✅ Conclusion: Temperature has MINIMAL effect on web search rate (<5pp change)")
else:
    print(f"  ⚠️  Conclusion: Temperature DOES affect web search rate (>5pp change)")

# Reasoning effect
reasoning_diff = results['reasoning_low']['web_search_rate'] - baseline_rate
reasoning_latency_increase = results['reasoning_low']['avg_latency'] - results['baseline']['avg_latency']

print(f"\nReasoning Level Effect:")
print(f"  reasoning=low vs none: {reasoning_diff:+.1f} percentage points")
print(f"  Latency increase: +{reasoning_latency_increase:.2f}s")

if abs(reasoning_diff) < 5:
    print(f"  ✅ Conclusion: Reasoning has MINIMAL effect on web search rate (<5pp change)")
else:
    print(f"  ⚠️  Conclusion: Reasoning DOES affect web search rate (>5pp change)")

# Save results
output = {
    'production_baseline_phoenix': data['baseline_rate'] * 100,
    'total_queries': len(all_queries),
    'results': results,
    'analysis': {
        'temperature_effect': {
            'temp_0_vs_baseline': temp_0_diff,
            'temp_1_vs_baseline': temp_1_diff,
            'conclusion': 'minimal' if abs(temp_0_diff) < 5 and abs(temp_1_diff) < 5 else 'significant'
        },
        'reasoning_effect': {
            'low_vs_none': reasoning_diff,
            'latency_increase': reasoning_latency_increase,
            'conclusion': 'minimal' if abs(reasoning_diff) < 5 else 'significant'
        }
    }
}

with open('production_baseline_test_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n✅ Results saved to: production_baseline_test_results.json")
print("="*80)
