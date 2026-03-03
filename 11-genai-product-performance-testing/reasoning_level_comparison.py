#!/usr/bin/env python3
"""
Compare reasoning="low" vs reasoning="none" to test:
1. How often web search is used
2. Output quality differences
3. Latency impact

Tests with 30 realistic Central Texas social services prompts.
"""

import os
import json
import time
from openai import OpenAI
from collections import defaultdict

# Check for API key
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY environment variable must be set")

# 30 realistic test prompts for Central Texas social services
TEST_PROMPTS = [
    # Housing & Homelessness (5 prompts)
    "Single mother with 2 kids facing eviction in Austin, needs emergency housing assistance",
    "Homeless veteran in Travis County needs transitional housing and job training",
    "Elderly couple on fixed income needs help with rising rent costs in Austin",
    "Family escaping domestic violence needs emergency shelter in Central Texas",
    "Young adult aging out of foster care needs affordable housing options in Austin",

    # Food Assistance (5 prompts)
    "Low-income family with 4 children needs food assistance in Austin, TX",
    "Elderly person living alone needs home-delivered meals in Travis County",
    "College student struggling with food insecurity needs food pantry locations near UT Austin",
    "Undocumented immigrant family needs food assistance that doesn't require legal status",
    "Person with diabetes needs food assistance with dietary restrictions in Austin",

    # Employment & Job Training (5 prompts)
    "Unemployed single parent needs job training and childcare assistance in Austin",
    "Ex-offender needs employment programs for people with criminal records in Travis County",
    "Person with disability needs supported employment services in Central Texas",
    "Recent immigrant needs ESL classes and job placement assistance in Austin",
    "Teenager needs summer job opportunities and youth employment programs",

    # Healthcare & Mental Health (5 prompts)
    "Uninsured family needs low-cost healthcare clinic in Austin, TX",
    "Person experiencing depression needs free or sliding-scale mental health counseling",
    "Senior citizen needs help navigating Medicare and prescription drug costs",
    "Pregnant woman without insurance needs prenatal care in Travis County",
    "Person with substance abuse issues needs addiction treatment programs in Austin",

    # Childcare & Education (5 prompts)
    "Working parent needs affordable childcare for toddler and preschooler in Austin",
    "Family needs after-school programs for elementary school children in East Austin",
    "Low-income family needs Head Start or Pre-K programs for 3-year-old",
    "Parent needs tutoring services for child struggling in school",
    "Family needs summer camp programs with financial assistance in Travis County",

    # Benefits & Legal (5 prompts)
    "Person recently laid off needs help applying for unemployment benefits in Texas",
    "Immigrant family needs help applying for SNAP and Medicaid benefits",
    "Senior citizen needs help with Social Security disability application",
    "Person facing wage theft needs legal aid services in Austin",
    "Low-income family needs help with tax preparation and EITC",
]

print("=" * 80)
print("REASONING LEVEL COMPARISON TEST")
print("Testing: reasoning='low' vs reasoning='none'")
print("=" * 80)
print(f"Model: gpt-5.1")
print(f"Temperature: 0.25 (optimal from previous testing)")
print(f"Web Search: ENABLED")
print(f"Test prompts: {len(TEST_PROMPTS)}")
print("=" * 80)

client = OpenAI()

def run_test(prompt_text: str, reasoning_effort: str, prompt_num: int) -> dict:
    """Run a single test with the given reasoning effort level."""

    # Format the prompt as the app would
    formatted_prompt = f"""You are a case worker assistant in Central Texas. Recommend 5-7 relevant social support resources.

Client: {prompt_text}

Return ONLY valid JSON in this exact format:
{{
  "resources": [
    {{
      "name": "Organization Name",
      "description": "Brief description",
      "website": "URL",
      "phones": ["phone number"],
      "emails": ["email"],
      "addresses": ["address"]
    }}
  ]
}}"""

    print(f"\n{'='*80}")
    print(f"Prompt {prompt_num}/30: {prompt_text[:80]}...")
    print(f"Reasoning: {reasoning_effort}")
    print(f"{'='*80}")

    try:
        start_time = time.time()

        api_params = {
            "model": "gpt-5.1",
            "input": formatted_prompt,
            "reasoning": {"effort": reasoning_effort},
            "tools": [{"type": "web_search"}],
            "temperature": 0.25
        }

        response = client.responses.create(**api_params)
        elapsed = time.time() - start_time

        # Parse response
        response_text = response.output_text

        # Extract JSON
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start != -1 and end != -1:
            json_str = response_text[start:end + 1]
            result_json = json.loads(json_str)
            resource_count = len(result_json.get("resources", []))
            resource_names = [r.get("name", "Unknown") for r in result_json.get("resources", [])]
        else:
            resource_count = 0
            resource_names = []
            result_json = None

        # Check for web search indicators in response
        # Note: This is a heuristic - actual trace inspection would be more accurate
        web_search_indicators = [
            "according to",
            "based on web search",
            "website shows",
            "online information",
            "current information",
            ".org",
            ".gov",
            ".com"
        ]

        likely_used_web_search = any(indicator in response_text.lower() for indicator in web_search_indicators)

        result = {
            "success": True,
            "reasoning_effort": reasoning_effort,
            "response_time": round(elapsed, 2),
            "resource_count": resource_count,
            "resource_names": resource_names,
            "output_text": response_text,
            "output_json": result_json,
            "likely_used_web_search": likely_used_web_search,
            "error": None
        }

        print(f"✅ SUCCESS!")
        print(f"⏱️  Response time: {elapsed:.2f}s")
        print(f"📊 Resources found: {resource_count}")
        print(f"🔍 Likely used web search: {'YES' if likely_used_web_search else 'NO'}")
        if resource_names:
            print(f"📝 Resources: {', '.join(resource_names[:3])}{'...' if len(resource_names) > 3 else ''}")

        return result

    except Exception as e:
        error_msg = str(e)
        print(f"❌ ERROR: {error_msg[:150]}")

        return {
            "success": False,
            "reasoning_effort": reasoning_effort,
            "response_time": 0,
            "resource_count": 0,
            "resource_names": [],
            "output_text": None,
            "output_json": None,
            "likely_used_web_search": False,
            "error": error_msg
        }

# Store all results
all_results = []

# Test each prompt with both reasoning levels
for i, prompt in enumerate(TEST_PROMPTS, 1):
    prompt_results = {
        "prompt_number": i,
        "prompt": prompt,
        "reasoning_none": None,
        "reasoning_low": None
    }

    # Test with reasoning="none" (current production config)
    prompt_results["reasoning_none"] = run_test(prompt, "none", i)

    # Small delay to avoid rate limits
    time.sleep(1)

    # Test with reasoning="low"
    prompt_results["reasoning_low"] = run_test(prompt, "low", i)

    # Small delay between prompts
    time.sleep(1)

    all_results.append(prompt_results)

print("\n" + "=" * 80)
print("ANALYSIS: Reasoning Level Comparison")
print("=" * 80)

# Aggregate statistics
stats = {
    "none": {
        "total_tests": 0,
        "successful_tests": 0,
        "total_latency": 0,
        "avg_latency": 0,
        "total_resources": 0,
        "avg_resources": 0,
        "web_search_count": 0,
        "web_search_rate": 0
    },
    "low": {
        "total_tests": 0,
        "successful_tests": 0,
        "total_latency": 0,
        "avg_latency": 0,
        "total_resources": 0,
        "avg_resources": 0,
        "web_search_count": 0,
        "web_search_rate": 0
    }
}

for result in all_results:
    for reasoning_level in ["none", "low"]:
        test_result = result[f"reasoning_{reasoning_level}"]
        stats[reasoning_level]["total_tests"] += 1

        if test_result["success"]:
            stats[reasoning_level]["successful_tests"] += 1
            stats[reasoning_level]["total_latency"] += test_result["response_time"]
            stats[reasoning_level]["total_resources"] += test_result["resource_count"]

            if test_result["likely_used_web_search"]:
                stats[reasoning_level]["web_search_count"] += 1

# Calculate averages
for reasoning_level in ["none", "low"]:
    successful = stats[reasoning_level]["successful_tests"]
    if successful > 0:
        stats[reasoning_level]["avg_latency"] = stats[reasoning_level]["total_latency"] / successful
        stats[reasoning_level]["avg_resources"] = stats[reasoning_level]["total_resources"] / successful
        stats[reasoning_level]["web_search_rate"] = (stats[reasoning_level]["web_search_count"] / successful) * 100

print("\n📊 REASONING='NONE' (Current Production Config):")
print(f"  Success rate: {stats['none']['successful_tests']}/{stats['none']['total_tests']} ({stats['none']['successful_tests']/stats['none']['total_tests']*100:.1f}%)")
print(f"  Avg latency: {stats['none']['avg_latency']:.2f}s")
print(f"  Avg resources per query: {stats['none']['avg_resources']:.1f}")
print(f"  Web search usage: {stats['none']['web_search_count']}/{stats['none']['successful_tests']} ({stats['none']['web_search_rate']:.1f}%)")

print("\n📊 REASONING='LOW':")
print(f"  Success rate: {stats['low']['successful_tests']}/{stats['low']['total_tests']} ({stats['low']['successful_tests']/stats['low']['total_tests']*100:.1f}%)")
print(f"  Avg latency: {stats['low']['avg_latency']:.2f}s")
print(f"  Avg resources per query: {stats['low']['avg_resources']:.1f}")
print(f"  Web search usage: {stats['low']['web_search_count']}/{stats['low']['successful_tests']} ({stats['low']['web_search_rate']:.1f}%)")

print("\n🔍 COMPARISON:")
latency_diff = stats['low']['avg_latency'] - stats['none']['avg_latency']
latency_pct = (latency_diff / stats['none']['avg_latency'] * 100) if stats['none']['avg_latency'] > 0 else 0
print(f"  Latency impact: {latency_diff:+.2f}s ({latency_pct:+.1f}%)")

web_search_diff = stats['low']['web_search_rate'] - stats['none']['web_search_rate']
print(f"  Web search usage difference: {web_search_diff:+.1f}pp")
print(f"    reasoning='none': {stats['none']['web_search_rate']:.1f}%")
print(f"    reasoning='low': {stats['low']['web_search_rate']:.1f}%")

resource_diff = stats['low']['avg_resources'] - stats['none']['avg_resources']
print(f"  Avg resources difference: {resource_diff:+.1f}")

print("\n" + "=" * 80)
print("KEY FINDINGS:")
print("=" * 80)

if stats['low']['web_search_rate'] > stats['none']['web_search_rate'] + 10:
    print("✅ reasoning='low' SIGNIFICANTLY increases web search usage")
    print(f"   +{web_search_diff:.1f}pp more web searches detected")
elif stats['low']['web_search_rate'] > stats['none']['web_search_rate']:
    print("⚠️  reasoning='low' slightly increases web search usage")
    print(f"   +{web_search_diff:.1f}pp more web searches detected")
else:
    print("❌ reasoning='low' does NOT increase web search usage")
    print(f"   No significant difference in web search detection")

if latency_pct > 50:
    print(f"\n⚠️  WARNING: reasoning='low' has MAJOR latency impact (+{latency_pct:.1f}%)")
elif latency_pct > 20:
    print(f"\n⚠️  reasoning='low' has moderate latency impact (+{latency_pct:.1f}%)")
else:
    print(f"\n✅ reasoning='low' has minimal latency impact (+{latency_pct:.1f}%)")

print("\n" + "=" * 80)
print("RECOMMENDATIONS:")
print("=" * 80)

if stats['low']['web_search_rate'] > stats['none']['web_search_rate'] + 10 and latency_pct < 50:
    print("✅ RECOMMENDATION: Switch to reasoning='low'")
    print("   - Significantly more web search usage")
    print("   - Acceptable latency impact")
    print("   - Should improve output quality")
elif stats['low']['web_search_rate'] > stats['none']['web_search_rate'] and latency_pct < 20:
    print("✅ RECOMMENDATION: Consider switching to reasoning='low'")
    print("   - More web search usage")
    print("   - Minimal latency impact")
    print("   - May improve output quality")
else:
    print("⚠️  RECOMMENDATION: Stay with reasoning='none'")
    print("   - No significant web search increase with reasoning='low'")
    print("   - Current config maintains speed with temperature control")

print("\n" + "=" * 80)
print("NEXT STEPS:")
print("=" * 80)
print("1. Review trace data in observability platform for accurate web search counts")
print("   - Look for 'OpenAIWebSearch' spans")
print("   - Check for 'web_search_...' child spans (0.10ms in traces)")
print("2. Manually review sample outputs to compare quality")
print("3. Test with real users if web search improvement is significant")
print("=" * 80)

# Save detailed results
output_file = "reasoning_level_comparison_results.json"
with open(output_file, "w") as f:
    json.dump({
        "metadata": {
            "model": "gpt-5.1",
            "temperature": 0.25,
            "total_prompts": len(TEST_PROMPTS),
            "reasoning_levels": ["none", "low"]
        },
        "statistics": stats,
        "detailed_results": all_results
    }, f, indent=2)

print(f"\n💾 Detailed results saved to: {output_file}")
print("\n" + "=" * 80)
print("IMPORTANT NOTE:")
print("=" * 80)
print("⚠️  This test uses HEURISTICS to detect web search usage (looking for")
print("   indicators in the response text). For ACCURATE web search counts,")
print("   you MUST check the trace spans in your observability platform.")
print("   ")
print("   Look for:")
print("   - 'OpenAIWebSearch' component spans")
print("   - 'web_search_...' child spans (highlighted in your screenshot)")
print("   - 'websearch' tokens in span/event/tool payloads")
print("=" * 80)
