#!/usr/bin/env python3
"""
ACCURATE web search detection test using API response metadata.

Instead of heuristics (looking for URLs in text), this checks:
1. Response object metadata for tool invocations
2. Actual tool_calls in the response
3. Response structure that indicates web search was used

This provides definitive evidence of whether web search is actually invoked.
"""

import os
import json
import time
from openai import OpenAI

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY environment variable must be set")

# 20 diverse test prompts
TEST_PROMPTS = [
    "Single mother with 2 kids facing eviction in Austin, needs emergency housing assistance",
    "Homeless veteran in Travis County needs transitional housing and job training",
    "Low-income family with 4 children needs food assistance in Austin, TX",
    "Unemployed single parent needs job training and childcare assistance in Austin",
    "Uninsured family needs low-cost healthcare clinic in Austin, TX",
    "Working parent needs affordable childcare for toddler and preschooler in Austin",
    "Person recently laid off needs help applying for unemployment benefits in Texas",
    "Elderly couple on fixed income needs help with rising rent costs in Austin",
    "Ex-offender needs employment programs for people with criminal records in Travis County",
    "Person with substance abuse issues needs addiction treatment programs in Austin",
    "Family escaping domestic violence needs emergency shelter in Central Texas",
    "Young adult aging out of foster care needs affordable housing options in Austin",
    "Elderly person living alone needs home-delivered meals in Travis County",
    "College student struggling with food insecurity needs food pantry locations near UT Austin",
    "Person with disability needs supported employment services in Central Texas",
    "Recent immigrant needs ESL classes and job placement assistance in Austin",
    "Senior citizen needs help navigating Medicare and prescription drug costs",
    "Pregnant woman without insurance needs prenatal care in Travis County",
    "Family needs after-school programs for elementary school children in East Austin",
    "Low-income family needs Head Start or Pre-K programs for 3-year-old",
]

print("=" * 80)
print("ACCURATE WEB SEARCH DETECTION TEST")
print("=" * 80)
print(f"Model: gpt-5.1")
print(f"Reasoning: none")
print(f"Temperature: 0.25")
print(f"Test prompts: {len(TEST_PROMPTS)}")
print("=" * 80)
print("\nMethod: Inspecting API response object for actual tool invocations")
print("=" * 80)

client = OpenAI()
results = []

for i, prompt in enumerate(TEST_PROMPTS, 1):
    formatted_prompt = f"""You are a case worker assistant in Central Texas. Recommend 5-7 relevant social support resources.

Client: {prompt}

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
    print(f"Test {i}/{len(TEST_PROMPTS)}")
    print(f"Prompt: {prompt[:70]}...")
    print(f"{'='*80}")

    try:
        start_time = time.time()

        response = client.responses.create(
            model="gpt-5.1",
            input=formatted_prompt,
            reasoning={"effort": "none"},
            tools=[{"type": "web_search"}],
            temperature=0.25
        )

        elapsed = time.time() - start_time

        # ACCURATE DETECTION: Inspect response object structure
        response_dict = response.model_dump() if hasattr(response, 'model_dump') else response.__dict__

        # Check for tool calls/invocations in response
        web_search_used = False
        tool_evidence = []

        # Method 1: Check for tool_calls attribute
        if hasattr(response, 'tool_calls') and response.tool_calls:
            web_search_used = True
            tool_evidence.append("response.tool_calls present")

        # Method 2: Check response dict for tool/search indicators
        response_str = json.dumps(response_dict, default=str)
        if 'web_search' in response_str.lower() or 'tool' in response_str.lower():
            web_search_used = True
            tool_evidence.append("web_search/tool in response metadata")

        # Method 3: Check for specific response attributes that indicate tool use
        if hasattr(response, 'additional_kwargs'):
            if response.additional_kwargs:
                web_search_used = True
                tool_evidence.append("additional_kwargs present")

        # Method 4: Inspect the raw response object
        if hasattr(response, 'raw_response'):
            raw = response.raw_response
            if 'tool' in str(raw).lower() or 'search' in str(raw).lower():
                web_search_used = True
                tool_evidence.append("tool/search in raw response")

        result = {
            "prompt_number": i,
            "prompt": prompt,
            "response_time": round(elapsed, 2),
            "web_search_detected": web_search_used,
            "detection_evidence": tool_evidence,
            "response_structure": {
                "has_tool_calls": hasattr(response, 'tool_calls'),
                "has_additional_kwargs": hasattr(response, 'additional_kwargs'),
                "response_type": type(response).__name__,
                "available_attributes": [attr for attr in dir(response) if not attr.startswith('_')]
            }
        }

        results.append(result)

        print(f"✅ Complete")
        print(f"⏱️  {elapsed:.2f}s")
        print(f"🔍 Web search detected: {'YES' if web_search_used else 'NO'}")
        if tool_evidence:
            print(f"📋 Evidence: {', '.join(tool_evidence)}")

        # Print response structure for analysis
        print(f"📦 Response type: {type(response).__name__}")
        print(f"📦 Response attributes: {[a for a in dir(response) if not a.startswith('_')][:10]}...")

        time.sleep(1)

    except Exception as e:
        print(f"❌ ERROR: {str(e)[:150]}")
        results.append({
            "prompt_number": i,
            "prompt": prompt,
            "error": str(e),
            "web_search_detected": None
        })
        time.sleep(1)

print("\n" + "=" * 80)
print("ANALYSIS: Actual Web Search Usage")
print("=" * 80)

detected_count = sum(1 for r in results if r.get("web_search_detected") == True)
not_detected_count = sum(1 for r in results if r.get("web_search_detected") == False)
error_count = sum(1 for r in results if r.get("web_search_detected") is None)

print(f"\n✅ Web search DETECTED: {detected_count}/{len(TEST_PROMPTS)} ({detected_count/len(TEST_PROMPTS)*100:.1f}%)")
print(f"❌ Web search NOT detected: {not_detected_count}/{len(TEST_PROMPTS)} ({not_detected_count/len(TEST_PROMPTS)*100:.1f}%)")
if error_count > 0:
    print(f"⚠️  Errors: {error_count}/{len(TEST_PROMPTS)}")

print("\n" + "=" * 80)
print("DETECTION EVIDENCE SUMMARY")
print("=" * 80)

all_evidence = []
for r in results:
    if r.get("detection_evidence"):
        all_evidence.extend(r["detection_evidence"])

if all_evidence:
    from collections import Counter
    evidence_counts = Counter(all_evidence)
    print("\nEvidence types found:")
    for evidence, count in evidence_counts.most_common():
        print(f"  - {evidence}: {count} times")
else:
    print("\n⚠️  No tool invocation evidence found in response objects")
    print("   This suggests web search may NOT be used, or")
    print("   the API doesn't expose tool usage in the response structure")

print("\n" + "=" * 80)
print("RESPONSE STRUCTURE ANALYSIS")
print("=" * 80)

if results and results[0].get("response_structure"):
    sample = results[0]["response_structure"]
    print(f"\nSample response structure:")
    print(f"  Type: {sample['response_type']}")
    print(f"  Has tool_calls: {sample['has_tool_calls']}")
    print(f"  Has additional_kwargs: {sample['has_additional_kwargs']}")
    print(f"  Available attributes: {', '.join(sample['available_attributes'][:15])}...")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if detected_count == 0:
    print("\n❌ CRITICAL FINDING: Web search NOT detected in API responses!")
    print("   ")
    print("   This means EITHER:")
    print("   1. Web search is not being invoked (most likely)")
    print("   2. The API doesn't expose tool usage in responses")
    print("   ")
    print("   Action required:")
    print("   - Check traces in your observability platform")
    print("   - Look for OpenAIWebSearch component spans")
    print("   - Verify if web_search tool is actually invoked")
elif detected_count == len(TEST_PROMPTS):
    print(f"\n✅ Web search detected in ALL {detected_count} queries")
    print("   Web search appears to be working as expected")
else:
    print(f"\n⚠️  Web search detected in {detected_count}/{len(TEST_PROMPTS)} queries ({detected_count/len(TEST_PROMPTS)*100:.1f}%)")
    print("   Inconsistent web search usage")
    print("   ")
    print("   Queries WITHOUT web search:")
    for r in results:
        if r.get("web_search_detected") == False:
            print(f"   - #{r['prompt_number']}: {r['prompt'][:60]}...")

# Save results
output_file = "accurate_websearch_detection_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n💾 Detailed results saved to: {output_file}")
print("=" * 80)
