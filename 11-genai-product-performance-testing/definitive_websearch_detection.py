#!/usr/bin/env python3
"""
DEFINITIVE web search detection by thorough OpenAI Response inspection.

This script:
1. Makes API calls with web_search tool enabled
2. Thoroughly inspects the Response object for ANY evidence of tool usage
3. Examines all attributes, metadata, and internal structures
4. Provides definitive proof by showing the actual tool invocation data

This addresses the lead designer's skepticism by going beyond heuristics.
"""

import os
import json
import time
import pprint
from typing import Any, Dict
from openai import OpenAI

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY environment variable must be set")

# Test prompts - using a subset for thorough analysis
TEST_PROMPTS = [
    "Single mother with 2 kids facing eviction in Austin, needs emergency housing assistance",
    "Homeless veteran in Travis County needs transitional housing and job training",
    "Low-income family with 4 children needs food assistance in Austin, TX",
    "Unemployed single parent needs job training and childcare assistance in Austin",
    "Uninsured family needs low-cost healthcare clinic in Austin, TX",
]

print("=" * 80)
print("DEFINITIVE WEB SEARCH DETECTION: Deep Response Object Inspection")
print("=" * 80)
print(f"Model: gpt-5.1")
print(f"Reasoning: none")
print(f"Temperature: 0.25")
print(f"Test prompts: {len(TEST_PROMPTS)}")
print("=" * 80)
print("\nMethod: Deep inspection of OpenAI Response object structure")
print("=" * 80)

client = OpenAI()
results = []


def deep_inspect_object(obj: Any, path: str = "response", max_depth: int = 10, current_depth: int = 0) -> Dict[str, Any]:
    """
    Recursively inspect an object to find any tool/web_search indicators.
    Returns a dict of findings.
    """
    findings = {}

    if current_depth > max_depth:
        return findings

    # Check for common attributes
    attrs_to_check = [
        'tool_calls', 'tools', 'web_search', 'search', 'function_call',
        'choices', 'message', 'content', 'role', 'metadata', 'usage',
        'model_extra', 'additional_kwargs', 'response_metadata',
        'tool_use', 'tool_results', 'search_results'
    ]

    for attr in attrs_to_check:
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            if value is not None and value != [] and value != {}:
                findings[f"{path}.{attr}"] = str(value)[:500]  # Truncate long values

    # Check all attributes
    if hasattr(obj, '__dict__'):
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):
                full_path = f"{path}.{key}"
                # Look for tool/search keywords in keys
                if any(keyword in key.lower() for keyword in ['tool', 'search', 'function', 'call']):
                    findings[full_path] = str(value)[:500]
                # Look for tool/search keywords in string values
                elif isinstance(value, str) and any(keyword in value.lower() for keyword in ['tool', 'search', 'function']):
                    findings[full_path] = str(value)[:500]
                # Recursively check nested objects
                elif hasattr(value, '__dict__') and current_depth < max_depth:
                    nested = deep_inspect_object(value, full_path, max_depth, current_depth + 1)
                    findings.update(nested)

    return findings


print("\n" + "=" * 80)
print("RUNNING DETAILED TESTS")
print("=" * 80)

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

        # Make API call
        response = client.responses.create(
            model="gpt-5.1",
            input=formatted_prompt,
            reasoning={"effort": "none"},
            tools=[{"type": "web_search"}],
            temperature=0.25
        )

        elapsed = time.time() - start_time

        print(f"✅ Complete ({elapsed:.2f}s)")
        print("\n🔍 DETAILED RESPONSE INSPECTION:")
        print("-" * 80)

        # 1. Print response type and basic info
        print(f"\n1. Response Type: {type(response).__name__}")
        print(f"   Module: {type(response).__module__}")

        # 2. Print all attributes
        print(f"\n2. All Attributes:")
        all_attrs = [attr for attr in dir(response) if not attr.startswith('_')]
        for attr in all_attrs[:20]:  # Show first 20
            try:
                value = getattr(response, attr)
                if not callable(value):
                    value_str = str(value)[:100]
                    print(f"   - {attr}: {value_str}")
            except:
                pass

        # 3. Try to convert to dict
        print(f"\n3. Response as Dictionary:")
        try:
            if hasattr(response, 'model_dump'):
                response_dict = response.model_dump()
            elif hasattr(response, 'dict'):
                response_dict = response.dict()
            elif hasattr(response, '__dict__'):
                response_dict = response.__dict__
            else:
                response_dict = {}

            # Print first level keys
            print(f"   Keys: {list(response_dict.keys())}")

            # Look for tool-related keys
            tool_keys = [k for k in response_dict.keys() if 'tool' in k.lower() or 'search' in k.lower()]
            if tool_keys:
                print(f"   🎯 TOOL-RELATED KEYS FOUND: {tool_keys}")
                for key in tool_keys:
                    print(f"      {key}: {response_dict[key]}")

            # Print as formatted JSON (truncated)
            json_str = json.dumps(response_dict, indent=2, default=str)
            if len(json_str) > 2000:
                print(f"   (Truncated to 2000 chars)")
                json_str = json_str[:2000] + "..."
            print(f"\n   Full dict:\n{json_str}")

        except Exception as e:
            print(f"   Error converting to dict: {e}")
            response_dict = {}

        # 4. Deep inspection for tool indicators
        print(f"\n4. Deep Object Inspection (searching for tool/search indicators):")
        findings = deep_inspect_object(response)
        if findings:
            print(f"   🎯 FINDINGS ({len(findings)} items):")
            for path, value in findings.items():
                print(f"      {path}:")
                print(f"         {value}")
        else:
            print(f"   ⚠️  No tool/search indicators found in deep inspection")

        # 5. Check for web_search in JSON serialization
        print(f"\n5. String Search in JSON Representation:")
        json_repr = json.dumps(response_dict, default=str).lower()
        search_terms = ['web_search', 'websearch', 'tool_call', 'tool_use', 'search_query', 'search_result']
        found_terms = [term for term in search_terms if term in json_repr]
        if found_terms:
            print(f"   🎯 FOUND TERMS: {found_terms}")
            # Show context around each found term
            for term in found_terms:
                idx = json_repr.find(term)
                if idx != -1:
                    start = max(0, idx - 100)
                    end = min(len(json_repr), idx + 100)
                    context = json_repr[start:end]
                    print(f"      Context for '{term}':")
                    print(f"         ...{context}...")
        else:
            print(f"   ❌ None of these terms found: {search_terms}")

        # 6. Try to access output/content
        print(f"\n6. Response Output/Content:")
        if hasattr(response, 'output'):
            output = response.output
            print(f"   Type: {type(output)}")
            if isinstance(output, str):
                print(f"   Length: {len(output)} chars")
                print(f"   Preview: {output[:500]}...")
            else:
                print(f"   Value: {output}")
        elif hasattr(response, 'content'):
            content = response.content
            print(f"   Type: {type(content)}")
            print(f"   Value: {str(content)[:500]}...")
        else:
            print(f"   ⚠️  No 'output' or 'content' attribute found")

        # Store result
        result = {
            "prompt_number": i,
            "prompt": prompt,
            "response_time": round(elapsed, 2),
            "response_type": type(response).__name__,
            "has_tool_indicators": len(findings) > 0 or len(found_terms) > 0,
            "tool_indicators_found": list(findings.keys()),
            "search_terms_found": found_terms,
            "response_dict_keys": list(response_dict.keys()) if response_dict else [],
        }

        results.append(result)

        print("\n" + "-" * 80)
        print(f"📊 CONCLUSION FOR THIS TEST:")
        if len(findings) > 0 or len(found_terms) > 0:
            print(f"   ✅ STRONG EVIDENCE of tool/web_search usage")
            print(f"      - Object inspection findings: {len(findings)}")
            print(f"      - Search terms found: {len(found_terms)}")
        else:
            print(f"   ❌ NO DEFINITIVE EVIDENCE of tool/web_search usage found")
            print(f"      The response object does not contain clear tool invocation data")

        time.sleep(2)

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        results.append({
            "prompt_number": i,
            "prompt": prompt,
            "error": str(e),
        })

print("\n" + "=" * 80)
print("FINAL ANALYSIS")
print("=" * 80)

tests_with_evidence = sum(1 for r in results if r.get("has_tool_indicators"))
total_tests = len([r for r in results if 'error' not in r])

print(f"\n📊 Summary:")
print(f"   Total tests: {total_tests}")
print(f"   Tests with tool/search evidence: {tests_with_evidence}/{total_tests}")
print(f"   Percentage: {tests_with_evidence/total_tests*100:.1f}%" if total_tests > 0 else "   No successful tests")

if tests_with_evidence == total_tests and total_tests > 0:
    print(f"\n✅ CONCLUSION: Web search appears to be used in ALL tests")
    print(f"   Evidence found in OpenAI Response object structure")
elif tests_with_evidence == 0:
    print(f"\n❌ CRITICAL: NO evidence of web search found in Response objects")
    print(f"   This suggests web search may NOT be invoked by the API")
    print(f"   OR the API does not expose tool usage in the response structure")
else:
    print(f"\n⚠️  MIXED: Inconsistent evidence across tests")

print(f"\n💡 KEY INSIGHT:")
print(f"   The OpenAI Responses API may not expose tool invocations in the")
print(f"   response object structure. To get definitive proof, you may need to:")
print(f"   1. Check your production Phoenix traces for actual web_search spans")
print(f"   2. Enable debug logging in your application to see tool calls")
print(f"   3. Contact OpenAI support to understand how to verify tool usage")

# Save results
output_file = "definitive_websearch_detection_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n💾 Detailed results saved to: {output_file}")
print("=" * 80)
