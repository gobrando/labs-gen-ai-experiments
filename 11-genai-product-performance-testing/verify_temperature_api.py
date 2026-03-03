#!/usr/bin/env python3
"""
Direct API test to verify if temperature parameter works with OpenAI Responses API
Testing with the new component code's approach
"""

import os
from openai import OpenAI

test_query = """You are a case worker assistant. Recommend 3 social support resources in Austin, TX.

Client: Single mother needs employment and childcare.

Return valid JSON: {"resources": [{"name": "..."}]}"""

print("=" * 80)
print("Testing OpenAI Responses API with temperature parameter")
print("Model: gpt-5.1 (no reasoning)")
print("=" * 80)

client = OpenAI()

# Test different temperature values
temperatures_to_test = [1.0, 0.5, 0.0]

for temp in temperatures_to_test:
    print(f"\nTesting temperature={temp}...")
    try:
        api_params = {
            "model": "gpt-5.1",
            "input": test_query,
            "reasoning": {"effort": "low"},
            "tools": [{"type": "web_search"}],
            "temperature": temp
        }

        response = client.responses.create(**api_params)

        print(f"✅ SUCCESS! Temperature {temp} was accepted by the API")
        print(f"   Response length: {len(response.output_text)} characters")

    except Exception as e:
        error_msg = str(e)
        if "temperature" in error_msg.lower() or "unsupported" in error_msg.lower():
            print(f"❌ REJECTED: Temperature not supported")
            print(f"   Error: {error_msg[:150]}")
        else:
            print(f"⚠️  Other error: {error_msg[:150]}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
print("If temperature 0.5 and 0.0 were ACCEPTED:")
print("  → The API now supports temperature! Your engineer's code will work!")
print("\nIf temperature 0.5 and 0.0 were REJECTED:")
print("  → The API still doesn't support it. The code can pass it, but API rejects it.")
print("=" * 80)
