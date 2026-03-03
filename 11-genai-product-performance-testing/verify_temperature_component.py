#!/usr/bin/env python3
"""
Quick test to verify if the new temperature parameter works with the updated component
"""

import os
from haystack.dataclasses.chat_message import ChatMessage
from src.common.components import OpenAIWebSearchGenerator

# Test query
test_query = """You are a case worker assistant in Central Texas. Recommend 5-7 relevant social support resources in Austin, TX.

Client: Single mother with two young children (ages 3 and 5) needs:
- Employment opportunities
- Affordable childcare while job searching
- Food assistance

Return valid JSON: {"resources": [{"name": "...", "description": "..."}]}"""

print("=" * 80)
print("Testing OpenAIWebSearchGenerator with temperature parameter")
print("=" * 80)

generator = OpenAIWebSearchGenerator()
message = ChatMessage.from_user(test_query)

# Test different temperature values
temperatures_to_test = [1.0, 0.5, 0.0]

for temp in temperatures_to_test:
    print(f"\nTesting temperature={temp}...")
    try:
        result = generator.run(
            messages=[message],
            model="gpt-5.1",
            reasoning_effort="low",
            temperature=temp
        )
        print(f"✅ SUCCESS! Temperature {temp} was accepted by the API")
        print(f"   Response length: {len(result['replies'][0].text)} characters")
    except Exception as e:
        error_msg = str(e)
        if "temperature" in error_msg.lower() or "unsupported" in error_msg.lower():
            print(f"❌ REJECTED: {error_msg[:100]}")
        else:
            print(f"⚠️  Other error: {error_msg[:100]}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("If temperature 0.5 and 0.0 were rejected, the API still doesn't support it")
print("If they were accepted, the component now enables temperature control!")
print("=" * 80)
