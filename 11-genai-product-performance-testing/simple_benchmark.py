#!/usr/bin/env python3
"""
Simplified benchmark test - bypasses Phoenix and uses direct model calls
"""

import time
import json
import os
from src.common.components import OpenAIWebSearchGenerator, GeminiWebSearchGenerator
from haystack.dataclasses.chat_message import ChatMessage

# API keys should be set as environment variables before running:
# export OPENAI_API_KEY="your-key-here"
# export GOOGLE_API_KEY="your-key-here"

# Test query
test_query = """
You are a case worker assistant. Based on this client description, recommend relevant social support resources.

Client: Single mother with two young children (ages 3 and 5) needs comprehensive support.
She recently lost her job and is struggling with:
- Finding new employment opportunities
- Accessing affordable childcare while job searching
- Getting food assistance for her family
- Finding resources for children's educational development
- Accessing healthcare services
Location: Seattle, WA

Recommend 5-7 resources with:
- Name
- Description
- Contact information (address, phone, email, website)
- Justification for why this resource matches the client's needs

Return ONLY valid JSON in this format:
{
  "resources": [
    {
      "name": "Resource Name",
      "description": "What they provide",
      "addresses": ["Address"],
      "phones": ["Phone"],
      "emails": ["Email"],
      "website": "URL",
      "justification": "Why this helps the client"
    }
  ]
}
"""

print("=" * 80)
print("SIMPLIFIED MODEL BENCHMARK TEST")
print("=" * 80 + "\n")

results = []

# Test ChatGPT 5.1
print("1. Testing ChatGPT 5.1 (no reasoning)...")
try:
    generator = OpenAIWebSearchGenerator()
    messages = [ChatMessage.from_user(test_query)]

    start = time.time()
    response = generator.run(messages=messages, model="gpt-5.1", reasoning_effort="low")
    duration = time.time() - start

    text = response["replies"][0].text
    try:
        data = json.loads(text)
        resource_count = len(data.get("resources", []))
    except:
        resource_count = 0

    results.append({
        "model": "gpt-5.1",
        "duration": duration,
        "resources": resource_count,
        "success": True
    })
    print(f"   ✅ {duration:.2f}s | {resource_count} resources\n")
except Exception as e:
    results.append({"model": "gpt-5.1", "duration": 0, "resources": 0, "success": False, "error": str(e)})
    print(f"   ❌ Error: {e}\n")

# Test Gemini 2.0 Flash (experimental - fastest)
print("2. Testing Gemini 2.0 Flash Experimental...")
try:
    generator = GeminiWebSearchGenerator()
    messages = [ChatMessage.from_user(test_query)]

    start = time.time()
    response = generator.run(messages=messages, model="gemini-2.0-flash-exp")
    duration = time.time() - start

    text = response["replies"][0].text
    try:
        # Try to extract JSON from response
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_text = text[start_idx:end_idx]
            data = json.loads(json_text)
            resource_count = len(data.get("resources", []))
        else:
            resource_count = 0
    except:
        resource_count = 0

    results.append({
        "model": "gemini-2.0-flash-exp",
        "duration": duration,
        "resources": resource_count,
        "success": True
    })
    print(f"   ✅ {duration:.2f}s | {resource_count} resources\n")
except Exception as e:
    results.append({"model": "gemini-2.0-flash-exp", "duration": 0, "resources": 0, "success": False, "error": str(e)})
    print(f"   ❌ Error: {e}\n")

# Test Gemini 1.5 Flash (stable)
print("3. Testing Gemini 1.5 Flash Latest...")
try:
    generator = GeminiWebSearchGenerator()
    messages = [ChatMessage.from_user(test_query)]

    start = time.time()
    response = generator.run(messages=messages, model="gemini-1.5-flash-latest")
    duration = time.time() - start

    text = response["replies"][0].text
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_text = text[start_idx:end_idx]
            data = json.loads(json_text)
            resource_count = len(data.get("resources", []))
        else:
            resource_count = 0
    except:
        resource_count = 0

    results.append({
        "model": "gemini-1.5-flash-latest",
        "duration": duration,
        "resources": resource_count,
        "success": True
    })
    print(f"   ✅ {duration:.2f}s | {resource_count} resources\n")
except Exception as e:
    results.append({"model": "gemini-1.5-flash-latest", "duration": 0, "resources": 0, "success": False, "error": str(e)})
    print(f"   ❌ Error: {e}\n")

# Results summary
print("=" * 80)
print("RESULTS SUMMARY")
print("=" * 80 + "\n")

successful = [r for r in results if r["success"]]
if successful:
    successful.sort(key=lambda x: x["duration"])

    for i, result in enumerate(successful, 1):
        print(f"{i}. {result['model']}")
        print(f"   Duration: {result['duration']:.2f}s")
        print(f"   Resources: {result['resources']}")
        print(f"   Speed: {result['resources']/result['duration']:.2f} resources/sec\n")

    print("=" * 80)
    print(f"🏆 WINNER: {successful[0]['model']}")
    print(f"   ({successful[0]['duration']:.2f}s - {(successful[1]['duration']/successful[0]['duration']-1)*100:.1f}% faster than #{2})")
    print("=" * 80)
else:
    print("❌ All models failed")
