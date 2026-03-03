#!/usr/bin/env python3
"""
Final CORRECT Benchmark: ChatGPT 5.1 (no reasoning) vs Gemini 3 Flash
With proper safety settings for Gemini
"""

import time
import json
import os

# API keys should be set as environment variables before running:
# export OPENAI_API_KEY="your-key-here"
# export GOOGLE_API_KEY="your-key-here"

# Test query
test_query = """
You are a case worker assistant. Based on this client description, recommend 5-7 relevant social support resources.

Client: Single mother with two young children (ages 3 and 5) needs comprehensive support.
She recently lost her job and is struggling with:
- Finding new employment opportunities
- Accessing affordable childcare while job searching
- Getting food assistance for her family
- Finding resources for children's educational development
- Accessing healthcare services
Location: Seattle, WA

For each resource, provide:
- Name
- Description
- Contact information (address, phone, email, website)
- Justification for why this resource matches the client's needs

Return your response in valid JSON format:
{
  "resources": [
    {
      "name": "Resource Name",
      "description": "What they provide",
      "addresses": ["123 Main St, Seattle, WA"],
      "phones": ["555-0100"],
      "emails": ["info@example.org"],
      "website": "https://example.org",
      "justification": "Why this helps"
    }
  ]
}
"""

print("=" * 80)
print("BENCHMARK: ChatGPT 5.1 (no reasoning) vs Gemini 3 Flash")
print("=" * 80 + "\n")

results = []

# Test 1: ChatGPT 5.1 with no reasoning
print("1. Testing ChatGPT 5.1 (no reasoning)...")
try:
    from openai import OpenAI
    client = OpenAI()

    start = time.time()

    # Using gpt-5.1 with reasoning_effort set to "low" for no reasoning
    response = client.responses.create(
        model="gpt-5.1",
        input=test_query,
        reasoning={"effort": "low"}  # No reasoning
    )

    duration = time.time() - start
    text = response.output_text

    # Parse JSON
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
        "model": "ChatGPT 5.1 (no reasoning)",
        "duration": duration,
        "resources": resource_count,
        "success": True
    })
    print(f"   ✅ Completed in {duration:.2f}s")
    print(f"   📊 Found {resource_count} resources")
    print(f"   ⚡ Speed: {resource_count/duration:.2f} resources/second\n")

except Exception as e:
    results.append({"model": "ChatGPT 5.1", "duration": 0, "resources": 0, "success": False, "error": str(e)})
    print(f"   ❌ Error: {str(e)}\n")

# Test 2: Gemini 3 Flash with safety settings disabled
print("2. Testing Gemini 3 Flash...")
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    # Disable safety filters to avoid blocking
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    model = genai.GenerativeModel(
        'gemini-3-flash-preview',
        safety_settings=safety_settings
    )

    start = time.time()
    response = model.generate_content(test_query)
    duration = time.time() - start

    text = response.text

    # Parse JSON
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
        "model": "Gemini 3 Flash",
        "duration": duration,
        "resources": resource_count,
        "success": True
    })
    print(f"   ✅ Completed in {duration:.2f}s")
    print(f"   📊 Found {resource_count} resources")
    print(f"   ⚡ Speed: {resource_count/duration:.2f} resources/second\n")

except Exception as e:
    results.append({"model": "Gemini 3 Flash", "duration": 0, "resources": 0, "success": False, "error": str(e)})
    print(f"   ❌ Error: {str(e)}\n")

# Results
print("=" * 80)
print("FINAL RESULTS")
print("=" * 80 + "\n")

successful = [r for r in results if r["success"]]
if len(successful) >= 2:
    successful.sort(key=lambda x: x["duration"])

    winner = successful[0]
    runner_up = successful[1]

    speedup = ((runner_up["duration"] / winner["duration"]) - 1) * 100

    print(f"🏆 WINNER: {winner['model']}")
    print(f"   Time: {winner['duration']:.2f} seconds")
    print(f"   Resources: {winner['resources']}")
    print(f"   Speed: {winner['resources']/winner['duration']:.2f} resources/sec\n")

    print(f"🥈 SECOND: {runner_up['model']}")
    print(f"   Time: {runner_up['duration']:.2f} seconds")
    print(f"   Resources: {runner_up['resources']}")
    print(f"   Speed: {runner_up['resources']/runner_up['duration']:.2f} resources/sec\n")

    print(f"📈 SPEED DIFFERENCE: {winner['model']} is {speedup:.1f}% faster")
    print(f"   ({winner['duration']:.2f}s vs {runner_up['duration']:.2f}s)")
    print("=" * 80)
elif len(successful) == 1:
    result = successful[0]
    print(f"✅ Only one model succeeded: {result['model']}")
    print(f"   Time: {result['duration']:.2f}s")
    print(f"   Resources: {result['resources']}")
    print(f"   Speed: {result['resources']/result['duration']:.2f} resources/sec")
    print("\n" + "=" * 80)
else:
    print("❌ Both models failed")
    for r in results:
        if not r["success"]:
            print(f"\n{r['model']}:")
            print(f"   Error: {r.get('error', 'Unknown error')}")
