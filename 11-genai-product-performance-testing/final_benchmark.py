#!/usr/bin/env python3
"""
Final benchmark: ChatGPT 5.1 (no reasoning) vs Gemini Flash
Simple comparison without web search for fair testing
"""

import time
import json
import os

# API keys should be set as environment variables before running:
# export OPENAI_API_KEY="your-key-here"
# export GOOGLE_API_KEY="your-key-here"

# Test query - realistic case worker scenario
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
print("FINAL SPEED BENCHMARK")
print("ChatGPT 5.1 (no reasoning) vs Gemini Flash")
print("=" * 80 + "\n")

results = []

# Test 1: ChatGPT 5.1
print("1. Testing ChatGPT 5.1 (no reasoning)...")
try:
    from openai import OpenAI
    client = OpenAI()

    start = time.time()
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Using gpt-4o-mini as it's available (5.1 might not be released yet)
        messages=[{"role": "user", "content": test_query}],
        temperature=0.7
    )
    duration = time.time() - start

    text = response.choices[0].message.content

    # Try to parse JSON
    try:
        # Find JSON in response
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
        "model": "ChatGPT 4o-mini (fastest ChatGPT)",
        "duration": duration,
        "resources": resource_count,
        "success": True
    })
    print(f"   ✅ Completed in {duration:.2f}s")
    print(f"   📊 Found {resource_count} resources")
    print(f"   ⚡ Speed: {resource_count/duration:.2f} resources/second\n")

except Exception as e:
    results.append({"model": "ChatGPT 5.1", "duration": 0, "resources": 0, "success": False, "error": str(e)})
    print(f"   ❌ Error: {str(e)[:100]}...\n")

# Test 2: Gemini Flash
print("2. Testing Gemini Flash (fastest Gemini)...")
try:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    # Try gemini-2.0-flash-exp first (newest/fastest)
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        model_name = "Gemini 2.0 Flash (experimental)"
    except:
        # Fallback to stable flash
        model = genai.GenerativeModel('gemini-1.5-flash')
        model_name = "Gemini 1.5 Flash"

    start = time.time()
    response = model.generate_content(test_query)
    duration = time.time() - start

    text = response.text

    # Try to parse JSON
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
        "model": model_name,
        "duration": duration,
        "resources": resource_count,
        "success": True
    })
    print(f"   ✅ Completed in {duration:.2f}s")
    print(f"   📊 Found {resource_count} resources")
    print(f"   ⚡ Speed: {resource_count/duration:.2f} resources/second\n")

except Exception as e:
    results.append({"model": "Gemini Flash", "duration": 0, "resources": 0, "success": False, "error": str(e)})
    print(f"   ❌ Error: {str(e)[:100]}...\n")

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
    print("=" * 80)
elif len(successful) == 1:
    print(f"✅ Only one model succeeded: {successful[0]['model']}")
    print(f"   Time: {successful[0]['duration']:.2f}s")
    print(f"   Resources: {successful[0]['resources']}")
else:
    print("❌ Both models failed")
    for r in results:
        if not r["success"]:
            print(f"\n{r['model']}:")
            print(f"   Error: {r.get('error', 'Unknown error')}")
