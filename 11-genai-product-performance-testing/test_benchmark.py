#!/usr/bin/env python3
"""
Quick benchmark test script to compare ChatGPT 5.1 vs Gemini models.
"""

import requests
import json

API_URL = "http://localhost:3000/benchmark_models/run"

# Test query - a realistic client description
test_query = """
Single mother with two young children (ages 3 and 5) needs comprehensive support.
She recently lost her job and is struggling with:
- Finding new employment opportunities
- Accessing affordable childcare while job searching
- Getting food assistance for her family
- Finding resources for children's educational development
- Accessing healthcare services
Location: Seattle, WA
"""

print("=" * 80)
print("MODEL BENCHMARK TEST")
print("=" * 80)
print(f"\nTest Query: {test_query[:100]}...\n")
print("Testing models:")
print("  1. gpt-5.1 (ChatGPT 5.1 - no reasoning)")
print("  2. gemini-1.5-flash (Gemini Flash)")
print("  3. gemini-1.5-flash-8b (Gemini Flash 8B - fastest)")
print("\n" + "=" * 80)
print("Starting benchmark... (this may take 1-2 minutes)")
print("=" * 80 + "\n")

try:
    response = requests.post(
        API_URL,
        json={
            "query": test_query,
            "user_email": "test@example.com",
            "models_to_test": [
                "gpt-5.1",
                "gemini-1.5-flash",
                "gemini-1.5-flash-8b"
            ]
        },
        timeout=300  # 5 minutes timeout
    )

    if response.status_code == 200:
        result = response.json()

        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80 + "\n")

        results = result.get("result", {}).get("results", [])
        winner = result.get("result", {}).get("winner")

        for i, model_result in enumerate(results, 1):
            model_name = model_result["model_name"]
            duration = model_result["duration_seconds"]
            resources = model_result["resources_count"]
            success = model_result["success"]
            error = model_result.get("error")

            print(f"{i}. {model_name}")
            print(f"   Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
            if success:
                print(f"   Duration: {duration:.2f} seconds")
                print(f"   Resources Found: {resources}")
                print(f"   Speed: {resources/duration:.2f} resources/second")
            else:
                print(f"   Error: {error}")
            print()

        print("=" * 80)
        print(f"🏆 WINNER: {winner}")
        print("=" * 80)

    else:
        print(f"❌ Error: API returned status code {response.status_code}")
        print(f"Response: {response.text}")

except requests.exceptions.Timeout:
    print("❌ Request timed out. The models might be taking too long to respond.")
except requests.exceptions.ConnectionError:
    print("❌ Could not connect to the API. Make sure Docker services are running:")
    print("   cd app && docker-compose ps")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
