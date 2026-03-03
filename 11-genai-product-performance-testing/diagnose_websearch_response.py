#!/usr/bin/env python3
"""
Diagnose why web search detection is failing
Examine raw API response structure to understand web search invocation
"""
import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Use a query that DID trigger web search in production
TEST_QUERY = "household goods"

# Try different system prompts
PROMPTS = {
    "simplified": """You are helping social services case workers in Central Texas find resources for their clients.

Based on the client's request, identify relevant social service resources.

Use web search when:
- You need current program availability
- You need up-to-date contact information
- You need to verify if a program still exists
- The query asks about specific current programs or schedules

Do NOT use web search when:
- The query is about general resource categories
- You have sufficient knowledge about well-established resources
- The information doesn't change frequently
""",

    "minimal": "You help find social service resources in Central Texas. Use web search when you need current information.",

    "none": "Find social service resources for this query."
}

print("="*80)
print("WEB SEARCH INVOCATION DIAGNOSTIC")
print("="*80)
print(f"Test query: '{TEST_QUERY}'")
print(f"Note: This query triggered web search in production\n")

for prompt_name, system_prompt in PROMPTS.items():
    print(f"\n{'='*80}")
    print(f"TESTING WITH: {prompt_name} prompt")
    print(f"{'='*80}")

    try:
        response = client.responses.create(
            model="gpt-5.1",
            reasoning={"effort": "none"},
            temperature=0.9,
            tools=[{"type": "web_search"}],
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": TEST_QUERY}
            ]
        )

        print(f"\n✅ API call succeeded")
        print(f"\nResponse type: {type(response)}")
        print(f"Response class: {response.__class__.__name__}")

        # List all attributes
        print(f"\n📋 All response attributes:")
        for attr in dir(response):
            if not attr.startswith('_'):
                try:
                    value = getattr(response, attr)
                    if not callable(value):
                        value_str = str(value)[:200]
                        print(f"  {attr}: {value_str}")
                except Exception as e:
                    print(f"  {attr}: <error accessing: {e}>")

        # Check specifically for web search indicators
        print(f"\n🔍 Web search detection:")

        # Check various possible attributes
        checks = [
            'web_search_queries',
            'web_search',
            'tool_calls',
            'tools',
            'searches',
            'search_queries'
        ]

        for check in checks:
            if hasattr(response, check):
                value = getattr(response, check)
                print(f"  ✅ Found '{check}': {value}")
            else:
                print(f"  ❌ No '{check}' attribute")

        # Try to convert to dict and examine
        try:
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
            print(f"\n📄 Response as dict (keys): {list(response_dict.keys())}")

            # Save full response for inspection
            with open(f'response_{prompt_name}.json', 'w') as f:
                json.dump(response_dict, f, indent=2, default=str)
            print(f"   Saved full response to: response_{prompt_name}.json")

        except Exception as e:
            print(f"\n⚠️  Could not convert response to dict: {e}")

        # Check response content/output
        if hasattr(response, 'output'):
            output = response.output
            print(f"\nResponse output preview: {str(output)[:300]}...")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*80}")
print("DIAGNOSTIC COMPLETE")
print(f"{'='*80}")
print("\nCheck the generated response_*.json files for full API response structure")
