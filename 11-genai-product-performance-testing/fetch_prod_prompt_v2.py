#!/usr/bin/env python3
"""
Fetch actual production prompt from Phoenix API
"""
import httpx
import json

PHOENIX_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJBcGlLZXk6MTMifQ.GA8Jh3OWNAAWd1hfHPUn4APm-TVMLClH_aeOYNQzm0Y"
PHOENIX_ENDPOINT = "https://phoenix.referral-pilot-dev.navateam.com:6006"
PROJECT_NAME = "pilot-prod"

print("Fetching production prompt from Phoenix...")
print(f"Endpoint: {PHOENIX_ENDPOINT}")
print(f"Project: {PROJECT_NAME}")
print("="*80)

# Try REST API endpoint
url = f"{PHOENIX_ENDPOINT}/v1/prompts"

headers = {
    "Authorization": f"Bearer {PHOENIX_API_KEY}",
    "Content-Type": "application/json"
}

try:
    # List all prompts first
    response = httpx.get(url, headers=headers, timeout=30.0)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        prompts = response.json()
        print(f"\nFound {len(prompts)} prompts:")
        for p in prompts:
            print(f"  - {p.get('name', 'unknown')}")

        # Find generate_referrals
        for prompt in prompts:
            if prompt.get('name') == 'generate_referrals':
                print(f"\n{'='*80}")
                print("GENERATE_REFERRALS PROMPT")
                print(f"{'='*80}")
                print(json.dumps(prompt, indent=2))

                # Save to file
                with open('production_prompt_full.json', 'w') as f:
                    json.dump(prompt, f, indent=2)

                # Extract and save human-readable version
                if 'template' in prompt:
                    template = prompt['template']
                    with open('production_prompt.txt', 'w') as f:
                        f.write("="*80 + "\n")
                        f.write("PRODUCTION PROMPT: generate_referrals\n")
                        f.write("="*80 + "\n\n")

                        if isinstance(template, dict) and 'messages' in template:
                            for msg in template['messages']:
                                role = msg.get('role', 'unknown')
                                content = msg.get('content', [])

                                f.write(f"\n{'='*80}\n")
                                f.write(f"{role.upper()}\n")
                                f.write(f"{'='*80}\n")

                                for item in content:
                                    if isinstance(item, dict) and item.get('type') == 'text':
                                        f.write(item.get('text', ''))
                                        f.write("\n")
                        else:
                            f.write(json.dumps(template, indent=2))

                print(f"\n✅ Saved to:")
                print("  - production_prompt_full.json")
                print("  - production_prompt.txt")
                break
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
