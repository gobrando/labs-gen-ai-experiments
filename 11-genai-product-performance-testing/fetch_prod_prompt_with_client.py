#!/usr/bin/env python3
"""
Fetch production prompt using Phoenix Python Client SDK
"""
from phoenix.client import Client
import json
import os

PHOENIX_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJBcGlLZXk6MTMifQ.GA8Jh3OWNAAWd1hfHPUn4APm-TVMLClH_aeOYNQzm0Y"
PHOENIX_ENDPOINT = "https://phoenix.referral-pilot-dev.navateam.com:6006"
PROMPT_VERSION_ID = "UHJvbXB0VmVyc2lvbjo0OA=="  # From app_config.py

print("Fetching production prompt from Phoenix...")
print(f"Endpoint: {PHOENIX_ENDPOINT}")
print(f"Prompt Version ID: {PROMPT_VERSION_ID}")
print("="*80)

try:
    # Create Phoenix client
    client = Client(base_url=PHOENIX_ENDPOINT, api_key=PHOENIX_API_KEY)

    # Fetch the specific prompt version
    prompt_version = client.prompts.get(prompt_version_id=PROMPT_VERSION_ID)

    print(f"\nPrompt ID: {prompt_version.id}")
    print(f"Description: {prompt_version._description}")
    print(f"\n{'='*80}")
    print("TEMPLATE:")
    print(f"{'='*80}\n")

    template = prompt_version._template
    print(json.dumps(template, indent=2))

    # Save to files
    with open('production_prompt_full_v48.json', 'w') as f:
        json.dump({
            'id': prompt_version.id,
            'description': prompt_version._description,
            'template': template
        }, f, indent=2, default=str)

    # Extract human-readable version
    with open('production_prompt_v48.txt', 'w') as f:
        f.write(f"Production Prompt Version 48\n")
        f.write(f"ID: {prompt_version.id}\n")
        f.write(f"Description: {prompt_version._description}\n")
        f.write("="*80 + "\n\n")

        if 'messages' in template:
            for msg in template['messages']:
                role = msg.get('role', 'unknown')
                content = msg.get('content', [])

                f.write(f"\n{'='*80}\n")
                f.write(f"{role.upper()}\n")
                f.write(f"{'='*80}\n\n")

                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        f.write(item.get('text', ''))
                        f.write("\n")
        else:
            f.write(json.dumps(template, indent=2))

    print(f"\n{'='*80}")
    print("✅ Production prompt saved to:")
    print("   - production_prompt_full_v48.json (full JSON)")
    print("   - production_prompt_v48.txt (human-readable)")
    print(f"{'='*80}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
