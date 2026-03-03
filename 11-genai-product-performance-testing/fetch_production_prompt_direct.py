#!/usr/bin/env python3
"""
Fetch actual production prompt from Phoenix API using direct HTTP request
"""
import os
import httpx
import json

PHOENIX_COLLECTOR_ENDPOINT = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "https://phoenix.referral-pilot-dev.navateam.com:6006")
PHOENIX_API_KEY = os.environ.get("PHOENIX_API_KEY")
PHOENIX_PROJECT_NAME = os.environ.get("PHOENIX_PROJECT_NAME", "pilot-prod")

print("Fetching production prompt for 'generate_referrals' from Phoenix...")
print(f"Endpoint: {PHOENIX_COLLECTOR_ENDPOINT}")
print(f"Project: {PHOENIX_PROJECT_NAME}")
print("="*80)

# Phoenix GraphQL query to get prompt template
query = """
query GetPrompt($projectName: String!, $promptName: String!) {
  project(name: $projectName) {
    prompts(names: [$promptName]) {
      edges {
        node {
          name
          latestVersion {
            template
            version
          }
        }
      }
    }
  }
}
"""

variables = {
    "projectName": PHOENIX_PROJECT_NAME,
    "promptName": "generate_referrals"
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {PHOENIX_API_KEY}"
}

try:
    response = httpx.post(
        f"{PHOENIX_COLLECTOR_ENDPOINT}/graphql",
        json={"query": query, "variables": variables},
        headers=headers,
        timeout=30.0
    )
    response.raise_for_status()

    data = response.json()

    if "errors" in data:
        print(f"❌ GraphQL errors: {data['errors']}")
    else:
        prompt_edges = data.get("data", {}).get("project", {}).get("prompts", {}).get("edges", [])

        if not prompt_edges:
            print("❌ No prompt found with name 'generate_referrals'")
        else:
            prompt_node = prompt_edges[0]["node"]
            prompt_name = prompt_node["name"]
            latest_version = prompt_node["latestVersion"]
            version = latest_version["version"]
            template = latest_version["template"]

            print(f"\nPrompt: {prompt_name}")
            print(f"Version: {version}\n")
            print("="*80)
            print("TEMPLATE:")
            print("="*80)
            print(json.dumps(template, indent=2))
            print("="*80)

            # Save to file
            with open('production_prompt_template.json', 'w') as f:
                json.dump(template, f, indent=2)

            # Also save human-readable version
            with open('production_prompt.txt', 'w') as f:
                f.write(f"Prompt: {prompt_name}\n")
                f.write(f"Version: {version}\n\n")
                f.write("="*80 + "\n")

                if "messages" in template:
                    for msg in template["messages"]:
                        role = msg.get("role", "unknown")
                        content = msg.get("content", [])

                        f.write(f"\n=== {role.upper()} ===\n")
                        for item in content:
                            if item.get("type") == "text":
                                f.write(item.get("text", ""))
                        f.write("\n\n")
                else:
                    f.write(json.dumps(template, indent=2))

            print(f"\n✅ Production prompt saved to:")
            print(f"   - production_prompt_template.json (raw JSON)")
            print(f"   - production_prompt.txt (human-readable)")

except httpx.HTTPStatusError as e:
    print(f"❌ HTTP error: {e}")
    print(f"Response: {e.response.text}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
