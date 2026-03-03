#!/usr/bin/env python3
"""
Extract full system prompt with resources from production Phoenix traces
"""
import os
import httpx
import json
import re

PHOENIX_URL = "https://phoenix.referral-pilot-dev.navateam.com:6006"
PHOENIX_PROJECT_NAME = "pilot-prod"
PHOENIX_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJBcGlLZXk6MTMifQ.GA8Jh3OWNAAWd1hfHPUn4APm-TVMLClH_aeOYNQzm0Y"

headers = {"Authorization": f"Bearer {PHOENIX_API_KEY}"}

print("="*80)
print("EXTRACTING RESOURCES FROM PRODUCTION TRACES")
print("="*80)
print(f"Phoenix URL: {PHOENIX_URL}")
print(f"Project: {PHOENIX_PROJECT_NAME}")
print("="*80)

# Fetch spans (limit to recent ones)
print("\nFetching LLM spans...")
all_spans = []
cursor = None
max_pages = 5  # Limit to avoid too much data

for page in range(max_pages):
    url = f"{PHOENIX_URL}/v1/projects/{PHOENIX_PROJECT_NAME}/spans"
    if cursor:
        url += f"?cursor={cursor}"

    try:
        response = httpx.get(url, headers=headers, verify=False, timeout=60.0)
        data = response.json()
        spans = data.get('data', [])

        all_spans.extend(spans)
        print(f"  Page {page+1}: {len(spans)} spans (total: {len(all_spans)})")

        cursor = data.get('next_cursor')
        if not cursor or len(spans) == 0:
            break
    except Exception as e:
        print(f"  Error fetching page {page+1}: {e}")
        break

print(f"\n✅ Total spans fetched: {len(all_spans)}\n")

# Look for spans with LLM input messages
resource_texts = []
largest_resource_section = ""

for idx, span in enumerate(all_spans):
    span_name = span.get('name', '')
    attrs = span.get('attributes', {})

    # Check for LLM input messages
    llm_input = attrs.get('llm.input_messages')

    if llm_input:
        try:
            # Parse if it's a JSON string
            if isinstance(llm_input, str):
                messages = json.loads(llm_input)
            else:
                messages = llm_input

            if isinstance(messages, list):
                for msg in messages:
                    if not isinstance(msg, dict):
                        continue

                    role = msg.get('role', '')
                    content = msg.get('content', '')

                    # Look for system messages with resources
                    if role == 'system' and isinstance(content, str) and '## Resources' in content:
                        print(f"\n{'='*80}")
                        print(f"FOUND SYSTEM MESSAGE WITH RESOURCES (Span: {span_name})")
                        print(f"{'='*80}")

                        # Try to extract the resources section
                        # Look for text between "## Resources" and "Career Advancement Training"
                        match = re.search(
                            r'## Resources\s*\nIn addition to.*?choose from following list of resources:\s*(.*?)(?:\n\nCareer Advancement Training|$)',
                            content,
                            re.DOTALL
                        )

                        if match:
                            resources = match.group(1).strip()
                            print(f"Extracted {len(resources)} characters of resources")
                            print(f"\nFirst 1000 characters:")
                            print(resources[:1000])

                            if len(resources) > len(largest_resource_section):
                                largest_resource_section = resources

                            resource_texts.append({
                                'span_name': span_name,
                                'resources': resources,
                                'full_system_message': content
                            })
                        else:
                            # Maybe the whole prompt is short, just save it
                            if len(content) > 5000:  # Substantial prompt
                                print(f"Saving full system message ({len(content)} chars)")
                                resource_texts.append({
                                    'span_name': span_name,
                                    'resources': "FULL_PROMPT",
                                    'full_system_message': content
                                })

        except json.JSONDecodeError as e:
            pass  # Skip if can't parse
        except Exception as e:
            print(f"  Error processing span {idx}: {e}")

print(f"\n{'='*80}")
print("EXTRACTION RESULTS")
print(f"{'='*80}")
print(f"Found {len(resource_texts)} system messages with resources")

if largest_resource_section:
    print(f"Largest resource section: {len(largest_resource_section)} characters")

if resource_texts:
    # Save the one with the most resources
    best = max(resource_texts, key=lambda x: len(x['full_system_message']))

    print(f"\nSaving largest prompt: {len(best['full_system_message'])} characters")

    with open('extracted_production_resources.txt', 'w') as f:
        f.write("# Extracted from Production Phoenix Trace\n")
        f.write(f"# Span: {best['span_name']}\n")
        f.write(f"# Full system message length: {len(best['full_system_message'])} chars\n")
        f.write("=" * 80 + "\n\n")
        f.write(best['full_system_message'])

    print(f"✅ Saved to: extracted_production_resources.txt")

    # Also save just the resources section if we found one
    if best['resources'] != "FULL_PROMPT":
        with open('extracted_resources_only.txt', 'w') as f:
            f.write("# Just the Resources Section\n")
            f.write("=" * 80 + "\n\n")
            f.write(best['resources'])
        print(f"✅ Also saved resources-only to: extracted_resources_only.txt")
else:
    print("\n⚠️  No system messages with resources found in traces")
    print("Note: Traces might not include full prompts, or resources might be in a different format")

print("\n" + "=" * 80)
