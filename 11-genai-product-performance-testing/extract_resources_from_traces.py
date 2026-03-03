#!/usr/bin/env python3
"""
Extract actual resources from production Phoenix traces
"""
from phoenix.client import Client
import json
import re
from collections import defaultdict

PHOENIX_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJBcGlLZXk6MTMifQ.GA8Jh3OWNAAWd1hfHPUn4APm-TVMLClH_aeOYNQzm0Y"
PHOENIX_ENDPOINT = "https://phoenix.referral-pilot-dev.navateam.com:6006"

print("Connecting to Phoenix to extract production resources...")
print("=" * 80)

try:
    client = Client(base_url=PHOENIX_ENDPOINT, api_key=PHOENIX_API_KEY)

    # Get recent spans from production project
    spans = client.get_spans_dataframe(
        project_name="pilot-prod",
        limit=100,  # Get more spans to find resource patterns
        filter_condition="span_kind == 'LLM'"  # Focus on LLM calls
    )

    print(f"Retrieved {len(spans)} LLM spans from production")
    print(f"Columns: {list(spans.columns)}")
    print()

    # Look for input/output columns that might contain the prompt
    resource_sets = []
    unique_resources = set()

    for idx, row in spans.iterrows():
        # Try to get input attributes
        if 'attributes.input.value' in spans.columns:
            input_val = row.get('attributes.input.value')
            if input_val:
                print(f"\n{'='*80}")
                print(f"Span {idx} - Input:")
                print(f"{'='*80}")
                print(str(input_val)[:500])  # Print first 500 chars

                # Try to extract resources between "## Resources" and next section
                if isinstance(input_val, str):
                    # Look for resource section
                    resource_match = re.search(
                        r'## Resources\s*\n.*?choose from following list of resources:(.*?)(?:Career Advancement Training|$)',
                        input_val,
                        re.DOTALL
                    )
                    if resource_match:
                        resources_text = resource_match.group(1)
                        print(f"\n{'='*40}")
                        print("FOUND RESOURCES SECTION:")
                        print(f"{'='*40}")
                        print(resources_text[:1000])
                        resource_sets.append(resources_text)

                        # Try to extract individual resource entries
                        # Look for common patterns like "Name:", "Description:", "Address:", etc.
                        resource_entries = re.findall(
                            r'(?:^|\n\n)([A-Z][^\n]+(?:\n(?!\n)[^\n]+)*)',
                            resources_text
                        )
                        unique_resources.update(resource_entries)

        # Also check for llm.input_messages
        if 'attributes.llm.input_messages' in spans.columns:
            messages = row.get('attributes.llm.input_messages')
            if messages:
                try:
                    if isinstance(messages, str):
                        messages = json.loads(messages)
                    if isinstance(messages, list):
                        for msg in messages:
                            if isinstance(msg, dict) and msg.get('role') == 'system':
                                content = msg.get('content', '')
                                if '## Resources' in content:
                                    print(f"\n{'='*80}")
                                    print(f"Span {idx} - System Message with Resources")
                                    print(f"{'='*80}")
                                    # Extract resources section
                                    resource_match = re.search(
                                        r'## Resources.*?In addition to.*?choose from following list of resources:(.*?)(?:Career Advancement Training|Additional resources|$)',
                                        content,
                                        re.DOTALL
                                    )
                                    if resource_match:
                                        resources_text = resource_match.group(1).strip()
                                        if resources_text and len(resources_text) > 50:
                                            print(f"Found {len(resources_text)} chars of resources")
                                            print(resources_text[:2000])
                                            resource_sets.append(resources_text)
                except Exception as e:
                    print(f"Error parsing messages: {e}")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total resource sets found: {len(resource_sets)}")
    print(f"Unique resource entries: {len(unique_resources)}")

    # Save the largest resource set (most comprehensive)
    if resource_sets:
        largest_set = max(resource_sets, key=len)
        print(f"\nLargest resource set: {len(largest_set)} characters")

        with open('extracted_production_resources.txt', 'w') as f:
            f.write("# Extracted from Production Phoenix Traces\n")
            f.write("# These are the actual resources injected into production prompts\n")
            f.write("=" * 80 + "\n\n")
            f.write(largest_set)

        print(f"\n✅ Saved to: extracted_production_resources.txt")
    else:
        print("\n⚠️  No resource sections found in traces")
        print("Trying alternative approach - checking span attributes...")

        # Print available columns to debug
        print("\nAvailable span columns:")
        for col in sorted(spans.columns):
            if not spans[col].isna().all():
                print(f"  - {col}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
