#!/usr/bin/env python3
"""
Fetch actual production prompt from Phoenix API
"""
import os
import sys
sys.path.insert(0, 'src')

from src.common import phoenix_utils, haystack_utils

# Fetch the production prompt
print("Fetching production prompt for 'generate_referrals' from Phoenix...")
print("="*80)

try:
    # Get prompt from Phoenix (same way production does it)
    prompt_messages = haystack_utils.get_phoenix_prompt("generate_referrals", "")

    print(f"Retrieved {len(prompt_messages)} messages\n")

    for i, msg in enumerate(prompt_messages, 1):
        print(f"\nMessage {i}:")
        print(f"  Role: {msg.role}")
        print(f"  Content length: {len(msg.text)} characters")
        print(f"\n{'-'*80}")
        print(msg.text)
        print(f"{'-'*80}\n")

    # Save to file for analysis
    with open('production_prompt.txt', 'w') as f:
        for msg in prompt_messages:
            f.write(f"=== {msg.role.upper()} ===\n")
            f.write(msg.text)
            f.write("\n\n")

    print("\n✅ Production prompt saved to: production_prompt.txt")

except Exception as e:
    print(f"❌ Error fetching prompt: {e}")
    import traceback
    traceback.print_exc()
