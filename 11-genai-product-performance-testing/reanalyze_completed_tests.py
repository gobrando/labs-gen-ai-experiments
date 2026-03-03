#!/usr/bin/env python3
"""
Re-analyze the completed corrected_temperature_test and corrected_reasoning_test
to extract actual web search invocation data that was missed by wrong detection code

Since the tests saved individual query result files, we can parse those
"""
import os
import json
import glob

print("="*80)
print("RE-ANALYZING COMPLETED TESTS WITH CORRECTED DETECTION")
print("="*80)

# The actual results would be in the raw output, but those aren't saved per-query
# We'd need to re-run the tests OR parse the output text files

# Let's check what files we have
print("\nLooking for saved result files...")

test_dirs = [
    '/Users/brandonworks/projects/labs-referral-pilot/app'
]

# Check for any saved JSON files from the tests
for test_dir in test_dirs:
    json_files = glob.glob(os.path.join(test_dir, 'corrected_*_results.json'))
    for f in json_files:
        print(f"  Found: {f}")
        with open(f, 'r') as file:
            data = json.load(file)
            print(f"    Keys: {list(data.keys())}")

# Check output text files
output_files = [
    'corrected_temperature_test_output.txt',
    'corrected_reasoning_test_output.txt'
]

for output_file in output_files:
    path = os.path.join('/Users/brandonworks/projects/labs-referral-pilot/app', output_file)
    if os.path.exists(path):
        print(f"\n{'='*80}")
        print(f"FILE: {output_file}")
        print(f"{'='*80}")
        with open(path, 'r') as f:
            lines = f.readlines()
            print(f"Total lines: {len(lines)}")

            # The tests logged "YES" or "NO" for each query - but with WRONG detection!
            # The format was: "   1.    NO |  20.04s | household goods"

            # Count the pattern
            yes_count = sum(1 for line in lines if '    YES |' in line)
            no_count = sum(1 for line in lines if '    NO |' in line)

            print(f"Lines with 'YES |': {yes_count}")
            print(f"Lines with 'NO |': {no_count}")

            # Extract summary section
            print("\nSummary section:")
            in_summary = False
            for line in lines:
                if 'SUMMARY' in line or 'RESULTS' in line:
                    in_summary = True
                if in_summary:
                    print(line.rstrip())

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
The completed tests likely DID invoke web search, but the wrong detection code
reported 0% web search.

To get accurate results, we have two options:
1. Re-run the full temperature and reasoning tests with corrected detection
2. Examine saved API response objects if they were cached

Since the tests take ~15-20 minutes each and we've confirmed the detection fix works,
we should RE-RUN both tests with the corrected detection code.
""")
