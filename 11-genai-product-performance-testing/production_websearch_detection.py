#!/usr/bin/env python3
"""
Production web search detection following the team's documented approach.

This script:
1. Queries the Phoenix API to fetch recent spans
2. Groups spans by trace_id
3. Looks for OpenAIWebSearchGenerator.run and web_search_call spans
4. Classifies each trace as: YES, NO, DISTANCE_ONLY, N/A, or UNKNOWN

This matches the approach documented in web_search_detection.md
"""

import os
import json
import time
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from typing import List, Dict
import httpx

# Phoenix configuration from environment or defaults
PHOENIX_URL = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "https://phoenix:6006")
PHOENIX_PROJECT_NAME = os.environ.get("PHOENIX_PROJECT_NAME", "default")
PHOENIX_API_KEY = os.environ.get("PHOENIX_API_KEY", "")

print("=" * 80)
print("PRODUCTION WEB SEARCH DETECTION")
print("=" * 80)
print(f"Phoenix URL: {PHOENIX_URL}")
print(f"Project Name: {PHOENIX_PROJECT_NAME}")
print(f"API Key configured: {'Yes' if PHOENIX_API_KEY else 'No'}")
print("=" * 80)


def detect_web_search(trace_spans: List[Dict]) -> str:
    """Detect whether web search was used in a trace.

    Returns: YES, NO, DISTANCE_ONLY, N/A, or UNKNOWN.

    This follows the logic documented in web_search_detection.md
    """
    has_generator = False
    web_search_calls = []

    for span in trace_spans:
        name = span.get('name', '')
        if name == 'OpenAIWebSearchGenerator.run':
            has_generator = True
        elif name == 'web_search_call':
            web_search_calls.append(span)

    if not has_generator and not web_search_calls:
        return 'N/A'

    if not web_search_calls:
        # Generator ran but LLM chose not to search
        return 'NO'

    # Classify web_search_call spans
    has_real_search = False
    has_distance = False

    for span in web_search_calls:
        attrs = span.get('attributes', {})
        # Check two attribute naming conventions (Phoenix version differences)
        action_type = (attrs.get('action_type', '')
                       or attrs.get('tool.parameters.action_type', ''))
        query = str(attrs.get('query', '')
                    or attrs.get('tool.parameters.query', ''))
        source_urls = (attrs.get('source_urls', '')
                       or attrs.get('tool.parameters.source_urls', ''))

        if action_type == 'search' and source_urls:
            has_real_search = True
        elif query.startswith('calculator:'):
            calc_rest = query.split(':', 1)[1].strip()
            if 'distance' in calc_rest:
                has_distance = True
            # calculator: 0, calculator: 1, calculator: 1+1 → no-op
        else:
            # Real query text (even without source_urls) = search attempted
            if query and not query.startswith('calculator'):
                has_real_search = True

    if has_real_search:
        return 'YES'
    if has_distance:
        return 'DISTANCE_ONLY'
    return 'NO'


def fetch_phoenix_spans(days_back: int = 2) -> List[Dict]:
    """Fetch recent spans from Phoenix, paginating through results."""
    headers = {}
    if PHOENIX_API_KEY:
        headers["Authorization"] = f"Bearer {PHOENIX_API_KEY}"

    all_spans = []
    cursor = None
    start_date = datetime.now(timezone.utc) - timedelta(days=days_back)

    print(f"\nFetching spans from {start_date.isoformat()} onwards...")
    print(f"Using endpoint: {PHOENIX_URL}/v1/projects/{PHOENIX_PROJECT_NAME}/spans")

    page_count = 0
    while True:
        page_count += 1
        url = f"{PHOENIX_URL}/v1/projects/{PHOENIX_PROJECT_NAME}/spans"
        if cursor:
            url += f"?cursor={cursor}"

        print(f"  Fetching page {page_count}...", end=" ")

        try:
            response = httpx.get(url, headers=headers, verify=False, timeout=60.0)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            print(f"\n❌ HTTP Error: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"   Response: {e.response.text[:500]}")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            break

        spans = data.get('data', [])
        print(f"got {len(spans)} spans")

        # Filter spans by date
        for span in spans:
            span_time_str = span.get('start_time')
            if span_time_str:
                try:
                    # Parse ISO format timestamp
                    span_date = datetime.fromisoformat(span_time_str.replace('Z', '+00:00'))
                    if span_date >= start_date:
                        all_spans.append(span)
                except (ValueError, AttributeError):
                    # If parsing fails, include the span anyway
                    all_spans.append(span)
            else:
                all_spans.append(span)

        cursor = data.get('next_cursor')
        if not cursor or len(spans) == 0:
            break

    print(f"\n✅ Fetched {len(all_spans)} total spans from last {days_back} days")
    return all_spans


def group_by_trace(spans: List[Dict]) -> Dict[str, List[Dict]]:
    """Group spans by their trace_id."""
    traces = defaultdict(list)
    for span in spans:
        trace_id = span.get('context', {}).get('trace_id', '')
        if trace_id:
            traces[trace_id].append(span)
    return traces


def analyze_traces(traces: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Analyze each trace for web search usage."""
    results = {}

    for trace_id, span_list in traces.items():
        web_search_status = detect_web_search(span_list)

        # Get trace metadata
        trace_start_times = [s.get('start_time') for s in span_list if s.get('start_time')]
        trace_start = min(trace_start_times) if trace_start_times else None

        # Find generator and search call spans
        generator_spans = [s for s in span_list if s.get('name') == 'OpenAIWebSearchGenerator.run']
        search_call_spans = [s for s in span_list if s.get('name') == 'web_search_call']

        results[trace_id] = {
            'web_search_used': web_search_status,
            'trace_start_time': trace_start,
            'total_spans': len(span_list),
            'has_generator': len(generator_spans) > 0,
            'search_call_count': len(search_call_spans),
            'search_calls': []
        }

        # Extract details from search calls
        for search_span in search_call_spans:
            attrs = search_span.get('attributes', {})
            results[trace_id]['search_calls'].append({
                'action_type': attrs.get('action_type') or attrs.get('tool.parameters.action_type'),
                'query': attrs.get('query') or attrs.get('tool.parameters.query'),
                'source_urls': attrs.get('source_urls') or attrs.get('tool.parameters.source_urls'),
            })

    return results


def main():
    print("\n🔍 Step 1: Fetching spans from Phoenix API...")
    spans = fetch_phoenix_spans(days_back=2)

    if not spans:
        print("\n❌ No spans found! This could mean:")
        print("   1. Phoenix is not accessible at the configured endpoint")
        print("   2. No traces have been logged in the last 2 days")
        print("   3. Project name is incorrect")
        print("   4. Authentication is failing")
        return

    print(f"\n📊 Step 2: Grouping spans by trace_id...")
    traces = group_by_trace(spans)
    print(f"✅ Found {len(traces)} unique traces")

    print(f"\n🔬 Step 3: Analyzing traces for web search usage...")
    results = analyze_traces(traces)

    # Generate statistics
    stats = {
        'YES': 0,
        'NO': 0,
        'DISTANCE_ONLY': 0,
        'N/A': 0,
        'UNKNOWN': 0
    }

    for result in results.values():
        status = result['web_search_used']
        stats[status] = stats.get(status, 0) + 1

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    total = len(results)
    print(f"\nTotal traces analyzed: {total}")
    print(f"\nWeb search usage breakdown:")
    print(f"  ✅ YES (web search used):           {stats['YES']:4d} ({stats['YES']/total*100:5.1f}%)")
    print(f"  ❌ NO (generator ran, no search):   {stats['NO']:4d} ({stats['NO']/total*100:5.1f}%)")
    print(f"  📏 DISTANCE_ONLY (calculator only): {stats['DISTANCE_ONLY']:4d} ({stats['DISTANCE_ONLY']/total*100:5.1f}%)")
    print(f"  ⚪ N/A (no generator):               {stats['N/A']:4d} ({stats['N/A']/total*100:5.1f}%)")
    print(f"  ❓ UNKNOWN:                          {stats['UNKNOWN']:4d} ({stats['UNKNOWN']/total*100:5.1f}%)")

    # Show some example traces
    print("\n" + "=" * 80)
    print("SAMPLE TRACES")
    print("=" * 80)

    for status in ['YES', 'NO', 'DISTANCE_ONLY', 'N/A']:
        examples = [tid for tid, r in results.items() if r['web_search_used'] == status][:3]
        if examples:
            print(f"\n{status} examples:")
            for trace_id in examples:
                result = results[trace_id]
                print(f"  Trace: {trace_id[:16]}...")
                print(f"    Time: {result['trace_start_time']}")
                print(f"    Spans: {result['total_spans']}, Generator: {result['has_generator']}, Search calls: {result['search_call_count']}")
                if result['search_calls']:
                    for i, call in enumerate(result['search_calls'][:2], 1):
                        print(f"      Call {i}: {call['action_type']} - Query: {str(call['query'])[:80]}")

    # Save detailed results
    output_file = "production_websearch_detection_results.json"
    with open(output_file, "w") as f:
        json.dump({
            'summary': stats,
            'total_traces': total,
            'traces': results,
            'metadata': {
                'phoenix_url': PHOENIX_URL,
                'project_name': PHOENIX_PROJECT_NAME,
                'analysis_time': datetime.now(timezone.utc).isoformat(),
                'days_back': 2
            }
        }, f, indent=2, default=str)

    print(f"\n💾 Detailed results saved to: {output_file}")
    print("=" * 80)

    # Final assessment
    print("\n📋 ASSESSMENT:")
    if stats['YES'] > 0:
        print(f"✅ Web search IS being used in production ({stats['YES']} traces)")
        if stats['NO'] > 0:
            print(f"⚠️  However, {stats['NO']} traces had the generator run but chose NOT to search")
            print(f"   This could be intentional (LLM determined search wasn't needed)")
    else:
        print("❌ NO web search usage detected in any traces!")
        if stats['NO'] > 0:
            print(f"   The generator ran {stats['NO']} times but LLM chose not to search")
        if stats['N/A'] == total:
            print("   No OpenAIWebSearchGenerator.run spans found at all")
            print("   This suggests the web search component isn't being invoked")


if __name__ == "__main__":
    main()
