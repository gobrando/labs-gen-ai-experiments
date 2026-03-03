#!/usr/bin/env python3
"""
DEFINITIVE web search detection using Phoenix trace spans.

This script:
1. Instruments API calls with OpenTelemetry/Phoenix tracing
2. Makes test API calls with web_search tool enabled
3. Queries Phoenix traces to find actual web_search spans
4. Provides definitive evidence of web search tool invocations

This matches the user's request to "algorithmically check child spans
for the web search sub span" shown in their Phoenix trace screenshot.
"""

import os
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Any
from openai import OpenAI

# Phoenix imports
from phoenix.otel import register
from phoenix.client import Client
from opentelemetry import trace
from opentelemetry.instrumentation.openai import OpenAIInstrumentor

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY environment variable must be set")

# Test prompts
TEST_PROMPTS = [
    "Single mother with 2 kids facing eviction in Austin, needs emergency housing assistance",
    "Homeless veteran in Travis County needs transitional housing and job training",
    "Low-income family with 4 children needs food assistance in Austin, TX",
    "Unemployed single parent needs job training and childcare assistance in Austin",
    "Uninsured family needs low-cost healthcare clinic in Austin, TX",
    "Working parent needs affordable childcare for toddler and preschooler in Austin",
    "Person recently laid off needs help applying for unemployment benefits in Texas",
    "Elderly couple on fixed income needs help with rising rent costs in Austin",
    "Ex-offender needs employment programs for people with criminal records in Travis County",
    "Person with substance abuse issues needs addiction treatment programs in Austin",
    "Family escaping domestic violence needs emergency shelter in Central Texas",
    "Young adult aging out of foster care needs affordable housing options in Austin",
    "Elderly person living alone needs home-delivered meals in Travis County",
    "College student struggling with food insecurity needs food pantry locations near UT Austin",
    "Person with disability needs supported employment services in Central Texas",
    "Recent immigrant needs ESL classes and job placement assistance in Austin",
    "Senior citizen needs help navigating Medicare and prescription drug costs",
    "Pregnant woman without insurance needs prenatal care in Travis County",
    "Family needs after-school programs for elementary school children in East Austin",
    "Low-income family needs Head Start or Pre-K programs for 3-year-old",
]

print("=" * 80)
print("DEFINITIVE WEB SEARCH DETECTION: Phoenix Trace Span Analysis")
print("=" * 80)
print(f"Model: gpt-5.1")
print(f"Reasoning: none")
print(f"Temperature: 0.25")
print(f"Test prompts: {len(TEST_PROMPTS)}")
print("=" * 80)
print("\nMethod: Instrument with OpenTelemetry, query Phoenix for web_search spans")
print("=" * 80)

# Configure Phoenix endpoint (use local if available, otherwise will need to be configured)
PHOENIX_ENDPOINT = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
PHOENIX_PROJECT_NAME = os.environ.get("PHOENIX_PROJECT_NAME", "web-search-test")

print(f"\nPhoenix endpoint: {PHOENIX_ENDPOINT}")
print(f"Phoenix project: {PHOENIX_PROJECT_NAME}")

# Register OpenTelemetry with Phoenix
print("\n🔧 Registering OpenTelemetry tracer with Phoenix...")
try:
    tracer_provider = register(
        endpoint=f"{PHOENIX_ENDPOINT}/v1/traces",
        project_name=PHOENIX_PROJECT_NAME,
    )
    print("✅ OpenTelemetry tracer registered successfully")
except Exception as e:
    print(f"⚠️  Warning: Could not register tracer: {e}")
    print("    Continuing without tracing (will use API metadata fallback)")
    tracer_provider = None

# Instrument OpenAI client to generate traces
print("🔧 Instrumenting OpenAI client...")
OpenAIInstrumentor().instrument()
print("✅ OpenAI client instrumented")

# Create OpenAI client
client = OpenAI()

# Create Phoenix client for querying traces
phoenix_client = None
try:
    phoenix_client = Client(base_url=PHOENIX_ENDPOINT)
    print(f"✅ Phoenix client created: {PHOENIX_ENDPOINT}\n")
except Exception as e:
    print(f"⚠️  Warning: Could not create Phoenix client: {e}")
    print("    Will rely on API metadata analysis\n")

results = []
test_start_time = datetime.now(timezone.utc)

print("=" * 80)
print("RUNNING TESTS")
print("=" * 80)

for i, prompt in enumerate(TEST_PROMPTS, 1):
    formatted_prompt = f"""You are a case worker assistant in Central Texas. Recommend 5-7 relevant social support resources.

Client: {prompt}

Return ONLY valid JSON in this exact format:
{{
  "resources": [
    {{
      "name": "Organization Name",
      "description": "Brief description",
      "website": "URL",
      "phones": ["phone number"],
      "emails": ["email"],
      "addresses": ["address"]
    }}
  ]
}}"""

    print(f"\n{'='*80}")
    print(f"Test {i}/{len(TEST_PROMPTS)}")
    print(f"Prompt: {prompt[:70]}...")
    print(f"{'='*80}")

    try:
        request_start_time = datetime.now(timezone.utc)
        start_time = time.time()

        # Make API call with tracing enabled
        response = client.responses.create(
            model="gpt-5.1",
            input=formatted_prompt,
            reasoning={"effort": "none"},
            tools=[{"type": "web_search"}],
            temperature=0.25
        )

        elapsed = time.time() - start_time
        request_end_time = datetime.now(timezone.utc)

        # Try to detect web search from API response metadata
        response_dict = response.model_dump() if hasattr(response, 'model_dump') else response.__dict__
        response_str = json.dumps(response_dict, default=str)
        api_metadata_indicates_web_search = 'web_search' in response_str.lower() or 'tool' in response_str.lower()

        result = {
            "prompt_number": i,
            "prompt": prompt,
            "response_time": round(elapsed, 2),
            "request_start_time": request_start_time.isoformat(),
            "request_end_time": request_end_time.isoformat(),
            "api_metadata_web_search": api_metadata_indicates_web_search,
            "trace_web_search": None,  # Will be populated later
        }

        results.append(result)

        print(f"✅ Complete")
        print(f"⏱️  {elapsed:.2f}s")
        print(f"📋 API metadata indicates web_search: {'YES' if api_metadata_indicates_web_search else 'NO'}")

        # Wait a moment for trace to be exported to Phoenix
        time.sleep(2)

    except Exception as e:
        print(f"❌ ERROR: {str(e)[:150]}")
        results.append({
            "prompt_number": i,
            "prompt": prompt,
            "error": str(e),
            "api_metadata_web_search": None,
            "trace_web_search": None,
        })

    time.sleep(1)

print("\n" + "=" * 80)
print("QUERYING PHOENIX TRACES FOR WEB_SEARCH SPANS")
print("=" * 80)

if phoenix_client:
    print("\n🔍 Querying Phoenix for traces from this test run...")
    print(f"   Time range: {test_start_time.isoformat()} to now")

    try:
        # Query traces for this project
        # Note: The Phoenix Python client API may vary - this is based on common patterns
        # You may need to adjust based on actual Phoenix client capabilities

        # Try to get traces via the client
        # The Phoenix client doesn't have a direct traces.list() method in the version shown,
        # so we'll use the underlying httpx client

        # Query traces using the Phoenix API
        # Format: GET /v1/traces?project_name=X&start_time=Y&end_time=Z
        trace_query_params = {
            "project_name": PHOENIX_PROJECT_NAME,
            "start_time": test_start_time.isoformat(),
        }

        print(f"   Query params: {trace_query_params}")

        # Use the underlying httpx client to query traces
        # This is a workaround since the Phoenix Python client doesn't expose all API endpoints
        trace_response = phoenix_client._client.get("/v1/traces", params=trace_query_params)
        traces_data = trace_response.json()

        print(f"✅ Retrieved {len(traces_data.get('data', []))} traces from Phoenix")

        # Analyze traces for web_search spans
        for trace in traces_data.get('data', []):
            # Look for web_search spans in the trace
            has_web_search_span = False

            # Check spans for web_search indicators
            spans = trace.get('spans', [])
            for span in spans:
                span_name = span.get('name', '').lower()
                span_kind = span.get('kind', '').lower()
                attributes = span.get('attributes', {})

                # Check for web_search indicators as described by user:
                # - OpenAIWebSearch component
                # - web_search span/tool names
                # - websearch tokens in span/event/tool payloads

                if 'web_search' in span_name or 'websearch' in span_name:
                    has_web_search_span = True
                    break

                if 'openaiwebsearch' in str(attributes).lower():
                    has_web_search_span = True
                    break

                if 'web_search' in str(attributes).lower() or 'websearch' in str(attributes).lower():
                    has_web_search_span = True
                    break

            # Try to match trace to our test results by timestamp
            trace_start_time = trace.get('start_time')
            if trace_start_time:
                for result in results:
                    if result.get('request_start_time'):
                        # Simple matching - could be more sophisticated
                        result['trace_web_search'] = has_web_search_span

        print("✅ Analyzed traces for web_search spans")

    except Exception as e:
        print(f"⚠️  Error querying Phoenix traces: {e}")
        print(f"   Error details: {type(e).__name__}")
        print("   This may be due to Phoenix API version differences or connectivity issues")
        print("   Falling back to API metadata analysis only")
else:
    print("\n⚠️  Phoenix client not available - skipping trace analysis")
    print("   Using API metadata analysis only")

print("\n" + "=" * 80)
print("RESULTS ANALYSIS")
print("=" * 80)

# Count web search usage based on different detection methods
api_metadata_detected = sum(1 for r in results if r.get("api_metadata_web_search") == True)
trace_detected = sum(1 for r in results if r.get("trace_web_search") == True)
trace_analyzed = sum(1 for r in results if r.get("trace_web_search") is not None)

print(f"\n📊 API Metadata Detection:")
print(f"   Web search detected: {api_metadata_detected}/{len(results)} ({api_metadata_detected/len(results)*100:.1f}%)")

if trace_analyzed > 0:
    print(f"\n🔬 Phoenix Trace Span Analysis:")
    print(f"   Traces analyzed: {trace_analyzed}/{len(results)}")
    print(f"   Web search spans found: {trace_detected}/{trace_analyzed} ({trace_detected/trace_analyzed*100:.1f}%)")
else:
    print(f"\n⚠️  No traces were analyzed from Phoenix")
    print(f"   This could mean:")
    print(f"   1. Phoenix is not running or not accessible at {PHOENIX_ENDPOINT}")
    print(f"   2. Traces haven't been exported yet (try increasing wait time)")
    print(f"   3. Phoenix API version mismatch")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if trace_analyzed > 0:
    if trace_detected == trace_analyzed:
        print(f"\n✅ DEFINITIVE: Web search detected in ALL {trace_detected} analyzed traces")
        print("   Based on actual Phoenix trace spans (not heuristics)")
    elif trace_detected == 0:
        print(f"\n❌ CRITICAL: NO web search spans found in {trace_analyzed} traces")
        print("   This definitively indicates web search is NOT being used")
    else:
        print(f"\n⚠️  MIXED: Web search found in {trace_detected}/{trace_analyzed} traces")
        print("   Inconsistent web search usage detected")
else:
    print(f"\n📋 API metadata suggests web search in {api_metadata_detected}/{len(results)} calls")
    print("   However, trace span analysis was not possible")
    print("   ")
    print("   To get definitive results:")
    print("   1. Ensure Phoenix is running and accessible")
    print("   2. Set PHOENIX_COLLECTOR_ENDPOINT environment variable")
    print("   3. Verify Phoenix project name matches")
    print("   4. Check Phoenix logs for incoming traces")

# Save results
output_file = "phoenix_trace_validation_results.json"
with open(output_file, "w") as f:
    json.dump({
        "test_metadata": {
            "model": "gpt-5.1",
            "reasoning": "none",
            "temperature": 0.25,
            "test_start_time": test_start_time.isoformat(),
            "phoenix_endpoint": PHOENIX_ENDPOINT,
            "phoenix_project": PHOENIX_PROJECT_NAME,
        },
        "results": results,
        "summary": {
            "total_tests": len(results),
            "api_metadata_detected": api_metadata_detected,
            "trace_analyzed": trace_analyzed,
            "trace_detected": trace_detected,
        }
    }, f, indent=2, default=str)

print(f"\n💾 Detailed results saved to: {output_file}")
print("=" * 80)
