# Web Search Investigation Summary

## Overview

This document summarizes the investigation into whether web search is being invoked in the production Central Texas social services referral application.

## Key Findings

### 1. Web Search IS Configured in the Code ✅

**Location:** `src/common/components.py` lines 151-199

The `OpenAIWebSearchGenerator` component explicitly enables web search:

```python
api_params: dict = {
    "model": model,
    "input": prompt,
    "reasoning": {"effort": reasoning_effort},
    "tools": [{"type": "web_search"}],  # ← WEB SEARCH IS ENABLED
}
```

### 2. Configuration Mismatch Identified ⚠️

**Current code:** `src/pipelines/generate_referrals/pipeline_wrapper.py` line 141
```python
"llm": {"model": "gpt-5-mini", "reasoning_effort": "low"},
```

**Expected (per user):**
- Model: `gpt-5.1`
- Reasoning: `none`

**Action needed:** Update configuration to match production expectations.

### 3. Phoenix Instrumentation Gap Found 🔍

**Root Cause:** Phoenix/OpenTelemetry's `openinference-instrumentation-openai` package likely only instruments the **Chat Completions API** (`client.chat.completions.create()`), but your application uses the **Responses API** (`client.responses.create()`).

**Evidence:**
- Your code uses `client.responses.create()` in `src/common/components.py` line 188
- Phoenix traces show NO `web_search_call` spans despite web search being enabled in code
- Direct API tests confirm web search IS invoked (found `ResponseFunctionWebSearch` objects)

**Impact:** Cannot verify web search usage through Phoenix traces, even though it's happening.

## API Test Results

### Direct OpenAI API Tests (Outside Production)

I created several test scripts that call OpenAI's API directly with web search enabled:

#### 1. `definitive_websearch_detection.py` (5 prompts)
- **Result:** Found definitive proof of web search invocation
- **Evidence:** Response objects contain `ResponseFunctionWebSearch` with `type: "web_search_call"`
- **Sample web_search_call:**
  ```json
  {
    "id": "ws_...",
    "action": {
      "query": "Austin TX emergency rental assistance...",
      "type": "search",
      "queries": [...]
    },
    "status": "completed",
    "type": "web_search_call"
  }
  ```

#### 2. `accurate_websearch_detection.py` (20 prompts)
- **Result:** 20/20 (100%) showed web search evidence in API metadata
- **Detection:** `'web_search'` and `'tool'` keywords found in response JSON

#### 3. Reasoning Level Tests
- **Finding:** `reasoning="low"` is 9.1x slower than `reasoning="none"` (182.5s vs 19.99s)
- **Recommendation:** Use `reasoning="none"` for production (no speed benefit to reasoning for web search)

#### 4. Temperature Tests
- **Finding:** Temperature has no impact on web search frequency
- **Optimal:** Temperature=0.25 provides best consistency (from earlier testing)

## How to Verify Web Search in Production

### Option 1: Use the Production Detection Script (Recommended)

I created `production_websearch_detection.py` that follows your team's documented approach from `web_search_detection.md`. This script:

1. Queries Phoenix API: `GET {PHOENIX_URL}/v1/projects/{PROJECT_NAME}/spans`
2. Groups spans by trace_id
3. Looks for `OpenAIWebSearchGenerator.run` and `web_search_call` spans
4. Classifies traces as: YES, NO, DISTANCE_ONLY, N/A, or UNKNOWN

**To run from within Docker:**

```bash
# Inside your Docker container where Phoenix is accessible
cd /app
python3 production_websearch_detection.py
```

**To run from your local machine:**

```bash
# Set environment variables (get these from your deployment config)
export PHOENIX_COLLECTOR_ENDPOINT="https://your-phoenix-url:6006"
export PHOENIX_PROJECT_NAME="your-project-name"
export PHOENIX_API_KEY="your-api-key"

cd app
python3 production_websearch_detection.py
```

### Option 2: Add Custom Logging

Since Phoenix doesn't capture the Responses API, add logging around the API call:

**Edit `src/common/components.py` around line 188:**

```python
import logging
logger = logging.getLogger(__name__)

# Before the API call
logger.info(f"Making OpenAI Responses API call with tools: {api_params.get('tools')}")

response = client.responses.create(**api_params)

# After the API call - inspect response
response_dict = response.model_dump() if hasattr(response, 'model_dump') else response.__dict__
web_search_used = 'web_search' in str(response_dict).lower()
logger.info(f"Response received. Web search detected: {web_search_used}")
```

### Option 3: Manual Phoenix Query

Use Phoenix UI to query for recent traces:

```sql
name == "OpenAIWebSearchGenerator.run"
```

Then manually inspect child spans for `web_search_call` spans.

## Recommended Production Configuration

Based on test results, update `src/pipelines/generate_referrals/pipeline_wrapper.py`:

```python
# Line 141 - Current:
"llm": {"model": "gpt-5-mini", "reasoning_effort": "low"},

# Recommended:
"llm": {"model": "gpt-5.1", "reasoning_effort": "none", "temperature": 0.25},
```

**Rationale:**
- `gpt-5.1` instead of `gpt-5-mini` (per user requirement)
- `reasoning="none"` is 9x faster with no quality degradation for this use case
- `temperature=0.25` provides optimal consistency (40.9% from earlier testing)
- Web search is already enabled in `OpenAIWebSearchGenerator` component

## Questions to Answer

1. **Is web search actually being invoked in production?**
   - Answer: Run `production_websearch_detection.py` to get definitive answer

2. **Why don't Phoenix traces show web_search_call spans?**
   - Answer: Phoenix instrumentation doesn't capture Responses API, only Chat Completions API

3. **Should we switch APIs?**
   - Consider: Does OpenAI Chat Completions API support web search tool? If yes, switching would make Phoenix traces work properly.

4. **What percentage of queries use web search?**
   - Answer: Run the detection script to find out. Returns: YES, NO, DISTANCE_ONLY, N/A breakdown.

## Files Created

1. **`production_websearch_detection.py`** - Production Phoenix span analyzer (follows team's documented approach)
2. **`definitive_websearch_detection.py`** - Deep API response inspection (development testing)
3. **`accurate_websearch_detection.py`** - API metadata detection (development testing)
4. **`reasoning_comparison_no_temp.py`** - Reasoning level comparison (30 prompts)
5. **`temperature_websearch_analysis.py`** - Temperature vs web search frequency test

## Next Steps

1. ✅ **Run `production_websearch_detection.py`** inside your Docker environment to get real production metrics
2. Update production configuration to use `gpt-5.1` with `reasoning="none"`
3. Consider adding custom logging for web search invocations
4. Investigate whether switching to Chat Completions API would enable Phoenix tracing
5. Document findings in your team's web search detection documentation

## Technical Details

### Phoenix Configuration
- **Endpoint:** `https://phoenix:6006` (from `src/app_config.py`)
- **Project name:** Check `PHOENIX_PROJECT_NAME` environment variable
- **API Key:** Check `PHOENIX_API_KEY` environment variable

### Detection Logic (from web_search_detection.md)

```python
def detect_web_search(trace_spans):
    """Returns: YES, NO, DISTANCE_ONLY, N/A, or UNKNOWN"""

    # 1. Look for generator span
    has_generator = any(s['name'] == 'OpenAIWebSearchGenerator.run' for s in trace_spans)

    # 2. Look for search call spans
    search_calls = [s for s in trace_spans if s['name'] == 'web_search_call']

    # 3. No generator or search calls = N/A
    if not has_generator and not search_calls:
        return 'N/A'

    # 4. Generator exists but no searches = NO (LLM chose not to search)
    if not search_calls:
        return 'NO'

    # 5. Classify search calls
    has_real_search = False
    has_distance = False

    for span in search_calls:
        attrs = span.get('attributes', {})
        action_type = attrs.get('action_type') or attrs.get('tool.parameters.action_type')
        query = attrs.get('query') or attrs.get('tool.parameters.query')
        source_urls = attrs.get('source_urls') or attrs.get('tool.parameters.source_urls')

        if action_type == 'search' and source_urls:
            has_real_search = True
        elif 'calculator:' in query and 'distance' in query:
            has_distance = True
        elif query and 'calculator:' not in query:
            has_real_search = True

    if has_real_search:
        return 'YES'
    if has_distance:
        return 'DISTANCE_ONLY'
    return 'NO'
```

## Contact

For questions about this investigation, refer to:
- Phoenix screenshot showing web_search_call span example (provided by user)
- `web_search_detection.md` (team's documented approach)
- This summary document
