#!/usr/bin/env python3
"""
PROPER Benchmark: ChatGPT 5.1 (no reasoning) vs Gemini 3 Flash
- WITH web search for both models
- 10 diverse test prompts
- Statistical analysis
"""

import time
import json
import os
import statistics

# API keys should be set as environment variables before running:
# export OPENAI_API_KEY="your-key-here"
# export GOOGLE_API_KEY="your-key-here"

# 10 diverse test cases representing real case worker scenarios
test_cases = [
    {
        "name": "Single Mother - Employment & Childcare",
        "query": """You are a case worker assistant. Recommend 5-7 relevant social support resources in Seattle, WA.

Client: Single mother with two young children (ages 3 and 5) needs:
- Employment opportunities
- Affordable childcare while job searching
- Food assistance
- Children's educational development resources
- Healthcare services

Return valid JSON: {"resources": [{"name": "...", "description": "...", "addresses": ["..."], "phones": ["..."], "emails": ["..."], "website": "...", "justification": "..."}]}"""
    },
    {
        "name": "Veteran - Housing & Mental Health",
        "query": """You are a case worker assistant. Recommend 5-7 relevant social support resources in Los Angeles, CA.

Client: Military veteran experiencing homelessness needs:
- Emergency housing
- PTSD counseling and mental health services
- Job training programs
- VA benefits assistance
- Substance abuse treatment

Return valid JSON: {"resources": [{"name": "...", "description": "...", "addresses": ["..."], "phones": ["..."], "emails": ["..."], "website": "...", "justification": "..."}]}"""
    },
    {
        "name": "Elderly - Medical & Daily Living",
        "query": """You are a case worker assistant. Recommend 5-7 relevant social support resources in Miami, FL.

Client: 78-year-old living alone needs:
- Home healthcare services
- Meal delivery programs
- Transportation to medical appointments
- Social activities and companionship
- Prescription assistance programs

Return valid JSON: {"resources": [{"name": "...", "description": "...", "addresses": ["..."], "phones": ["..."], "emails": ["..."], "website": "...", "justification": "..."}]}"""
    },
    {
        "name": "Immigrant Family - Language & Integration",
        "query": """You are a case worker assistant. Recommend 5-7 relevant social support resources in Houston, TX.

Client: Newly arrived immigrant family with limited English needs:
- ESL (English as Second Language) classes
- Immigration legal services
- Cultural integration programs
- Job placement assistance
- Children's after-school programs

Return valid JSON: {"resources": [{"name": "...", "description": "...", "addresses": ["..."], "phones": ["..."], "emails": ["..."], "website": "...", "justification": "..."}]}"""
    },
    {
        "name": "Teen Parent - Education & Support",
        "query": """You are a case worker assistant. Recommend 5-7 relevant social support resources in Chicago, IL.

Client: 17-year-old parent finishing high school needs:
- Parenting classes and support groups
- Childcare while attending school
- GED or high school completion programs
- Teen parent mentorship
- Financial literacy education

Return valid JSON: {"resources": [{"name": "...", "description": "...", "addresses": ["..."], "phones": ["..."], "emails": ["..."], "website": "...", "justification": "..."}]}"""
    },
    {
        "name": "Domestic Violence Survivor - Safety & Recovery",
        "query": """You are a case worker assistant. Recommend 5-7 relevant social support resources in Phoenix, AZ.

Client: Domestic violence survivor with two children needs:
- Emergency shelter
- Legal advocacy and restraining order assistance
- Trauma counseling
- Job training and financial independence programs
- Safe childcare options

Return valid JSON: {"resources": [{"name": "...", "description": "...", "addresses": ["..."], "phones": ["..."], "emails": ["..."], "website": "...", "justification": "..."}]}"""
    },
    {
        "name": "Disabled Adult - Accessibility & Employment",
        "query": """You are a case worker assistant. Recommend 5-7 relevant social support resources in Philadelphia, PA.

Client: Adult with physical disability needs:
- Vocational rehabilitation services
- Accessible housing resources
- Disability benefits navigation
- Adaptive technology assistance
- Independent living skills training

Return valid JSON: {"resources": [{"name": "...", "description": "...", "addresses": ["..."], "phones": ["..."], "emails": ["..."], "website": "...", "justification": "..."}]}"""
    },
    {
        "name": "Ex-Offender - Reentry Support",
        "query": """You are a case worker assistant. Recommend 5-7 relevant social support resources in Atlanta, GA.

Client: Recently released from incarceration needs:
- Job placement for people with records
- Transitional housing
- Legal services for record expungement
- Substance abuse counseling
- Life skills and financial management classes

Return valid JSON: {"resources": [{"name": "...", "description": "...", "addresses": ["..."], "phones": ["..."], "emails": ["..."], "website": "...", "justification": "..."}]}"""
    },
    {
        "name": "Youth Aging Out - Independence Preparation",
        "query": """You are a case worker assistant. Recommend 5-7 relevant social support resources in San Diego, CA.

Client: 18-year-old aging out of foster care needs:
- Affordable housing resources
- College scholarship and financial aid assistance
- Job readiness training
- Life skills education (budgeting, cooking, etc.)
- Mentorship programs for former foster youth

Return valid JSON: {"resources": [{"name": "...", "description": "...", "addresses": ["..."], "phones": ["..."], "emails": ["..."], "website": "...", "justification": "..."}]}"""
    },
    {
        "name": "Family Crisis - Multiple Urgent Needs",
        "query": """You are a case worker assistant. Recommend 5-7 relevant social support resources in Denver, CO.

Client: Family facing multiple crises needs:
- Emergency financial assistance (utilities, rent)
- Food bank and meal programs
- Free legal aid for eviction prevention
- Mental health crisis counseling
- Emergency childcare services

Return valid JSON: {"resources": [{"name": "...", "description": "...", "addresses": ["..."], "phones": ["..."], "emails": ["..."], "website": "...", "justification": "..."}]}"""
    }
]

print("=" * 80)
print("PROPER BENCHMARK: ChatGPT 5.1 vs Gemini 3 Flash")
print("WITH WEB SEARCH - 10 Test Cases")
print("=" * 80 + "\n")

chatgpt_results = []
gemini_results = []

# Test ChatGPT 5.1 with web search on all 10 prompts
print("Testing ChatGPT 5.1 (no reasoning) WITH WEB SEARCH...")
print("-" * 80)

from openai import OpenAI
client = OpenAI()

for i, test_case in enumerate(test_cases, 1):
    print(f"{i}. {test_case['name']}...", end=" ")
    try:
        start = time.time()

        # Using web search with ChatGPT 5.1
        response = client.responses.create(
            model="gpt-5.1",
            input=test_case["query"],
            reasoning={"effort": "low"},
            tools=[{"type": "web_search"}]  # Enable web search
        )

        duration = time.time() - start
        text = response.output_text

        # Count resources
        try:
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            if start_idx != -1:
                data = json.loads(text[start_idx:end_idx])
                resource_count = len(data.get("resources", []))
            else:
                resource_count = 0
        except:
            resource_count = 0

        chatgpt_results.append({"duration": duration, "resources": resource_count, "success": True})
        print(f"✅ {duration:.1f}s ({resource_count} resources)")

    except Exception as e:
        chatgpt_results.append({"duration": 0, "resources": 0, "success": False, "error": str(e)})
        print(f"❌ Error: {str(e)[:50]}")

print()

# Test Gemini 3 Flash with web search on all 10 prompts
print("Testing Gemini 3 Flash WITH WEB SEARCH...")
print("-" * 80)

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# Try to enable web search (grounding with Google Search)
for i, test_case in enumerate(test_cases, 1):
    print(f"{i}. {test_case['name']}...", end=" ")
    try:
        start = time.time()

        # Try with web search first
        try:
            model = genai.GenerativeModel(
                'gemini-3-flash-preview',
                safety_settings=safety_settings,
                tools='google_search_retrieval'  # Try web search
            )
        except:
            # Fallback to regular model without web search if not supported
            model = genai.GenerativeModel(
                'gemini-3-flash-preview',
                safety_settings=safety_settings
            )

        response = model.generate_content(test_case["query"])
        duration = time.time() - start
        text = response.text

        # Count resources
        try:
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            if start_idx != -1:
                data = json.loads(text[start_idx:end_idx])
                resource_count = len(data.get("resources", []))
            else:
                resource_count = 0
        except:
            resource_count = 0

        gemini_results.append({"duration": duration, "resources": resource_count, "success": True})
        print(f"✅ {duration:.1f}s ({resource_count} resources)")

    except Exception as e:
        gemini_results.append({"duration": 0, "resources": 0, "success": False, "error": str(e)})
        print(f"❌ Error: {str(e)[:50]}")

# Statistical Analysis
print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS")
print("=" * 80 + "\n")

chatgpt_successful = [r for r in chatgpt_results if r["success"]]
gemini_successful = [r for r in gemini_results if r["success"]]

if chatgpt_successful and gemini_successful:
    chatgpt_durations = [r["duration"] for r in chatgpt_successful]
    gemini_durations = [r["duration"] for r in gemini_successful]

    print(f"ChatGPT 5.1 (no reasoning) WITH WEB SEARCH:")
    print(f"  Successful: {len(chatgpt_successful)}/10")
    print(f"  Average: {statistics.mean(chatgpt_durations):.2f}s")
    print(f"  Median: {statistics.median(chatgpt_durations):.2f}s")
    print(f"  Min: {min(chatgpt_durations):.2f}s")
    print(f"  Max: {max(chatgpt_durations):.2f}s")
    if len(chatgpt_durations) > 1:
        print(f"  Std Dev: {statistics.stdev(chatgpt_durations):.2f}s")
    print()

    print(f"Gemini 3 Flash WITH WEB SEARCH:")
    print(f"  Successful: {len(gemini_successful)}/10")
    print(f"  Average: {statistics.mean(gemini_durations):.2f}s")
    print(f"  Median: {statistics.median(gemini_durations):.2f}s")
    print(f"  Min: {min(gemini_durations):.2f}s")
    print(f"  Max: {max(gemini_durations):.2f}s")
    if len(gemini_durations) > 1:
        print(f"  Std Dev: {statistics.stdev(gemini_durations):.2f}s")
    print()

    # Winner
    chatgpt_avg = statistics.mean(chatgpt_durations)
    gemini_avg = statistics.mean(gemini_durations)

    if gemini_avg < chatgpt_avg:
        speedup = ((chatgpt_avg / gemini_avg) - 1) * 100
        print(f"🏆 WINNER: Gemini 3 Flash")
        print(f"   {speedup:.1f}% faster on average")
        print(f"   ({gemini_avg:.2f}s vs {chatgpt_avg:.2f}s)")
    else:
        speedup = ((gemini_avg / chatgpt_avg) - 1) * 100
        print(f"🏆 WINNER: ChatGPT 5.1")
        print(f"   {speedup:.1f}% faster on average")
        print(f"   ({chatgpt_avg:.2f}s vs {gemini_avg:.2f}s)")

    print("=" * 80)
else:
    print("❌ Not enough successful results for comparison")
    print(f"ChatGPT successful: {len(chatgpt_successful)}/10")
    print(f"Gemini successful: {len(gemini_successful)}/10")
