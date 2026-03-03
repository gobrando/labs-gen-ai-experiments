#!/usr/bin/env python3
"""
Test how temperature and reasoning parameters affect web search decisions
Baseline: 96.4% web search with production prompt (CAT/Excel resources only)
"""
import os
import json
import time
from openai import OpenAI

# Load real production queries
with open('sample_production_queries.json', 'r') as f:
    data = json.load(f)

all_queries = data['web_search_queries'] + data['no_web_search_queries']
print(f"Loaded {len(all_queries)} real production queries")
print(f"Production baseline: {data['baseline_rate']*100:.1f}% web search")
print(f"Test baseline (CAT/Excel only): 96.4% web search\n")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Production prompt with CAT/Excel resources
PRODUCTION_PROMPT = """You are an API endpoint for Goodwill Central Texas Referral and you return only a JSON object.
You are designed to help career case managers provide high-quality, local resource referrals to client's in Central Texas.
Your role is to support case managers working with low-income job seekers and learners in Austin and surrounding counties (Bastrop, Blanco, Burnet, Caldwell, DeWitt, Fayette, Gillespie, Gonzales, Hays, Lavaca, Lee, Llano, Mason, Travis, Williamson).

## Task Checklist
- Evaluate the client's needs and consider their eligibility for each resource, such as the client's age, income, disability, immigration/veteran status, and number of dependents.
- Suggest recommended resources and rank by proximity and eligibility.
- Never invent or fabricate resources. If none are available, state this clearly. Use trusted sources such as Goodwill, government, vetted nonprofits, and trusted news outlets (Findhelp, 211, Connect ATX permitted). Never use unreliable websites (e.g., shelterlistings.org, needhelppayingbills.com, thehelplist.com). Prefer direct sources rather than websites that aggregate listings.
- NEVER invent or guess URLs. Use only verified URLs that will actually work.
- NEVER offer Texas Workforce Commission OR Capital IDEA unless there's a more specific resource that these services specifically offer that GoodWill does not offer.
- NEVER recommend a resource that is no longer available (e.g., a course with a start date in the past) OR a resource that is unlikely to be available soon (e.g., a site opening in 2027.)

## Response Constraints
- Your response should ONLY include resources from the list below or resources you find searching the web.
- If no resources are found, return only an empty JSON list without any extra text.
- Do not summarize your assessment of the clients needs.
- Limit the description for a resource to be less than 255 words.
- Set referral_type to: "goodwill" if the resource offered by Goodwill (such as the Goodwill Career and Training Academy), "government" for resources provided by the city, county, or state, and "external" for all others.

## Resources
In addition to what you find searching the web, choose from following list of resources:

Career Advancement Training (CAT)
Free short-term training courses (1-4 weeks) covering essential workplace skills and prerequisites. CAT serves as both standalone skill-building and as required preparation for GCTA programs.

⚠️ CRITICAL DISTINCTION: CAT ≠ GCTA CAT trainings are NOT the same as GCTA trainings. Key differences:
 - Duration: CAT classes are much shorter (1-4 weeks) vs GCTA programs (4-12 weeks)
 - Enrollment: CAT has a simpler, faster enrollment process - often just requires online registration through Wufoo forms
 - Prerequisites: CAT courses often serve as prerequisites TO GCTA programs (e.g., Career Advancement Essentials must be completed before GCTA enrollment)
 - Certification: GCTA leads to industry certifications and job placement; CAT builds foundational skills
 - Complexity: GCTA requires extensive documentation, assessments (Wonderlic/CASAS), and multi-level approvals; CAT enrollment is streamlined

CAT Class Registration Links by Location:

Goodwill Resource Center (GRC/South):
 - Career Advancement Essentials: https://gwcareeradvancement.wufoo.com/forms/grc-career-advancement-essentials/
 - Computer Basics/Keyboarding: https://gwcareeradvancement.wufoo.com/forms/grc-computer-basics/
 - Digital Skills 1:1: https://gwcareeradvancement.wufoo.com/forms/grc-digital-skills-11/
 - Financial Empowerment Training: https://gwcareeradvancement.wufoo.com/forms/grc-11-financial-empowerment-trainings/
 - Indeed Lab: https://gwcareeradvancement.wufoo.com/forms/grc-indeed-lab/
 - Interview Preparation & Practice: https://gwcareeradvancement.wufoo.com/forms/grc-interview-preparation-and-practice/
 - Job Preparation 1:1: https://gwcareeradvancement.wufoo.com/forms/grc-job-preparation-11/
 - Online Safety: https://gwcareeradvancement.wufoo.com/forms/grc-online-safety/
 - Virtual Career Advancement Essentials: https://gwcareeradvancement.wufoo.com/forms/virtual-career-advancement-essentials/

Goodwill Community Center (GCC/North):
 - Career Advancement Essentials: https://gwcareeradvancement.wufoo.com/forms/gcc-career-advancement-essentials/
 - Computer Basics/Keyboarding: https://gwcareeradvancement.wufoo.com/forms/gcc-computer-basics/
 - Digital Skills 1:1: https://gwcareeradvancement.wufoo.com/forms/gcc-digital-skills-11/
 - Financial Empowerment Training: https://gwcareeradvancement.wufoo.com/forms/gcc-11-financial-empowerment-trainings/
 - Indeed Lab: https://gwcareeradvancement.wufoo.com/forms/gcc-indeed-lab/
 - Interview Preparation & Practice: https://gwcareeradvancement.wufoo.com/forms/gcc-interview-preparation-and-practice/
 - Job Preparation 1:1: https://gwcareeradvancement.wufoo.com/forms/gcc-job-preparation-11/
 - Wonderlic Prep & Practice: https://gwcareeradvancement.wufoo.com/forms/gcc-wonderlic-prep-and-practice/
 - AI Basics: https://gwcareeradvancement.wufoo.com/forms/zjgi3bu0u7t757/
 - Online Safety: https://gwcareeradvancement.wufoo.com/forms/zs43hn608egpxa/
 - Virtual Career Advancement Essentials: https://gwcareeradvancement.wufoo.com/forms/virtual-career-advancement-essentials/

When recommending CAT classes:
 - Direct clients to the appropriate location-specific registration link
 - GRC serves South Austin and surrounding areas
 - GCC serves North Austin, Round Rock, Georgetown, and surrounding areas
 - Most classes require pre-registration through the Wufoo forms
 - Classes run on monthly schedules - check with Career Case Manager for current availability

Excel Center High School
Goodwill's tuition-free high school completion program for adults ages 18-50:
- Earn accredited high school diploma (not GED)
- Flexible schedules designed for working adults
- Free childcare during classes
- Career coaching integrated into curriculum
- College prep included
- Small class sizes (15-20 students)
- Usually 12-18 months to complete
- Website: https://excelcenterhighschool.org/
When to recommend: Clients without high school diploma asking about GED should be informed about Excel Center as a superior alternative to traditional GED programs.
"""

def test_configuration(config_name, model, reasoning=None, temperature=None):
    """Test a specific parameter configuration"""
    print(f"\n{'='*80}")
    print(f"TESTING: {config_name}")
    print(f"{'='*80}")
    print(f"Model: {model}")
    if reasoning:
        print(f"Reasoning: {reasoning}")
    if temperature is not None:
        print(f"Temperature: {temperature}")
    print(f"{'='*80}")

    web_search_count = 0
    total_calls = 0
    latencies = []
    errors = []

    for i, query_data in enumerate(all_queries[:20], 1):  # Test on subset for speed
        query = query_data['query']

        try:
            start = time.time()

            # Build request parameters
            request_params = {
                "model": model,
                "tools": [{"type": "web_search"}],
                "input": [
                    {"role": "system", "content": PRODUCTION_PROMPT},
                    {"role": "user", "content": f"Client needs: {query}"}
                ]
            }

            # Add reasoning or temperature (mutually exclusive)
            if reasoning:
                request_params["reasoning"] = {"effort": reasoning}
            elif temperature is not None:
                request_params["temperature"] = temperature
                request_params["reasoning"] = {"effort": "none"}  # Required for temperature

            response = client.responses.create(**request_params)

            latency = time.time() - start
            latencies.append(latency)
            total_calls += 1

            # Check if web search was used
            used_web_search = False
            if response.output:
                for item in response.output:
                    if hasattr(item, 'type') and item.type == 'web_search_call':
                        used_web_search = True
                        web_search_count += 1
                        break

            status = "WEB" if used_web_search else " NO"
            print(f"  {i:2d}. {status} | {latency:6.2f}s | {query[:50]}")

            time.sleep(0.3)  # Rate limiting

        except Exception as e:
            error_msg = str(e)[:80]
            print(f"  {i:2d}. ❌ ERROR: {error_msg}")
            errors.append({"query": query, "error": error_msg})
            continue

    # Calculate stats
    web_search_rate = (web_search_count / total_calls * 100) if total_calls > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"  Web search rate: {web_search_rate:.1f}% ({web_search_count}/{total_calls})")
    print(f"  Avg latency: {avg_latency:.2f}s")
    print(f"  Errors: {len(errors)}")
    print(f"{'='*80}")

    return {
        "config": config_name,
        "model": model,
        "reasoning": reasoning,
        "temperature": temperature,
        "web_search_rate": web_search_rate,
        "web_search_count": web_search_count,
        "total_queries": total_calls,
        "avg_latency": avg_latency,
        "errors": errors
    }

# Run tests
print("\n" + "="*80)
print("PARAMETER EFFECT TESTING")
print("="*80)
print("Goal: Measure how temperature and reasoning affect web search decisions")
print("Baseline: 96.4% web search (production prompt with CAT/Excel only)")
print("="*80)

results = []

# Test 1: Baseline (reasoning=none)
results.append(test_configuration(
    "Baseline (reasoning=none)",
    model="gpt-5.1",
    reasoning="none"
))

# Test 2-6: Temperature variations (only work with reasoning=none)
for temp in [0.0, 0.5, 1.0, 1.5, 2.0]:
    results.append(test_configuration(
        f"Temperature {temp}",
        model="gpt-5.1",
        temperature=temp
    ))

# Test 7-9: Reasoning variations (temperature not compatible)
for reasoning_level in ["low", "medium", "high"]:
    results.append(test_configuration(
        f"Reasoning {reasoning_level}",
        model="gpt-5.1",
        reasoning=reasoning_level
    ))

# Summary comparison
print("\n" + "="*80)
print("SUMMARY COMPARISON")
print("="*80)
print(f"{'Configuration':<25} {'Web Search %':<15} {'Avg Latency':<15}")
print("-"*80)

baseline_rate = results[0]['web_search_rate']

for result in results:
    config = result['config']
    rate = result['web_search_rate']
    latency = result['avg_latency']
    diff = rate - baseline_rate
    diff_str = f"({diff:+.1f}pp)" if diff != 0 else ""

    print(f"{config:<25} {rate:>6.1f}% {diff_str:<7} {latency:>6.2f}s")

print("="*80)

# Save results
output = {
    "baseline": {
        "description": "Production prompt with CAT/Excel resources only",
        "web_search_rate": 96.4,
        "note": "Missing ~hundreds of database-injected resources"
    },
    "tests": results,
    "summary": {
        "temperature_range": "0.0 to 2.0 (with reasoning=none)",
        "reasoning_levels": ["none", "low", "medium", "high"],
        "queries_per_test": 20,
        "total_tests": len(results)
    }
}

with open('parameter_effect_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n✅ Full results saved to: parameter_effect_results.json")
print("="*80)
