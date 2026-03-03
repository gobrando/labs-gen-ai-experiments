#!/usr/bin/env python3
"""
Test with actual production prompt (Version 48)
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
print(f"Production baseline: {data['baseline_rate']*100:.1f}% web search\n")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Actual production prompt from Version 48
# Note: Simplified - production injects database resources via {% for s in supports %}
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

[NOTE: Production injects hundreds of database resources here via template. For this test, we include the static resources from the prompt.]

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

print("="*80)
print("PRODUCTION PROMPT TEST (Version 48)")
print("="*80)
print("Config: gpt-5.1 + reasoning='none'")
print("Note: Simplified version - production injects database resources")
print("="*80)

web_search_count = 0
total_calls = 0
latencies = []
match_count = 0

for i, query_data in enumerate(all_queries, 1):
    query = query_data['query']
    expected_web_search = query_data['used_web_search']

    try:
        start = time.time()

        # Production configuration
        response = client.responses.create(
            model="gpt-5.1",
            reasoning={"effort": "none"},
            tools=[{"type": "web_search"}],
            input=[
                {"role": "system", "content": PRODUCTION_PROMPT},
                {"role": "user", "content": f"Client needs: {query}"}
            ]
        )

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

        # Check if behavior matches production
        if used_web_search == expected_web_search:
            match_count += 1
            match = "✅"
        else:
            match = "⚠️ "

        status = "WEB" if used_web_search else " NO"
        expected = "WEB" if expected_web_search else " NO"

        print(f"  {i:2d}. {match} {status} (prod:{expected}) | {latency:6.2f}s | {query[:50]}")

        time.sleep(0.5)  # Rate limiting

    except Exception as e:
        print(f"  {i:2d}. ❌ ERROR: {str(e)[:80]}")
        continue

# Calculate stats
web_search_rate = (web_search_count / total_calls * 100) if total_calls > 0 else 0
avg_latency = sum(latencies) / len(latencies) if latencies else 0
baseline_diff = web_search_rate - (data['baseline_rate'] * 100)
match_rate = (match_count / total_calls * 100) if total_calls > 0 else 0

print(f"\n{'='*80}")
print("RESULTS")
print(f"{'='*80}")
print(f"  Web search rate: {web_search_rate:.1f}% ({web_search_count}/{total_calls})")
print(f"  Production baseline: {data['baseline_rate']*100:.1f}%")
print(f"  Difference: {baseline_diff:+.1f} percentage points")
print(f"  Match rate with production: {match_rate:.1f}%")
print(f"  Avg latency: {avg_latency:.2f}s")

print(f"\n{'='*80}")
print("ANALYSIS")
print(f"{'='*80}")

if abs(baseline_diff) < 10:
    print(f"✅ GOOD: Test web search rate ({web_search_rate:.1f}%) is within 10pp of production")
    print(f"   The production prompt with embedded resources reduces web search reliance!")
else:
    print(f"⚠️  MISMATCH: Test web search rate ({web_search_rate:.1f}%) differs by {abs(baseline_diff):.1f}pp")
    print(f"   Likely cause: Missing database resources that production injects via template")
    print(f"   Production has hundreds of pre-loaded resources for common queries")

# Save results
output = {
    'production_baseline': data['baseline_rate'] * 100,
    'test_web_search_rate': web_search_rate,
    'difference': baseline_diff,
    'match_rate': match_rate,
    'total_queries': total_calls,
    'avg_latency': avg_latency,
    'config': {
        'model': 'gpt-5.1',
        'reasoning_effort': 'none',
        'prompt_version': 48,
        'note': 'Simplified - missing database-injected resources'
    }
}

with open('production_prompt_test_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n✅ Results saved to: production_prompt_test_results.json")
print("="*80)
