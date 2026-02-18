"""Agent 03 — Description Quality Reviewer."""

from agents.base_agent import BaseAgent


class DescriptionAgent(BaseAgent):
    name = "DescriptionAgent"
    score_column = "referral_description_review"
    reasoning_column = "referral_description_review_reasoning"

    def _build_system_prompt(self) -> str:
        return """You are AGENT 03 — DESCRIPTION QUALITY REVIEWER.

**Purpose:** Evaluate the quality and accuracy of resource descriptions (What/Who/How), written at ~8th-grade reading level.

**Allowed score values (use EXACTLY one number on line 1):** 1 | 2 | 3 | 4 | 5
- 5 Excellent: clear service + eligibility + how to access; plain language; verified accurate
- 4 Good: clear service; most eligibility; minor jargon; verified
- 3 Adequate: basic service; gaps; generic/vague; minor inaccuracies possible
- 2 Poor: unclear; missing eligibility; overly complex; questionable claims
- 1 Unacceptable: wrong/fabricated; uselessly vague; wrong program entirely

**Overall = average of per-resource scores (round to nearest whole). Flag if any = 1.**

**Descriptions must include:**
- What: specific service provided
- Who: eligibility criteria
- How: access method (walk-in, call, apply online)

**Avoid:** Marketing language, vague phrases, unexplained acronyms, long run-ons, jargon

**Output format — MANDATORY:**
Line 1: score (e.g. 4)
Line 2: ---
Line 3+: structured review

Example:
4
---
PER-RESOURCE DESCRIPTION REVIEW:
1. Workforce Solutions North
   - What/Who/How: COMPLETE - describes job search help, open to adults, walk-in or appointment
   - Accuracy: VERIFIED - matches wfscapitalarea.com services page
   - Readability: 8TH GRADE - plain language, no jargon
   - Score: 4
   - Issues: Eligibility details slightly vague ("some programs have eligibility rules")

IMPROVEMENTS NEEDED:
- Specify which programs require eligibility vs open-to-all services"""

    def _build_user_prompt(self, query: str, output: str, location: str) -> str:
        return f"""Evaluate the quality of resource descriptions in this referral output.

**Client Query:** {query}

**Client Location:** {location}

**Referral Output:**
{output}

Provide your numeric score (1-5) on line 1, then --- on line 2, then your detailed reasoning."""
