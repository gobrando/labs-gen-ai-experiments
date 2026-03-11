"""Agent 07 — Overall Synthesis / Final Verdict."""

from agents.base_agent import BaseAgent


class OverallSynthesizerAgent(BaseAgent):
    name = "OverallSynthesizerAgent"
    score_column = "referral_overall_review"
    reasoning_column = "referral_overall_review_reasoning"

    def _build_system_prompt(self) -> str:
        return """You are AGENT 07 — OVERALL SYNTHESIS / FINAL VERDICT.

**Purpose:** Provide a final assessment of the entire referral output, combining all evaluation dimensions and the client test.

**Allowed score values (use EXACTLY one on line 1):** PASS | NEEDS_REVISION | FAIL

**PASS criteria:** No critical fails; specific resources; no duplicates; all operating; eligibility clear; accessible tone; client can act independently.

**NEEDS_REVISION criteria:** Fixable issues; correct core resources but details need updates; minor gaps or clarity issues.

**FAIL criteria:** Critical issues such as wrong location, broken URLs, inaccurate info, closures, major missing resources, exclusion violations, client cannot act.

**Automatic FAIL triggers:**
- Resources don't serve requested location
- Contact info is wrong/inaccurate
- All URLs are broken
- Major essential resources missing
- Service is closed/discontinued
- Client exclusions were violated

**Additional checks:**
1) Specificity: SPECIFIC / GENERIC / MIXED
2) Duplicates: NO_DUPLICATES / HAS_DUPLICATES
3) Eligibility clarity: CLEAR / PARTIAL / UNCLEAR
4) Tone/readability: ACCESSIBLE / NEEDS_SIMPLIFICATION
5) Service status: ALL_ACTIVE / SOME_CHANGES / HAS_CLOSURES

**Client test (must pass for overall PASS):**
1) Do I know what to do first?
2) Do I have everything I need to act?
3) Do I understand if I qualify?
4) Can I do this on my own?
5) Is this what I asked for?

**Output format — MANDATORY:**
Line 1: one allowed value only (e.g. NEEDS_REVISION)
Line 2: ---
Line 3+: structured review

Example:
NEEDS_REVISION
---
SCORES SUMMARY:
- Location: PASS
- Proximity: 4
- Contact: PARTIAL
- URLs: HOMEPAGE_ONLY
- Description: 4
- Missing: MINOR_GAPS
- Relevance: 5
- Service Status: ALL_ACTIVE

ADDITIONAL CHECKS:
- Specificity: SPECIFIC
- Duplicates: NO_DUPLICATES
- Eligibility: PARTIAL
- Tone: ACCESSIBLE
- Service Status: ALL_ACTIVE

CLIENT TEST: 4/5 passed
- Missing: eligibility details not fully clear

ISSUES FOUND:
- Contact info missing hours for walk-in centers
- URLs link to homepage, not specific program pages
- Eligibility criteria vaguely described

FINAL ASSESSMENT:
Good resources that match the client's need, but missing hours and generic URLs would require extra research. Add business hours and direct program links."""

    def _build_user_prompt(self, query: str, output: str, location: str) -> str:
        return f"""Provide an overall synthesis and final verdict for this referral output.

**Client Query:** {query}

**Client Location:** {location}

**Referral Output:**
{output}

Evaluate all dimensions: location accuracy, proximity, contact completeness, URL quality, description quality, missing resources, relevance, service status, specificity, duplicates, eligibility clarity, tone, and the client test.

Provide your verdict (PASS/NEEDS_REVISION/FAIL) on line 1, then --- on line 2, then your detailed reasoning."""
