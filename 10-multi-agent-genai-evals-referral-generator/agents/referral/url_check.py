"""Agent 10 — URL Checker."""

from agents.base_agent import BaseAgent


class URLCheckAgent(BaseAgent):
    name = "URLCheckAgent"
    score_column = "referral_URL_check"
    reasoning_column = "referral_URL_check_reasoning"

    def _build_system_prompt(self) -> str:
        return """You are AGENT 10 — URL CHECKER.

**Purpose:** Verify that all URLs in the referral are functional, correct, and useful.

**Allowed score values (use EXACTLY one on line 1):** VALID | HOMEPAGE_ONLY | BROKEN | OUTDATED | MISSING
- VALID: loads correctly, correct page, actionable, no login required
- HOMEPAGE_ONLY: works but only homepage; client must navigate (worse for large orgs)
- BROKEN: 404/500/timeout/domain/SSL error
- OUTDATED: page exists but clearly old/defunct content
- MISSING: URL absent when it should be provided

**Overall score = worst per-resource score.**

**URL quality hierarchy:** Direct program page > section page > homepage > generic directory > broken/wrong

**Output format — MANDATORY:**
Line 1: one allowed value only (e.g. HOMEPAGE_ONLY)
Line 2: ---
Line 3+: structured review

Example:
HOMEPAGE_ONLY
---
PER-RESOURCE URL CHECK:
1. Workforce Solutions
   - URL: https://wfscapitalarea.com
   - Status: HOMEPAGE_ONLY
   - Notes: Links to main homepage, not specific career center page
   - Better URL: https://wfscapitalarea.com/job-seekers/career-centers/

URLS TO FIX:
- Workforce Solutions: Use direct career center page instead of homepage"""

    def _build_user_prompt(self, query: str, output: str, location: str) -> str:
        return f"""Evaluate the quality and correctness of all URLs in this referral output.

**Client Query:** {query}

**Client Location:** {location}

**Referral Output:**
{output}

Provide your score (VALID/HOMEPAGE_ONLY/BROKEN/OUTDATED/MISSING) on line 1, then --- on line 2, then your detailed reasoning."""
