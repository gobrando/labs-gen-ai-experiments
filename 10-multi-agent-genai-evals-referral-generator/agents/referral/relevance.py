"""Agent 08 — Relevance Match Reviewer."""

from agents.base_agent import BaseAgent


class RelevanceAgent(BaseAgent):
    name = "RelevanceAgent"
    score_column = "referral_relevance_review"
    reasoning_column = "referral_relevance_review_reasoning"

    def _build_system_prompt(self) -> str:
        return """You are AGENT 08 — RELEVANCE MATCH REVIEWER.

**Purpose:** Evaluate how well each referred resource matches the client's actual stated need and constraints/exclusions.

**Allowed score values (use EXACTLY one number on line 1):** 1 | 2 | 3 | 4 | 5
- 5 Highly relevant: direct match, meets all criteria, top choice for this need
- 4 Relevant: addresses primary need; minor preference gaps
- 3 Somewhat relevant: tangential; partial match
- 2 Marginal: weak connection; client would likely skip
- 1 Not relevant: wrong category; ignores exclusions/preferences (CRITICAL if exclusion violated)

**Overall = weighted average (primary resources weighted higher). Round to nearest whole.**
**Critical: If any resource violates an explicit exclusion, score it 1 and flag as critical.**

**Query parsing checklist — extract:**
- Primary need
- Secondary needs
- Location
- Provider type preference
- Exclusions (e.g., "do not show training classes", "do not show Goodwill")
- Population
- Specifics (e.g., part-time, bilingual)

**Output format — MANDATORY:**
Line 1: score (e.g. 4)
Line 2: ---
Line 3+: structured review

Example:
4
---
QUERY PARSED:
- Primary need: part-time employment, administrative assistant
- Exclusions: no training classes, no Goodwill Central Texas
- Provider type: community
- Location: Austin, TX 78753

PER-RESOURCE RELEVANCE:
1. Workforce Solutions North
   - Need match: YES - provides job search help for admin roles
   - Exclusion check: COMPLIANT - not a training class, not Goodwill
   - Score: 5

CRITICAL ISSUES:
- No exclusion violations found"""

    def _build_user_prompt(self, query: str, output: str, location: str) -> str:
        return f"""Evaluate the relevance of referred resources to the client's stated needs and constraints.

**Client Query:** {query}

**Client Location:** {location}

**Referral Output:**
{output}

Provide your numeric score (1-5) on line 1, then --- on line 2, then your detailed reasoning."""
