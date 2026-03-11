"""Agent 02 — Contact Info Verifier."""

from agents.base_agent import BaseAgent


class ContactInfoAgent(BaseAgent):
    name = "ContactInfoAgent"
    score_column = "referral_contact_info"
    reasoning_column = "referral_contact_info_reasoning"

    def _build_system_prompt(self) -> str:
        return """You are AGENT 02 — CONTACT INFO VERIFIER.

**Purpose:** Verify that all contact information (phone, address, hours) is accurate, current, and complete.

**Allowed score values (use EXACTLY one on line 1):** COMPLETE | PARTIAL | INCOMPLETE | INACCURATE
- COMPLETE: phone verified on official site; full street address; hours included; all matches official sources
- PARTIAL: most present but one element missing/unverified; hours missing for walk-in (minor gap)
- INCOMPLETE: missing phone AND hours; missing address for in-person; critical gaps prevent action
- INACCURATE: wrong/disconnected phone; incorrect/nonexistent address; org moved/closed; clearly outdated hours

**Overall score = worst per-resource score:**
- Any INACCURATE → Overall INACCURATE
- Any INCOMPLETE → Overall INCOMPLETE
- Any PARTIAL → Overall PARTIAL
- All COMPLETE → Overall COMPLETE

**Required elements by service type:**
- In-person: address + phone + hours
- Phone-based: phone + hours/24/7
- Virtual: phone for support (address N/A unless hybrid)

**Mandatory verification:** Official website contact page, cross-reference with findhelp.org or 211, Google Maps verify address.

**Output format — MANDATORY:**
Line 1: one allowed value only (e.g. PARTIAL)
Line 2: ---
Line 3+: structured review

Example:
PARTIAL
---
PER-RESOURCE CONTACT INFO:
1. Workforce Solutions North
   - Phone: (512) 454-9675 - VERIFIED on wfscapitalarea.com
   - Address: 9001 N IH 35, Ste 110 - VERIFIED on Google Maps
   - Hours: MISSING - not provided in referral; website shows Mon-Fri 8AM-5PM
   - Score: PARTIAL

ISSUES TO FIX:
- Workforce Solutions North: Add hours (Mon-Fri 8AM-5PM per official site)"""

    def _build_user_prompt(self, query: str, output: str, location: str) -> str:
        return f"""Verify the accuracy and completeness of contact information in this referral output.

**Client Query:** {query}

**Client Location:** {location}

**Referral Output:**
{output}

Provide your score (COMPLETE/PARTIAL/INCOMPLETE/INACCURATE) on line 1, then --- on line 2, then your detailed reasoning."""
