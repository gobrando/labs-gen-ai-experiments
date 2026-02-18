"""Agent 09 — Service Status Verifier."""

from agents.base_agent import BaseAgent


class ServiceStatusAgent(BaseAgent):
    name = "ServiceStatusAgent"
    score_column = "referral_service_status"
    reasoning_column = "referral_service_status_reasoning"

    def _build_system_prompt(self) -> str:
        return """You are AGENT 09 — SERVICE STATUS VERIFIER.

**Purpose:** Verify that each referred resource is currently operating as described.

**Allowed score values (use EXACTLY one on line 1):** ALL_ACTIVE | SOME_CHANGES | HAS_CLOSURES
- ALL_ACTIVE: confirmed operating; hours/location match; programs still offered
- SOME_CHANGES: operating but changed hours/location/program naming not reflected in referral
- HAS_CLOSURES: one or more permanently closed / discontinued / not accepting new clients

**Overall = worst score (any CLOSED → HAS_CLOSURES).**

**Mandatory verification for each resource:**
1) Google Maps: open/closed flags, hours, address match
2) Official website: alerts, contact page, programs page
3) News search: "{resource} closed/closing" (past 12 months)
4) Backup: findhelp.org, 211, social media last post date

**Output format — MANDATORY:**
Line 1: one allowed value only (e.g. ALL_ACTIVE)
Line 2: ---
Line 3+: structured review

Example:
SOME_CHANGES
---
PER-RESOURCE STATUS:
1. Workforce Solutions North
   - Google Maps: OPEN
   - Website: ACTIVE - current contact info and programs listed
   - Status: ACTIVE

2. East Career Center
   - Google Maps: OPEN
   - Website: CHANGED - hours now show Mon-Thu only (was Mon-Fri)
   - Status: CHANGED

REQUIRED UPDATES:
- East Career Center: Update hours to Mon-Thu per current website"""

    def _build_user_prompt(self, query: str, output: str, location: str) -> str:
        return f"""Verify the operational status of each referred resource.

**Client Query:** {query}

**Client Location:** {location}

**Referral Output:**
{output}

Provide your score (ALL_ACTIVE/SOME_CHANGES/HAS_CLOSURES) on line 1, then --- on line 2, then your detailed reasoning."""
