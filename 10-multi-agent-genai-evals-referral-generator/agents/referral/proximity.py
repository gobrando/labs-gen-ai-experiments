"""Agent 05 — Proximity / Distance Scorer."""

from agents.base_agent import BaseAgent


class ProximityAgent(BaseAgent):
    name = "ProximityAgent"
    score_column = "referral_location_proximity"
    reasoning_column = "referral_location_proximity_reasoning"

    def _build_system_prompt(self) -> str:
        return """You are AGENT 05 — PROXIMITY SCORER.

**Purpose:** Assess how close/convenient the referred resources are to the client's stated location, considering transportation barriers.

**Allowed score values (use EXACTLY one number on line 1):** 1 | 2 | 3 | 4 | 5 | N/A
- 5 Excellent: within 5 miles / same ZIP; on transit; virtual option
- 4 Good: within 10 miles; same city; transit ≤1 transfer
- 3 Adequate: within 15-20 miles; multiple transfers / car likely
- 2 Poor: 20-30 miles; different city; hard without vehicle
- 1 Unacceptable: >30 miles; different county/region; impractical
- N/A: virtual/phone-only; no physical visit required

**Overall = weighted average of non-N/A scores, rounded to nearest whole number.**

**Mandatory:** Estimate Google Maps distance from client ZIP/city to each resource address.

**Output format — MANDATORY:**
Line 1: score (e.g. 4)
Line 2: ---
Line 3+: structured review

Example:
4
---
CLIENT LOCATION: Austin, TX 78753

PER-RESOURCE PROXIMITY:
1. North Career Center: 5 - ~1 mile from 78753, same ZIP, on bus route 1
2. East Career Center: 3 - ~8 miles, different area, requires bus transfer

NOTES:
- Both resources are within Austin city limits but East Center requires transit transfer"""

    def _build_user_prompt(self, query: str, output: str, location: str) -> str:
        return f"""Evaluate proximity/convenience of referred resources to the client's location.

**Client Query:** {query}

**Client Location:** {location}

**Referral Output:**
{output}

Provide your numeric score (1-5 or N/A) on line 1, then --- on line 2, then your detailed reasoning."""
