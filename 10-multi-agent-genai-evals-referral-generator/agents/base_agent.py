"""Base agent class for all evaluation agents."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, Optional

import litellm


class BaseAgent:
    """Base class for all evaluation agents.

    Each agent has a name, a rubric (system prompt), and produces
    a score + reasoning pair for its assigned column.
    """

    name: str = "BaseAgent"
    score_column: str = ""
    reasoning_column: str = ""

    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        self.provider = provider
        self.model = model

    def _get_model_string(self) -> str:
        """Build the LiteLLM model string."""
        if self.provider == "openai":
            return self.model
        elif self.provider == "anthropic":
            return f"anthropic/{self.model}"
        return self.model

    def _build_system_prompt(self) -> str:
        """Return the system prompt / rubric for this agent.
        Override in subclasses.
        """
        raise NotImplementedError

    def _build_user_prompt(self, query: str, output: str, location: str) -> str:
        """Return the user prompt with the trace data to evaluate.
        Override in subclasses.
        """
        raise NotImplementedError

    def _parse_response(self, raw_response: str) -> dict[str, str]:
        """Parse the LLM response into score + reasoning.

        Expects format:
            SCORE_VALUE
            ---
            Reasoning text...

        Returns dict with keys matching score_column and reasoning_column.
        """
        lines = raw_response.strip().split("\n", 2)

        score = lines[0].strip() if lines else "ERROR"
        # Find reasoning after the --- separator
        reasoning = ""
        if "---" in raw_response:
            parts = raw_response.split("---", 1)
            reasoning = parts[1].strip() if len(parts) > 1 else ""
        elif len(lines) > 1:
            reasoning = "\n".join(lines[1:]).strip()

        return {
            self.score_column: score,
            self.reasoning_column: reasoning,
        }

    async def evaluate(
        self, query: str, output: str, location: str
    ) -> dict[str, str]:
        """Run evaluation and return score + reasoning dict."""
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, output, location)

        try:
            response = await litellm.acompletion(
                model=self._get_model_string(),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content or ""
            return self._parse_response(raw)
        except Exception as e:
            return {
                self.score_column: "ERROR",
                self.reasoning_column: f"Agent error: {str(e)}",
            }
