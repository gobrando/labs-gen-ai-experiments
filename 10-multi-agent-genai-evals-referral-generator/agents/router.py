"""Router to classify trace output type and dispatch to appropriate agents."""

from typing import Literal


OutputType = Literal["referral", "actionplan", "unknown"]


class OutputTypeRouter:
    """Classify whether a trace log row is a referral or action plan
    based on the prompt_type column.
    """

    REFERRAL_KEYWORDS = ["referral", "referraltx", "referralkeystone", "referralpa"]
    ACTIONPLAN_KEYWORDS = ["actionplan", "action_plan", "action plan"]

    @staticmethod
    def classify(prompt_type: str) -> OutputType:
        """Classify the output type from the prompt_type field.

        Args:
            prompt_type: The prompt_type value from the CSV row.

        Returns:
            'referral', 'actionplan', or 'unknown'
        """
        if not prompt_type:
            return "unknown"

        pt = prompt_type.strip().lower()

        for kw in OutputTypeRouter.REFERRAL_KEYWORDS:
            if kw in pt:
                return "referral"

        for kw in OutputTypeRouter.ACTIONPLAN_KEYWORDS:
            if kw in pt:
                return "actionplan"

        return "unknown"
