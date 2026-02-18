"""Text cleaning utilities for readability analysis."""

import re


def clean_text(raw_output: str) -> str:
    """Clean raw referral/action plan output for readability scoring.

    Strips structural markers, URLs, addresses, phone numbers, and
    formatting artifacts, leaving only the descriptive prose that
    a readability formula should measure.

    Args:
        raw_output: The full_output text from the trace log.

    Returns:
        Cleaned text suitable for readability metrics.
    """
    if not raw_output or not isinstance(raw_output, str):
        return ""

    text = raw_output

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove phone numbers (various formats)
    text = re.sub(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", "", text)

    # Remove standalone field labels (Resource 1:, Name:, Address:, etc.)
    text = re.sub(r"^(Resource \d+|Name|Address|Phone|Website|Description):\s*", "", text, flags=re.MULTILINE)

    # Remove "Resources Found: N" header
    text = re.sub(r"Resources Found:\s*\d+", "", text)

    # Remove markdown formatting
    text = re.sub(r"[*#_`>]", "", text)

    # Remove bullet markers
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)

    # Remove numbered list markers at start of line
    text = re.sub(r"^\s*\d+[.)]\s*", "", text, flags=re.MULTILINE)

    # Collapse multiple newlines and whitespace
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Strip leading/trailing whitespace per line and overall
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    text = " ".join(lines)

    return text
