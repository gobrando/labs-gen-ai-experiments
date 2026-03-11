"""Readability metrics calculator using textstat."""

from __future__ import annotations

from typing import Dict, Union

import textstat


def calculate_readability_metrics(text: str) -> Dict[str, Union[float, str]]:
    """Calculate all readability metrics for a cleaned text.

    Args:
        text: Cleaned text (from text_cleaner.clean_text).

    Returns:
        Dict with keys: flesch_ease, flesch_grade, gunning_fog,
        smog_index, dale_chall.
    """
    if not text or len(text.split()) < 10:
        return {
            "flesch_ease": "",
            "flesch_grade": "",
            "gunning_fog": "",
            "smog_index": "",
            "dale_chall": "",
        }

    return {
        "flesch_ease": round(textstat.flesch_reading_ease(text), 2),
        "flesch_grade": round(textstat.flesch_kincaid_grade(text), 2),
        "gunning_fog": round(textstat.gunning_fog(text), 2),
        "smog_index": round(textstat.smog_index(text), 2),
        "dale_chall": round(textstat.dale_chall_readability_score(text), 2),
    }
