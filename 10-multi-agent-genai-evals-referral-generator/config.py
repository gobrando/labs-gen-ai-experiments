"""Configuration settings for the Phoenix Logs Agent Evals app."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load env from local project first, then common fallback locations.
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()  # current process defaults
load_dotenv(Path.home() / ".env")
load_dotenv(Path.home() / "phoenix-logs-agent-evals" / ".env")

# LLM Provider Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "openai")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")

# Model options per provider
MODEL_OPTIONS = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ],
    "anthropic": [
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-haiku-20240307",
    ],
}

# Column mappings for evaluation results
REFERRAL_COLUMNS = [
    "referral_service_area_eligibility",
    "referral_service_area_eligibility_reasoning",
    "referral_location_proximity",
    "referral_location_proximity_reasoning",
    "referral_contact_info",
    "referral_contact_info_reasoning",
    "referral_URL_check",
    "referral_URL_check_reasoning",
    "referral_description_review",
    "referral_description_review_reasoning",
    "referral_missing_resources",
    "referral_missing_resources_reasoning",
    "referral_relevance_review",
    "referral_relevance_review_reasoning",
    "referral_service_status",
    "referral_service_status_reasoning",
    "referral_overall_review",
    "referral_overall_review_reasoning",
]

ACTION_PLAN_COLUMNS = [
    "actionplan_overallreview",
    "actionplan_overallreview_reasoning",
]

READABILITY_COLUMNS = [
    "flesch_ease",
    "flesch_grade",
    "gunning_fog",
    "smog_index",
    "dale_chall",
    "cleaned_text",
]

TRACE_METADATA_COLUMNS = [
    "web_search_performed",
    "web_search_used",
    "web_search_detection_source",
    "prompted_categories",
    "region_bucket",
]

# All evaluation columns
ALL_EVAL_COLUMNS = (
    REFERRAL_COLUMNS
    + ACTION_PLAN_COLUMNS
    + READABILITY_COLUMNS
    + TRACE_METADATA_COLUMNS
)
