# Phoenix Logs Multi-Agent Evaluator

A multi-agent workflow application that evaluates the quality of GenAI outputs from Phoenix Arize trace logs. The system uses specialized AI agents running in parallel to grade referrals and action plans against detailed rubrics.

## Features

- **Multi-Agent Evaluation**: 9 specialized agents for referrals + 1 for action plans
- **Parallel Processing**: Agents run concurrently via asyncio for fast evaluation
- **Multiple LLM Providers**: Supports OpenAI and Anthropic models via LiteLLM
- **Reasoning Columns**: Each score is paired with a text explanation of why the agent gave that score
- **Trace Metadata Columns**: Adds `web_search_performed`, `prompted_categories`, and `region_bucket`
- **Readability Metrics**: Automatic Flesch, Gunning Fog, SMOG, and Dale-Chall scores
- **Streamlit Dashboard**: Interactive web UI with color-coded results, expandable row details
- **Macro Analytics**: Keystone vs Texas web-search proportions, web-search rates, and top prompted categories
- **Export Options**: Download results as CSV or formatted Excel with conditional formatting

If your CSV does not include span payloads, enable **Phoenix Enrichment** in the sidebar and provide your Phoenix base URL to resolve `web_search_performed` from trace/span APIs.

## Installation

```bash
cd phoenix-logs-agent-evals

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Edit `.env` file with your credentials:

```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
DEFAULT_PROVIDER=openai
DEFAULT_MODEL=gpt-4o-mini
```

## Usage

### Streamlit Web App

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

1. Enter your API key in the sidebar
2. Select your LLM provider and model
3. Upload your Phoenix trace log CSV
4. Choose which rows to evaluate
5. Click "Run Evaluation"
6. View scores and reasoning in the dashboard
7. Download as CSV or Excel

## Evaluation Agents

### Referral Agents (9 total)

| Agent | Column | Score Type |
|-------|--------|------------|
| Service Area Checker | `referral_service_area_eligibility` | PASS/FAIL/PARTIAL/N/A |
| Proximity Scorer | `referral_location_proximity` | 1-5 |
| Contact Info Verifier | `referral_contact_info` | COMPLETE/PARTIAL/INCOMPLETE/INACCURATE |
| URL Checker | `referral_URL_check` | VALID/HOMEPAGE_ONLY/BROKEN/OUTDATED/MISSING |
| Description Reviewer | `referral_description_review` | 1-5 |
| Missing Resources Auditor | `referral_missing_resources` | NONE_MISSING/MINOR_GAPS/MAJOR_GAPS |
| Relevance Reviewer | `referral_relevance_review` | 1-5 |
| Service Status Verifier | `referral_service_status` | ALL_ACTIVE/SOME_CHANGES/HAS_CLOSURES |
| Overall Synthesizer | `referral_overall_review` | PASS/NEEDS_REVISION/FAIL |

Each score column has a paired `*_reasoning` column with the agent's text explanation.

### Action Plan Agent

| Agent | Column | Score Type |
|-------|--------|------------|
| Action Plan Reviewer | `actionplan_overallreview` | PASS/NEEDS_REVISION/FAIL |

Also includes `actionplan_overallreview_reasoning` with detailed review text.

### Readability Metrics (Automatic, no LLM needed)

| Metric | Column | Description |
|--------|--------|-------------|
| Flesch Reading Ease | `flesch_ease` | 0-100 (higher = easier) |
| Flesch-Kincaid Grade | `flesch_grade` | US grade level |
| Gunning Fog Index | `gunning_fog` | Years of education needed |
| SMOG Index | `smog_index` | Years of education needed |
| Dale-Chall Score | `dale_chall` | Readability score |
| Cleaned Text | `cleaned_text` | Cleaned text used for analysis |

## Input CSV Format

The input CSV should contain these columns (from Phoenix export):

| Column | Required | Description |
|--------|----------|-------------|
| `prompt_type` | Yes | Type identifier (e.g., "referraltx", "actionplan") |
| `natural_language_query` | Yes | The client's original query |
| `full_output` | Yes | The GenAI-generated output to evaluate |
| `location` | Recommended | Client's location for geographic checks |
| `trace_id` | Recommended | Phoenix trace identifier |

## Project Structure

```
phoenix-logs-agent-evals/
├── app.py                      # Streamlit web application
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── .env.example                # Environment template
├── agents/
│   ├── base_agent.py           # Base agent class with LiteLLM
│   ├── router.py               # Output type classifier (referral vs action plan)
│   ├── referral/
│   │   ├── service_area.py     # referral_service_area_eligibility
│   │   ├── proximity.py        # referral_location_proximity
│   │   ├── contact_info.py     # referral_contact_info
│   │   ├── url_check.py        # referral_URL_check
│   │   ├── description.py      # referral_description_review
│   │   ├── missing_resources.py# referral_missing_resources
│   │   ├── relevance.py        # referral_relevance_review
│   │   ├── service_status.py   # referral_service_status
│   │   └── overall.py          # referral_overall_review
│   └── actionplan/
│       └── overall_review.py   # actionplan_overallreview
├── utils/
│   ├── csv_handler.py          # CSV/Excel read/write with formatting
│   ├── readability.py          # Readability metrics via textstat
│   └── text_cleaner.py         # Text preprocessing for readability
├── rubrics/                    # Evaluation rubric documents
│   ├── referral_rubrics.md
│   ├── referral_review_rubric.md
│   └── action_plan_rubric.md
└── data/                       # Data files (gitignored)
```

## License

MIT
