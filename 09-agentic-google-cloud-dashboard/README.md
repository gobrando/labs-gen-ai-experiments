# 09 - Agentic Google Cloud Dashboard

A Streamlit-powered dashboard for monitoring AI agent trace logs from Google Cloud BigQuery.

## Features

- **Real-time BigQuery Integration** — Live data from Claude API trace logs
- **Smart Caching** — Parquet-based local cache to avoid re-fetching thousands of traces
- **Email Authentication** — Only authorized users can access the dashboard
- **Interactive Charts** — Request volume, token usage, model distribution, cost analysis
- **Weekly Reports** — Auto-generated executive summaries with week-over-week metrics
- **File Upload** — Non-technical users can upload CSV/Excel trace logs directly

## Quick Start

### Local Development

```bash
cd 09-agentic-google-cloud-dashboard
pip install -r requirements.txt

# Add your service account key
cp /path/to/your/service-account.json .

# Run the dashboard
streamlit run app.py --server.port 8510
```

### Streamlit Cloud Deployment

1. Connect your GitHub repo at [share.streamlit.io](https://share.streamlit.io)
2. Set the main file path: `09-agentic-google-cloud-dashboard/app.py`
3. Add secrets in the Streamlit Cloud dashboard (see `.streamlit/secrets.toml.example`)

## Authentication

Default credentials (change in production via `st.secrets`):
- Email: `brandoncanniff@navapbc.com`
- Email: `christinewilkes@navapbc.com`
- Password: `nava2026`

## Architecture

- **Data Source**: Google Cloud BigQuery (`nava-labs.anthropic_logging.request_response_logging`)
- **Auth**: Email/password via `st.secrets` or hardcoded defaults
- **Caching**: Local parquet files with 1-hour TTL
- **Deployment**: Streamlit Community Cloud (free tier)
