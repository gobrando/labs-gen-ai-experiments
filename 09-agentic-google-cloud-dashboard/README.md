# ⚡ Agentic AI Dashboard

A real-time monitoring dashboard for Nava's AI agents, built with Streamlit and connected to Google BigQuery. Deployed on Hugging Face Spaces.

🔗 **Live Dashboard:** [huggingface.co/spaces/brandonava/agentic-dashboard](https://huggingface.co/spaces/brandonava/agentic-dashboard)

## Features

- 📊 **Real-time metrics** — Request volume, token usage, success rates pulled live from BigQuery
- 📈 **Trend visualization** — Daily and hourly request patterns with interactive Plotly charts
- 🎯 **Model distribution** — See which models are being used and how
- 💰 **Cost tracking** — Estimated API costs based on token usage
- 📅 **Date range filtering** — Filter traces across any date range since the product started
- 🗂️ **File upload** — Upload local CSV or Excel trace logs for non-BigQuery analysis
- 📝 **Weekly Report** — Executive summary, usage & engagement metrics, error summaries, and user feedback
- 🔐 **Authentication** — Email-based login to restrict access
- 🌙 **Dark theme** — Easy on the eyes

## Access

The dashboard is protected by email authentication. Reach out to Brandon or Christine to be added as a user.

## Data Source

The dashboard connects to BigQuery by default:
- **Project**: `nava-labs`
- **Dataset**: `anthropic_logging`
- **Table**: `request_response_logging`

Alternatively, you can upload a local CSV or Excel trace log file directly from the sidebar.

## Local Development

### 1. Install Dependencies

```bash
cd agentic-dashboard
pip install -r requirements.txt
```

### 2. Set Up Credentials

```bash
# Option A: Use a service account key file
cp service-account.json.example service-account.json
# (replace with your actual key)

# Option B: Use gcloud CLI
gcloud auth application-default login
```

### 3. Run the Dashboard

```bash
streamlit run app.py --server.port 8510
```

The dashboard will open at `http://localhost:8510`

## Deployment (Hugging Face Spaces)

The app is deployed as a Docker-based Hugging Face Space.

### Environment Variables (set in HF Space Settings → Variables and Secrets)

| Variable | Description |
|----------|-------------|
| `GCP_SERVICE_ACCOUNT_JSON` | Full JSON contents of the GCP service account key (set as a **Secret**) |

### Key Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Builds the Docker image — installs deps, configures Streamlit, and runs the app on port 7860 |
| `app.py` | Main Streamlit application with dashboard logic, data parsing, and visualizations |
| `requirements.txt` | Python dependencies |

### Redeploying

Push changes to the Hugging Face Space repo, or upload updated files through the HF web UI. The Space will automatically rebuild.

## Project Structure

```
agentic-dashboard/
├── app.py                 # Main Streamlit application
├── Dockerfile             # Docker config for Hugging Face Spaces
├── requirements.txt       # Python dependencies
├── .streamlit/
│   ├── config.toml        # Streamlit theme & server config
│   └── secrets.toml.example  # Example secrets for local dev
└── README.md              # This file
```

## Metrics Tracked

| Metric | Description |
|--------|-------------|
| Total Requests | Sum of all API requests in the selected date range |
| Avg Daily Requests | Average requests per day |
| Peak Hour Load | Maximum requests in a single hour |
| Success Rate | Percentage of successful responses |
| Token Usage | Total input/output tokens consumed |
| Estimated Cost | Projected API costs based on model and token usage |

---

Built with ❤️ using Streamlit, Plotly, and Google BigQuery · Deployed on [Hugging Face Spaces](https://huggingface.co/spaces/brandonava/agentic-dashboard)
