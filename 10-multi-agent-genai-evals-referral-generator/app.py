"""Phoenix Logs Agent Evals — Streamlit Application.

Multi-agent workflow that evaluates GenAI trace log outputs
(referrals and action plans) against quality rubrics.
"""

import asyncio
import base64
import os
import re
import sys
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
import pandas as pd
import streamlit as st

# Ensure project root is on path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ALL_EVAL_COLUMNS,
    MODEL_OPTIONS,
    REFERRAL_COLUMNS,
    ACTION_PLAN_COLUMNS,
    READABILITY_COLUMNS,
)
from agents.router import OutputTypeRouter
from agents.referral import (
    ServiceAreaAgent,
    ProximityAgent,
    ContactInfoAgent,
    URLCheckAgent,
    DescriptionAgent,
    MissingResourcesAgent,
    RelevanceAgent,
    ServiceStatusAgent,
    OverallSynthesizerAgent,
)
from agents.actionplan import ActionPlanReviewerAgent
from utils.csv_handler import load_csv, ensure_eval_columns, save_csv, save_excel
from utils.text_cleaner import clean_text
from utils.readability import calculate_readability_metrics


# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Phoenix Logs Agent Evals",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp {
        max-width: 100%;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 0.75rem;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-card p {
        margin: 0.25rem 0 0;
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .pass-card { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .warn-card { background: linear-gradient(135deg, #F2994A 0%, #F2C94C 100%); }
    .fail-card { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_referral_agents(provider: str, model: str) -> list:
    """Instantiate all referral evaluation agents."""
    return [
        ServiceAreaAgent(provider, model),
        ProximityAgent(provider, model),
        ContactInfoAgent(provider, model),
        URLCheckAgent(provider, model),
        DescriptionAgent(provider, model),
        MissingResourcesAgent(provider, model),
        RelevanceAgent(provider, model),
        ServiceStatusAgent(provider, model),
        OverallSynthesizerAgent(provider, model),
    ]


def get_actionplan_agents(provider: str, model: str) -> list:
    """Instantiate action plan evaluation agents."""
    return [
        ActionPlanReviewerAgent(provider, model),
    ]


def infer_region_bucket(prompt_type: str, location: str) -> str:
    """Infer whether a row belongs to Texas vs Keystone."""
    pt = (prompt_type or "").lower()
    loc = (location or "").lower()
    if "keystone" in pt or "referralpa" in pt or "pennsylvania" in loc or " pa" in loc:
        return "keystone"
    if "tx" in pt or "texas" in loc or " austin" in loc:
        return "texas"
    return "unknown"


def extract_prompted_categories(query: str, existing_category_type: str = "") -> str:
    """Extract category/categories from natural_language_query and fallback columns."""
    query_text = query or ""
    categories = []

    patterns = [
        r"Include resources that support the following categories:\s*(.+)",
        r"categories:\s*(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query_text, flags=re.IGNORECASE)
        if match:
            raw = match.group(1).strip()
            # Stop at next line if multiline.
            raw = raw.splitlines()[0].strip()
            pieces = re.split(r",|/|&| and ", raw, flags=re.IGNORECASE)
            categories.extend([p.strip() for p in pieces if p.strip()])
            break

    # Fallback from existing Category_type values like "employment_tx; training_tx"
    if not categories and existing_category_type:
        raw_items = [x.strip() for x in str(existing_category_type).split(";") if x.strip()]
        for item in raw_items:
            cleaned = re.sub(r"_(tx|pa)$", "", item, flags=re.IGNORECASE).replace("_", " ").strip()
            if cleaned:
                categories.append(cleaned)

    # Deduplicate preserving order.
    deduped = []
    seen = set()
    for cat in categories:
        key = cat.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(cat)

    return "; ".join(deduped)


def detect_web_search_performed(row: pd.Series) -> str:
    """Detect whether web search tool execution appears in span-like columns.

    Returns:
        YES | NO | UNKNOWN
    """
    span_like_cols = [
        col
        for col in row.index
        if any(token in col.lower() for token in ("span", "event", "tool", "trace_json", "trace_data"))
    ]

    if not span_like_cols:
        return "UNKNOWN"

    searchable = " ".join(str(row.get(col, "")) for col in span_like_cols).lower()
    if not searchable.strip():
        return "UNKNOWN"

    if any(token in searchable for token in ("openaiwebsearch", "web_search", "websearch")):
        return "YES"
    return "NO"


def _extract_strings_recursive(obj: Any) -> str:
    """Flatten nested json-like objects into a lowercase searchable string."""
    if isinstance(obj, dict):
        return " ".join(_extract_strings_recursive(v) for v in obj.values())
    if isinstance(obj, list):
        return " ".join(_extract_strings_recursive(v) for v in obj)
    return str(obj)


def _contains_web_search_marker(content_text: str) -> bool:
    """Match common span/event names that indicate web search tool use."""
    normalized = content_text.lower()
    return any(
        marker in normalized
        for marker in (
            "openaiwebsearch",
            "openai_web_search",
            "web_search",
            "websearch",
            "web search",
        )
    )


def _decode_selected_span_node_id(encoded_id: str) -> str:
    """Decode selectedSpanNodeId (base64 like 'U3BhbjozMTcyMw==') to numeric span id."""
    if not encoded_id:
        return ""
    try:
        decoded = base64.b64decode(encoded_id).decode("utf-8")
    except Exception:
        return ""
    if ":" in decoded:
        return decoded.split(":")[-1].strip()
    return decoded.strip()


def parse_phoenix_link(link: str) -> dict:
    """Parse phoenix_link into base URL and ids used by API fallbacks."""
    if not link:
        return {"base_url": "", "project_id_token": "", "span_path_id": "", "selected_span_id": ""}

    selected_span_id = ""
    selected_match = re.search(r"selectedSpanNodeId=([^&]+)", link)
    if selected_match:
        selected_span_id = _decode_selected_span_node_id(selected_match.group(1))

    project_match = re.search(r"/projects/([^/]+)/", link)
    span_match = re.search(r"/spans/([^/?]+)", link)

    parsed = urlparse(link)
    base_url = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else ""

    return {
        "base_url": base_url,
        "project_id_token": project_match.group(1) if project_match else "",
        "span_path_id": span_match.group(1) if span_match else "",
        "selected_span_id": selected_span_id,
    }


async def _lookup_trace_web_search_status(
    client: httpx.AsyncClient,
    base_url: str,
    trace_id: str,
    span_path_id: str = "",
    selected_span_id: str = "",
    project_id_token: str = "",
) -> str:
    """Try several Phoenix endpoints and infer YES/NO/UNKNOWN from response content."""
    if not trace_id:
        return "UNKNOWN"

    base = base_url.rstrip("/")
    candidates = [
        f"{base}/v1/traces/{trace_id}",
        f"{base}/v1/spans?trace_id={trace_id}",
        f"{base}/v1/spans/{trace_id}",
        f"{base}/v1/spans?root_span_id={trace_id}",
        f"{base}/api/v1/traces/{trace_id}",
        f"{base}/api/v1/spans?trace_id={trace_id}",
    ]
    if span_path_id:
        candidates.extend(
            [
                f"{base}/v1/spans/{span_path_id}",
                f"{base}/v1/spans?span_id={span_path_id}",
                f"{base}/api/v1/spans/{span_path_id}",
            ]
        )
    if selected_span_id:
        candidates.extend(
            [
                f"{base}/v1/spans/{selected_span_id}",
                f"{base}/v1/spans?span_id={selected_span_id}",
                f"{base}/api/v1/spans/{selected_span_id}",
            ]
        )
    if project_id_token:
        candidates.extend(
            [
                f"{base}/v1/projects/{project_id_token}/spans?trace_id={trace_id}",
                f"{base}/v1/projects/{project_id_token}/traces/{trace_id}",
            ]
        )

    got_parseable_response = False
    for url in candidates:
        try:
            resp = await client.get(url)
        except Exception:
            continue

        if resp.status_code >= 400:
            continue

        got_parseable_response = True
        content_text = ""
        try:
            content_text = _extract_strings_recursive(resp.json()).lower()
        except Exception:
            content_text = (resp.text or "").lower()

        if _contains_web_search_marker(content_text):
            return "YES"

    if got_parseable_response:
        return "NO"
    return "UNKNOWN"


async def probe_trace_web_search_debug(
    base_url: str,
    trace_id: str,
    span_path_id: str,
    selected_span_id: str,
    project_id_token: str,
    api_key: str = "",
) -> dict:
    """Debug probe for one trace; returns endpoint-level status details."""
    result = {
        "trace_id": trace_id,
        "base_url": base_url,
        "detected": "UNKNOWN",
        "matched_endpoint": "",
        "attempts": [],
    }
    if not base_url:
        result["attempts"].append({"endpoint": "(none)", "status": "missing_base_url"})
        return result

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["x-api-key"] = api_key

    timeout = httpx.Timeout(12.0)
    limits = httpx.Limits(max_connections=8, max_keepalive_connections=4)
    base = base_url.rstrip("/")
    endpoints = [
        f"{base}/v1/traces/{trace_id}",
        f"{base}/v1/spans?trace_id={trace_id}",
        f"{base}/v1/spans/{trace_id}",
        f"{base}/v1/spans?root_span_id={trace_id}",
        f"{base}/api/v1/traces/{trace_id}",
        f"{base}/api/v1/spans?trace_id={trace_id}",
    ]
    if span_path_id:
        endpoints.extend(
            [
                f"{base}/v1/spans/{span_path_id}",
                f"{base}/v1/spans?span_id={span_path_id}",
                f"{base}/api/v1/spans/{span_path_id}",
            ]
        )
    if selected_span_id:
        endpoints.extend(
            [
                f"{base}/v1/spans/{selected_span_id}",
                f"{base}/v1/spans?span_id={selected_span_id}",
                f"{base}/api/v1/spans/{selected_span_id}",
            ]
        )
    if project_id_token:
        endpoints.extend(
            [
                f"{base}/v1/projects/{project_id_token}/spans?trace_id={trace_id}",
                f"{base}/v1/projects/{project_id_token}/traces/{trace_id}",
            ]
        )

    async with httpx.AsyncClient(headers=headers, timeout=timeout, limits=limits) as client:
        for endpoint in endpoints:
            try:
                resp = await client.get(endpoint)
            except Exception as exc:
                result["attempts"].append({"endpoint": endpoint, "status": f"error: {exc}"})
                continue

            attempt = {"endpoint": endpoint, "status_code": resp.status_code, "detected": "NO"}
            if resp.status_code >= 400:
                result["attempts"].append(attempt)
                continue

            content_text = ""
            try:
                content_text = _extract_strings_recursive(resp.json()).lower()
            except Exception:
                content_text = (resp.text or "").lower()

            if _contains_web_search_marker(content_text):
                attempt["detected"] = "YES"
                result["attempts"].append(attempt)
                result["detected"] = "YES"
                result["matched_endpoint"] = endpoint
                return result

            result["attempts"].append(attempt)

    # If we got any successful endpoint but no marker, it's NO; otherwise UNKNOWN.
    any_success = any(a.get("status_code", 500) < 400 for a in result["attempts"] if "status_code" in a)
    result["detected"] = "NO" if any_success else "UNKNOWN"
    return result


async def enrich_web_search_from_phoenix_api(
    df: pd.DataFrame,
    phoenix_base_url: str,
    phoenix_api_key: str,
    max_unknown_rows: int = 300,
) -> pd.DataFrame:
    """Resolve UNKNOWN web_search_performed rows via Phoenix API lookups."""
    if not phoenix_base_url and "phoenix_link" not in df.columns:
        return df

    unknown_mask = (
        df.get("web_search_performed", pd.Series(["UNKNOWN"] * len(df)))
        .astype(str)
        .str.upper()
        .eq("UNKNOWN")
    )
    candidate_indices = df[unknown_mask].index.tolist()[:max_unknown_rows]
    if not candidate_indices:
        return df

    headers = {}
    if phoenix_api_key:
        headers["Authorization"] = f"Bearer {phoenix_api_key}"
        headers["x-api-key"] = phoenix_api_key

    timeout = httpx.Timeout(12.0)
    limits = httpx.Limits(max_connections=12, max_keepalive_connections=6)

    async with httpx.AsyncClient(headers=headers, timeout=timeout, limits=limits) as client:
        tasks = []
        trace_ids = []
        for idx in candidate_indices:
            trace_id = str(df.at[idx, "trace_id"]) if "trace_id" in df.columns else ""
            phoenix_link = str(df.at[idx, "phoenix_link"]) if "phoenix_link" in df.columns else ""
            parsed_link = parse_phoenix_link(phoenix_link)
            row_base_url = phoenix_base_url or parsed_link["base_url"]
            trace_ids.append(trace_id)
            tasks.append(
                _lookup_trace_web_search_status(
                    client=client,
                    base_url=row_base_url,
                    trace_id=trace_id,
                    span_path_id=parsed_link["span_path_id"],
                    selected_span_id=parsed_link["selected_span_id"],
                    project_id_token=parsed_link["project_id_token"],
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, result in zip(candidate_indices, results):
            if isinstance(result, str):
                df.at[idx, "web_search_performed"] = result

    return df


def enrich_trace_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add/refresh metadata columns used for macro trace analytics."""
    df = df.copy()
    for col in ("web_search_performed", "prompted_categories", "region_bucket"):
        if col not in df.columns:
            df[col] = ""

    for idx in range(len(df)):
        row = df.iloc[idx]
        prompt_type = str(row.get("prompt_type", ""))
        location = str(row.get("location", ""))
        query = str(row.get("natural_language_query", ""))
        category_type = str(row.get("Category_type", ""))

        df.at[idx, "region_bucket"] = infer_region_bucket(prompt_type, location)
        df.at[idx, "prompted_categories"] = extract_prompted_categories(query, category_type)
        df.at[idx, "web_search_performed"] = detect_web_search_performed(row)

    return df


def unique_preserve_order(items: list) -> list:
    """Return unique values preserving first-seen order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


async def evaluate_row(
    row: pd.Series,
    provider: str,
    model: str,
) -> dict:
    """Evaluate a single trace log row with appropriate agents.

    Returns a dict of column_name -> value for all eval columns.
    """
    query = str(row.get("natural_language_query", ""))
    output = str(row.get("full_output", ""))
    location = str(row.get("location", ""))
    prompt_type = str(row.get("prompt_type", ""))

    output_type = OutputTypeRouter.classify(prompt_type)
    results = {}

    # Run appropriate agents in parallel
    if output_type == "referral":
        agents = get_referral_agents(provider, model)
        tasks = [agent.evaluate(query, output, location) for agent in agents]
        agent_results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in agent_results:
            if isinstance(res, dict):
                results.update(res)
            elif isinstance(res, Exception):
                pass  # Errors already handled in base agent

    elif output_type == "actionplan":
        agents = get_actionplan_agents(provider, model)
        tasks = [agent.evaluate(query, output, location) for agent in agents]
        agent_results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in agent_results:
            if isinstance(res, dict):
                results.update(res)

    # Readability metrics (always run, no LLM needed)
    cleaned = clean_text(output)
    results["cleaned_text"] = cleaned
    readability = calculate_readability_metrics(cleaned)
    results.update(readability)

    return results


async def evaluate_dataframe(
    df: pd.DataFrame,
    provider: str,
    model: str,
    progress_bar,
    status_text,
) -> pd.DataFrame:
    """Evaluate all rows in the DataFrame."""
    df = ensure_eval_columns(df, ALL_EVAL_COLUMNS)
    total = len(df)

    for idx in range(total):
        row = df.iloc[idx]
        trace_id = row.get("trace_id", f"row-{idx}")
        prompt_type = str(row.get("prompt_type", "unknown"))
        output_type = OutputTypeRouter.classify(prompt_type)

        status_text.text(f"Evaluating row {idx + 1}/{total}  |  trace: {trace_id}  |  type: {output_type}")

        results = await evaluate_row(row, provider, model)

        for col, val in results.items():
            if col in df.columns:
                df.at[idx, col] = val

        progress_bar.progress((idx + 1) / total)

    status_text.text(f"Evaluation complete — {total} rows processed.")
    return df


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Settings")

    st.subheader("LLM Configuration")

    provider = st.selectbox(
        "Provider",
        options=["openai", "anthropic"],
        index=0,
    )

    model = st.selectbox(
        "Model",
        options=MODEL_OPTIONS.get(provider, ["gpt-4o-mini"]),
        index=0,
    )

    # API key inputs
    st.subheader("API Keys")
    openai_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    anthropic_key = st.text_input("Anthropic API Key", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""))

    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key

    st.divider()
    st.subheader("Phoenix Enrichment (Optional)")
    phoenix_enrich = st.checkbox(
        "Resolve unknown web-search status from Phoenix API",
        value=True,
        help="Calls Phoenix trace/span endpoints and checks span payloads for web-search markers like OpenAIWebSearch/web_search.",
    )
    phoenix_base_url = st.text_input(
        "Phoenix Base URL",
        value=os.getenv("PHOENIX_BASE_URL", ""),
        placeholder="https://phoenix.your-domain.com:6006",
    )
    phoenix_api_key = st.text_input(
        "Phoenix API Key (optional)",
        type="password",
        value=os.getenv("PHOENIX_API_KEY", ""),
    )
    st.caption(
        "Tip: If your CSV has `phoenix_link`, base URL can be auto-derived. "
        "If your Phoenix API requires auth, provide API key for accurate YES/NO detection."
    )
    test_phoenix = st.button("Test Phoenix access on one trace", use_container_width=True)

    st.divider()
    st.subheader("About")
    st.markdown("""
    **Phoenix Logs Agent Evals** evaluates GenAI
    trace log outputs against quality rubrics using
    specialized AI agents running in parallel.

    - **Referrals**: 9 specialized agents
    - **Action Plans**: 1 comprehensive agent
    - **Readability**: 5 automated metrics
    """)


# ── Main Content ─────────────────────────────────────────────────────────────

st.title("Phoenix Logs Agent Evals")
st.markdown("Upload Phoenix Arize trace logs and evaluate output quality with specialized AI agents.")

# ── Upload ───────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload Phoenix trace log CSV",
    type=["csv"],
    help="Upload the CSV export from Phoenix Arize containing trace logs.",
)

if uploaded_file is not None:
    df = load_csv(uploaded_file)
    df = enrich_trace_metadata(df)

    if phoenix_enrich and phoenix_base_url:
        with st.spinner("Resolving unknown web-search status from Phoenix API..."):
            df = asyncio.run(
                enrich_web_search_from_phoenix_api(
                    df=df,
                    phoenix_base_url=phoenix_base_url,
                    phoenix_api_key=phoenix_api_key,
                )
            )

    st.success(f"Loaded **{len(df)} rows** from `{uploaded_file.name}`")

    # Show data preview
    with st.expander("Preview raw data", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

    # Phoenix access debug test for one trace from uploaded CSV.
    if test_phoenix:
        if "trace_id" not in df.columns:
            st.error("CSV is missing `trace_id`; cannot test Phoenix API access.")
        else:
            sample_row = df[df["trace_id"].notna() & (df["trace_id"].astype(str).str.strip() != "")]
            if sample_row.empty:
                st.error("No non-empty `trace_id` found in CSV.")
            else:
                row = sample_row.iloc[0]
                trace_id = str(row.get("trace_id", ""))
                parsed_link = parse_phoenix_link(str(row.get("phoenix_link", "")))
                debug_base = phoenix_base_url or parsed_link["base_url"]
                with st.spinner(f"Testing Phoenix access for trace `{trace_id}`..."):
                    debug_result = asyncio.run(
                        probe_trace_web_search_debug(
                            base_url=debug_base,
                            trace_id=trace_id,
                            span_path_id=parsed_link["span_path_id"],
                            selected_span_id=parsed_link["selected_span_id"],
                            project_id_token=parsed_link["project_id_token"],
                            api_key=phoenix_api_key,
                        )
                    )
                st.markdown("**Phoenix Access Test Result**")
                st.json(debug_result, expanded=False)

    # Show type distribution
    if "prompt_type" in df.columns:
        type_counts = df["prompt_type"].apply(OutputTypeRouter.classify).value_counts()
        cols = st.columns(len(type_counts))
        for i, (typ, count) in enumerate(type_counts.items()):
            with cols[i]:
                css_class = "metric-card"
                if typ == "referral":
                    css_class += " pass-card"
                elif typ == "actionplan":
                    css_class += " warn-card"
                else:
                    css_class += " fail-card"
                st.markdown(
                    f'<div class="{css_class}"><h3>{count}</h3><p>{typ}</p></div>',
                    unsafe_allow_html=True,
                )

    # ── Row selection ────────────────────────────────────────────────────────

    st.subheader("Row Selection")
    eval_mode = st.radio(
        "Which rows to evaluate?",
        ["All rows", "Only rows without existing scores", "Select range"],
        horizontal=True,
    )

    if eval_mode == "Select range":
        col1, col2 = st.columns(2)
        with col1:
            start_row = st.number_input("Start row", min_value=0, max_value=len(df) - 1, value=0)
        with col2:
            end_row = st.number_input("End row", min_value=0, max_value=len(df) - 1, value=min(9, len(df) - 1))

    # ── Run evaluation ───────────────────────────────────────────────────────

    if st.button("Run Evaluation", type="primary", use_container_width=True):
        # Validate API key
        if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            st.error("Please provide an OpenAI API key in the sidebar.")
            st.stop()
        elif provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            st.error("Please provide an Anthropic API key in the sidebar.")
            st.stop()

        # Filter rows based on selection
        if eval_mode == "Only rows without existing scores":
            # Find rows where referral_overall_review and actionplan_overallreview are empty
            mask = (
                df.get("referral_overall_review", pd.Series([""] * len(df))).fillna("").astype(str).str.strip().eq("")
                & df.get("actionplan_overallreview", pd.Series([""] * len(df))).fillna("").astype(str).str.strip().eq("")
            )
            eval_df = df[mask].copy()
            eval_indices = df[mask].index.tolist()
        elif eval_mode == "Select range":
            eval_df = df.iloc[start_row:end_row + 1].copy()
            eval_indices = list(range(start_row, end_row + 1))
        else:
            eval_df = df.copy()
            eval_indices = list(range(len(df)))

        if len(eval_df) == 0:
            st.warning("No rows to evaluate based on your selection.")
            st.stop()

        st.info(f"Evaluating **{len(eval_df)} rows** with **{provider}/{model}**...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Run async evaluation
        evaluated_df = asyncio.run(
            evaluate_dataframe(eval_df, provider, model, progress_bar, status_text)
        )

        # Merge results back into the original DataFrame
        df = ensure_eval_columns(df, ALL_EVAL_COLUMNS)
        for i, orig_idx in enumerate(eval_indices):
            for col in ALL_EVAL_COLUMNS:
                if col in evaluated_df.columns:
                    df.at[orig_idx, col] = evaluated_df.iloc[i][col]

        df = enrich_trace_metadata(df)
        if phoenix_enrich and phoenix_base_url:
            with st.spinner("Refreshing web-search metadata from Phoenix API..."):
                df = asyncio.run(
                    enrich_web_search_from_phoenix_api(
                        df=df,
                        phoenix_base_url=phoenix_base_url,
                        phoenix_api_key=phoenix_api_key,
                    )
                )
        st.session_state["evaluated_df"] = df
        st.session_state["eval_complete"] = True
        st.rerun()

# ── Results Dashboard ────────────────────────────────────────────────────────

if st.session_state.get("eval_complete") and "evaluated_df" in st.session_state:
    df = st.session_state["evaluated_df"]

    st.divider()
    st.header("Evaluation Results")

    # ── Summary Statistics ───────────────────────────────────────────────────

    st.subheader("Summary Statistics")

    # Overall verdict distribution
    score_cols_categorical = [
        "referral_overall_review",
        "actionplan_overallreview",
    ]

    col1, col2, col3, col4 = st.columns(4)

    # Count PASS/NEEDS_REVISION/FAIL across both verdict columns
    all_verdicts = []
    for col_name in score_cols_categorical:
        if col_name in df.columns:
            vals = df[col_name].dropna().astype(str).str.strip()
            vals = vals[vals != ""]
            all_verdicts.extend(vals.tolist())

    pass_count = sum(1 for v in all_verdicts if "PASS" in v and "NEEDS" not in v)
    revision_count = sum(1 for v in all_verdicts if "NEEDS_REVISION" in v)
    fail_count = sum(1 for v in all_verdicts if "FAIL" in v)
    total_evaluated = pass_count + revision_count + fail_count

    with col1:
        st.markdown(
            f'<div class="metric-card"><h3>{total_evaluated}</h3><p>Total Evaluated</p></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card pass-card"><h3>{pass_count}</h3><p>PASS</p></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="metric-card warn-card"><h3>{revision_count}</h3><p>NEEDS REVISION</p></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="metric-card fail-card"><h3>{fail_count}</h3><p>FAIL</p></div>',
            unsafe_allow_html=True,
        )

    # ── Macro Trace Stats ────────────────────────────────────────────────────
    st.subheader("Macro Trace Stats")

    macro_cols = st.columns(4)
    web_vals = df.get("web_search_performed", pd.Series(["UNKNOWN"] * len(df))).astype(str).str.upper()
    yes_web = (web_vals == "YES").sum()
    no_web = (web_vals == "NO").sum()
    unknown_web = (web_vals == "UNKNOWN").sum()

    with macro_cols[0]:
        st.metric("Traces With Web Search", int(yes_web))
    with macro_cols[1]:
        st.metric("Traces Without Web Search", int(no_web))
    with macro_cols[2]:
        st.metric("Unknown Web Search Status", int(unknown_web))
    with macro_cols[3]:
        pct_web = round((yes_web / len(df)) * 100, 1) if len(df) else 0.0
        st.metric("Web Search Rate", f"{pct_web}%")

    # Keystone vs Texas web search proportions.
    region_series = df.get("region_bucket", pd.Series(["unknown"] * len(df))).astype(str).str.lower()
    regional_rows = []
    for region_name in ("keystone", "texas"):
        mask = region_series == region_name
        region_total = int(mask.sum())
        region_yes = int(((web_vals == "YES") & mask).sum())
        region_unknown = int(((web_vals == "UNKNOWN") & mask).sum())
        proportion = round((region_yes / region_total) * 100, 1) if region_total else 0.0
        regional_rows.append(
            {
                "region": region_name,
                "total_prompts": region_total,
                "web_search_yes": region_yes,
                "web_search_unknown": region_unknown,
                "web_search_proportion_pct": proportion,
            }
        )
    st.markdown("**Web Search Proportion (Keystone vs Texas)**")
    st.dataframe(pd.DataFrame(regional_rows), use_container_width=True, hide_index=True)

    if unknown_web > 0:
        st.caption(
            "Note: many rows are UNKNOWN because span payloads are not present in the CSV. "
            "Enable Phoenix Enrichment in the sidebar to resolve from trace/span APIs."
        )

    # Most common prompted categories.
    cat_values = df.get("prompted_categories", pd.Series([""] * len(df))).fillna("").astype(str)
    cat_counts = {}
    for raw in cat_values:
        for part in [p.strip() for p in raw.split(";") if p.strip()]:
            key = part.lower()
            cat_counts[key] = cat_counts.get(key, 0) + 1
    if cat_counts:
        top_categories = (
            pd.DataFrame(
                [{"category": k, "count": v} for k, v in cat_counts.items()]
            )
            .sort_values("count", ascending=False)
            .head(15)
        )
        st.markdown("**Top Prompted Categories**")
        st.dataframe(top_categories, use_container_width=True, hide_index=True)

    # ── Per-Agent Score Distributions ────────────────────────────────────────

    st.subheader("Score Distributions by Agent")

    score_display_cols = [
        "referral_service_area_eligibility",
        "referral_location_proximity",
        "referral_contact_info",
        "referral_URL_check",
        "referral_description_review",
        "referral_missing_resources",
        "referral_relevance_review",
        "referral_service_status",
        "referral_overall_review",
        "actionplan_overallreview",
    ]

    # Build a summary table
    summary_rows = []
    for col_name in score_display_cols:
        if col_name not in df.columns:
            continue
        vals = df[col_name].dropna().astype(str).str.strip()
        vals = vals[vals != ""]
        if len(vals) == 0:
            continue
        value_counts = vals.value_counts().to_dict()
        summary_rows.append({
            "Agent": col_name.replace("referral_", "").replace("actionplan_", "AP: "),
            "Evaluated": len(vals),
            **value_counts,
        })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).fillna(0)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ── Readability Averages ─────────────────────────────────────────────────

    readability_cols = ["flesch_ease", "flesch_grade", "gunning_fog", "smog_index", "dale_chall"]
    readability_data = {}
    for col_name in readability_cols:
        if col_name in df.columns:
            numeric_vals = pd.to_numeric(df[col_name], errors="coerce").dropna()
            if len(numeric_vals) > 0:
                readability_data[col_name] = round(numeric_vals.mean(), 2)

    if readability_data:
        st.subheader("Average Readability Metrics")
        r_cols = st.columns(len(readability_data))
        for i, (name, val) in enumerate(readability_data.items()):
            with r_cols[i]:
                st.metric(name.replace("_", " ").title(), val)

    # ── Detailed Results Table ───────────────────────────────────────────────

    st.subheader("Detailed Results")

    # Color-code the dataframe
    def color_scores(val):
        """Apply color styling to score cells."""
        if not isinstance(val, str):
            val = str(val)
        val = val.strip()

        green_vals = {"PASS", "COMPLETE", "VALID", "ALL_ACTIVE", "NONE_MISSING", "5", "4"}
        yellow_vals = {"NEEDS_REVISION", "PARTIAL", "HOMEPAGE_ONLY", "SOME_CHANGES", "MINOR_GAPS", "3", "N/A"}
        red_vals = {"FAIL", "INCOMPLETE", "INACCURATE", "BROKEN", "OUTDATED", "MISSING", "HAS_CLOSURES", "MAJOR_GAPS", "1", "2", "ERROR"}

        if val in green_vals:
            return "background-color: #C6EFCE; color: #006100"
        elif val in yellow_vals:
            return "background-color: #FFEB9C; color: #9C5700"
        elif val in red_vals:
            return "background-color: #FFC7CE; color: #9C0006"
        return ""

    # Show only eval-relevant columns plus identifiers
    display_cols = [
        "trace_id",
        "prompt_type",
        "region_bucket",
        "web_search_performed",
        "prompted_categories",
        "natural_language_query",
    ]
    display_cols += [c for c in ALL_EVAL_COLUMNS if c in df.columns]
    display_cols = [c for c in display_cols if c in df.columns]
    display_cols = unique_preserve_order(display_cols)

    # Styler requires unique index/columns; normalize index and dedupe columns.
    table_df = df[display_cols].copy().reset_index(drop=True)

    styled_df = table_df.style.applymap(
        color_scores,
        subset=[c for c in score_display_cols if c in display_cols],
    )
    st.dataframe(styled_df, use_container_width=True, height=500)

    # ── Expandable Row Details ───────────────────────────────────────────────

    st.subheader("Row Details")
    st.markdown("Expand any row to see full evaluation reasoning.")

    for idx in range(min(len(df), 100)):  # Cap at 100 to avoid UI overload
        row = df.iloc[idx]
        trace_id = row.get("trace_id", f"row-{idx}")
        prompt_type = str(row.get("prompt_type", ""))
        output_type = OutputTypeRouter.classify(prompt_type)
        overall = row.get("referral_overall_review", "") or row.get("actionplan_overallreview", "")

        label = f"Row {idx} | {trace_id} | {output_type} | {overall}"

        with st.expander(label, expanded=False):
            st.markdown(f"**Query:** {row.get('natural_language_query', 'N/A')}")
            st.markdown(f"**Location:** {row.get('location', 'N/A')}")
            st.markdown(f"**Output Type:** {output_type}")
            st.markdown(f"**Region Bucket:** {row.get('region_bucket', 'unknown')}")
            st.markdown(f"**Web Search Performed:** {row.get('web_search_performed', 'UNKNOWN')}")
            st.markdown(f"**Prompted Categories:** {row.get('prompted_categories', '') or 'N/A'}")

            st.divider()

            # Show scores and reasoning side by side
            if output_type == "referral":
                agent_pairs = [
                    ("Service Area Eligibility", "referral_service_area_eligibility"),
                    ("Location Proximity", "referral_location_proximity"),
                    ("Contact Info", "referral_contact_info"),
                    ("URL Check", "referral_URL_check"),
                    ("Description Review", "referral_description_review"),
                    ("Missing Resources", "referral_missing_resources"),
                    ("Relevance Review", "referral_relevance_review"),
                    ("Service Status", "referral_service_status"),
                    ("Overall Review", "referral_overall_review"),
                ]
            else:
                agent_pairs = [
                    ("Action Plan Overall", "actionplan_overallreview"),
                ]

            for agent_name, col_name in agent_pairs:
                score = row.get(col_name, "")
                reasoning = row.get(f"{col_name}_reasoning", "")
                if score or reasoning:
                    c1, c2 = st.columns([1, 4])
                    with c1:
                        st.markdown(f"**{agent_name}**")
                        st.code(str(score))
                    with c2:
                        st.markdown(str(reasoning) if reasoning else "_No reasoning available_")

            # Readability
            st.divider()
            st.markdown("**Readability Metrics**")
            r_cols_ui = st.columns(5)
            for i, metric in enumerate(readability_cols):
                with r_cols_ui[i]:
                    val = row.get(metric, "")
                    st.metric(metric.replace("_", " ").title(), val if val != "" else "N/A")

    # ── Export Options ───────────────────────────────────────────────────────

    st.divider()
    st.subheader("Export Results")

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        csv_bytes = save_csv(df)
        st.download_button(
            label="Download as CSV",
            data=csv_bytes,
            file_name=f"phoenix_evals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with exp_col2:
        excel_bytes = save_excel(df)
        st.download_button(
            label="Download as Excel",
            data=excel_bytes,
            file_name=f"phoenix_evals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

elif uploaded_file is None:
    # Show landing state
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1. Upload")
        st.markdown("Upload your Phoenix Arize trace log CSV file.")
    with col2:
        st.markdown("### 2. Evaluate")
        st.markdown("AI agents evaluate each trace output against quality rubrics in parallel.")
    with col3:
        st.markdown("### 3. Export")
        st.markdown("View scores in-app, then download as CSV or formatted Excel.")

    st.markdown("---")
    st.markdown("#### Evaluation Agents")
    st.markdown("""
    | Agent | Column | Scores |
    |-------|--------|--------|
    | Service Area Check | `referral_service_area_eligibility` | PASS / FAIL / PARTIAL / N/A |
    | Proximity Scorer | `referral_location_proximity` | 1-5 |
    | Contact Info Verifier | `referral_contact_info` | COMPLETE / PARTIAL / INCOMPLETE / INACCURATE |
    | URL Checker | `referral_URL_check` | VALID / HOMEPAGE_ONLY / BROKEN / OUTDATED / MISSING |
    | Description Reviewer | `referral_description_review` | 1-5 |
    | Missing Resources | `referral_missing_resources` | NONE_MISSING / MINOR_GAPS / MAJOR_GAPS |
    | Relevance Reviewer | `referral_relevance_review` | 1-5 |
    | Service Status | `referral_service_status` | ALL_ACTIVE / SOME_CHANGES / HAS_CLOSURES |
    | Overall Synthesizer | `referral_overall_review` | PASS / NEEDS_REVISION / FAIL |
    | Action Plan Reviewer | `actionplan_overallreview` | PASS / NEEDS_REVISION / FAIL |

    Each score column is paired with a `*_reasoning` column containing the agent's explanation.
    """)
