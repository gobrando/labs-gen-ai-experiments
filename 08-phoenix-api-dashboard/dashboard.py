"""
Phoenix Arize Product Dashboard
Interactive dashboard for analyzing GenAI product logs
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import os
from dotenv import load_dotenv

from phoenix_client import PhoenixClient
from data_analyzer import TraceAnalyzer

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Phoenix Product Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .big-metric {
        font-size: 2.5rem !important;
        font-weight: bold !important;
    }
    .metric-label {
        font-size: 0.9rem !important;
        color: #666 !important;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_phoenix_client(api_url, api_key):
    """Initialize Phoenix client (cached)"""
    return PhoenixClient(api_url, api_key)


@st.cache_data(ttl=43200)  # Cache for 12 hours (trace data changes slowly)
def load_data(_client, project_id, start_time, end_time, max_spans, cache_bust: str = "0"):
    """Load spans data from Phoenix API"""
    if start_time and end_time and end_time <= start_time:
        raise ValueError("end_time must be after start_time")
    spans = _client.get_all_spans(
        project_id=project_id,
        start_time=start_time,
        end_time=end_time,
        max_spans=max_spans
    )
    return spans


def main():
    st.title("📊 Phoenix Product Analytics Dashboard")
    st.markdown("*Analyze your GenAI product usage and quality metrics*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Configuration
        api_url = st.text_input(
            "Phoenix API URL",
            value=os.getenv('PHOENIX_API_URL', 'https://your-phoenix-instance.arize.com'),
            help="Your Phoenix instance URL"
        )
        
        api_key = st.text_input(
            "API Key (optional)",
            value=os.getenv('PHOENIX_API_KEY', ''),
            type="password",
            help="Leave empty if no authentication required"
        )
        
        st.divider()
        
        # Data filters
        st.header("🔍 Data Filters")
        
        # Time range
        time_range = st.selectbox(
            "Time Range",
            ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Custom"]
        )
        
        if time_range == "Custom":
            start_date = st.date_input("Start Date", value=datetime.now(timezone.utc).date() - timedelta(days=7))
            end_date = st.date_input("End Date", value=datetime.now(timezone.utc).date())
            # Combine dates with time and make timezone-aware (UTC)
            start_time = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
            end_time = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc)
        else:
            days_map = {"Last 24 Hours": 1, "Last 7 Days": 7, "Last 30 Days": 30}
            days = days_map.get(time_range, 7)
            # Use UTC-aware datetimes for proper timezone handling
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=days)
        
        max_spans = st.number_input(
            "Max Spans to Load",
            min_value=100,
            max_value=50000,
            value=10000,
            step=1000,
            help="Maximum number of traces to analyze"
        )
        
        project_id = st.text_input(
            "Project ID (optional)",
            value=os.getenv('PHOENIX_PROJECT_ID', 'UHJvamVjdDoxOQ=='),
            help="Filter by specific project (default: pilot-prod)"
        )
        
        st.divider()
        
        # Load/refresh controls
        if 'cache_bust' not in st.session_state:
            st.session_state.cache_bust = "0"

        col1, col2 = st.columns(2)
        with col1:
            load_button = st.button("🔄 Load Data", type="primary", use_container_width=True)
        with col2:
            force_refresh = st.button(
                "♻️ Force Refresh",
                help="Bypasses the cache and re-pulls from Phoenix immediately.",
                use_container_width=True
            )
            if force_refresh:
                st.session_state.cache_bust = str(datetime.now(timezone.utc).timestamp())
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Load data
    if load_button or st.session_state.data_loaded:
        try:
            with st.spinner("Connecting to Phoenix API..."):
                client = get_phoenix_client(api_url, api_key if api_key else None)
            
            with st.spinner("Loading trace data..."):
                spans = load_data(
                    client,
                    project_id if project_id else None,
                    start_time,
                    end_time,
                    max_spans,
                    st.session_state.get('cache_bust', "0")
                )
            
            if not spans:
                st.error("No data found. Please check your configuration and filters.")
                return
            
            st.session_state.analyzer = TraceAnalyzer(spans)
            st.session_state.data_loaded = True
            st.success(f"✅ Loaded {len(spans)} spans successfully!")
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return
    
    if not st.session_state.data_loaded:
        st.info("👈 Configure your settings in the sidebar and click 'Load Data' to begin")
        return
    
    analyzer = st.session_state.analyzer
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Executive Summary",
        "📊 Usage Analytics",
        "📋 Usage Report",
        "⚡ Performance Metrics",
        "🔎 Log Explorer",
        "🧠 Advanced Analytics"
    ])
    
    # TAB 1: Executive Summary
    with tab1:
        st.header("Executive Summary")
        st.markdown("*High-level KPIs for leadership reporting*")
        
        stats = analyzer.get_macro_statistics()
        
        if stats:
            # Key metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Requests",
                    f"{stats['total_requests']:,}",
                    help="Total number of API requests"
                )
            
            with col2:
                st.metric(
                    "Unique Traces",
                    f"{stats['unique_traces']:,}",
                    help="Number of unique user sessions"
                )
            
            with col3:
                st.metric(
                    "Success Rate",
                    f"{stats['success_rate']:.1f}%",
                    help="Percentage of successful requests"
                )
            
            with col4:
                st.metric(
                    "Avg Latency",
                    f"{stats['avg_latency_s']:.2f}s",
                    help="Average response time"
                )
            
            st.divider()
            
            # Second row of metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Tokens",
                    f"{stats['total_tokens']:,}",
                    help="Total tokens consumed"
                )
            
            with col2:
                st.metric(
                    "Avg Tokens/Request",
                    f"{stats['avg_tokens_per_request']:.0f}",
                    help="Average tokens per request"
                )
            
            with col3:
                st.metric(
                    "P95 Latency",
                    f"{stats['p95_latency_s']:.2f}s",
                    help="95th percentile latency"
                )
            
            with col4:
                st.metric(
                    "P99 Latency",
                    f"{stats['p99_latency_s']:.2f}s",
                    help="99th percentile latency"
                )
            
            st.divider()
            
            # Trace counts by type
            if 'trace_counts' in stats:
                st.subheader("Trace Counts by Type")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Generating Referrals",
                        f"{stats['trace_counts']['referrals']:,}",
                        help="Number of traces for generating referrals"
                    )
                
                with col2:
                    st.metric(
                        "Generating Action Plans",
                        f"{stats['trace_counts']['action_plans']:,}",
                        help="Number of traces for generating action plans"
                    )
                
                with col3:
                    st.metric(
                        "Emailing Results",
                        f"{stats['trace_counts'].get('email_results', 0):,}",
                        help="Number of traces for emailing results to users"
                    )
                
                with col4:
                    st.metric(
                        "Other Traces",
                        f"{stats['trace_counts']['other']:,}",
                        help="Number of other trace types"
                    )
            
            st.divider()
            
            # Date range
            if stats['date_range']['start'] and stats['date_range']['end']:
                st.info(f"📅 Data Range: {stats['date_range']['start'].strftime('%Y-%m-%d %H:%M')} to {stats['date_range']['end'].strftime('%Y-%m-%d %H:%M')}")
            
            # Model usage breakdown
            if stats.get('models_used'):
                st.subheader("Model Usage Distribution")
                model_df = pd.DataFrame(
                    list(stats['models_used'].items()),
                    columns=['Model', 'Requests']
                )
                fig = px.pie(
                    model_df,
                    values='Requests',
                    names='Model',
                    title='Requests by Model'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Usage Analytics (Trace-based)
    with tab2:
        st.header("Usage Analytics")
        st.markdown("*User-level analytics based on traces*")
        
        # User Analytics Table
        st.subheader("👥 User Usage Summary")
        user_analytics = analyzer.get_user_analytics()
        
        if not user_analytics.empty:
            # Display user table with key metrics
            display_cols = ['user_name', 'user_email', 'total_traces', 'referrals_count', 'action_plans_count', 
                          'email_results_count', 'first_trace', 'last_trace', 'avg_duration_s', 'total_tokens']
            
            user_display = user_analytics[display_cols].copy()
            user_display.columns = ['Name', 'Email', 'Total Traces', 'Referrals', 'Action Plans', 
                                  'Emails', 'First Trace', 'Last Trace', 'Avg Duration (s)', 'Total Tokens']
            
            # Format dates and handle NaT
            user_display['First Trace'] = pd.to_datetime(user_display['First Trace'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
            user_display['Last Trace'] = pd.to_datetime(user_display['Last Trace'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
            user_display['Avg Duration (s)'] = user_display['Avg Duration (s)'].round(2).fillna(0)
            user_display = user_display.fillna('N/A')
            
            st.dataframe(
                user_display,
                use_container_width=True,
                hide_index=True
            )
            
            # User selection for detailed view
            st.subheader("🔍 User Detail View")
            selected_user = st.selectbox(
                "Select a user to view detailed trace history",
                options=user_analytics['user_email'].tolist(),
                format_func=lambda x: f"{user_analytics[user_analytics['user_email']==x]['user_name'].iloc[0]} ({x})"
            )
            
            if selected_user:
                user_traces = analyzer.get_user_trace_details(selected_user)
                
                if not user_traces.empty:
                    st.write(f"**Traces for {selected_user}**")
                    
                    # Display trace details
                    display_cols = ['trace_start', 'trace_type', 'query', 'category',
                                   'location_preference', 'zip_code',
                                   'trace_duration_s', 'total_tokens', 'status']
                    available_cols = [col for col in display_cols if col in user_traces.columns]
                    
                    trace_display = user_traces[available_cols].copy()
                    col_names = [
                        'Timestamp', 'Type', 'Query', 'Category', 'Location Pref',
                        'Zip Code', 'Duration (s)', 'Tokens', 'Status'
                    ]
                    trace_display.columns = col_names[:len(available_cols)]
                    
                    if 'Timestamp' in trace_display.columns:
                        trace_display['Timestamp'] = pd.to_datetime(trace_display['Timestamp'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
                    if 'Duration (s)' in trace_display.columns:
                        trace_display['Duration (s)'] = pd.to_numeric(trace_display['Duration (s)'], errors='coerce').round(2)
                    
                    trace_display = trace_display.fillna('N/A')
                    st.dataframe(trace_display, use_container_width=True, hide_index=True)
                    
                    # Show summary stats for this user
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Traces", len(user_traces))
                    with col2:
                        referrals = len(user_traces[user_traces['trace_type'] == 'referrals'])
                        st.metric("Referrals", referrals)
                    with col3:
                        action_plans = len(user_traces[user_traces['trace_type'] == 'action_plans'])
                        st.metric("Action Plans", action_plans)
                    with col4:
                        avg_dur = user_traces['trace_duration_s'].mean()
                        st.metric("Avg Duration", f"{avg_dur:.2f}s" if pd.notna(avg_dur) else "N/A")
        
        # Time series based on traces
        st.subheader("📈 Trace Volume Over Time")
        col1, col2 = st.columns([1, 3])
        with col1:
            freq = st.selectbox(
                "Time Granularity",
                ["Hourly", "Daily", "Weekly"],
                index=1,
                key="trace_freq"
            )
        
        freq_map = {"Hourly": "H", "Daily": "D", "Weekly": "W"}
        trace_time_series = analyzer.get_trace_time_series(freq=freq_map[freq])
        
        if not trace_time_series.empty:
            # Trace volume over time with type breakdown
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Trace Volume Over Time", "Trace Types Breakdown"),
                vertical_spacing=0.15
            )
            
            fig.add_trace(
                go.Scatter(
                    x=trace_time_series['timestamp'],
                    y=trace_time_series['trace_count'],
                    name='Total Traces',
                    fill='tozeroy',
                    line=dict(color='#1f77b4')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=trace_time_series['timestamp'],
                    y=trace_time_series['referrals_count'],
                    name='Referrals',
                    fill='tozeroy',
                    line=dict(color='#2ca02c')
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=trace_time_series['timestamp'],
                    y=trace_time_series['action_plans_count'],
                    name='Action Plans',
                    fill='tozeroy',
                    line=dict(color='#ff7f0e')
                ),
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_yaxes(title_text="Trace Count", row=1, col=1)
            fig.update_yaxes(title_text="Count by Type", row=2, col=1)
            
            fig.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Average duration over time
            st.subheader("⏱️ Average Trace Duration Over Time")
            fig_duration = px.line(
                trace_time_series,
                x='timestamp',
                y='avg_duration',
                title='Average Trace Duration',
                labels={'avg_duration': 'Duration (s)', 'timestamp': 'Time'}
            )
            st.plotly_chart(fig_duration, use_container_width=True)

        # Organic usage (exclude planned advisory board sessions)
        st.divider()
        st.subheader("🌿 Organic Usage (excluding NAB/CAB meeting windows)")
        st.caption("Filters out traces that occurred during scheduled advisory board meetings, so you can see organic (non-planned) usage trends.")

        exclude_meetings = st.checkbox(
            "Exclude advisory board meeting windows",
            value=True,
            help="If enabled, traces during the listed meeting windows will be counted as 'planned' and removed from 'organic' usage."
        )
        meeting_duration_min = st.slider(
            "Assumed meeting session length (minutes)",
            min_value=15,
            max_value=180,
            value=60,
            step=15,
            help="Used to convert each meeting start time into an exclusion window."
        )

        # Extracted from: Goodwill NAB Sessions.pdf (shared by user)
        # Source content includes times listed as PT/ET; we normalize using the ET time.
        default_meeting_starts_et = [
            # September/October 2025 - NAB Session 2
            "2025-09-29 13:00", "2025-09-29 15:00",
            "2025-09-30 14:00", "2025-09-30 15:00",
            "2025-10-02 14:30",
            "2025-10-06 13:00",
            # November 2025 - NAB Session 3
            "2025-11-04 14:15", "2025-11-04 15:00",
            "2025-11-05 14:15", "2025-11-05 13:30",
            "2025-11-06 12:00",
            "2025-11-07 13:30",
            "2025-11-10 12:00", "2025-11-10 12:45", "2025-11-10 13:30",
            # December 2025 - NAB Session 4
            "2025-12-08 13:00", "2025-12-08 14:00",
            "2025-12-09 14:00",
            "2025-12-10 14:00",
            "2025-12-11 14:00",
        ]

        with st.expander("📅 Meeting window schedule (editable)", expanded=False):
            st.markdown("Paste additional meeting start times below (ET), one per line as `YYYY-MM-DD HH:MM`.")
            additional_starts_text = st.text_area(
                "Additional meeting starts (ET)",
                value="",
                height=120,
                key="additional_meeting_starts_et"
            )
            show_default = st.checkbox("Show default meeting starts extracted from the PDF", value=False)
            if show_default:
                st.code("\n".join(default_meeting_starts_et))

        # Build meeting windows in UTC
        ny_tz = ZoneInfo("America/New_York")
        meeting_starts_et = list(default_meeting_starts_et)
        if additional_starts_text.strip():
            for line in additional_starts_text.splitlines():
                s = line.strip()
                if s:
                    meeting_starts_et.append(s)

        windows_utc = []
        for s in meeting_starts_et:
            try:
                dt_local = datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=ny_tz)
                start_utc = dt_local.astimezone(timezone.utc)
                end_utc = start_utc + timedelta(minutes=int(meeting_duration_min))
                windows_utc.append((start_utc, end_utc))
            except Exception:
                continue

        if analyzer.traces_df.empty or 'trace_start' not in analyzer.traces_df.columns:
            st.info("No trace timestamps available to compute organic usage.")
        else:
            traces_for_org = analyzer.traces_df.copy()
            ts_utc = pd.to_datetime(traces_for_org['trace_start'], errors='coerce', utc=True)
            traces_for_org = traces_for_org.assign(_trace_start_utc=ts_utc).dropna(subset=['_trace_start_utc'])

            planned_mask = pd.Series(False, index=traces_for_org.index)
            if exclude_meetings and windows_utc:
                for start_utc, end_utc in windows_utc:
                    planned_mask = planned_mask | (
                        (traces_for_org['_trace_start_utc'] >= pd.Timestamp(start_utc)) &
                        (traces_for_org['_trace_start_utc'] < pd.Timestamp(end_utc))
                    )

            organic_df = traces_for_org[~planned_mask].copy()
            planned_df = traces_for_org[planned_mask].copy()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Traces", f"{len(traces_for_org):,}")
            with col2:
                st.metric("Planned (meeting) Traces", f"{len(planned_df):,}")
            with col3:
                st.metric("Organic Traces", f"{len(organic_df):,}")
            with col4:
                pct = (len(organic_df) / len(traces_for_org) * 100) if len(traces_for_org) else 0
                st.metric("% Organic", f"{pct:.1f}%")

            # Time series (use the same granularity selection as the main trace chart)
            ts_freq = freq_map.get(freq, "D")

            total_series = traces_for_org.set_index('_trace_start_utc').resample(ts_freq).size().rename("total_traces")
            organic_series = organic_df.set_index('_trace_start_utc').resample(ts_freq).size().rename("organic_traces")
            planned_series = planned_df.set_index('_trace_start_utc').resample(ts_freq).size().rename("planned_traces")

            org_ts = pd.concat([total_series, organic_series, planned_series], axis=1).fillna(0).reset_index().rename(columns={'_trace_start_utc': 'timestamp'})
            org_ts[['total_traces', 'organic_traces', 'planned_traces']] = org_ts[['total_traces', 'organic_traces', 'planned_traces']].astype(int)

            fig_org = go.Figure()
            fig_org.add_trace(go.Scatter(
                x=org_ts['timestamp'], y=org_ts['organic_traces'],
                name="Organic Traces", line=dict(color="#2ca02c")
            ))
            fig_org.add_trace(go.Scatter(
                x=org_ts['timestamp'], y=org_ts['total_traces'],
                name="Total Traces", line=dict(color="#1f77b4"), opacity=0.5
            ))
            fig_org.add_trace(go.Bar(
                x=org_ts['timestamp'], y=org_ts['planned_traces'],
                name="Planned (meeting) Traces", marker_color="#ff7f0e", opacity=0.25
            ))
            fig_org.update_layout(
                title="Organic vs Planned Usage Over Time",
                barmode="overlay",
                height=450,
                xaxis_title="Time",
                yaxis_title="Trace Count",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_org, use_container_width=True)
    
    # TAB 3: Usage Report
    with tab3:
        st.header("📋 Comprehensive Usage Report")
        st.markdown("*Detailed breakdown by user level, resource type, and geography*")
        
        # Define cohorts with default values
        # G1 Cohort - Wave 1 users
        g1_emails_default = """deaun.hoffman@gwctx.org
herminio.chaparro@gwctx.org
jeremy.hunt@gwctx.org
jordan.kelch@gwctx.org
jorge.ortiz@gwctx.org
lisa.chavez@gwctx.org
mary.medrano@gwctx.org
michelle.clark@gwctx.org
nessa.martin@gwctx.org
nora.moreno@gwctx.org
presley.price@gwctx.org
rachel.mignemi@gwctx.org
robert.bachman@gwctx.org
sandra.mcdowell@gwctx.org
scarlett.miears@gwctx.org
tavonn.uresti@gwctx.org"""

        # G2 Cohort - Wave 2 users
        g2_emails_default = """adryan.mcguire@gwctx.org
alyssa.cabello@gwctx.org
amber.carrizales@gwctx.org
brenda.warner@gwctx.org
daniel.ayala@gwctx.org
dora.mcafee@gwctx.org
eric.sherman@gwctx.org
farah.alabdallah@gwctx.org
jerry.harris@gwctx.org
josette.krebuszewski@gwctx.org
marea.warren-hernandez@gwctx.org
morgan.marley@gwctx.org
orlando.perez@gwctx.org
rafal.cygankow@gwctx.org
roddrick.gaines@gwctx.org
vincent.giddens@gwctx.org"""

        # G3 Cohort - Wave 3 users
        g3_emails_default = """ahmarlay.myint@gwctx.org
alyxandria.currington@gwctx.org
beth.burnett@gwctx.org
brian.shade@gwctx.org
daniella.owens@gwctx.org
elyse.waugh@gwctx.org
erin.delorme@gwctx.org
gyanu.gautam@gwctx.org
jonah.benedict@gwctx.org
kennedy.pasquinzo@gwctx.org
mary.rudinsky@gwctx.org
natalie.watkins@gwctx.org
reva.conley@gwctx.org
samirrah.cooke@gwctx.org
zwany.batista@gwctx.org"""

        # CAB (Client Advisory Board) users
        cab_emails_default = """jswann40@yahoo.com
lukearodriguez07@gmail.com
malikjkobethomas@gmail.com
marielacalanchi3@gmail.com
mtmaguire@gmail.com
mleake86@gmail.com
chanelunknown1@gmail.com
ritadecarlo675@gmail.com"""

        # CAB name lookup (for display purposes)
        cab_name_lookup = {
            'jswann40@yahoo.com': 'Jodi Swann',
            'lukearodriguez07@gmail.com': 'Luke Rodriguez',
            'malikjkobethomas@gmail.com': 'Malik Jkobe Thomas',
            'marielacalanchi3@gmail.com': 'Mariela Calanchi',
            'mtmaguire@gmail.com': 'Mark Maguire',
            'mleake86@gmail.com': 'Mary Leake',
            'chanelunknown1@gmail.com': 'Nova Sannoh',
            'ritadecarlo675@gmail.com': 'Rita DeCarlo'
        }
        
        # Report configuration
        with st.expander("⚙️ Cohort Configuration", expanded=False):
            cohort_tab1, cohort_tab2, cohort_tab3, cohort_tab4 = st.tabs(["G1 Users", "G2 Users", "G3 Users", "CAB Users"])
            
            with cohort_tab1:
                g1_emails_text = st.text_area(
                    "G1 User Emails (one per line)",
                    value=g1_emails_default,
                    height=200,
                    key="g1_config"
                )
            
            with cohort_tab2:
                g2_emails_text = st.text_area(
                    "G2 User Emails (one per line)",
                    value=g2_emails_default,
                    height=200,
                    key="g2_config"
                )
            
            with cohort_tab3:
                g3_emails_text = st.text_area(
                    "G3 User Emails (one per line)",
                    value=g3_emails_default,
                    height=200,
                    key="g3_config"
                )
            
            with cohort_tab4:
                cab_emails_text = st.text_area(
                    "CAB User Emails (one per line)",
                    value=cab_emails_default,
                    height=200,
                    key="cab_config"
                )
        
        # Parse emails
        g1_emails = [e.strip() for e in g1_emails_text.split('\n') if e.strip()]
        g2_emails = [e.strip() for e in g2_emails_text.split('\n') if e.strip()]
        g3_emails = [e.strip() for e in g3_emails_text.split('\n') if e.strip()]
        cab_emails = [e.strip() for e in cab_emails_text.split('\n') if e.strip()]
        
        # Generate comprehensive report
        usage_breakdown = analyzer.get_usage_breakdown_by_level()
        resource_categories = analyzer.get_resource_category_breakdown()
        geo_data = analyzer.extract_zip_codes()
        
        # Overall metrics
        st.subheader("✅ Total Usage Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_users = (len(usage_breakdown['high']) + 
                      len(usage_breakdown['medium']) + 
                      len(usage_breakdown['low']))
        
        with col1:
            st.metric("Unique Users", total_users)
        with col2:
            st.metric("Total Traces", len(analyzer.traces_df))
        with col3:
            referrals = len(analyzer.traces_df[analyzer.traces_df['trace_type'] == 'referrals'])
            st.metric("Referrals", referrals)
        with col4:
            action_plans = len(analyzer.traces_df[analyzer.traces_df['trace_type'] == 'action_plans'])
            st.metric("Action Plans", action_plans)
        with col5:
            email_results = len(analyzer.traces_df[analyzer.traces_df['trace_type'] == 'email_results'])
            st.metric("Emails Sent", email_results)
        
        st.divider()
        
        # User breakdown by level
        st.subheader("🧑‍💼 User Breakdown by Usage Level")
        
        # High-usage staff (4+ traces)
        if usage_breakdown['high']:
            st.markdown("### ⭐ High-Usage Staff (4+ traces)")
            high_df = pd.DataFrame(usage_breakdown['high'])
            high_df.columns = ['Email', 'Name', 'Total Traces']
            st.dataframe(high_df, use_container_width=True, hide_index=True)
        
        # Medium-usage staff (2-3 traces)
        if usage_breakdown['medium']:
            st.markdown("### 🧑‍💼 Medium-Usage Staff (2-3 traces)")
            medium_df = pd.DataFrame(usage_breakdown['medium'])
            medium_df.columns = ['Email', 'Name', 'Total Traces']
            st.dataframe(medium_df, use_container_width=True, hide_index=True)
        
        # Low-usage staff (1 trace)
        if usage_breakdown['low']:
            st.markdown("### 🧑‍💼 Low-Usage Staff (1 trace)")
            low_df = pd.DataFrame(usage_breakdown['low'])
            low_df.columns = ['Email', 'Name', 'Total Traces']
            st.dataframe(low_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # All Cohorts Analysis
        st.subheader("👥 User Cohort Analysis")
        
        # Create tabs for different cohorts
        cohort_display_tabs = st.tabs(["📊 Overview", "G1 Users", "G2 Users", "G3 Users", "🎯 CAB Users"])
        
        # Get all cohort analyses
        g1_cohort = analyzer.get_cohort_analysis(g1_emails, "G1") if g1_emails else {}
        g2_cohort = analyzer.get_cohort_analysis(g2_emails, "G2") if g2_emails else {}
        g3_cohort = analyzer.get_cohort_analysis(g3_emails, "G3") if g3_emails else {}
        cab_cohort = analyzer.get_cohort_analysis(cab_emails, "CAB") if cab_emails else {}
        
        # Overview Tab
        with cohort_display_tabs[0]:
            st.markdown("### 📊 Cohort Adoption Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                g1_rate = g1_cohort.get('adoption_rate', 0) if g1_cohort else 0
                st.metric("G1 Adoption", f"{g1_rate:.0f}%", 
                         f"{g1_cohort.get('active_count', 0)}/{g1_cohort.get('total_cohort', 0)}")
            with col2:
                g2_rate = g2_cohort.get('adoption_rate', 0) if g2_cohort else 0
                st.metric("G2 Adoption", f"{g2_rate:.0f}%",
                         f"{g2_cohort.get('active_count', 0)}/{g2_cohort.get('total_cohort', 0)}")
            with col3:
                g3_rate = g3_cohort.get('adoption_rate', 0) if g3_cohort else 0
                st.metric("G3 Adoption", f"{g3_rate:.0f}%",
                         f"{g3_cohort.get('active_count', 0)}/{g3_cohort.get('total_cohort', 0)}")
            with col4:
                cab_rate = cab_cohort.get('adoption_rate', 0) if cab_cohort else 0
                st.metric("CAB Adoption", f"{cab_rate:.0f}%",
                         f"{cab_cohort.get('active_count', 0)}/{cab_cohort.get('total_cohort', 0)}")
            
            # Combined activity table
            st.markdown("### 📈 All Active Users Across Cohorts")
            all_active = []
            for cohort_data, cohort_name in [(g1_cohort, 'G1'), (g2_cohort, 'G2'), (g3_cohort, 'G3'), (cab_cohort, 'CAB')]:
                if cohort_data and cohort_data.get('active_users'):
                    for user in cohort_data['active_users']:
                        user_copy = user.copy()
                        user_copy['cohort'] = cohort_name
                        # Use CAB name lookup for CAB users
                        if cohort_name == 'CAB' and user['email'].lower() in [e.lower() for e in cab_name_lookup.keys()]:
                            for orig_email, name in cab_name_lookup.items():
                                if user['email'].lower() == orig_email.lower():
                                    user_copy['name'] = name
                                    break
                        all_active.append(user_copy)
            
            if all_active:
                all_active_df = pd.DataFrame(all_active)
                display_cols = ['cohort', 'name', 'email', 'trace_count', 'first_trace', 'last_trace']
                available_cols = [c for c in display_cols if c in all_active_df.columns]
                all_active_display = all_active_df[available_cols].copy()
                all_active_display.columns = ['Cohort', 'Name', 'Email', 'Traces', 'First Use', 'Last Use'][:len(available_cols)]
                if 'First Use' in all_active_display.columns:
                    all_active_display['First Use'] = pd.to_datetime(all_active_display['First Use']).dt.strftime('%Y-%m-%d')
                if 'Last Use' in all_active_display.columns:
                    all_active_display['Last Use'] = pd.to_datetime(all_active_display['Last Use']).dt.strftime('%Y-%m-%d')
                all_active_display = all_active_display.sort_values('Traces', ascending=False)
                st.dataframe(all_active_display, use_container_width=True, hide_index=True)
            else:
                st.info("No active users found across any cohort")
        
        # Helper function to display cohort details
        def display_cohort(cohort_data, cohort_name, name_lookup=None):
            if not cohort_data:
                st.info(f"No {cohort_name} cohort data configured")
                return
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"Total {cohort_name} Users", cohort_data.get('total_cohort', 0))
            with col2:
                st.metric("Active Users", cohort_data.get('active_count', 0))
            with col3:
                adoption = cohort_data.get('adoption_rate', 0)
                st.metric("Adoption Rate", f"{adoption:.1f}%")
            
            # Active users
            if cohort_data.get('active_users'):
                st.markdown(f"### ✅ {cohort_name} Users Who HAVE Used the Tool")
                active_df = pd.DataFrame(cohort_data['active_users'])
                
                # Apply name lookup if provided
                if name_lookup:
                    active_df['name'] = active_df['email'].apply(
                        lambda e: name_lookup.get(e.lower(), name_lookup.get(e, e.split('@')[0]))
                    )
                
                display_cols = ['name', 'email', 'trace_count', 'first_trace', 'last_trace']
                available_cols = [c for c in display_cols if c in active_df.columns]
                active_display = active_df[available_cols].copy()
                active_display.columns = ['Name', 'Email', 'Traces', 'First Use', 'Last Use'][:len(available_cols)]
                if 'First Use' in active_display.columns:
                    active_display['First Use'] = pd.to_datetime(active_display['First Use']).dt.strftime('%Y-%m-%d')
                if 'Last Use' in active_display.columns:
                    active_display['Last Use'] = pd.to_datetime(active_display['Last Use']).dt.strftime('%Y-%m-%d')
                st.dataframe(active_display, use_container_width=True, hide_index=True)
            
            # Inactive users
            if cohort_data.get('non_users'):
                st.markdown(f"### ❌ {cohort_name} Users Who Have NOT Used the Tool")
                non_users_df = pd.DataFrame(cohort_data['non_users'])
                
                # Apply name lookup if provided
                if name_lookup:
                    non_users_df['name'] = non_users_df['email'].apply(
                        lambda e: name_lookup.get(e.lower(), name_lookup.get(e, e.split('@')[0]))
                    )
                
                non_users_display = non_users_df[['name', 'email']].copy()
                non_users_display.columns = ['Name', 'Email']
                st.dataframe(non_users_display, use_container_width=True, hide_index=True)
        
        # G1 Tab
        with cohort_display_tabs[1]:
            st.markdown("### 🔵 G1 User Cohort (Wave 1)")
            display_cohort(g1_cohort, "G1")
        
        # G2 Tab
        with cohort_display_tabs[2]:
            st.markdown("### 🟢 G2 User Cohort (Wave 2)")
            display_cohort(g2_cohort, "G2")
        
        # G3 Tab
        with cohort_display_tabs[3]:
            st.markdown("### 🟡 G3 User Cohort (Wave 3)")
            display_cohort(g3_cohort, "G3")
        
        # CAB Tab
        with cohort_display_tabs[4]:
            st.markdown("### 🎯 Client Advisory Board (CAB)")
            # Create lowercase lookup for CAB names
            cab_name_lookup_lower = {k.lower(): v for k, v in cab_name_lookup.items()}
            display_cohort(cab_cohort, "CAB", cab_name_lookup_lower)
        
        st.divider()
        
        # Resource category breakdown
        st.subheader("🎯 Resource Type Demand")
        
        if resource_categories:
            # Summary table
            cat_summary = []
            for category, data in resource_categories.items():
                cat_summary.append({
                    'Resource Type': category,
                    'Count': data['count'],
                    '% of Total': f"{data['percentage']:.1f}%"
                })
            
            cat_df = pd.DataFrame(cat_summary)
            st.dataframe(cat_df, use_container_width=True, hide_index=True)
            
            # Visualization
            fig = px.bar(
                cat_df,
                x='Resource Type',
                y='Count',
                title='Resource Demand by Category',
                text='Count'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed breakdown
            st.markdown("### 🧭 Category Breakdown with Examples")
            for category, data in resource_categories.items():
                with st.expander(f"{category} — {data['count']} traces ({data['percentage']:.1f}%)"):
                    if data.get('examples'):
                        st.markdown("**Sample queries:**")
                        for i, example in enumerate(data['examples'], 1):
                            st.markdown(f"{i}. **{example['user']}**: {example['query']}")
        
        st.divider()
        
        # Geographic analysis
        st.subheader("📍 Geographic Coverage")
        
        if geo_data.get('all_zips'):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Unique Zip Codes", geo_data['total_unique'])
                
                # Texas zips
                if geo_data.get('texas'):
                    st.markdown("**Central Texas Cluster:**")
                    for item in geo_data['texas'][:10]:
                        city = item.get('city', 'Unknown')
                        st.text(f"{item['zip']} - {city} ({item['count']} traces)")
            
            with col2:
                # Out of state
                if geo_data.get('out_of_state'):
                    st.markdown("**Out-of-Region:**")
                    for item in geo_data['out_of_state']:
                        city = item.get('city', 'Unknown')
                        st.text(f"{item['zip']} - {city} ({item['count']} traces)")
        else:
            st.info("No zip codes detected in trace queries")
        
        # Download report
        st.divider()
        st.subheader("📥 Export Report")
        
        # Prepare export data
        _export_data = {
            'Usage Summary': pd.DataFrame([{
                'Total Users': total_users,
                'Total Traces': len(analyzer.traces_df),
                'Referrals': referrals,
                'Action Plans': action_plans
            }]),
            'High Usage Users': high_df if usage_breakdown['high'] else pd.DataFrame(),
            'Medium Usage Users': medium_df if usage_breakdown['medium'] else pd.DataFrame(),
            'Low Usage Users': low_df if usage_breakdown['low'] else pd.DataFrame()
        }
        
        # Create download button for each sheet
        col1, col2 = st.columns(2)
        with col1:
            if usage_breakdown['high']:
                csv = high_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download High-Usage Users CSV",
                    csv,
                    "high_usage_users.csv",
                    "text/csv"
                )
        with col2:
            if resource_categories:
                cat_csv = cat_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Resource Categories CSV",
                    cat_csv,
                    "resource_categories.csv",
                    "text/csv"
                )
    
    # TAB 4: Performance Metrics
    with tab4:
        st.header("Performance Metrics")
        st.markdown("*Detailed performance analysis*")
        
        # Latency distribution
        st.subheader("Latency Distribution")
        latency_dist = analyzer.get_latency_distribution()
        
        if latency_dist:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.histogram(
                    x=latency_dist['histogram'],
                    nbins=50,
                    title="Latency Distribution",
                    labels={'x': 'Latency (s)', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Latency Statistics**")
                st.metric("Mean", f"{latency_dist['mean']:.2f}s")
                st.metric("Median", f"{latency_dist['median']:.2f}s")
                st.metric("Std Dev", f"{latency_dist['std']:.2f}s")
                st.metric("Min", f"{latency_dist['min']:.2f}s")
                st.metric("Max", f"{latency_dist['max']:.2f}s")
        
        # Percentiles
        if latency_dist and 'percentiles' in latency_dist:
            st.subheader("Latency Percentiles")
            percentiles_df = pd.DataFrame(
                list(latency_dist['percentiles'].items()),
                columns=['Percentile', 'Latency (s)']
            )
            
            fig = px.bar(
                percentiles_df,
                x='Percentile',
                y='Latency (s)',
                title='Latency Percentiles'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison
        st.subheader("Model Performance Comparison")
        model_comparison = analyzer.get_model_performance_comparison()
        
        if not model_comparison.empty:
            st.dataframe(
                model_comparison.style.format({
                    'total_requests': '{:,.0f}',
                    'avg_latency': '{:.2f}s',
                    'median_latency': '{:.2f}s',
                    'avg_tokens': '{:.2f}',
                    'total_tokens': '{:,.0f}',
                    'error_rate': '{:.2f}%'
                }),
                use_container_width=True
            )
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    model_comparison,
                    x='model',
                    y='avg_latency',
                    title='Average Latency by Model',
                    labels={'avg_latency': 'Avg Latency (s)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    model_comparison,
                    x='model',
                    y='error_rate',
                    title='Error Rate by Model',
                    labels={'error_rate': 'Error Rate (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Error analysis
        st.subheader("Error Analysis")
        error_analysis = analyzer.get_error_analysis()
        
        if error_analysis and error_analysis['total_errors'] > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Errors", error_analysis['total_errors'])
                st.metric("Error Rate", f"{error_analysis['error_rate']:.2f}%")
            
            with col2:
                if error_analysis['errors_by_type']:
                    errors_df = pd.DataFrame(
                        list(error_analysis['errors_by_type'].items()),
                        columns=['Error Type', 'Count']
                    )
                    fig = px.pie(
                        errors_df,
                        values='Count',
                        names='Error Type',
                        title='Errors by Type'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No errors detected in the analyzed period!")
    
    # TAB 5: Log Explorer
    with tab5:
        st.header("Log Explorer")
        st.markdown("*Exportable trace list + latency bottleneck breakdown*")

        if analyzer.traces_df.empty:
            st.info("No trace data available to explore.")
        else:
            st.subheader("📄 All Traces (User, Query, Trace Latency)")
            st.caption("This is designed to be your download-ready dataset of what users asked + how long the end-to-end trace took.")

            only_query_traces = st.checkbox(
                "Only include traces with a non-empty query (recommended for a 'questions' dataset)",
                value=True
            )

            traces = analyzer.traces_df.copy()
            # Keep the most useful columns for export / regression testing.
            preferred_cols = [
                'trace_start', 'trace_id', 'user_name', 'user_email',
                'trace_type', 'query', 'category', 'location_preference',
                'zip_code', 'trace_duration_s', 'status', 'total_tokens', 'tokens_estimated'
            ]
            cols = [c for c in preferred_cols if c in traces.columns]
            traces = traces[cols].copy()

            if 'query' in traces.columns:
                traces['query'] = traces['query'].fillna('').astype(str)
                if only_query_traces:
                    traces = traces[traces['query'].str.strip().str.len() > 0]

            if 'trace_start' in traces.columns:
                traces = traces.sort_values('trace_start', ascending=False)

            display = traces.copy()
            if 'trace_start' in display.columns:
                display['trace_start'] = pd.to_datetime(display['trace_start'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
                display = display.rename(columns={'trace_start': 'timestamp'})

            if 'trace_duration_s' in display.columns:
                display['trace_duration_s'] = pd.to_numeric(display['trace_duration_s'], errors='coerce').round(3)
                display = display.rename(columns={'trace_duration_s': 'trace_latency_s'})

            if 'total_tokens' in display.columns:
                display['total_tokens'] = (
                    pd.to_numeric(display['total_tokens'], errors='coerce')
                    .fillna(0)
                    .astype(int)
                )

            rename_map = {
                'user_name': 'user',
                'user_email': 'email',
            }
            display = display.rename(columns=rename_map)

            st.dataframe(display.fillna(''), use_container_width=True, hide_index=True)

            csv_bytes = display.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download trace list (CSV)",
                csv_bytes,
                "phoenix_trace_list.csv",
                "text/csv",
                key="download-trace-list-csv"
            )

            st.divider()
            st.subheader("📈 Trends & Correlation (Latency vs Tokens)")

            # Build a numeric/time series-friendly view (separate from the display table formatting).
            trend_df = traces.copy()
            if 'trace_start' in trend_df.columns:
                trend_df['trace_start'] = pd.to_datetime(trend_df['trace_start'], errors='coerce')
            if 'trace_duration_s' in trend_df.columns:
                trend_df['trace_duration_s'] = pd.to_numeric(trend_df['trace_duration_s'], errors='coerce')
            if 'total_tokens' in trend_df.columns:
                trend_df['total_tokens'] = pd.to_numeric(trend_df['total_tokens'], errors='coerce')

            trend_df = trend_df.dropna(subset=[c for c in ['trace_start', 'trace_duration_s', 'total_tokens'] if c in trend_df.columns])

            if trend_df.empty or 'trace_start' not in trend_df.columns:
                st.info("Not enough data to plot trends.")
            else:
                trend_df = trend_df.sort_values('trace_start')

                col1, col2 = st.columns(2)
                with col1:
                    fig = px.line(
                        trend_df,
                        x='trace_start',
                        y='trace_duration_s',
                        title='Trace Latency Over Time',
                        labels={'trace_start': 'Time', 'trace_duration_s': 'Latency (s)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    if 'total_tokens' in trend_df.columns:
                        fig = px.line(
                            trend_df,
                            x='trace_start',
                            y='total_tokens',
                            title='Tokens Over Time',
                            labels={'trace_start': 'Time', 'total_tokens': 'Tokens'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No token data available for tokens-over-time chart.")

                if 'total_tokens' in trend_df.columns:
                    st.subheader("🔎 Latency vs Tokens (Correlation)")
                    corr = trend_df['trace_duration_s'].corr(trend_df['total_tokens'])
                    st.metric("Pearson correlation (latency vs tokens)", f"{corr:.3f}" if pd.notna(corr) else "N/A")

                    fig = px.scatter(
                        trend_df,
                        x='total_tokens',
                        y='trace_duration_s',
                        title='Latency vs Tokens',
                        labels={'total_tokens': 'Tokens', 'trace_duration_s': 'Latency (s)'},
                        hover_data=[c for c in ['user_email', 'user_name', 'trace_type', 'query', 'trace_id'] if c in trend_df.columns]
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("⏱️ Average Latency by Trace Step (Span Name)")
            st.caption("Aggregated across the currently listed traces; use this to spot the biggest latency bottlenecks.")

            if analyzer.df.empty or 'name' not in analyzer.df.columns:
                st.info("Span-level data is not available for step breakdown.")
            else:
                spans_df = analyzer.df.copy()
                if 'trace_id' in spans_df.columns and 'trace_id' in traces.columns:
                    spans_df = spans_df[spans_df['trace_id'].isin(traces['trace_id'].dropna().unique())]

                if 'latency_s' in spans_df.columns:
                    spans_df['latency_s'] = pd.to_numeric(spans_df['latency_s'], errors='coerce')

                spans_df = spans_df.dropna(subset=['name', 'latency_s'])

                if spans_df.empty:
                    st.info("No spans with latency were found for the selected traces.")
                else:
                    step_df = spans_df.groupby('name').agg(
                        avg_latency_s=('latency_s', 'mean'),
                        p95_latency_s=('latency_s', lambda x: x.quantile(0.95)),
                        count_spans=('latency_s', 'count'),
                        count_traces=('trace_id', pd.Series.nunique) if 'trace_id' in spans_df.columns else ('latency_s', 'count'),
                    ).reset_index().rename(columns={'name': 'step'})

                    step_df = step_df.sort_values('avg_latency_s', ascending=False)
                    step_display = step_df.copy()
                    for c in ['avg_latency_s', 'p95_latency_s']:
                        if c in step_display.columns:
                            step_display[c] = pd.to_numeric(step_display[c], errors='coerce').round(3)

                    st.dataframe(step_display, use_container_width=True, hide_index=True)

                    step_csv = step_display.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download step latency breakdown (CSV)",
                        step_csv,
                        "phoenix_span_step_latency.csv",
                        "text/csv",
                        key="download-step-latency-csv"
                    )

    # TAB 6: Advanced Analytics
    with tab6:
        st.header("🧠 Advanced Analytics")
        st.markdown("*Deep insights into query patterns, resource effectiveness, and user behavior*")
        
        # Feature #1: Query Pattern Analysis
        st.divider()
        st.subheader("🔍 Query Pattern Analysis & Search Intelligence")
        
        with st.spinner("Analyzing query patterns..."):
            query_analysis = analyzer.analyze_query_patterns()
        
        if query_analysis:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Unique Queries", query_analysis.get('total_unique_queries', 0))
            with col2:
                st.metric("Avg Query Length", f"{query_analysis.get('avg_query_length', 0):.0f} chars")
            with col3:
                st.metric("Refinement Rate", f"{query_analysis.get('refinement_rate', 0):.1f}%")
            with col4:
                st.metric("No Action Rate", f"{query_analysis.get('no_action_rate', 0):.1f}%")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔥 Most Common Query Terms**")
                top_terms = query_analysis.get('top_terms', {})
                if top_terms:
                    term_df = pd.DataFrame([
                        {'Term': term, 'Count': count} 
                        for term, count in list(top_terms.items())[:15]
                    ])
                    st.dataframe(term_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**🔄 Query Refinements** (users re-querying within 15 min)")
                refinements = query_analysis.get('refinements', [])
                if refinements:
                    for ref in refinements[:5]:
                        with st.expander(f"{ref.get('user', 'Unknown')} - {ref.get('time_gap_minutes', 0)} min gap"):
                            st.text(f"Query 1: {ref.get('query1', '')}")
                            st.text(f"Query 2: {ref.get('query2', '')}")
                            st.text(f"Category: {ref.get('category', 'N/A')}")
                else:
                    st.info("No query refinements detected")
            
            st.markdown("**⚠️ Queries Without Follow-up Actions** (Potential Unmet Needs)")
            no_action = query_analysis.get('no_action_examples', [])
            if no_action:
                no_action_df = pd.DataFrame(no_action)
                st.dataframe(no_action_df, use_container_width=True, hide_index=True)
            else:
                st.success("All queries have follow-up actions!")
        else:
            st.info("No query data available for analysis")
        
        # Feature #2: Resource Effectiveness
        st.divider()
        st.subheader("⚡ Resource Effectiveness Scoring")
        
        with st.spinner("Analyzing resource effectiveness..."):
            resource_analysis = analyzer.analyze_resource_effectiveness()
        
        if resource_analysis and 'message' not in resource_analysis:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Referrals", resource_analysis.get('total_referrals', 0))
            with col2:
                st.metric("Action Plans Created", resource_analysis.get('total_action_plans', 0))
            with col3:
                st.metric("Conversion Rate", f"{resource_analysis.get('overall_conversion_rate', 0):.1f}%")
            with col4:
                users_both = resource_analysis.get('users_both', 0)
                users_searched = resource_analysis.get('users_searched', 1)
                st.metric("User Conversion", f"{users_both}/{users_searched}")
            
            st.markdown("**📊 Category Effectiveness** (Referral → Action Plan Conversion)")
            effectiveness = resource_analysis.get('category_effectiveness', [])
            if effectiveness:
                eff_df = pd.DataFrame(effectiveness)
                
                # Create bar chart
                import altair as alt
                chart = alt.Chart(eff_df).mark_bar().encode(
                    x=alt.X('conversion_rate:Q', title='Conversion Rate (%)'),
                    y=alt.Y('category:N', sort='-x', title='Category'),
                    color=alt.Color('conversion_rate:Q', scale=alt.Scale(scheme='blues')),
                    tooltip=['category', 'referrals', 'action_plans', 'conversion_rate']
                ).properties(height=400)
                
                st.altair_chart(chart, use_container_width=True)
                
                st.dataframe(eff_df, use_container_width=True, hide_index=True)
        else:
            st.info(resource_analysis.get('message', 'No resource effectiveness data available'))
        
        # Feature #3: User Journey & Workflow Analytics
        st.divider()
        st.subheader("🛣️ User Journey & Workflow Analytics")
        
        with st.spinner("Analyzing user journeys..."):
            journey_analysis = analyzer.analyze_user_journeys()
        
        if journey_analysis:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Sessions", journey_analysis.get('total_sessions', 0))
            with col2:
                st.metric("Avg Session Duration", f"{journey_analysis.get('avg_session_duration', 0):.1f} min")
            with col3:
                st.metric("Multi-Category Rate", f"{journey_analysis.get('multi_category_rate', 0):.1f}%")
            with col4:
                st.metric("Drop-off Rate", f"{journey_analysis.get('drop_off_rate', 0):.1f}%")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**⏱️ Time to Action** (First referral → First action plan)")
                st.metric("Average", f"{journey_analysis.get('avg_time_to_action', 0):.1f} minutes")
                st.metric("Median", f"{journey_analysis.get('median_time_to_action', 0):.1f} minutes")
                
                time_examples = journey_analysis.get('time_to_action_examples', [])
                if time_examples:
                    st.markdown("**Fastest Users:**")
                    for ex in time_examples:
                        st.text(f"{ex.get('user', 'Unknown')}: {ex.get('time_minutes', 0)} min")
            
            with col2:
                st.markdown("**🚪 Drop-off Analysis**")
                users_dropped = journey_analysis.get('users_dropped', 0)
                st.metric("Users Who Searched But Never Acted", users_dropped)
                
                dropped_cats = journey_analysis.get('dropped_user_categories', {})
                if dropped_cats:
                    st.markdown("**Categories where users dropped off:**")
                    for cat, count in list(dropped_cats.items())[:5]:
                        st.text(f"{cat}: {count} users")
            
            st.markdown("**📝 Sample User Sessions**")
            session_examples = journey_analysis.get('session_examples', [])
            if session_examples:
                for sess in session_examples[:3]:
                    with st.expander(f"{sess.get('user', 'Unknown')} - {sess.get('duration_minutes', 0):.1f} min session"):
                        st.text(f"Traces: {sess.get('trace_count', 0)}")
                        st.text(f"Types: {', '.join(sess.get('types', []))}")
                        st.text(f"Categories: {', '.join(sess.get('categories', []))}")
        else:
            st.info("No user journey data available for analysis")
        
        # Feature #4: Performance & Quality Metrics
        st.divider()
        st.subheader("🎯 Performance & Quality Metrics")
        
        with st.spinner("Analyzing performance..."):
            perf_analysis = analyzer.analyze_performance_quality()
        
        if perf_analysis:
            # Key performance metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Error Rate", f"{perf_analysis.get('overall_error_rate', 0):.1f}%")
            with col2:
                benchmarks = perf_analysis.get('benchmarks', {})
                st.metric("Median Response Time", f"{benchmarks.get('p50', 0):.2f}s")
            with col3:
                st.metric("P95 Response Time", f"{benchmarks.get('p95', 0):.2f}s")
            with col4:
                llm_perf = perf_analysis.get('llm_performance', {})
                st.metric("Total LLM Calls", llm_perf.get('total_llm_calls', 0))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**⏱️ Response Time Benchmarks**")
                benchmarks = perf_analysis.get('benchmarks', {})
                if benchmarks:
                    bench_df = pd.DataFrame([
                        {'Percentile': 'P50 (Median)', 'Time (s)': f"{benchmarks.get('p50', 0):.2f}"},
                        {'Percentile': 'P75', 'Time (s)': f"{benchmarks.get('p75', 0):.2f}"},
                        {'Percentile': 'P90', 'Time (s)': f"{benchmarks.get('p90', 0):.2f}"},
                        {'Percentile': 'P95', 'Time (s)': f"{benchmarks.get('p95', 0):.2f}"},
                        {'Percentile': 'P99', 'Time (s)': f"{benchmarks.get('p99', 0):.2f}"}
                    ])
                    st.dataframe(bench_df, use_container_width=True, hide_index=True)
                
                st.markdown("**🔥 LLM Performance**")
                llm_perf = perf_analysis.get('llm_performance', {})
                if llm_perf:
                    st.text(f"Avg LLM Latency: {llm_perf.get('avg_llm_latency_s', 0):.2f}s")
                    st.text(f"Median LLM Latency: {llm_perf.get('median_llm_latency_s', 0):.2f}s")
                    st.text(f"P95 LLM Latency: {llm_perf.get('p95_llm_latency_s', 0):.2f}s")
            
            with col2:
                st.markdown("**📊 Token Usage**")
                token_usage = perf_analysis.get('token_usage', {})
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Total Tokens", f"{token_usage.get('total_tokens', 0):,}")
                with col_b:
                    st.metric("Avg per Trace", f"{token_usage.get('avg_tokens_per_trace', 0):,.0f}")
                
                st.markdown("**🚨 Error Rates by Category**")
                error_by_cat = perf_analysis.get('error_rate_by_category', {})
                if error_by_cat:
                    error_df = pd.DataFrame([
                        {'Category': cat, 'Total': data['total'], 'Errors': data['errors'], 'Error Rate': f"{data['error_rate']:.1f}%"}
                        for cat, data in error_by_cat.items() if cat and data['total'] > 0
                    ])
                    if not error_df.empty:
                        st.dataframe(error_df.head(10), use_container_width=True, hide_index=True)
            
            # Slow traces
            slow_traces = perf_analysis.get('slow_traces', [])
            if slow_traces:
                st.markdown(f"**🐢 Slowest Traces** (>{perf_analysis.get('slow_threshold_s', 10):.1f}s)")
                slow_df = pd.DataFrame(slow_traces)
                st.dataframe(slow_df, use_container_width=True, hide_index=True)
        else:
            st.info("No performance data available")
        
        # Feature #5: Geographic Service Gap Analysis
        st.divider()
        st.subheader("🗺️ Geographic Service Gap Analysis")
        
        with st.spinner("Analyzing geographic coverage..."):
            geo_analysis = analyzer.analyze_geographic_gaps()
        
        if geo_analysis:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Zip Mentions", geo_analysis.get('total_zip_mentions', 0))
            with col2:
                st.metric("Unique Zip Codes", geo_analysis.get('unique_zips', 0))
            with col3:
                st.metric("In-Region Zips", geo_analysis.get('in_region_zips', 0))
            with col4:
                st.metric("Out-of-Region", geo_analysis.get('out_of_region_zips', 0))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🏠 Central Texas Coverage**")
                coverage = geo_analysis.get('coverage_summary', {})
                if coverage:
                    coverage_df = pd.DataFrame([
                        {'Area': 'Austin Metro', 'Zip Codes Served': coverage.get('austin_metro', 0)},
                        {'Area': 'Round Rock', 'Zip Codes Served': coverage.get('round_rock', 0)},
                        {'Area': 'Georgetown', 'Zip Codes Served': coverage.get('georgetown', 0)},
                        {'Area': 'Pflugerville', 'Zip Codes Served': coverage.get('pflugerville', 0)},
                        {'Area': 'Cedar Park/Leander', 'Zip Codes Served': coverage.get('cedar_park_leander', 0)},
                        {'Area': 'South Austin (Buda/Kyle)', 'Zip Codes Served': coverage.get('south_austin', 0)}
                    ])
                    st.dataframe(coverage_df, use_container_width=True, hide_index=True)
                
                st.markdown("**📍 In-Region Details**")
                in_region = geo_analysis.get('in_region_details', {})
                if in_region:
                    region_df = pd.DataFrame([
                        {'Zip': z, 'City': data['city'], 'Requests': data['count'], 'Categories': ', '.join(data['categories'][:2])}
                        for z, data in list(in_region.items())[:10]
                    ])
                    st.dataframe(region_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**🌎 Out-of-Region Requests**")
                out_region = geo_analysis.get('out_of_region_details', {})
                if out_region:
                    out_df = pd.DataFrame([
                        {'Zip': z, 'City': data['city'], 'Requests': data['count'], 'Categories': ', '.join(data['categories'][:2])}
                        for z, data in list(out_region.items())[:10]
                    ])
                    st.dataframe(out_df, use_container_width=True, hide_index=True)
                else:
                    st.success("All requests are within Central Texas region!")
                
                st.markdown("**⚠️ Locations Without Zip Codes** (Potential Service Gaps)")
                no_zip = geo_analysis.get('location_without_zip', {})
                if no_zip:
                    for loc, count in list(no_zip.items())[:5]:
                        st.text(f"{loc}: {count} mentions")
                else:
                    st.success("All location requests have associated zip codes")
            
            # Category by region
            cat_by_region = geo_analysis.get('category_by_region', {})
            if cat_by_region:
                st.markdown("**📊 Category Demand by Region**")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("*Central Texas*")
                    ctx = cat_by_region.get('central_texas', {})
                    if ctx:
                        for cat, count in sorted(ctx.items(), key=lambda x: x[1], reverse=True)[:5]:
                            st.text(f"{cat}: {count}")
                with col2:
                    st.markdown("*Out-of-Region*")
                    oor = cat_by_region.get('out_of_region', {})
                    if oor:
                        for cat, count in sorted(oor.items(), key=lambda x: x[1], reverse=True)[:5]:
                            st.text(f"{cat}: {count}")
        else:
            st.info("No geographic data available")
        
        # Feature #6: Comparative Period Analysis
        st.divider()
        st.subheader("📅 Comparative Period Analysis")
        
        st.markdown("*Compare metrics between two time periods*")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Period 1 (Previous)**")
            _p1_days = st.slider("Days ago (start)", 7, 30, 14, key="p1_days")
        with col2:
            st.markdown("**Period 2 (Recent)**")
            p2_days = st.slider("Days to compare", 3, 14, 7, key="p2_days")
        
        # Calculate period dates
        now = datetime.now(timezone.utc)
        p2_end = now
        p2_start = now - pd.Timedelta(days=p2_days)
        p1_end = p2_start
        p1_start = p1_end - pd.Timedelta(days=p2_days)
        
        with st.spinner("Comparing periods..."):
            compare_analysis = analyzer.analyze_comparative_periods(p1_start, p1_end, p2_start, p2_end)
        
        if compare_analysis:
            # Period summaries
            p1 = compare_analysis.get('period1', {})
            p2 = compare_analysis.get('period2', {})
            changes = compare_analysis.get('changes', {})
            
            st.markdown(f"**Comparing:** {p1.get('start', '')[:10]} to {p1.get('end', '')[:10]} vs {p2.get('start', '')[:10]} to {p2.get('end', '')[:10]}")
            
            # Growth trajectory indicator
            trajectory = compare_analysis.get('growth_trajectory', 'stable')
            trajectory_icon = '📈' if trajectory == 'growing' else ('📉' if trajectory == 'declining' else '➡️')
            st.markdown(f"**Overall Trajectory:** {trajectory_icon} {trajectory.upper()}")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            _p1_metrics = p1.get('metrics', {})
            p2_metrics = p2.get('metrics', {})
            
            with col1:
                delta = changes.get('traces_change', 0)
                st.metric("Traces", p2_metrics.get('total_traces', 0), 
                         f"{delta:+.1f}%" if delta != 0 else "0%")
            with col2:
                delta = changes.get('users_change', 0)
                st.metric("Users", p2_metrics.get('unique_users', 0),
                         f"{delta:+.1f}%" if delta != 0 else "0%")
            with col3:
                delta = changes.get('referrals_change', 0)
                st.metric("Referrals", p2_metrics.get('referrals', 0),
                         f"{delta:+.1f}%" if delta != 0 else "0%")
            with col4:
                delta = changes.get('action_plans_change', 0)
                st.metric("Action Plans", p2_metrics.get('action_plans', 0),
                         f"{delta:+.1f}%" if delta != 0 else "0%")
            with col5:
                delta = changes.get('duration_change', 0)
                st.metric("Avg Duration", f"{p2_metrics.get('avg_duration_s', 0):.1f}s",
                         f"{delta:+.1f}%" if delta != 0 else "0%")
            
            # Category changes
            cat_changes = compare_analysis.get('category_changes', {})
            if cat_changes:
                st.markdown("**Category Shifts**")
                cat_df = pd.DataFrame([
                    {
                        'Category': cat,
                        'Period 1': data['period1'],
                        'Period 2': data['period2'],
                        'Change': f"{data['change_pct']:+.0f}%"
                    }
                    for cat, data in cat_changes.items() 
                    if cat and (data['period1'] > 0 or data['period2'] > 0)
                ])
                if not cat_df.empty:
                    st.dataframe(cat_df.sort_values('Period 2', ascending=False).head(10), 
                                use_container_width=True, hide_index=True)
            
            # New/dropped categories
            col1, col2 = st.columns(2)
            with col1:
                new_cats = compare_analysis.get('new_categories', [])
                if new_cats:
                    st.markdown("**🆕 New Categories in Period 2:**")
                    for cat in new_cats:
                        st.text(f"• {cat}")
            with col2:
                dropped_cats = compare_analysis.get('dropped_categories', [])
                if dropped_cats:
                    st.markdown("**📉 Categories Not in Period 2:**")
                    for cat in dropped_cats:
                        st.text(f"• {cat}")
        else:
            st.info("Not enough data for period comparison")
        
        # Feature #8: Real-time Alerting
        st.divider()
        st.subheader("🚨 System Health & Alerts")
        
        with st.spinner("Checking system health..."):
            alert_analysis = analyzer.analyze_realtime_alerts()
        
        if alert_analysis:
            status = alert_analysis.get('status', 'healthy')
            
            # Status indicator
            status_colors = {
                'healthy': '🟢',
                'attention': '🟡', 
                'warning': '🟠',
                'critical': '🔴'
            }
            
            st.markdown(f"### System Status: {status_colors.get(status, '⚪')} {status.upper()}")
            
            # Health checks
            health_checks = alert_analysis.get('health_checks', {})
            if health_checks:
                cols = st.columns(5)
                check_icons = {'ok': '✅', 'issue': '⚠️'}
                
                for i, (check, result) in enumerate(health_checks.items()):
                    with cols[i]:
                        st.markdown(f"{check_icons.get(result, '❓')} **{check.replace('_', ' ').title()}**")
            
            # Alert counts
            alert_counts = alert_analysis.get('alert_counts', {})
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Critical Alerts", alert_counts.get('critical', 0))
            with col2:
                st.metric("Warnings", alert_counts.get('warning', 0))
            with col3:
                st.metric("Info Notices", alert_counts.get('info', 0))
            
            # Alert details
            alerts = alert_analysis.get('alerts', [])
            if alerts:
                st.markdown("**📋 Active Alerts**")
                for alert in alerts:
                    severity = alert.get('severity', 'info')
                    icon = {'critical': '🔴', 'warning': '🟠', 'info': '🔵'}[severity]
                    
                    with st.expander(f"{icon} [{severity.upper()}] {alert.get('type', 'Unknown').replace('_', ' ').title()}"):
                        st.markdown(f"**Message:** {alert.get('message', '')}")
                        st.markdown(f"**Value:** {alert.get('value', 'N/A')}")
                        st.markdown(f"**Threshold:** {alert.get('threshold', 'N/A')}")
            else:
                st.success("🎉 No alerts! All systems operating normally.")
        else:
            st.info("Unable to perform health check")
        
        # Feature #9: AI-Powered Query Understanding
        st.divider()
        st.subheader("🤖 AI-Powered Query Understanding")
        st.markdown("*Deep analysis of query patterns, intents, entities, and quality*")
        
        with st.spinner("Analyzing queries with AI..."):
            query_intel = analyzer.analyze_query_intelligence()
        
        if query_intel:
            # Top-level metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Queries Analyzed", query_intel.get('total_queries_analyzed', 0))
            with col2:
                quality_stats = query_intel.get('quality_stats', {})
                st.metric("Avg Quality Score", f"{quality_stats.get('avg', 0):.0f}/100")
            with col3:
                complexity_stats = query_intel.get('complexity_stats', {})
                st.metric("Avg Complexity", f"{complexity_stats.get('avg', 0):.1f}/10")
            with col4:
                st.metric("Need Improvement", f"{quality_stats.get('needs_improvement_pct', 0):.0f}%")
            
            # AI Insights
            insights = query_intel.get('insights', [])
            if insights:
                st.markdown("### 💡 AI-Generated Insights")
                for insight in insights:
                    st.info(insight)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Intent Distribution
                st.markdown("**🎯 Intent Classification**")
                intent_dist = query_intel.get('intent_distribution', {})
                if intent_dist:
                    intent_df = pd.DataFrame([
                        {'Intent': k.replace('_', ' ').title(), 'Count': v}
                        for k, v in sorted(intent_dist.items(), key=lambda x: x[1], reverse=True)
                    ])
                    st.dataframe(intent_df, use_container_width=True, hide_index=True)
                
                # Urgency Distribution
                st.markdown("**⏰ Urgency Levels**")
                urgency_dist = query_intel.get('urgency_distribution', {})
                if urgency_dist:
                    urgency_icons = {'critical': '🔴', 'high': '🟠', 'moderate': '🟡', 'normal': '🟢'}
                    for level in ['critical', 'high', 'moderate', 'normal']:
                        count = urgency_dist.get(level, 0)
                        if count > 0:
                            st.text(f"{urgency_icons.get(level, '⚪')} {level.title()}: {count}")
            
            with col2:
                # Entity Extraction Summary
                st.markdown("**🏷️ Extracted Entities**")
                entity_summary = query_intel.get('entity_summary', {})
                
                with st.expander("👤 Client Demographics"):
                    demos = entity_summary.get('top_demographics', {})
                    if demos:
                        for demo, count in demos.items():
                            st.text(f"{demo.replace('_', ' ').title()}: {count}")
                    else:
                        st.text("No demographics detected")
                
                with st.expander("🔧 Services Requested"):
                    services = entity_summary.get('top_services', {})
                    if services:
                        for service, count in services.items():
                            st.text(f"{service.replace('_', ' ').title()}: {count}")
                    else:
                        st.text("No services detected")
                
                with st.expander("📍 Locations Mentioned"):
                    locations = entity_summary.get('top_locations', {})
                    if locations:
                        for loc, count in locations.items():
                            st.text(f"{loc}: {count}")
                    else:
                        st.text("No locations detected")
            
            # Common Query Patterns
            st.markdown("**📊 Common Query Patterns**")
            patterns = query_intel.get('common_patterns', {})
            if patterns:
                pattern_df = pd.DataFrame([
                    {'Pattern': k, 'Frequency': len(v), 'Example': v[0][:60] + '...' if len(v[0]) > 60 else v[0]}
                    for k, v in list(patterns.items())[:8]
                ])
                st.dataframe(pattern_df, use_container_width=True, hide_index=True)
            
            # Quality Analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**✅ High-Quality Query Examples**")
                high_quality = query_intel.get('sample_high_quality', [])
                if high_quality:
                    for q in high_quality[:3]:
                        with st.expander(f"Score: {q['quality_score']}/100 - {q['query'][:40]}..."):
                            st.text(f"Query: {q['query']}")
                            st.text(f"Services: {', '.join(q['services']) if q['services'] else 'None'}")
                            st.text(f"Demographics: {', '.join(q['demographics']) if q['demographics'] else 'None'}")
                            st.text(f"Locations: {', '.join(q['locations']) if q['locations'] else 'None'}")
                            st.text(f"Complexity: {q['complexity']}/10")
                else:
                    st.info("No high-quality queries found")
            
            with col2:
                st.markdown("**⚠️ Queries Needing Improvement**")
                needs_improvement = query_intel.get('sample_needs_improvement', [])
                if needs_improvement:
                    for q in needs_improvement[:3]:
                        with st.expander(f"Score: {q['quality_score']}/100 - {q['query'][:40]}..."):
                            st.text(f"Query: {q['query']}")
                            if q['suggestions']:
                                st.markdown("**Suggestions:**")
                                for sug in q['suggestions']:
                                    st.text(f"• {sug}")
                else:
                    st.success("All queries meet quality standards!")
            
            # Detailed Query Browser
            st.markdown("**🔍 Query Detail Browser**")
            all_analyzed = query_intel.get('all_analyzed', [])
            if all_analyzed:
                # Filter options
                col1, col2, col3 = st.columns(3)
                with col1:
                    intent_filter = st.selectbox(
                        "Filter by Intent",
                        ["All"] + list(set(q['intent'] for q in all_analyzed)),
                        key="intent_filter"
                    )
                with col2:
                    urgency_filter = st.selectbox(
                        "Filter by Urgency",
                        ["All", "critical", "high", "moderate", "normal"],
                        key="urgency_filter"
                    )
                with col3:
                    quality_filter = st.selectbox(
                        "Filter by Quality",
                        ["All", "High (70+)", "Medium (50-69)", "Low (<50)"],
                        key="quality_filter"
                    )
                
                # Apply filters
                filtered = all_analyzed
                if intent_filter != "All":
                    filtered = [q for q in filtered if q['intent'] == intent_filter]
                if urgency_filter != "All":
                    filtered = [q for q in filtered if q['urgency'] == urgency_filter]
                if quality_filter == "High (70+)":
                    filtered = [q for q in filtered if q['quality_score'] >= 70]
                elif quality_filter == "Medium (50-69)":
                    filtered = [q for q in filtered if 50 <= q['quality_score'] < 70]
                elif quality_filter == "Low (<50)":
                    filtered = [q for q in filtered if q['quality_score'] < 50]
                
                st.text(f"Showing {len(filtered)} queries")
                
                # Display filtered queries
                if filtered:
                    query_display = pd.DataFrame([
                        {
                            'Query': q['query'][:50] + '...' if len(q['query']) > 50 else q['query'],
                            'Intent': q['intent'].replace('_', ' ').title(),
                            'Urgency': q['urgency'].title(),
                            'Quality': q['quality_score'],
                            'Complexity': q['complexity'],
                            'Services': ', '.join(q['services'][:2]) if q['services'] else '-',
                            'Demographics': ', '.join(q['demographics'][:2]) if q['demographics'] else '-'
                        }
                        for q in filtered[:20]
                    ])
                    st.dataframe(query_display, use_container_width=True, hide_index=True)
        else:
            st.info("No query data available for AI analysis")
        
        # Output Quality Analysis
        st.divider()
        st.subheader("📊 Output Quality Analysis")
        st.markdown("*Analyzing the quality, relevance, and completeness of AI responses*")
        
        with st.spinner("Analyzing output quality..."):
            output_quality = analyzer.analyze_output_quality()
        
        if output_quality:
            # Top-level metrics
            quality_stats = output_quality.get('quality_stats', {})
            alignment_stats = output_quality.get('alignment_stats', {})
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Responses Analyzed", output_quality.get('total_analyzed', 0))
            with col2:
                avg_score = quality_stats.get('avg_score', 0)
                st.metric("Avg Quality Score", f"{avg_score:.0f}/100",
                         delta="Good" if avg_score >= 70 else ("Fair" if avg_score >= 50 else "Needs Work"))
            with col3:
                st.metric("Location Match Rate", f"{alignment_stats.get('location_match_rate', 0):.0f}%")
            with col4:
                st.metric("Category Match Rate", f"{alignment_stats.get('category_match_rate', 0):.0f}%")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Has Resources", f"{alignment_stats.get('has_resources_rate', 0):.0f}%")
            with col2:
                st.metric("Has Contact Info", f"{alignment_stats.get('actionable_rate', 0):.0f}%")
            with col3:
                st.metric("High Quality", quality_stats.get('high_quality_count', 0))
            with col4:
                st.metric("Low Quality", quality_stats.get('low_quality_count', 0))
            
            # Insights
            insights = output_quality.get('insights', [])
            if insights:
                st.markdown("### 💡 Output Quality Insights")
                for insight in insights:
                    if insight.startswith("⚠️") or insight.startswith("📍") or insight.startswith("🏷️") or insight.startswith("📋") or insight.startswith("🔄"):
                        st.warning(insight)
                    elif insight.startswith("✅"):
                        st.success(insight)
                    else:
                        st.info(insight)
            
            # Issue breakdown
            st.markdown("### 🚨 Issue Detection")
            issue_counts = output_quality.get('issue_counts', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("No Resources", issue_counts.get('no_resources', 0),
                         delta="issue" if issue_counts.get('no_resources', 0) > 5 else None,
                         delta_color="inverse")
                st.metric("Generic Responses", issue_counts.get('generic_response', 0),
                         delta="issue" if issue_counts.get('generic_response', 0) > 3 else None,
                         delta_color="inverse")
            with col2:
                st.metric("Location Mismatch", issue_counts.get('location_mismatch', 0),
                         delta="issue" if issue_counts.get('location_mismatch', 0) > 5 else None,
                         delta_color="inverse")
                st.metric("Category Mismatch", issue_counts.get('category_mismatch', 0),
                         delta="issue" if issue_counts.get('category_mismatch', 0) > 5 else None,
                         delta_color="inverse")
            with col3:
                st.metric("Low Quality", issue_counts.get('low_quality', 0),
                         delta="issue" if issue_counts.get('low_quality', 0) > 5 else None,
                         delta_color="inverse")
                st.metric("Too Short", issue_counts.get('too_short', 0),
                         delta="issue" if issue_counts.get('too_short', 0) > 3 else None,
                         delta_color="inverse")
            
            # Issue examples
            issues = output_quality.get('issues', {})
            if any(issues.values()):
                st.markdown("### 🔍 Issue Examples")
                
                issue_tabs = st.tabs(["Location Mismatch", "Category Mismatch", "No Resources", "Generic", "Low Quality"])
                
                with issue_tabs[0]:
                    loc_issues = issues.get('location_mismatch', [])
                    if loc_issues:
                        for item in loc_issues[:3]:
                            with st.expander(f"Score: {item['quality_score']} - {item['query'][:50]}..."):
                                st.text(f"User: {item['user']}")
                                st.text(f"Query: {item['query']}")
                                st.text(f"Factors: {', '.join(item.get('factors', []))}")
                    else:
                        st.success("No location mismatch issues!")
                
                with issue_tabs[1]:
                    cat_issues = issues.get('category_mismatch', [])
                    if cat_issues:
                        for item in cat_issues[:3]:
                            with st.expander(f"Score: {item['quality_score']} - {item['query'][:50]}..."):
                                st.text(f"User: {item['user']}")
                                st.text(f"Query: {item['query']}")
                                st.text(f"Factors: {', '.join(item.get('factors', []))}")
                    else:
                        st.success("No category mismatch issues!")
                
                with issue_tabs[2]:
                    no_res = issues.get('no_resources', [])
                    if no_res:
                        for item in no_res[:3]:
                            with st.expander(f"Score: {item['quality_score']} - {item['query'][:50]}..."):
                                st.text(f"User: {item['user']}")
                                st.text(f"Query: {item['query']}")
                    else:
                        st.success("All responses include resources!")
                
                with issue_tabs[3]:
                    generic = issues.get('generic_response', [])
                    if generic:
                        for item in generic[:3]:
                            with st.expander(f"Score: {item['quality_score']} - {item['query'][:50]}..."):
                                st.text(f"User: {item['user']}")
                                st.text(f"Query: {item['query']}")
                                st.text(f"Factors: {', '.join(item.get('factors', []))}")
                    else:
                        st.success("No generic/template responses detected!")
                
                with issue_tabs[4]:
                    low_qual = issues.get('low_quality', [])
                    if low_qual:
                        for item in low_qual[:3]:
                            with st.expander(f"Score: {item['quality_score']} - {item['query'][:50]}..."):
                                st.text(f"User: {item['user']}")
                                st.text(f"Query: {item['query']}")
                                st.text(f"Factors: {', '.join(item.get('factors', []))}")
                    else:
                        st.success("No low quality responses!")
            
            # Best vs Worst responses
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🏆 Best Quality Responses**")
                best = output_quality.get('best_responses', [])
                if best:
                    for item in best[:5]:
                        with st.expander(f"✅ Score: {item['quality_score']}/100"):
                            st.text(f"Query: {item['query']}")
                            st.text(f"User: {item['user']}")
                            st.text(f"Resources: {item['resource_count']}")
                            st.text(f"Has Contact Info: {'Yes' if item['has_actionable_info'] else 'No'}")
                            st.text(f"Positive Factors: {', '.join([f for f in item.get('factors', []) if not f.startswith('-')])}")
            
            with col2:
                st.markdown("**⚠️ Responses Needing Review**")
                worst = output_quality.get('all_analyses', [])
                if worst:
                    for item in worst[:5]:
                        with st.expander(f"⚠️ Score: {item['quality_score']}/100"):
                            st.text(f"Query: {item['query']}")
                            st.text(f"User: {item['user']}")
                            st.text(f"Issues: {', '.join(item.get('factors', []))}")
        else:
            st.info("No output data available for quality analysis")
        
        # Resource Recommendation Analysis
        st.divider()
        st.subheader("📚 Resource Recommendation Analysis")
        st.markdown("*Tracking which resources are recommended and their effectiveness*")
        
        with st.spinner("Analyzing resource recommendations..."):
            resource_analysis = analyzer.analyze_resource_recommendations()
        
        if resource_analysis:
            # Check for note (limited data)
            if resource_analysis.get('note'):
                st.warning(resource_analysis['note'])
            
            # Top-level metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Recommendations", resource_analysis.get('total_recommendations', 0))
            with col2:
                st.metric("Unique Resources", resource_analysis.get('unique_resources', 0))
            with col3:
                st.metric("Avg per Trace", f"{resource_analysis.get('recommendations_per_trace', 0):.1f}")
            with col4:
                concentration = resource_analysis.get('concentration_ratio', 0)
                st.metric("Top 5 Concentration", f"{concentration:.0f}%",
                         delta="High" if concentration > 50 else ("Balanced" if concentration > 30 else "Diverse"))
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Diversity Score", f"{resource_analysis.get('diversity_score', 0):.0f}/100")
            with col2:
                st.metric("Location Match Rate", f"{resource_analysis.get('location_match_rate', 0):.0f}%")
            with col3:
                st.metric("Traces with Resources", resource_analysis.get('traces_with_resources', 0))
            with col4:
                st.metric("Traces without Resources", resource_analysis.get('traces_without_resources', 0))
            
            # Insights
            insights = resource_analysis.get('insights', [])
            if insights:
                st.markdown("### 💡 Resource Insights")
                for insight in insights:
                    if insight.startswith("⚠️"):
                        st.warning(insight)
                    elif insight.startswith("✅"):
                        st.success(insight)
                    else:
                        st.info(insight)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Top recommended resources
                st.markdown("**🏆 Most Recommended Resources**")
                top_resources = resource_analysis.get('top_resources', [])
                if top_resources:
                    resource_df = pd.DataFrame([
                        {
                            'Resource': r['name'][:40] + '...' if len(r['name']) > 40 else r['name'],
                            'Count': r['count'],
                            '%': f"{r['percentage']:.1f}%"
                        }
                        for r in top_resources[:10]
                    ])
                    st.dataframe(resource_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No resource data available")
                
                # Category distribution
                st.markdown("**📂 Resource Categories**")
                cat_dist = resource_analysis.get('category_distribution', {})
                if cat_dist:
                    cat_df = pd.DataFrame([
                        {'Category': k, 'Count': v}
                        for k, v in list(cat_dist.items())[:10]
                    ])
                    st.dataframe(cat_df, use_container_width=True, hide_index=True)
            
            with col2:
                # Geographic distribution
                st.markdown("**📍 Geographic Distribution**")
                geo_dist = resource_analysis.get('geographic_distribution', {})
                
                if geo_dist:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Central Texas", geo_dist.get('central_texas_count', 0))
                    with col_b:
                        st.metric("Out of Region", geo_dist.get('out_of_region_count', 0))
                    
                    by_city = geo_dist.get('by_city', {})
                    if by_city:
                        st.markdown("**By City:**")
                        city_df = pd.DataFrame([
                            {'City': k, 'Count': v}
                            for k, v in list(by_city.items())[:8]
                        ])
                        st.dataframe(city_df, use_container_width=True, hide_index=True)
                    
                    by_zip = geo_dist.get('by_zip', {})
                    if by_zip:
                        with st.expander("📮 By Zip Code"):
                            zip_df = pd.DataFrame([
                                {'Zip': k, 'Count': v}
                                for k, v in list(by_zip.items())[:10]
                            ])
                            st.dataframe(zip_df, use_container_width=True, hide_index=True)
                    
                    out_of_region = geo_dist.get('out_of_region_zips', [])
                    if out_of_region:
                        st.warning(f"Out-of-region zips: {', '.join(out_of_region)}")
            
            # Resource detail browser
            st.markdown("**🔍 Resource Detail Browser**")
            sample_resources = resource_analysis.get('sample_resources', [])
            if sample_resources:
                res_display = pd.DataFrame([
                    {
                        'Name': r.get('name', 'Unknown')[:30],
                        'Category': r.get('category', '-'),
                        'City': r.get('city', '-'),
                        'Zip': r.get('zip_code', '-'),
                        'Phone': '✓' if r.get('phone') else '-',
                        'Website': '✓' if r.get('website') else '-'
                    }
                    for r in sample_resources[:15]
                ])
                st.dataframe(res_display, use_container_width=True, hide_index=True)
        else:
            st.info("No resource recommendation data available")


if __name__ == "__main__":
    main()

