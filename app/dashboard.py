"""
App Store Agent — Unified dashboard with analysis and chatbot.
Run with: streamlit run app/dashboard.py
Built by Sudhakar.G
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import DASHBOARD_USERNAME, DASHBOARD_PASSWORD
from app.database import (
    initialize_database, store_reviews, get_metadata,
    get_all_period_analyses, get_themes_for_period, get_reviews_for_period,
    count_reviews_for_period, count_unanalyzed_reviews, get_review_date_range,
    list_analyzed_apps, delete_app_data, delete_analysis_only,
    get_last_scraped_date, get_analyzed_months,
    aggregate_themes_from_monthly, store_themes,
)
from app.scraper import scrape_google_play, scrape_apple_app_store
from app.processor import (
    run_analysis, get_month_ranges, get_quarter_ranges, get_year_ranges,
    process_period, process_period_stats_only,
)
from app.llm_client import call_llm

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="App Store Agent",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLING — applied ONCE at the top of every render
# ============================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

/* Hide footer only — keep header so sidebar toggle works */
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}

/* Claude-inspired warm palette */
:root {
    --bg-primary: #1a1915;
    --bg-secondary: #242320;
    --bg-elevated: #2d2b26;
    --border: rgba(255,235,205,0.08);
    --text-primary: #e8e0d5;
    --text-secondary: #9c9588;
    --text-muted: #6b6560;
    --accent: #d97757;
    --accent-hover: #e8895f;
    --accent-subtle: rgba(217,119,87,0.12);
}

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--bg-elevated);
    border: 1px solid var(--border); border-radius: 14px;
    padding: 18px 22px; box-shadow: 0 2px 12px rgba(0,0,0,0.2);
}
[data-testid="stMetric"] label {
    color: var(--text-secondary) !important; font-weight: 500; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.06em;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text-primary) !important; font-weight: 700; font-size: 1.5rem;
}

/* Buttons */
.stButton > button {
    border-radius: 10px; font-weight: 600; transition: all 0.15s ease;
    border: 1px solid var(--border);
}
.stButton > button:hover {
    transform: translateY(-1px); box-shadow: 0 4px 16px rgba(217,119,87,0.12);
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: var(--accent) !important; color: #fff !important; border: none;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    background: var(--accent-hover) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #16140f 0%, #1f1d18 100%);
    border-right: 1px solid var(--border);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom-color: var(--border); }
.stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 8px 16px; font-weight: 500; }
.stTabs [aria-selected="true"] { border-bottom: 2px solid var(--accent) !important; }

/* Expanders */
.streamlit-expanderHeader { font-weight: 500; border-radius: 10px; }

/* Dividers */
hr { border-color: var(--border); }

/* Chat messages */
[data-testid="stChatMessage"] { border-radius: 12px; border: 1px solid var(--border); }

/* Selectbox */
.stSelectbox [data-baseweb="select"] { border-radius: 8px; }

/* Alert boxes */
.stSuccess, .stInfo, .stWarning { border-radius: 10px; }
</style>
"""

# Apply CSS globally — runs on EVERY rerender regardless of page state
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def apply_chart_style(fig):
    fig.update_layout(
        font=dict(family="Inter, sans-serif", color="#9c9588"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=14, color="#e8e0d5"),
        xaxis=dict(gridcolor="rgba(255,235,205,0.04)", linecolor="rgba(255,235,205,0.08)",
                   tickfont=dict(color="#9c9588")),
        yaxis=dict(gridcolor="rgba(255,235,205,0.04)", linecolor="rgba(255,235,205,0.08)",
                   tickfont=dict(color="#9c9588")),
        legend=dict(font=dict(color="#9c9588", size=10)),
        margin=dict(l=40, r=20, t=45, b=35),
    )
    return fig


# ============================================================
# SESSION STATE HELPERS
# ============================================================

# Persistent auth — survives browser F5 refresh (server-side cache)
@st.cache_resource
def _get_auth_store():
    return {"authenticated": False}

def _save_auth(state: bool):
    _get_auth_store()["authenticated"] = state

def _check_auth() -> bool:
    return _get_auth_store().get("authenticated", False)

def _clear_app_state():
    """Clear all app-specific session state when switching apps."""
    keys_to_clear = [
        "chat_history", "last_period_count",
        "pos_filter", "neg_filter",
        "pos_theme_sel", "neg_theme_sel",
        "pos_theme_selection", "neg_theme_selection",
        "confirm_delete", "confirm_clear_analysis",
    ]
    for k in keys_to_clear:
        st.session_state.pop(k, None)


# ============================================================
# LOGIN
# ============================================================
def render_login():
    # Compact login — everything in first viewport fold
    st.markdown("""
    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center;
                padding-top:4vh; text-align:center;">
        <div style="font-size:2rem; margin-bottom:0.2rem; color:#d97757;">◆</div>
        <h1 style="font-size:2rem; font-weight:700; margin:0; letter-spacing:-0.02em;
                    color:#e8e0d5;">
            App Store Agent</h1>
        <p style="color:#9c9588; font-size:0.9rem; margin:0.2rem 0 0.1rem; font-weight:300;">
            AI-powered review intelligence for any app</p>
        <p style="color:#6b6560; font-size:0.65rem; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:1rem;">
            Built by Sudhakar.G</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.3, 1, 1.3])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Username", label_visibility="collapsed")
            password = st.text_input("Password", type="password", placeholder="Password", label_visibility="collapsed")
            submitted = st.form_submit_button("Sign in", use_container_width=True, type="primary")
            if submitted:
                if username == DASHBOARD_USERNAME and password == DASHBOARD_PASSWORD:
                    st.session_state.authenticated = True
                    _save_auth(True)
                    st.rerun()
                else:
                    st.error("Invalid credentials.")


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar():
    # Brand
    st.sidebar.markdown("""
    <div style="text-align:center; padding:0.5rem 0 0.3rem;">
        <span style="color:#d97757; font-size:1.4rem;">◆</span>
        <span style="font-size:1.1rem; font-weight:700; color:#e8e0d5; margin-left:6px;">App Store Agent</span>
    </div>""", unsafe_allow_html=True)
    st.sidebar.markdown("---")

    existing_apps = list_analyzed_apps()
    app_name_input = ""
    store = "Google Play Store"

    if existing_apps:
        st.sidebar.markdown("**Your apps**")
        app_options = {}
        for a in existing_apps:
            display = a.get("app_name", "Unknown")
            # Handle duplicate names by appending store
            if display in app_options:
                display = f"{display} ({a.get('store', '')})"
            app_options[display] = a
        app_options["＋ Add new app"] = None
        selected_label = st.sidebar.selectbox("Select app", list(app_options.keys()),
                                               label_visibility="collapsed", key="sidebar_app_select")

        # Detect app switch → clear stale session state
        prev_app = st.session_state.get("_current_sidebar_app", None)
        if prev_app is not None and prev_app != selected_label:
            _clear_app_state()
        st.session_state["_current_sidebar_app"] = selected_label

        if selected_label == "＋ Add new app":
            mode = "new"
        else:
            mode = "existing"
            selected_app = app_options[selected_label]
    else:
        mode = "new"

    if mode == "new":
        st.sidebar.markdown("**Add new app**")
        store = st.sidebar.selectbox("Store", ["Google Play Store", "Apple App Store"])
        if store == "Google Play Store":
            app_id = st.sidebar.text_input("App ID", value="com.spotify.music",
                                           help="Package name from the Play Store URL")
        else:
            app_id = st.sidebar.text_input("App ID (numeric)", value="324684580")
            app_name_input = st.sidebar.text_input("App name", value="Spotify")

        # Add Now CTA — registers the app in the database so it appears in the list
        if app_id:
            if st.sidebar.button("➕ Add to dashboard", use_container_width=True, type="primary", key="btn_add_app"):
                store_code = "google_play" if store == "Google Play Store" else "apple_app_store"
                name_for_init = app_name_input if app_name_input else app_id
                initialize_database(app_id, name_for_init, store_code)
                st.sidebar.success(f"**{name_for_init}** added!")
                st.rerun()
    else:
        app_id = selected_app.get("app_id", "")
        store = "Google Play Store" if selected_app.get("store") == "google_play" else "Apple App Store"

    # App status summary
    st.sidebar.markdown("---")
    meta = get_metadata(app_id)
    if meta and meta.get("app_name"):
        st.sidebar.markdown(f"**{meta['app_name']}**")
        total_in_db = meta.get("total_reviews_stored", "0")
        last_analyzed = meta.get("last_analyzed_date", "Never")
        min_d, max_d = get_review_date_range(app_id)
        st.sidebar.caption(f"Reviews in DB: **{total_in_db}**")
        st.sidebar.caption(f"Last analyzed: **{last_analyzed}**")
        if min_d and max_d:
            st.sidebar.caption(f"Data: {min_d[:10]} → {max_d[:10]}")
    else:
        st.sidebar.caption("No data yet. Start by scraping reviews.")

    # Logout
    st.sidebar.markdown("---")
    if st.sidebar.button("Sign out", use_container_width=True, key="btn_logout"):
        _save_auth(False)
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    return app_id, store, app_name_input


# ============================================================
# CHARTS
# ============================================================
def chart_rating_distribution(analyses):
    if not analyses: return
    totals = {s: sum(a.get(f"rating_{s}", 0) for a in analyses) for s in range(1, 6)}
    total = sum(totals.values())
    fig = go.Figure(go.Bar(
        x=[f"{s}★" for s in range(1, 6)], y=[totals[s] for s in range(1, 6)],
        marker_color=["#c45c4a", "#d97757", "#c9a85c", "#8aad6e", "#5a9e6f"],
        text=[totals[s] for s in range(1, 6)], textposition="outside",
        textfont=dict(color="#9c9588", size=11),
    ))
    fig.update_layout(title=f"Rating distribution ({total:,} reviews)", height=370, yaxis_title="Count", xaxis_title="")
    apply_chart_style(fig)
    st.plotly_chart(fig, use_container_width=True)

def chart_rating_trend(analyses):
    if len(analyses) < 2: return
    df = pd.DataFrame(analyses).sort_values("period_start")
    fig = go.Figure(go.Scatter(
        x=df["period_label"], y=df["avg_rating"], mode="lines+markers+text",
        text=[f"{v:.1f}" for v in df["avg_rating"]], textposition="top center",
        textfont=dict(size=9, color="#9c9588"),
        line=dict(color="#d97757", width=3, shape="spline"),
        marker=dict(size=7, color="#d97757", line=dict(width=2, color="#2d2b26")),
    ))
    fig.update_layout(title="Average rating trend", height=370, yaxis_range=[1, 5], yaxis_title="Rating")
    apply_chart_style(fig)
    st.plotly_chart(fig, use_container_width=True)

def chart_star_breakdown(analyses):
    if len(analyses) < 2: return
    df = pd.DataFrame(analyses).sort_values("period_start")
    colors = {"1":"#c45c4a","2":"#d97757","3":"#c9a85c","4":"#8aad6e","5":"#5a9e6f"}
    fig = go.Figure()
    for s in range(1, 6):
        fig.add_trace(go.Scatter(x=df["period_label"], y=df[f"rating_{s}"], mode="lines+markers",
            name=f"{s}★", line=dict(color=colors[str(s)], width=2, shape="spline"), marker=dict(size=4)))
    fig.update_layout(title="Star rating breakdown", height=400, yaxis_title="Reviews",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    apply_chart_style(fig)
    st.plotly_chart(fig, use_container_width=True)

def chart_volume(analyses):
    if len(analyses) < 2: return
    df = pd.DataFrame(analyses).sort_values("period_start")
    fig = go.Figure(go.Bar(x=df["period_label"], y=df["total_reviews"], marker_color="#b8856c",
        text=df["total_reviews"], textposition="outside", textfont=dict(color="#9c9588", size=9)))
    fig.update_layout(title="Review volume", height=340, yaxis_title="Reviews")
    apply_chart_style(fig)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# THEME DEDUP + CHARTS
# ============================================================
def _merge_similar_themes(theme_data: dict, threshold: float = 0.6) -> dict:
    from difflib import SequenceMatcher

    normalized = {}
    for name, data in theme_data.items():
        key = name.lower().strip()
        if key not in normalized:
            normalized[key] = {"sentiment": data["sentiment"], "periods": {}, "total": 0, "display_name": name}
        for period, count in data["periods"].items():
            normalized[key]["periods"][period] = normalized[key]["periods"].get(period, 0) + count
        normalized[key]["total"] += data["total"]
        if data["total"] > (normalized[key]["total"] - data["total"]):
            normalized[key]["display_name"] = name

    keys = list(normalized.keys())
    merged = {}
    used = set()

    for i, k1 in enumerate(keys):
        if k1 in used:
            continue
        group = [k1]
        for j, k2 in enumerate(keys):
            if j <= i or k2 in used:
                continue
            if normalized[k1]["sentiment"] != normalized[k2]["sentiment"]:
                continue
            ratio = SequenceMatcher(None, k1, k2).ratio()
            contained = k1 in k2 or k2 in k1
            words1, words2 = set(k1.split()), set(k2.split())
            word_overlap = len(words1 & words2) / max(len(words1 | words2), 1)
            if ratio >= threshold or contained or word_overlap >= 0.5:
                group.append(k2)
                used.add(k2)

        canonical = min(group, key=lambda g: len(g))
        display = normalized[canonical]["display_name"]
        merged_entry = {"sentiment": normalized[k1]["sentiment"], "periods": {}, "total": 0}
        for g in group:
            for period, count in normalized[g]["periods"].items():
                merged_entry["periods"][period] = merged_entry["periods"].get(period, 0) + count
            merged_entry["total"] += normalized[g]["total"]
        merged[display.lower()] = merged_entry
        used.add(k1)

    return merged


def _render_theme_chart(themes_list, period_labels, sentiment, colors, key_prefix):
    """Render a single theme trend chart with filter. Reused for positive and negative."""
    if not themes_list:
        st.caption("None detected.")
        return

    all_names = [name for name, _ in themes_list]

    # The widget key — Streamlit stores the multiselect value under this key.
    # After first render, `default` is ignored. To change the selection,
    # we must write directly to this key in session_state.
    widget_key = f"{key_prefix}_filter"

    # Initialize on first render
    if widget_key not in st.session_state:
        st.session_state[widget_key] = all_names.copy()

    # Select All / Clear All — write directly to the widget key
    fc1, fc2, fc3 = st.columns([1, 1, 4])
    with fc1:
        if st.button("Select all", key=f"{key_prefix}_sel_all", use_container_width=True):
            st.session_state[widget_key] = all_names.copy()
            st.rerun()
    with fc2:
        if st.button("Clear all", key=f"{key_prefix}_clr_all", use_container_width=True):
            st.session_state[widget_key] = []
            st.rerun()

    selected = st.multiselect(
        f"Filter {sentiment} themes", all_names,
        key=widget_key, label_visibility="collapsed"
    )

    if selected:
        fig = go.Figure()
        idx = 0
        for name, data in themes_list:
            if name not in selected:
                continue
            fig.add_trace(go.Scatter(
                x=period_labels,
                y=[data["periods"].get(p, 0) for p in period_labels],
                mode="lines+markers", name=name,
                line=dict(width=2, color=colors[idx % len(colors)], shape="spline"),
                marker=dict(size=5)
            ))
            idx += 1
        fig.update_layout(
            height=420, title="", yaxis_title="Mentions",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        apply_chart_style(fig)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No themes selected. Click **Select all** or pick from the list.")


def chart_theme_trends(app_id, analyses, period_type):
    if not analyses: return
    theme_data = {}
    period_labels = sorted([a["period_label"] for a in analyses])
    for a in analyses:
        for t in get_themes_for_period(app_id, period_type, a["period_label"]):
            name = t["theme"]
            if name not in theme_data:
                theme_data[name] = {"sentiment": t["sentiment"], "periods": {}, "total": 0}
            theme_data[name]["periods"][a["period_label"]] = t.get("mention_count", 0)
            theme_data[name]["total"] += t.get("mention_count", 0)
    if not theme_data:
        st.caption("No themes found.")
        return

    theme_data = _merge_similar_themes(theme_data)

    positive = sorted([(k, v) for k, v in theme_data.items() if v["sentiment"] == "positive"],
                       key=lambda x: x[1]["total"], reverse=True)[:10]
    negative = sorted([(k, v) for k, v in theme_data.items() if v["sentiment"] == "negative"],
                       key=lambda x: x[1]["total"], reverse=True)[:10]

    pos_colors = ["#5a9e6f", "#8aad6e", "#6db58a", "#4e8e6a", "#7bb87a",
                  "#3d8b6e", "#9aba72", "#69a878", "#84c48a", "#5fb87a"]
    neg_colors = ["#c45c4a", "#d97757", "#b84a3a", "#a0453c", "#c46b4a",
                  "#8c3e32", "#d4826a", "#e89a7a", "#c97860", "#a65c4c"]

    st.markdown("##### Positive themes")
    _render_theme_chart(positive, period_labels, "positive", pos_colors, "pos")

    st.markdown("---")

    st.markdown("##### Negative themes")
    _render_theme_chart(negative, period_labels, "negative", neg_colors, "neg")


def render_period_themes(app_id, period_type, period_label):
    themes = get_themes_for_period(app_id, period_type, period_label)
    if not themes:
        st.caption("No themes for this period.")
        return
    pos = [t for t in themes if t["sentiment"] == "positive"][:5]
    neg = [t for t in themes if t["sentiment"] == "negative"][:5]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### ✅ Positive")
        for i, t in enumerate(pos, 1):
            with st.expander(f"{i}. {t['theme']} — {t['mention_count']} mentions"):
                try:
                    samples = json.loads(t.get("sample_reviews", "[]"))
                except (TypeError, json.JSONDecodeError):
                    samples = []
                for s in samples[:3]:
                    st.markdown(f"> *\"{s}\"*")
    with c2:
        st.markdown("##### ❌ Negative")
        for i, t in enumerate(neg, 1):
            with st.expander(f"{i}. {t['theme']} — {t['mention_count']} mentions"):
                try:
                    samples = json.loads(t.get("sample_reviews", "[]"))
                except (TypeError, json.JSONDecodeError):
                    samples = []
                for s in samples[:3]:
                    st.markdown(f"> *\"{s}\"*")


# ============================================================
# CHATBOT
# ============================================================
CHATBOT_SYSTEM = """You are an expert app review analyst. Answer using ONLY the data provided.
Rules: 1. Cite numbers, periods, themes. 2. Be concise and actionable. 3. Use customer quotes when possible.
4. If data doesn't cover the question, say so. 5. Use bullets and bold for structure."""

def render_chatbot(app_id):
    st.markdown("#### Ask anything about this app's reviews")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Suggested questions when chat is empty
    if not st.session_state.chat_history:
        st.caption("Try asking:")
        sq1, sq2, sq3 = st.columns(3)
        with sq1:
            if st.button("What are the top complaints?", key="sq1", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": "What are the top complaints?"})
                st.rerun()
        with sq2:
            if st.button("How did sentiment change?", key="sq2", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": "How did sentiment change over time?"})
                st.rerun()
        with sq3:
            if st.button("What do users love most?", key="sq3", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": "What do users love most about this app?"})
                st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    q = st.chat_input("Ask about the reviews...")
    if q:
        st.session_state.chat_history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                ctx = []
                meta = get_metadata(app_id)
                ctx.append(f"App: {meta.get('app_name', app_id)}, Total reviews: {meta.get('total_reviews_stored', '?')}")
                for m in get_all_period_analyses(app_id, "monthly")[-12:]:
                    themes = get_themes_for_period(app_id, "monthly", m["period_label"])
                    p = [t["theme"] for t in themes if t["sentiment"] == "positive"][:3]
                    n = [t["theme"] for t in themes if t["sentiment"] == "negative"][:3]
                    ctx.append(f"{m['period_label']}: {m['total_reviews']} reviews, avg {m['avg_rating']}, +[{','.join(p)}] -[{','.join(n)}]")
                for qq in get_all_period_analyses(app_id, "quarterly"):
                    themes = get_themes_for_period(app_id, "quarterly", qq["period_label"])
                    p = [f"{t['theme']}({t['mention_count']})" for t in themes if t["sentiment"] == "positive"][:5]
                    n = [f"{t['theme']}({t['mention_count']})" for t in themes if t["sentiment"] == "negative"][:5]
                    ctx.append(f"{qq['period_label']}: {qq['total_reviews']} reviews, avg {qq['avg_rating']}, +[{','.join(p)}] -[{','.join(n)}]")
                resp = call_llm(CHATBOT_SYSTEM, f"DATA:\n{chr(10).join(ctx)}\n\nQUESTION: {q}", temperature=0.2, expect_json=False)
                st.markdown(resp)
                st.session_state.chat_history.append({"role": "assistant", "content": resp})


# ============================================================
# MAIN DASHBOARD
# ============================================================
def render_dashboard(app_id, store, app_name_input):
    meta = get_metadata(app_id)
    app_name = meta.get("app_name", app_id) if meta else app_id

    # Header
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:0.2rem;">
        <span style="font-size:1.3rem; color:#d97757;">◆</span>
        <span style="font-size:1.3rem; font-weight:700; color:#e8e0d5;">{app_name}</span>
        <span style="color:#6b6560; font-size:0.8rem; margin-left:auto;">App Store Agent</span>
    </div>""", unsafe_allow_html=True)

    # Tabs
    tab_scrape, tab_analysis, tab_chat, tab_manage = st.tabs([
        "⬇ Scrape & analyze", "📊 Dashboard", "💬 Ask AI", "⚙ Manage"
    ])

    # ============================================================
    # TAB 1: SCRAPE & ANALYZE
    # ============================================================
    with tab_scrape:
        today = datetime.now()
        default_start = (today - relativedelta(years=1)).replace(day=1)

        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("Period start", value=default_start, key="sa_start")
            start_date = start_date.replace(day=1)
        with c2:
            end_date = st.date_input("Period end", value=today, key="sa_end")

        sd_str = start_date.strftime("%Y-%m-%d")
        ed_str = end_date.strftime("%Y-%m-%d")

        # Always use fresh DB count for the selected period — no stale cache
        period_count = count_reviews_for_period(app_id, sd_str, ed_str)
        last_scraped = get_last_scraped_date(app_id)
        already_analyzed_months = set(get_analyzed_months(app_id))
        has_data = period_count > 0

        st.markdown("---")

        if not has_data:
            # ---- NO DATA: Scrape first ----
            st.markdown("### Scrape reviews")
            st.caption("No reviews in the database for this period. Scrape to get started.")
            run_scrape = st.button("⬇ Scrape all available reviews", use_container_width=True,
                                   type="primary", key="btn_scrape")
            if run_scrape:
                _do_scrape(app_id, store, app_name_input, sd_str, ed_str)
                st.rerun()  # Auto-refresh to show analysis options

        else:
            # ---- HAS DATA: Analysis first ----
            months = get_month_ranges(sd_str, ed_str)
            new_months = [(l, s, e) for l, s, e in months if l not in already_analyzed_months]
            done_months = [(l, s, e) for l, s, e in months if l in already_analyzed_months]
            unanalyzed_count = sum(count_reviews_for_period(app_id, s, e) for l, s, e in new_months)
            analyzed_count = period_count - unanalyzed_count

            st.markdown("### AI analysis")
            st.caption(f"Selected period: **{sd_str}** to **{ed_str}** · "
                       f"{len(months)} month{'s' if len(months) != 1 else ''}")

            m1, m2, m3 = st.columns(3)
            m1.metric("Reviews in period", f"{period_count:,}")
            m2.metric("Already analyzed", f"{analyzed_count:,}",
                      help=f"{len(done_months)} months done")
            m3.metric("To be sent to LLM", f"{unanalyzed_count:,}",
                      help=f"{len(new_months)} new months")

            if unanalyzed_count > 0:
                use_limit = st.checkbox("Limit reviews sent to AI", value=False, key="use_limit",
                                        help="Uncheck = process all. Check = set a cap.")
                review_limit = unanalyzed_count
                if use_limit:
                    review_limit = st.slider("Max reviews to analyze", 100, unanalyzed_count,
                                             unanalyzed_count, 100, key="analyze_limit")

                ac1, ac2 = st.columns(2)
                with ac1:
                    run_analysis_btn = st.button(
                        f"⚡ Analyze {len(new_months)} new month{'s' if len(new_months) != 1 else ''}",
                        use_container_width=True, type="primary", key="btn_analyze",
                        help=f"Sends {review_limit:,} reviews across {len(new_months)} months to AI.")
                with ac2:
                    rerun_btn = st.button("🔄 Re-analyze from scratch", use_container_width=True, key="btn_rerun",
                                          help="Clears previous analysis, re-processes everything.")
            else:
                st.success(f"All **{len(months)}** months already analyzed. Go to **Dashboard** tab.")
                run_analysis_btn = False
                rerun_btn = st.button("🔄 Re-analyze from scratch", use_container_width=True, key="btn_rerun",
                                      help="Clears previous analysis, re-processes everything.")

            if (unanalyzed_count > 0 and run_analysis_btn) or rerun_btn:
                force = bool(rerun_btn)
                if force:
                    delete_analysis_only(app_id)
                progress = st.progress(0, text="Starting analysis...")
                def cb(cur, tot, msg):
                    progress.progress(int((cur / tot) * 100) if tot else 0, text=msg)
                try:
                    result = run_analysis(app_id, sd_str, ed_str,
                                          force_rerun=force, progress_callback=cb)
                    progress.progress(100, text="Complete!")
                    st.success(f"**{result['months_analyzed']}** months analyzed · "
                               f"**{result['months_skipped']}** skipped · "
                               f"**{result['quarters']}** quarters · **{result['years']}** years")
                    st.rerun()  # Refresh to show updated state
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

            # Scrape — secondary, in expander
            st.markdown("---")
            with st.expander(f"⬇ Scrape more reviews ({period_count:,} in DB for this period)"):
                st.caption("Fetch new reviews from the store. Duplicates auto-skipped.")
                if last_scraped:
                    st.caption(f"Latest review in DB: **{last_scraped[:10]}**")
                run_scrape = st.button("⬇ Scrape all available reviews", use_container_width=True, key="btn_scrape")
                if run_scrape:
                    _do_scrape(app_id, store, app_name_input, sd_str, ed_str)

    # ============================================================
    # TAB 2: DASHBOARD
    # ============================================================
    with tab_analysis:
        period_view = st.selectbox("View by", ["monthly", "quarterly", "yearly"], index=0, key="dash_period")
        analyses = get_all_period_analyses(app_id, period_type=period_view)

        if not analyses:
            st.info("No analysis data yet. Go to **Scrape & analyze** tab first.")
            return

        # Data freshness bar
        first_period = analyses[0]["period_label"] if analyses else "—"
        last_period = analyses[-1]["period_label"] if analyses else "—"
        last_analyzed_date = meta.get("last_analyzed_date", "—") if meta else "—"
        st.caption(f"Showing **{len(analyses)}** {period_view} periods · "
                   f"{first_period} → {last_period} · Last analyzed: {last_analyzed_date}")

        # Summary metrics
        total_reviews = sum(a.get("total_reviews", 0) for a in analyses)
        wavg = sum(a.get("avg_rating", 0) * a.get("total_reviews", 0) for a in analyses)
        avg = round(wavg / total_reviews, 2) if total_reviews else 0
        text_r = sum(a.get("reviews_with_text", 0) for a in analyses)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total reviews", f"{total_reviews:,}")
        m2.metric("Avg rating", f"{avg} ★")
        m3.metric("Periods", len(analyses))
        m4.metric("With text", f"{text_r:,}")

        st.markdown("---")

        # ---- Section 1: Ratings ----
        st.markdown("### Ratings")
        c1, c2 = st.columns(2)
        with c1:
            chart_rating_distribution(analyses)
        with c2:
            chart_rating_trend(analyses)

        # ---- Section 2: Volume & Breakdown ----
        st.markdown("### Volume & breakdown")
        c1, c2 = st.columns(2)
        with c1:
            chart_volume(analyses)
        with c2:
            chart_star_breakdown(analyses)

        st.markdown("---")

        # ---- Section 3: Period detail (BEFORE theme trends — most actionable) ----
        st.markdown("### Period detail")
        if analyses:
            labels = [a["period_label"] for a in analyses]
            sel = st.selectbox("Select period", labels, index=len(labels) - 1, key="pd_sel")
            render_period_themes(app_id, period_view, sel)

        st.markdown("---")

        # ---- Section 4: Theme trends ----
        st.markdown("### Theme trends")
        chart_theme_trends(app_id, analyses, period_view)

    # ============================================================
    # TAB 3: CHATBOT
    # ============================================================
    with tab_chat:
        render_chatbot(app_id)

    # ============================================================
    # TAB 4: MANAGE
    # ============================================================
    with tab_manage:
        st.markdown("### Manage app data")
        meta = get_metadata(app_id)
        if meta:
            total_in_db = meta.get("total_reviews_stored", "0")
            min_d, max_d = get_review_date_range(app_id)
            last_analyzed = meta.get("last_analyzed_date", "Never")
            baseline = "Yes" if meta.get("seagull_analysis_complete") == "true" else "No"

            c1, c2, c3 = st.columns(3)
            c1.metric("Reviews in database", f"{int(total_in_db):,}" if total_in_db else "0")
            c2.metric("Last analyzed", last_analyzed)
            c3.metric("Analysis complete", baseline)

            if min_d and max_d:
                st.caption(f"Review date range: **{min_d[:10]}** → **{max_d[:10]}** · "
                           f"This count includes all reviews stored across all periods.")

            st.markdown("---")
            st.markdown("**Danger zone**")

            dc1, dc2 = st.columns(2)
            with dc1:
                if st.button("🗑 Delete all data for this app", use_container_width=True):
                    st.session_state["confirm_delete"] = app_id

            with dc2:
                if st.button("🔄 Clear analysis only (keep reviews)", use_container_width=True):
                    st.session_state["confirm_clear_analysis"] = app_id

            # Delete confirmation
            if st.session_state.get("confirm_delete") == app_id:
                st.warning(f"This will permanently delete **{total_in_db} reviews** and all analysis "
                           f"for **{meta.get('app_name', app_id)}**.")
                cc1, cc2, cc3 = st.columns([1, 1, 2])
                with cc1:
                    if st.button("Yes, delete everything", type="primary", key="confirm_del_yes"):
                        delete_app_data(app_id)
                        st.session_state.pop("confirm_delete", None)
                        _clear_app_state()
                        st.rerun()
                with cc2:
                    if st.button("Cancel", key="confirm_del_no"):
                        st.session_state.pop("confirm_delete", None)
                        st.rerun()

            # Clear analysis confirmation
            if st.session_state.get("confirm_clear_analysis") == app_id:
                st.warning(f"This will clear all analysis results but keep the **{total_in_db} reviews** intact. "
                           f"You can re-run analysis after.")
                cc1, cc2, cc3 = st.columns([1, 1, 2])
                with cc1:
                    if st.button("Yes, clear analysis", type="primary", key="confirm_clear_yes"):
                        delete_analysis_only(app_id)
                        st.session_state.pop("confirm_clear_analysis", None)
                        st.success("Analysis cleared. Reviews preserved.")
                        st.rerun()
                with cc2:
                    if st.button("Cancel", key="confirm_clear_no"):
                        st.session_state.pop("confirm_clear_analysis", None)
                        st.rerun()
        else:
            st.info("No data for this app yet. Go to **Scrape & analyze** to get started.")


def _do_scrape(app_id, store, app_name_input, sd_str, ed_str):
    """Shared scraping logic. Scrapes only reviews within the selected period."""
    since = datetime.strptime(sd_str, "%Y-%m-%d")
    until = datetime.strptime(ed_str, "%Y-%m-%d")

    # Estimate a reasonable count based on period length
    # ~500 reviews/month is a generous estimate for most apps
    months_in_period = max(1, ((until.year - since.year) * 12 + until.month - since.month))
    estimated_count = min(months_in_period * 2000, 40000)

    progress = st.progress(0, text=f"Scraping reviews from {sd_str} to {ed_str}...")
    try:
        if store == "Google Play Store":
            app_info, fetched_reviews = scrape_google_play(
                app_id, count=estimated_count, since_date=since, until_date=until
            )
        else:
            app_info, raw_reviews = scrape_apple_app_store(
                app_id, app_name=app_name_input or "Unknown", count=10000
            )
            # Apple RSS doesn't support date filters — filter after fetch
            fetched_reviews = [r for r in raw_reviews
                               if since <= (r.date.replace(tzinfo=None) if hasattr(r.date, 'tzinfo') and r.date.tzinfo else r.date) <= until]

        progress.progress(50, text=f"Found {len(fetched_reviews)} reviews in period. Storing...")
        initialize_database(app_info.app_id, app_info.app_name, app_info.store)
        stored = store_reviews(app_info.app_id, fetched_reviews)
        progress.progress(100, text="Done!")
        new_period_count = count_reviews_for_period(app_id, sd_str, ed_str)
        st.success(f"**{len(fetched_reviews):,}** reviews found in period · **{stored:,}** new · "
                   f"**{len(fetched_reviews) - stored:,}** duplicates skipped")
        st.info(f"**{new_period_count:,}** total reviews now in your selected period.")
    except Exception as e:
        import traceback
        st.error(f"Scraping failed: {e}")
        st.code(traceback.format_exc())  # Show full error for debugging


# ============================================================
# MAIN
# ============================================================
def main():
    # Restore auth from server-side cache (survives F5 refresh)
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = _check_auth()

    if not st.session_state.authenticated:
        render_login()
    else:
        app_id, store, app_name_input = render_sidebar()
        render_dashboard(app_id, store, app_name_input)

if __name__ == "__main__":
    main()
