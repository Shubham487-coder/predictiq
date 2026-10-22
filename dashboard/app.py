"""
PredictIQ — Streamlit Dashboard
Deep-dark engineering UI with glow effects, grid background, and custom design tokens.
Reads from PostgreSQL (latest pipeline run) and can trigger new DAG runs.
"""
from __future__ import annotations

import json
import random
import subprocess
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PredictIQ",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Design tokens ──────────────────────────────────────────────────────── */
:root {
  --bg:           #03050a;
  --panel:        #090e1a;
  --panel-strong: #0d1526;
  --ink:          #f0f8ff;
  --muted:        #7ea4c8;
  --line:         #1e3254;
  --soft-line:    #142240;
  --green:        #00f5a0;
  --blue:         #3d9fff;
  --red:          #ff3b4e;
  --amber:        #ffc020;
  --cyan:         #00e5ff;
  --shadow:       0 20px 60px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.06);
  --glow-green:   0 0 32px rgba(0,245,160,0.35);
  --glow-blue:    0 0 32px rgba(61,159,255,0.35);
  --glow-red:     0 0 32px rgba(255,59,78,0.38);
  --glow-cyan:    0 0 28px rgba(0,229,255,0.30);
}

/* ── Hide Streamlit chrome ──────────────────────────────────────────────── */
[data-testid="stHeader"] { display: none; }
[data-testid="stToolbar"] { display: none; }
#MainMenu { display: none; }
footer { display: none; }

/* ── Full-app background with grid ─────────────────────────────────────── */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background:
        radial-gradient(circle at 18% 10%, rgba(0,245,160,0.13), transparent 28%),
        radial-gradient(circle at 88% 4%,  rgba(61,159,255,0.18), transparent 30%),
        radial-gradient(circle at 50% 90%, rgba(0,229,255,0.07), transparent 35%),
        linear-gradient(rgba(255,255,255,0.028) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.028) 1px, transparent 1px),
        #03050a;
    background-size: auto, auto, auto, 34px 34px, 34px 34px, auto;
    color: var(--ink);
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10,17,30,0.99), rgba(3,5,10,0.99));
    border-right: 1px solid #1e3254;
    box-shadow: inset -1px 0 0 rgba(61,159,255,0.25);
}
[data-testid="stSidebar"] * { color: var(--ink) !important; }
[data-testid="stSidebar"] .stRadio label { color: #ccdcf0 !important; font-weight: 700; }

/* ── Metric / panel cards ───────────────────────────────────────────────── */
.metric-card {
    border: 1px solid var(--line);
    border-top: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    background: linear-gradient(160deg, rgba(13,21,40,0.97), rgba(5,9,18,0.98));
    box-shadow: var(--shadow);
    padding: 16px 18px;
    margin-bottom: 12px;
}
.metric-label {
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 800;
    margin-bottom: 10px;
}
.metric-value {
    font-size: 34px;
    font-weight: 900;
    line-height: 1;
}
.metric-critical { color: var(--red);   text-shadow: 0 0 28px rgba(255,59,78,0.55); }
.metric-warning  { color: var(--amber); text-shadow: 0 0 24px rgba(255,192,32,0.48); }
.metric-healthy  { color: var(--green); text-shadow: 0 0 28px rgba(0,245,160,0.50); }
.metric-neutral  { color: var(--cyan);  text-shadow: 0 0 24px rgba(0,229,255,0.48); }

/* ── Section titles ─────────────────────────────────────────────────────── */
.section-title {
    font-size: 11px;
    font-weight: 900;
    color: var(--cyan);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin: 22px 0 10px 0;
    border-bottom: 1px solid var(--soft-line);
    padding-bottom: 7px;
    text-shadow: var(--glow-cyan);
}

/* ── Status badges ──────────────────────────────────────────────────────── */
.badge {
    display: inline-flex;
    align-items: center;
    min-height: 22px;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.05em;
    white-space: nowrap;
}
.badge-critical {
    border: 1px solid rgba(255,59,78,0.70);
    background: rgba(255,59,78,0.18);
    color: var(--red);
    box-shadow: 0 0 12px rgba(255,59,78,0.25);
}
.badge-monitor {
    border: 1px solid rgba(255,192,32,0.60);
    background: rgba(255,192,32,0.14);
    color: var(--amber);
    box-shadow: 0 0 12px rgba(255,192,32,0.20);
}
.badge-healthy {
    border: 1px solid rgba(0,245,160,0.55);
    background: rgba(0,245,160,0.12);
    color: var(--green);
    box-shadow: 0 0 12px rgba(0,245,160,0.22);
}

/* ── Streamlit native overrides ─────────────────────────────────────────── */
/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00b37a, #0099cc);
    color: #ffffff;
    border: none;
    border-radius: 6px;
    font-weight: 900;
    padding: 8px 18px;
    box-shadow: var(--glow-green);
    letter-spacing: 0.04em;
    text-shadow: 0 1px 3px rgba(0,0,0,0.4);
    transition: transform 120ms ease, box-shadow 120ms ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 44px rgba(0,245,160,0.55);
}

/* Text input */
[data-testid="stTextInput"] input {
    background: #040810;
    border: 1px solid #1e3254;
    border-radius: 5px;
    color: var(--cyan);
    padding: 8px 10px;
    font-weight: 700;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--blue);
    box-shadow: 0 0 0 3px rgba(61,159,255,0.22);
}
[data-testid="stTextInput"] label { color: var(--cyan) !important; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em; text-shadow: var(--glow-cyan); }

/* Select / multiselect */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: #060b13;
    border-color: #263a5b;
    color: var(--ink);
}
[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label { color: var(--muted) !important; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; }

/* Radio */
[data-testid="stRadio"] > div { gap: 4px; }
[data-testid="stRadio"] label {
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 9px 11px;
    font-weight: 700;
    color: #b9c7db;
    transition: all 120ms;
}
[data-testid="stRadio"] label:hover {
    border-color: rgba(23,216,139,0.38);
    background: rgba(23,216,139,0.07);
    color: var(--green);
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--panel);
}

/* Spinner */
.stSpinner > div { border-top-color: var(--blue) !important; }

/* Divider */
hr { border-color: var(--soft-line); }

/* Success / error */
[data-testid="stAlert"] {
    background: rgba(23,216,139,0.08);
    border: 1px solid rgba(23,216,139,0.30);
    border-radius: 8px;
    color: var(--green);
}
</style>
""", unsafe_allow_html=True)

# ── Design tokens for Python ──────────────────────────────────────────────────
C_GREEN  = "#00f5a0"
C_BLUE   = "#3d9fff"
C_RED    = "#ff3b4e"
C_AMBER  = "#ffc020"
C_CYAN   = "#00e5ff"
C_MUTED  = "#7ea4c8"
C_INK    = "#f0f8ff"
C_LINE   = "#1e3254"
C_BG     = "#03050a"
C_PANEL  = "#090e1a"

STATUS_COLOR = {
    "CRITICAL": C_RED,
    "MONITOR":  C_AMBER,
    "HEALTHY":  C_GREEN,
}
STATUS_BADGE = {
    "CRITICAL": "badge-critical",
    "MONITOR":  "badge-monitor",
    "HEALTHY":  "badge-healthy",
}

PLOTLY_THEME = dict(
    paper_bgcolor=C_BG,
    plot_bgcolor=C_PANEL,
    font_color=C_INK,
    font_size=12,
)

# ── DB config ─────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host": "127.0.0.1", "port": 5432, "dbname": "predictiq_db",
    "user": "postgres", "password": "Goodluck@1234",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_latest_run() -> tuple[pd.DataFrame, pd.DataFrame, datetime | None]:
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        try:
            cur = conn.cursor()
            cur.execute("SELECT MAX(pipeline_run_at) FROM classifier_results")
            latest_run_at = cur.fetchone()[0]
            if not latest_run_at:
                return pd.DataFrame(), pd.DataFrame(), None

            clf = pd.read_sql("""
                SELECT machine_id, overall_status, priority,
                       comp1_prob, comp1_alert, comp1_status,
                       comp2_prob, comp2_alert, comp2_status,
                       comp3_prob, comp3_alert, comp3_status,
                       comp4_prob, comp4_alert, comp4_status,
                       pipeline_run_at
                FROM classifier_results
                WHERE pipeline_run_at = %s
                ORDER BY machine_id
            """, conn, params=(latest_run_at,))

            rul = pd.read_sql("""
                SELECT machine_id, component, predicted_rul_hours,
                       day_bucket, day_label, interpretation,
                       pipeline_run_at
                FROM rul_results
                WHERE pipeline_run_at = %s
                ORDER BY machine_id, component
            """, conn, params=(latest_run_at,))

            return clf, rul, latest_run_at
        finally:
            conn.close()
    except Exception as e:
        st.error(f"DB connection failed: {e}")
        return pd.DataFrame(), pd.DataFrame(), None


def trigger_dag(as_of: str) -> bool:
    exec_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    result = subprocess.run(
        ["airflow", "dags", "test", "predictiq_pipeline", exec_date,
         "--conf", json.dumps({"as_of": as_of})],
        capture_output=True, text=True, timeout=600,
    )
    return result.returncode == 0


def random_as_of() -> str:
    start = pd.Timestamp("2015-01-15")
    end   = pd.Timestamp("2015-11-15")
    delta = int((end - start).total_seconds() / 3600)
    ts    = start + pd.Timedelta(hours=random.randint(0, delta))
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def dark_table(df: pd.DataFrame, color_fns: dict | None = None) -> None:
    """Render a fully dark-themed HTML table matching the dashboard palette."""
    STATUS_BG  = {"CRITICAL":"rgba(255,59,78,0.14)","MONITOR":"rgba(255,192,32,0.12)","HEALTHY":"rgba(0,245,160,0.10)"}
    STATUS_CLR = {"CRITICAL":C_RED,"MONITOR":C_AMBER,"HEALTHY":C_GREEN}

    th_style = (
        f"padding:9px 14px;text-align:left;font-size:11px;font-weight:800;"
        f"text-transform:uppercase;letter-spacing:0.10em;color:{C_MUTED};"
        f"border-bottom:1px solid {C_LINE};background:{C_PANEL};white-space:nowrap;"
    )
    td_base = f"padding:9px 14px;font-size:13px;border-bottom:1px solid {C_LINE};"

    headers = "".join(f"<th style='{th_style}'>{c}</th>" for c in df.columns)
    rows_html = ""
    for _, row in df.iterrows():
        cells = ""
        for col in df.columns:
            val = row[col]
            extra = ""
            if color_fns and col in color_fns:
                extra = color_fns[col](val)
            cells += f"<td style='{td_base}{extra}'>{val}</td>"
        rows_html += f"<tr style='transition:background 120ms;'>{cells}</tr>"

    html = f"""
    <div style="border:1px solid {C_LINE};border-radius:8px;overflow:hidden;
                background:{C_PANEL};box-shadow:0 8px 32px rgba(0,0,0,0.4);">
      <table style="width:100%;border-collapse:collapse;font-family:inherit;">
        <thead><tr>{headers}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def plotly_bar(values, labels, colors, yaxis_title="", height=280):
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        marker_line_width=0,
    ))
    fig.update_layout(
        **PLOTLY_THEME,
        margin=dict(l=20, r=20, t=20, b=20),
        height=height,
        showlegend=False,
        yaxis_title=yaxis_title,
        xaxis=dict(gridcolor=C_LINE, tickfont_color=C_MUTED, linecolor=C_LINE),
        yaxis=dict(gridcolor=C_LINE, tickfont_color=C_MUTED, linecolor=C_LINE),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""
    <div style="padding:6px 0 18px 0; border-bottom:1px solid {C_LINE}; margin-bottom:16px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="width:36px; height:36px; border-radius:8px;
                        background:linear-gradient(135deg,{C_GREEN},{C_BLUE});
                        display:grid; place-items:center;
                        color:#031018; font-weight:900; font-size:16px;
                        box-shadow:0 0 24px rgba(23,216,139,0.24);">⚡</div>
            <div>
                <div style="font-size:16px; font-weight:850; color:{C_INK};">PredictIQ</div>
                <div style="font-size:11px; color:{C_MUTED};">Predictive Maintenance</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["Overview", "Machine Detail", "Alerts"],
        label_visibility="collapsed",
    )

    st.markdown(f'<div class="section-title">Run Pipeline</div>', unsafe_allow_html=True)

    if "as_of" not in st.session_state:
        st.session_state.as_of = random_as_of()

    as_of_input = st.text_input(
        "AS_OF DATETIME",
        value=st.session_state.as_of,
        help="YYYY-MM-DD HH:MM:SS — range 2015-01-15 to 2015-11-15",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎲 Random"):
            st.session_state.as_of = random_as_of()
            st.rerun()
    with col2:
        run_clicked = st.button("▶ Run DAG")

    if run_clicked:
        with st.spinner(f"Running pipeline…"):
            ok = trigger_dag(as_of_input)
        if ok:
            st.success("Pipeline complete!")
            load_latest_run.clear()
            st.rerun()
        else:
            st.error("DAG failed — check Airflow logs")

    st.markdown("---")
    st.markdown(f'<div style="font-size:11px; color:{C_MUTED}; margin-bottom:6px;">Auto-refresh every 30s</div>', unsafe_allow_html=True)
    if st.button("🔄 Refresh"):
        load_latest_run.clear()
        st.rerun()


# ── Load data ─────────────────────────────────────────────────────────────────

clf, rul, run_at = load_latest_run()

if clf.empty or rul.empty:
    st.markdown(f"""
    <div style="display:grid; min-height:60vh; place-items:center; align-content:center;
                gap:12px; text-align:center;">
        <div style="font-size:52px;">⚡</div>
        <div style="font-size:22px; font-weight:800; color:{C_INK};">No pipeline runs yet</div>
        <div style="font-size:14px; color:{C_MUTED};">
            Pick an as_of datetime in the sidebar and click ▶ Run DAG
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Derived ───────────────────────────────────────────────────────────────────

status_counts = clf["overall_status"].value_counts().to_dict()
n_critical = status_counts.get("CRITICAL", 0)
n_monitor  = status_counts.get("MONITOR",  0)
n_healthy  = status_counts.get("HEALTHY",  0)
n_total    = len(clf)

rul_min = rul.groupby("machine_id")["predicted_rul_hours"].min().reset_index()
rul_min.columns = ["machine_id", "min_rul_hours"]
clf_rich = clf.merge(rul_min, on="machine_id", how="left")

priority_order = {"CRITICAL": 0, "MONITOR": 1, "HEALTHY": 2}
clf_rich["_prio"] = clf_rich["overall_status"].map(priority_order)
clf_rich_sorted = clf_rich.sort_values(["_prio", "min_rul_hours"]).drop(columns="_prio")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

if page == "Overview":
    run_at_str = run_at.strftime("%Y-%m-%d %H:%M UTC") if run_at else "—"

    st.markdown(f"""
    <div style="display:flex; align-items:flex-start; justify-content:space-between;
                margin-bottom:4px;">
        <div>
            <h1 style="margin:0 0 4px; font-size:25px; color:#fff;
                       text-shadow:0 0 22px rgba(47,140,255,0.28);">Fleet Overview</h1>
            <p style="margin:0; color:{C_MUTED}; font-size:13px;">
                Real-time health across {n_total} machines
            </p>
        </div>
        <div style="border:1px solid {C_LINE}; border-radius:8px;
                    background:rgba(10,19,32,0.78); padding:8px 14px; text-align:right;">
            <div style="font-size:10px; color:{C_MUTED}; text-transform:uppercase;
                        letter-spacing:0.1em; margin-bottom:2px;">Last Run</div>
            <div style="font-size:13px; color:{C_CYAN}; font-weight:700;">{run_at_str}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("Total Machines", n_total,   "metric-neutral"),
        ("Critical",       n_critical,"metric-critical"),
        ("Monitor",        n_monitor, "metric-warning"),
        ("Healthy",        n_healthy, "metric-healthy"),
    ]
    for col, (label, val, cls) in zip([c1,c2,c3,c4], kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value {cls}">{val}</div>
            </div>""", unsafe_allow_html=True)

    # Charts row
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown('<div class="section-title">Fleet Status Split</div>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=["CRITICAL","MONITOR","HEALTHY"],
            values=[n_critical, n_monitor, n_healthy],
            marker_colors=[C_RED, C_AMBER, C_GREEN],
            hole=0.58,
            textinfo="percent",
            textfont=dict(size=13, color=C_INK),
        ))
        fig_pie.update_layout(
            **PLOTLY_THEME,
            margin=dict(l=10, r=10, t=10, b=10),
            height=260,
            legend=dict(font_color=C_MUTED, bgcolor="rgba(0,0,0,0)",
                        orientation="h", y=-0.08),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-title">Avg Failure Probability by Component</div>', unsafe_allow_html=True)
        comp_probs = {c: clf[f"{c}_prob"].mean() for c in ["comp1","comp2","comp3","comp4"]}
        bar_colors = [C_RED if v>0.5 else C_AMBER if v>0.3 else C_GREEN for v in comp_probs.values()]
        fig_bar = plotly_bar(
            [v*100 for v in comp_probs.values()],
            list(comp_probs.keys()),
            bar_colors,
            yaxis_title="Avg Failure Prob (%)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Top 10 table
    st.markdown('<div class="section-title">Top 10 Machines by Risk</div>', unsafe_allow_html=True)
    top10 = clf_rich_sorted.head(10)[[
        "machine_id","overall_status","priority","min_rul_hours",
        "comp1_prob","comp2_prob","comp3_prob","comp4_prob",
    ]].copy()
    top10["min_rul_hours"] = top10["min_rul_hours"].round(1)
    for c in ["comp1_prob","comp2_prob","comp3_prob","comp4_prob"]:
        top10[c] = (top10[c]*100).round(1)
    top10.columns = ["Machine","Status","Priority","Min RUL (h)",
                     "Comp1 %","Comp2 %","Comp3 %","Comp4 %"]

    STATUS_BG = {"CRITICAL":"rgba(255,59,78,0.14)","MONITOR":"rgba(255,192,32,0.12)","HEALTHY":"rgba(0,245,160,0.10)"}
    def _top10_fns(col):
        if col == "Status":
            return lambda v: (f"background:{STATUS_BG.get(v,'')};color:{STATUS_COLOR.get(v,C_INK)};"
                              f"font-weight:800;")
        if col == "Min RUL (h)":
            return lambda v: (f"color:{C_RED};font-weight:700;" if isinstance(v,(int,float)) and v<48
                              else f"color:{C_AMBER};font-weight:700;" if isinstance(v,(int,float)) and v<72
                              else f"color:{C_INK};")
        return None
    dark_table(top10, {c: _top10_fns(c) for c in top10.columns if _top10_fns(c)})

    # RUL histogram
    st.markdown('<div class="section-title">RUL Distribution — All Machines</div>', unsafe_allow_html=True)
    fig_hist = px.histogram(
        rul, x="predicted_rul_hours", nbins=40,
        color="interpretation",
        color_discrete_map={"HEALTHY": C_GREEN, "MONITOR": C_AMBER, "URGENT": C_RED},
        labels={"predicted_rul_hours":"Predicted RUL (hours)"},
    )
    fig_hist.update_layout(
        **PLOTLY_THEME, height=280,
        margin=dict(l=20,r=20,t=20,b=20), bargap=0.05,
        legend=dict(font_color=C_MUTED, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=C_LINE, tickfont_color=C_MUTED),
        yaxis=dict(gridcolor=C_LINE, tickfont_color=C_MUTED),
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MACHINE DETAIL
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Machine Detail":
    machine_ids = sorted(clf["machine_id"].unique().tolist())

    col_h, col_s = st.columns([3,1])
    with col_h:
        st.markdown(f'<h1 style="margin:0 0 4px; font-size:25px; color:#fff; text-shadow:0 0 22px rgba(47,140,255,0.28);">Machine Detail</h1>', unsafe_allow_html=True)
    with col_s:
        selected_id = st.selectbox("Machine", machine_ids, label_visibility="collapsed")

    machine_clf = clf[clf["machine_id"] == selected_id].iloc[0]
    machine_rul = rul[rul["machine_id"] == selected_id].sort_values("component")

    status  = machine_clf["overall_status"]
    color   = STATUS_COLOR.get(status, C_INK)
    glow    = f"0 0 28px {color}44"

    st.markdown(f"""
    <div class="metric-card" style="border-color:{color}55; box-shadow:{glow}; margin-bottom:20px;">
        <div style="display:flex; align-items:center; gap:16px;">
            <div style="font-size:34px; font-weight:900; color:{color};
                        text-shadow:0 0 22px {color}66;">M{selected_id:03d}</div>
            <div>
                <div style="font-size:11px; color:{C_MUTED}; text-transform:uppercase;
                            letter-spacing:0.1em;">Overall Status</div>
                <div style="font-size:20px; font-weight:800; color:{color};">{status}</div>
            </div>
            <div style="margin-left:auto; text-align:right;">
                <div style="font-size:11px; color:{C_MUTED}; text-transform:uppercase;
                            letter-spacing:0.1em;">Priority</div>
                <div style="font-size:20px; font-weight:800; color:{C_CYAN};">
                    {int(machine_clf['priority'])}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Component cards
    st.markdown('<div class="section-title">Component Health</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (_, crow) in enumerate(machine_rul.iterrows()):
        comp      = crow["component"]
        rul_h     = crow["predicted_rul_hours"]
        interp    = crow["interpretation"]
        day_label = crow["day_label"]
        prob      = machine_clf[f"{comp}_prob"]
        cstatus   = machine_clf[f"{comp}_status"]
        alert     = machine_clf[f"{comp}_alert"]
        c         = STATUS_COLOR.get(cstatus, C_INK)

        with cols[i]:
            st.markdown(f"""
            <div class="metric-card" style="border-color:{c}55; box-shadow:0 0 20px {c}22;">
                <div style="font-size:12px; font-weight:850; color:{c};
                            text-transform:uppercase; letter-spacing:0.08em; margin-bottom:10px;">
                    {comp} {"🔴" if alert else ""}
                </div>
                <div style="font-size:10px; color:{C_MUTED}; margin-bottom:2px;">Failure Prob</div>
                <div style="font-size:24px; font-weight:800; color:{c};
                            text-shadow:0 0 16px {c}55; margin-bottom:10px;">
                    {prob*100:.1f}%
                </div>
                <div style="font-size:10px; color:{C_MUTED}; margin-bottom:2px;">RUL</div>
                <div style="font-size:20px; font-weight:700; color:{C_INK}; margin-bottom:4px;">
                    {rul_h:.1f}h
                </div>
                <div style="font-size:10px; color:{C_MUTED};">{day_label} · {interp}</div>
            </div>
            """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">RUL by Component</div>', unsafe_allow_html=True)
        interps    = machine_rul["interpretation"].tolist()
        bar_colors = [C_RED if i=="URGENT" else C_AMBER if i=="MONITOR" else C_GREEN for i in interps]
        fig_rul = plotly_bar(
            machine_rul["predicted_rul_hours"].tolist(),
            machine_rul["component"].tolist(),
            bar_colors, yaxis_title="Hours",
        )
        fig_rul.add_hline(y=72, line_dash="dot", line_color=C_BLUE,
                          annotation_text="3-day", annotation_font_color=C_BLUE)
        st.plotly_chart(fig_rul, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">Failure Probability</div>', unsafe_allow_html=True)
        prob_vals  = [machine_clf[f"{c}_prob"]*100 for c in ["comp1","comp2","comp3","comp4"]]
        prob_color = [C_RED if v>50 else C_AMBER if v>30 else C_GREEN for v in prob_vals]
        fig_prob   = plotly_bar(prob_vals, ["comp1","comp2","comp3","comp4"],
                                prob_color, yaxis_title="Failure Prob (%)")
        fig_prob.add_hline(y=50, line_dash="dot", line_color=C_RED,
                           annotation_text="Alert", annotation_font_color=C_RED)
        fig_prob.add_hline(y=30, line_dash="dot", line_color=C_AMBER,
                           annotation_text="Monitor", annotation_font_color=C_AMBER)
        st.plotly_chart(fig_prob, use_container_width=True)

    # Fleet box comparison
    st.markdown('<div class="section-title">Fleet Position — RUL</div>', unsafe_allow_html=True)
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(
        y=rul["predicted_rul_hours"], name="Fleet",
        marker_color=C_BLUE, line_color=C_BLUE,
        fillcolor="rgba(47,140,255,0.08)",
    ))
    for _, r in machine_rul.iterrows():
        c = C_RED if r["interpretation"]=="URGENT" else C_AMBER if r["interpretation"]=="MONITOR" else C_GREEN
        fig_box.add_trace(go.Scatter(
            x=["Fleet"], y=[r["predicted_rul_hours"]],
            mode="markers", name=r["component"],
            marker=dict(size=13, color=c, symbol="diamond",
                        line=dict(width=2, color=C_BG)),
        ))
    fig_box.update_layout(
        **PLOTLY_THEME, height=300,
        margin=dict(l=20,r=20,t=20,b=20),
        yaxis=dict(title="RUL (hours)", gridcolor=C_LINE, tickfont_color=C_MUTED),
        xaxis=dict(gridcolor=C_LINE, tickfont_color=C_MUTED),
        legend=dict(font_color=C_MUTED, bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_box, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ALERTS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Alerts":
    st.markdown(f"""
    <h1 style="margin:0 0 4px; font-size:25px; color:#fff;
               text-shadow:0 0 22px rgba(47,140,255,0.28);">Alerts</h1>
    <p style="margin:0 0 20px; color:{C_MUTED}; font-size:13px;">
        Machines requiring immediate or near-term attention
    </p>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="metric-card" style="border-color:rgba(255,77,94,0.40);
                                        box-shadow:var(--glow-red);">
            <div class="metric-label">🔴 Critical</div>
            <div class="metric-value metric-critical">{n_critical}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card" style="border-color:rgba(255,176,32,0.35);">
            <div class="metric-label">🟡 Monitor</div>
            <div class="metric-value metric-warning">{n_monitor}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Filter</div>', unsafe_allow_html=True)
    cf1, cf2 = st.columns(2)
    with cf1:
        status_filter = st.multiselect("STATUS", ["CRITICAL","MONITOR","HEALTHY"],
                                       default=["CRITICAL","MONITOR"])
    with cf2:
        comp_filter = st.multiselect("COMPONENT", ["comp1","comp2","comp3","comp4"],
                                     default=["comp1","comp2","comp3","comp4"])

    # Build alerts
    alert_rows = []
    for _, row in clf_rich_sorted.iterrows():
        if row["overall_status"] not in status_filter:
            continue
        for comp in ["comp1","comp2","comp3","comp4"]:
            if comp not in comp_filter or not row[f"{comp}_alert"]:
                continue
            cr = rul[(rul["machine_id"]==row["machine_id"]) & (rul["component"]==comp)]
            rul_h  = cr["predicted_rul_hours"].values[0] if len(cr) else None
            interp = cr["interpretation"].values[0] if len(cr) else "—"
            alert_rows.append({
                "Machine":     f"M{int(row['machine_id']):03d}",
                "Status":      row["overall_status"],
                "Component":   comp.upper(),
                "Fail Prob %": round(row[f"{comp}_prob"]*100, 1),
                "Comp Status": row[f"{comp}_status"],
                "RUL (h)":    round(rul_h,1) if rul_h is not None else "—",
                "Outlook":    interp,
                "Priority":   int(row["priority"]),
            })

    if not alert_rows:
        st.markdown(f"""
        <div style="text-align:center; padding:40px; color:{C_GREEN};
                    font-size:16px; font-weight:700;">
            ✓ No alerts for the selected filters
        </div>
        """, unsafe_allow_html=True)
    else:
        alerts_df = pd.DataFrame(alert_rows).sort_values(["Priority","RUL (h)"])

        STATUS_BG2 = {"CRITICAL":"rgba(255,59,78,0.14)","MONITOR":"rgba(255,192,32,0.12)","HEALTHY":"rgba(0,245,160,0.10)"}
        def _alert_fns(col):
            if col in ("Status","Comp Status"):
                return lambda v: (f"background:{STATUS_BG2.get(v,'')};color:{STATUS_COLOR.get(v,C_INK)};font-weight:800;")
            if col == "RUL (h)":
                return lambda v: (f"color:{C_RED};font-weight:700;" if isinstance(v,(int,float)) and v<48
                                  else f"color:{C_AMBER};font-weight:700;" if isinstance(v,(int,float)) and v<72
                                  else f"color:{C_INK};")
            return None
        dark_table(alerts_df, {c: _alert_fns(c) for c in alerts_df.columns if _alert_fns(c)})
        st.markdown(f'<div style="font-size:11px; color:{C_MUTED}; margin-top:6px;">'
                    f'{len(alert_rows)} alert(s) · {len(set(r["Machine"] for r in alert_rows))} machine(s)</div>',
                    unsafe_allow_html=True)

    # Risk scatter
    st.markdown('<div class="section-title">Risk Scatter — Probability vs Min RUL</div>', unsafe_allow_html=True)
    sdf = clf_rich.copy()
    sdf["max_prob"] = sdf[["comp1_prob","comp2_prob","comp3_prob","comp4_prob"]].max(axis=1)*100

    fig_sc = px.scatter(
        sdf, x="min_rul_hours", y="max_prob",
        color="overall_status",
        color_discrete_map=STATUS_COLOR,
        hover_data={"machine_id":True,"priority":True},
        labels={"min_rul_hours":"Min RUL (h)","max_prob":"Max Failure Prob (%)","overall_status":"Status"},
    )
    fig_sc.update_traces(marker=dict(size=9, opacity=0.88, line=dict(width=0.5, color=C_BG)))
    fig_sc.update_layout(
        **PLOTLY_THEME, height=380,
        margin=dict(l=20,r=20,t=20,b=20),
        legend=dict(font_color=C_MUTED, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=C_LINE, tickfont_color=C_MUTED),
        yaxis=dict(gridcolor=C_LINE, tickfont_color=C_MUTED),
    )
    fig_sc.add_vline(x=72, line_dash="dot", line_color=C_MUTED, opacity=0.45)
    fig_sc.add_hline(y=50, line_dash="dot", line_color=C_MUTED, opacity=0.45)
    st.plotly_chart(fig_sc, use_container_width=True)