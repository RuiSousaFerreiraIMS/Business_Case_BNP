import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Client Intelligence · BNP Paribas",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state init — MUST be before any widget ───────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Portfolio Overview"
if "selected_client" not in st.session_state:
    st.session_state.selected_client = None
if "sort_col" not in st.session_state:
    st.session_state.sort_col = "REVENUE_AT_RISK"
if "sort_asc" not in st.session_state:
    st.session_state.sort_asc = False
if "page_num" not in st.session_state:
    st.session_state.page_num = 0
if "hide_unassigned" not in st.session_state:
    st.session_state.hide_unassigned = True

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Source Sans 3', 'Segoe UI', sans-serif !important;
    background-color: #FFFFFF !important;
    color: #1A2B22 !important;
}
.stApp { background-color: #FFFFFF !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"]  { display: none !important; }
[data-testid="stToolbar"]  { display: none !important; }

.block-container {
    padding-top: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    padding-bottom: 3rem !important;
    max-width: 90% !important;
}

/* ── Top bar (brand) ── */
.topbar {
    width: 100%;
    background: #FFFFFF;
    border-bottom: 3px solid #00915A;
    padding: 12px 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-sizing: border-box;
}
.topbar-left  { display: flex; align-items: center; gap: 12px; }
.topbar-icon  { width: 32px; height: 32px; background: #00915A; border-radius: 6px;
                display: flex; align-items: center; justify-content: center; }
.topbar-icon-inner { width: 13px; height: 13px; background: #fff; border-radius: 2px; }
.topbar-title { font-size: 15px; font-weight: 700; color: #1A2B22; letter-spacing: -.02em; }
.topbar-sep   { width: 1px; height: 16px; background: #D1D9E0; margin: 0 4px; }
.topbar-sub   { font-size: 12px; color: #718096; }
.topbar-tag   { font-size: 11px; color: #718096; background: #F5F7FA;
                border: 1px solid #E2E8F0; padding: 3px 10px; border-radius: 20px; }

/* ── All Streamlit buttons — white base style ── */
.stButton > button {
    background: #FFFFFF !important;
    color: #2D3748 !important;
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    height: 38px !important;
    padding: 0 16px !important;
    font-family: 'Source Sans 3', sans-serif !important;
    box-shadow: none !important;
    transition: border-color .15s, color .15s, background .15s !important;
    white-space: nowrap !important;
    width: 100% !important;
}
.stButton > button:hover {
    border-color: #00915A !important;
    color: #00915A !important;
    background: #F0FBF7 !important;
}
.stButton > button p {
    color: inherit !important;
    font-size: 13px !important;
    font-weight: inherit !important;
}

/* ── Navbar background strip (targets the row that contains nav buttons) ── */
[data-testid="stHorizontalBlock"]:has([data-testid^="stButton-nav_"]) {
    background: #F8FAFB !important;
    border-bottom: 2px solid #E2E8F0 !important;
    padding: 0 !important;
    margin: 0 !important;
    gap: 0 !important;
}

/* ── Nav tab buttons (override base) ── */
[data-testid^="stButton-nav_"] > button {
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    border-radius: 0 !important;
    color: #718096 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 22px !important;
    height: 46px !important;
    box-shadow: none !important;
    width: 100% !important;
    font-family: 'Source Sans 3', sans-serif !important;
    transition: color .15s, border-color .15s !important;
    white-space: nowrap !important;
}
[data-testid^="stButton-nav_"] > button:hover {
    color: #00915A !important;
    background: transparent !important;
    border-color: transparent !important;
    border-bottom-color: #D1FAE5 !important;
}
[data-testid^="stButton-nav_"] > button p {
    color: inherit !important;
    font-weight: inherit !important;
    font-size: 13px !important;
}

/* Active nav tab */
[data-testid="stButton-nav_active"] > button {
    color: #00915A !important;
    border-bottom: 3px solid #00915A !important;
    font-weight: 700 !important;
    background: transparent !important;
}
[data-testid="stButton-nav_active"] > button p {
    color: #00915A !important;
    font-weight: 700 !important;
}

/* ── Page body ── */
.page-body { padding: 1.75rem 2.5rem; }
.page-title { font-size: 21px; font-weight: 700; color: #1A2B22; letter-spacing: -.02em; margin-bottom: 3px; }
.page-desc  { font-size: 13px; color: #718096; margin-bottom: 1.5rem; }

/* ── Metric cards ── */
.metrics-grid { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 14px; margin-bottom: 1.75rem; }
.mc { background: #fff; border: 1px solid #E2E8F0; border-radius: 10px;
      padding: 1.1rem 1.35rem; border-left: 4px solid #E2E8F0; }
.mc.green { border-left-color: #00915A; }
.mc.red   { border-left-color: #C0392B; }
.mc.blue  { border-left-color: #0066CC; }
.mc.amber { border-left-color: #B7791F; }
.mc-label { font-size: 11px; font-weight: 600; color: #718096; letter-spacing:.07em;
            text-transform: uppercase; margin-bottom: 7px; }
.mc-value { font-size: 27px; font-weight: 700; color: #1A2B22; line-height: 1; }
.mc-value.green { color: #00915A; }
.mc-value.red   { color: #C0392B; }
.mc-value.blue  { color: #0066CC; }
.mc-value.amber { color: #B7791F; }
.mc-sub { font-size: 12px; color: #718096; margin-top: 4px; }

/* ── Section label ── */
.slabel { font-size: 11px; font-weight: 700; color: #718096; letter-spacing:.08em;
          text-transform: uppercase; margin-bottom: 10px; padding-bottom: 7px;
          border-bottom: 1px solid #E2E8F0; }

/* ── Divider ── */
.divider { height: 1px; background: #E2E8F0; margin: 1.25rem 0; }

/* ── Sortable table ── */
.ctable { width: 100%; border-collapse: collapse; font-size: 13px; }
.ctable th {
    font-size: 11px; font-weight: 700; color: #718096; letter-spacing:.06em;
    text-transform: uppercase; padding: 0 10px 8px 0; border-bottom: 2px solid #E2E8F0;
    text-align: left; white-space: nowrap; cursor: pointer; user-select: none;
}
.ctable th:hover { color: #00915A; }
.ctable th.sort-asc::after  { content: " ↑"; color: #00915A; }
.ctable th.sort-desc::after { content: " ↓"; color: #00915A; }
.ctable td { padding: 9px 10px 9px 0; border-bottom: 1px solid #E2E8F0;
             color: #1A2B22; vertical-align: middle; }
.ctable tr:last-child td { border-bottom: none; }
.ctable tr:hover td { background: #F5F7FA; }

/* ── Gauge ── */
.gauge-wrap { background: #F5F7FA; border-radius: 10px; padding: 1.2rem; }
.gauge-title { font-size: 11px; font-weight: 700; color: #718096; letter-spacing:.08em;
               text-transform: uppercase; margin-bottom: 8px; }
.gauge-value { font-size: 30px; font-weight: 700; line-height: 1; margin-bottom: 5px; }
.gauge-bar-track { height: 7px; background: #E2E8F0; border-radius: 4px;
                   overflow: hidden; margin-bottom: 5px; }
.gauge-bar-fill  { height: 100%; border-radius: 4px; }
.gauge-label { font-size: 12px; color: #718096; }

/* ── Action card ── */
.action-card { border-radius: 10px; padding: 1.15rem 1.4rem; }
.action-card.urgent   { background: #FEE2E2; border-left: 4px solid #C0392B; }
.action-card.offer    { background: #E8F5EE; border-left: 4px solid #00915A; }
.action-card.upsell   { background: #DBEAFE; border-left: 4px solid #0066CC; }
.action-card.monitor  { background: #FEF9C3; border-left: 4px solid #B7791F; }
.action-card.standard { background: #F5F7FA; border-left: 4px solid #718096; }
.action-title { font-size: 14px; font-weight: 700; margin-bottom: 4px; }
.action-desc  { font-size: 13px; line-height: 1.55; }

/* ── Todo ── */
.todo { background: #F5F7FA; border: 1.5px dashed #D1D9E0; border-radius: 10px;
        padding: 2.5rem 2rem; text-align: center; }
.todo-t { font-size: 15px; font-weight: 600; color: #2D3748; margin-bottom: 6px; }
.todo-s { font-size: 13px; color: #718096; line-height: 1.6; }

/* ── View button — green accent ── */
[data-testid^="stButton-view_"] > button {
    background: #00915A !important;
    color: #FFFFFF !important;
    border-color: #00915A !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    height: 32px !important;
}
[data-testid^="stButton-view_"] > button:hover {
    background: #007A4C !important;
    border-color: #007A4C !important;
    color: #FFFFFF !important;
}
[data-testid^="stButton-view_"] > button p { color: #FFFFFF !important; }

/* ── Back button ── */
[data-testid="stButton-back_btn"] > button {
    background: #F5F7FA !important;
    color: #718096 !important;
    border-color: #E2E8F0 !important;
    width: auto !important;
    font-size: 12px !important;
}

/* ── Selectbox styling ── */
[data-testid="stSelectbox"] > div > div {
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    min-height: 42px !important;
}

/* ── Search input ── */
[data-testid="stTextInput"] input {
    border-radius: 8px !important;
    border: 1.5px solid #E2E8F0 !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
    height: 42px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #00915A !important;
    box-shadow: 0 0 0 3px rgba(0,145,90,.1) !important;
}

/* ── Checkbox label ── */
[data-testid="stCheckbox"] label {
    font-size: 13px !important;
    color: #2D3748 !important;
    font-weight: 500 !important;
}
[data-testid="stCheckbox"] label:hover {
    color: #00915A !important;
}

/* ── Table row columns — remove vertical gap ── */
[data-testid="stHorizontalBlock"]:has([data-testid^="stButton-view_"]) {
    gap: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}
[data-testid="stHorizontalBlock"]:has([data-testid^="stButton-view_"]) > div {
    padding: 0 !important;
    margin: 0 !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def safe_get(series, key, default="—"):
    return series[key] if key in series.index else default

def action_style(action):
    return {
        "Urgent restructuring": ("urgent",   "#C0392B"),
        "Urgent retention":     ("urgent",   "#C0392B"),
        "Debt monitoring":      ("monitor",  "#92400E"),
        "Refinancing offer":    ("offer",    "#00693E"),
        "Upsell opportunity":   ("upsell",   "#0066CC"),
        "Competitive proposal": ("upsell",   "#0066CC"),
        "Consolidation offer":  ("offer",    "#00693E"),
        "Renewal campaign":     ("monitor",  "#92400E"),
        "Cross-sell":           ("upsell",   "#0066CC"),
        "Monitor":              ("monitor",  "#92400E"),
        "Standard follow-up":   ("standard", "#718096"),
    }.get(action, ("standard", "#718096"))

def action_description(action, p_san=None, p_churn=None, cluster=None):
    san_str   = f"{p_san:.0%}"   if p_san   is not None else "—"
    churn_str = f"{p_churn:.0%}" if p_churn is not None else "—"
    return {
        "Urgent restructuring":  f"Delinquency risk is critical (P(Churn)={churn_str}). Contact immediately to renegotiate debt terms before the client defaults or exits.",
        "Urgent retention":      f"High-value client at serious risk — P(SAN)={san_str} and P(Churn)={churn_str}. Offer a retention incentive or restructured product before they act.",
        "Debt monitoring":       f"Financial stress signals detected (P(Churn)={churn_str}). Schedule a proactive outreach call — intervene before delinquency escalates.",
        "Refinancing offer":     f"Client likely to settle early (P(SAN)={san_str}) but low churn risk — they want to stay. Contact with a refinancing offer before the decision is made.",
        "Upsell opportunity":    f"Stable high-value client — P(SAN)={san_str}, P(Churn)={churn_str}. Good moment to propose a new product or credit line expansion.",
        "Competitive proposal":  f"Significant external credit exposure with P(Churn)={churn_str}. Present a counter-proposal before the client migrates to a competitor.",
        "Consolidation offer":   f"Client spreads credit across multiple providers (P(SAN)={san_str}). Offer debt consolidation to increase share of wallet at Cetelem.",
        "Renewal campaign":      f"Contract nearing end — P(Churn)={churn_str}. No financial pressure detected. Target with a renewal offer before natural exit.",
        "Cross-sell":            f"Stable and low-risk client — P(SAN)={san_str}, P(Churn)={churn_str}. Good moment to introduce a complementary product.",
        "Monitor":               f"No immediate action required. P(Churn)={churn_str} — monitor for changes in financial behaviour.",
        "Standard follow-up":    f"No immediate action required. Include in standard periodic review cycle.",
    }.get(action, "—")

CLUSTER_PROFILES = {
    1: ("High Risk / Overdue",    "#E24B4A", "Significant overdue amounts and recent delinquency signals (MONTVENC, RISK_EVER). Churn 36.7% — mixed early and natural exits."),
    2: ("High Value",             "#378ADD", "Multiple high-value contracts, highest income and LTI=24. Early settlement rate 46.6% — by far the highest of all segments."),
    3: ("High External Credit",   "#1D9E75", "High external consumer credit and total debt. Moderate churn 38.4%, spreading credit across competing providers."),
    4: ("Base / Dormant",         "#BA7517", "Low engagement, passive exits. Natural churn dominates (21.1%). Complete contracts but do not renew — largest segment (50%)."),
}

# Intervention windows (start%, end% of contract lifecycle) + action per segment
CLUSTER_INTERVENTION = {
    1: (25, 75,  "Recovery restructuring",  "Debt renegotiation and restructuring before delinquency escalates further."),
    2: (25, 50,  "Refinancing offer",       "New contract or refinancing proposal before the early settlement decision is made. LTI=24 confirms financial capacity."),
    3: (50, 75,  "Competitive proposal",    "Address external credit alternatives. Client manages multiple providers — offer a compelling counter-proposal."),
    4: (75, 100, "Renewal campaign",        "Cross-sell and renewal offer in the final quarter, before natural contract end."),
}

# Table column config for Client Search
TABLE_COLS = {
    "CONTRIB":          "Client ID",
    "P_SAN":            "P(Early Settlement)",
    "P_CHURN":          "P(Churn)",
    "CLUSTER":          "Cluster",
    "ACTION":           "Action",
    "REVENUE_AT_RISK":  "Revenue at risk",
}


# ══════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():

    def assign_action(p_san, p_churn, cluster):
        if cluster == 1:
            return "Urgent restructuring" if p_churn >= 0.60 else "Debt monitoring"
        elif cluster == 2:
            if p_san >= 0.65 and p_churn >= 0.55:
                return "Urgent retention"
            elif p_san >= 0.65:
                return "Refinancing offer"
            else:
                return "Upsell opportunity"
        elif cluster == 3:
            if p_churn >= 0.55:
                return "Competitive proposal"
            elif p_san >= 0.55:
                return "Consolidation offer"
            else:
                return "Monitor"
        elif cluster == 4:
            if p_churn >= 0.60:
                return "Renewal campaign"
            elif p_san < 0.35 and p_churn < 0.35:
                return "Cross-sell"
            else:
                return "Standard follow-up"
        else:
            return "Standard follow-up"

    try:
        df = pd.read_parquet("data/prepared/active_clients_scored.parquet")
        df = df.rename(columns={
            "Prob_SAN":      "P_SAN",
            "Prob_Churn":    "P_CHURN",
            "segment_final": "CLUSTER",
        })

        df["CLUSTER"] = df["CLUSTER"].fillna(0).astype(int)
        df["P_SAN"] = df["P_SAN"].fillna(0)
        df["P_CHURN"] = df["P_CHURN"].fillna(0)

        # Action — derived from cluster + scores
        df["ACTION"] = df.apply(
            lambda r: assign_action(r["P_SAN"], r["P_CHURN"], r["CLUSTER"]), axis=1
        )

        # Revenue at risk — interest lost if client settles early
        INTEREST_RATE = 0.08
        df["REVENUE_AT_RISK"] = (
            df["TOTAL_MTFINO"]
            * INTEREST_RATE
            * (df["MEDIAN_DURDEG"] / 12)
            * df["P_SAN"]
        ).round(2)

        return df

    except FileNotFoundError:
        # ── Synthetic data ──────────────────────────────────────────────
        rng = np.random.default_rng(42)
        n = 150
        cluster = rng.choice([1, 2, 3, 4], n, p=[0.20, 0.16, 0.14, 0.50])
        cluster_san_base   = {1: 0.255, 2: 0.466, 3: 0.269, 4: 0.228}
        cluster_churn_base = {1: 0.367, 2: 0.519, 3: 0.384, 4: 0.435}
        p_san   = np.clip([cluster_san_base[c]   + rng.normal(0, 0.10) for c in cluster], 0.02, 0.98)
        p_churn = np.clip([cluster_churn_base[c] + rng.normal(0, 0.10) for c in cluster], 0.02, 0.98)
        total_mtfino   = np.round(rng.uniform(5000, 30000, n), 2)
        median_durdeg  = np.round(rng.uniform(12, 60, n), 1)
        actions = [assign_action(s, ch, c) for s, ch, c in zip(p_san, p_churn, cluster)]
        revenue = np.round(total_mtfino * 0.08 * (median_durdeg / 12) * p_san, 2)
        return pd.DataFrame({
            "CONTRIB":        [f"C-{10000+i}" for i in range(n)],
            "sdem_age":       rng.integers(25, 70, n),
            "PRODALP":        rng.choice(["AUTO", "PESSOAL", "HABITACAO"], n),
            "TOTAL_MTFINO":   total_mtfino,
            "MEDIAN_DURDEG":  median_durdeg,
            "P_SAN":          np.round(p_san, 3),
            "P_CHURN":        np.round(p_churn, 3),
            "CLUSTER":        cluster,
            "ACTION":         actions,
            "REVENUE_AT_RISK": revenue,
        })

df_all = load_data()


# ══════════════════════════════════════════════════════════════════════════
# NAVIGATION — real top navbar with buttons
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="topbar">
  <div class="topbar-left">
    <div class="topbar-icon"><div class="topbar-icon-inner"></div></div>
    <span class="topbar-title">Client Intelligence</span>
    <div class="topbar-sep"></div>
    <span class="topbar-sub">BNP Paribas · Risk Analytics</span>
  </div>
  <span class="topbar-tag">Internal · v1.0 · March 2026</span>
</div>
""", unsafe_allow_html=True)

# Nav tabs rendered as buttons — centered with spacer columns on each side
PAGES = ["Portfolio Overview", "Client Search", "Clustering", "Model Metrics"]
nav_cols = st.columns([3, 2, 2, 2, 2, 3])   # spacers on left/right to center the 4 tabs
for i, pg in enumerate(PAGES):
    # Give active tab a distinct key so CSS [data-testid="stButton-nav_active"] fires
    key = "nav_active" if st.session_state.page == pg else f"nav_{i}"
    with nav_cols[i + 1]:
        if st.button(pg, key=key, use_container_width=True):
            st.session_state.page = pg
            st.session_state.selected_client = None
            st.rerun()

st.markdown("<div class='divider' style='margin:0'></div>", unsafe_allow_html=True)

page = st.session_state.page


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — PORTFOLIO OVERVIEW
# ══════════════════════════════════════════════════════════════════════════

if page == "Portfolio Overview":
    st.markdown("<div class='page-body'>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Portfolio Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-desc'>Active client portfolio — early settlement risk, churn probability, and revenue impact.</div>", unsafe_allow_html=True)

    n_total      = len(df_all)
    n_san_risk   = (df_all["P_SAN"] >= 0.6).sum()
    n_churn_risk = (df_all["P_CHURN"] >= 0.6).sum()
    rev_at_risk  = df_all.loc[df_all["P_SAN"] >= 0.6, "REVENUE_AT_RISK"].sum()

    st.markdown(f"""
    <div class="metrics-grid">
        <div class="mc">
            <div class="mc-label">Active clients</div>
            <div class="mc-value">{n_total:,}</div>
            <div class="mc-sub">in current portfolio</div>
        </div>
        <div class="mc green">
            <div class="mc-label">Early settlement risk</div>
            <div class="mc-value green">{n_san_risk:,}</div>
            <div class="mc-sub">P(SAN) ≥ 60%</div>
        </div>
        <div class="mc red">
            <div class="mc-label">Churn risk</div>
            <div class="mc-value red">{n_churn_risk:,}</div>
            <div class="mc-sub">P(Churn) ≥ 60%</div>
        </div>
        <div class="mc amber">
            <div class="mc-label">Revenue at risk</div>
            <div class="mc-value amber">€{rev_at_risk:,.0f}</div>
            <div class="mc-sub">from high SAN clients</div>
        </div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.markdown("<div class='slabel'>Action priority breakdown</div>", unsafe_allow_html=True)
        action_counts = df_all["ACTION"].value_counts().reset_index()
        action_counts.columns = ["Action", "Clients"]
        priority_order = [
            "Urgent restructuring", "Urgent retention",
            "Debt monitoring", "Refinancing offer",
            "Upsell opportunity", "Competitive proposal",
            "Consolidation offer", "Renewal campaign",
            "Cross-sell", "Monitor", "Standard follow-up",
        ]
        action_counts["Action"] = pd.Categorical(action_counts["Action"], categories=priority_order, ordered=True)
        action_counts = action_counts.sort_values("Action")
        colors = {
            "Urgent restructuring": "#C0392B",
            "Urgent retention": "#C0392B",
            "Debt monitoring": "#92400E",
            "Refinancing offer": "#00693E",
            "Upsell opportunity": "#0066CC",
            "Competitive proposal": "#0066CC",
            "Consolidation offer": "#00693E",
            "Renewal campaign": "#92400E",
            "Cross-sell": "#0066CC",
            "Monitor": "#B7791F",
            "Standard follow-up": "#718096",
        }
        for _, row in action_counts.iterrows():
            pct = row["Clients"] / n_total * 100
            c   = colors.get(row["Action"], "#718096")
            st.markdown(f"""
            <div style='margin-bottom:11px'>
              <div style='display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px'>
                <span style='font-weight:600;color:{c}'>{row["Action"]}</span>
                <span style='color:#718096'>{row["Clients"]} clients · {pct:.1f}%</span>
              </div>
              <div style='height:7px;background:#E2E8F0;border-radius:4px;overflow:hidden'>
                <div style='height:100%;width:{pct}%;background:{c};border-radius:4px'></div>
              </div>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='slabel'>Top 10 clients by revenue at risk</div>", unsafe_allow_html=True)
        top10 = df_all.nlargest(10, "REVENUE_AT_RISK")[["CONTRIB", "P_SAN", "P_CHURN", "ACTION", "REVENUE_AT_RISK"]]
        rows_html = ""
        for _, r in top10.iterrows():
            sc = "#00915A" if r["P_SAN"] >= 0.6 else "#718096"
            cc = "#C0392B" if r["P_CHURN"] >= 0.6 else "#718096"
            rows_html += f"""<tr>
                <td style='font-weight:600'>{r['CONTRIB']}</td>
                <td style='color:{sc};font-weight:600'>{r['P_SAN']:.0%}</td>
                <td style='color:{cc};font-weight:600'>{r['P_CHURN']:.0%}</td>
                <td style='font-weight:600'>€{r['REVENUE_AT_RISK']:,.0f}</td>
            </tr>"""
        st.markdown(f"""
        <table class='ctable'>
          <thead><tr><th>Client</th><th>P(SAN)</th><th>P(Churn)</th><th>Revenue at risk</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — CLIENT SEARCH
# ══════════════════════════════════════════════════════════════════════════

elif page == "Client Search":
    st.markdown("<div class='page-body'>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Client Search</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-desc'>Filter and explore your active client portfolio · click <b>View</b> to open the full risk profile.</div>", unsafe_allow_html=True)

    # ── If a client is selected, show profile ──────────────────────────
    selected_id = st.session_state.selected_client

    if selected_id and selected_id in df_all["CONTRIB"].values:
        client = df_all[df_all["CONTRIB"] == selected_id].iloc[0]

        if st.button("← Back to list", key="back_btn"):
            st.session_state.selected_client = None
            st.rerun()

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        age_str  = f"Age {int(client['sdem_age'])}" if "sdem_age" in client.index else ""
        prod_str = safe_get(client, "PRODALP", safe_get(client, "TYPEPROD", ""))
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:16px;margin-bottom:1.5rem'>
            <div style='width:50px;height:50px;background:#E8F5EE;border-radius:50%;
                        display:flex;align-items:center;justify-content:center;
                        font-size:17px;font-weight:700;color:#00693E'>
                {str(client['CONTRIB'])[-2:]}
            </div>
            <div>
                <div style='font-size:20px;font-weight:700;color:#1A2B22'>{client['CONTRIB']}</div>
                <div style='font-size:13px;color:#718096'>{age_str} · {prod_str}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        p_san   = float(client["P_SAN"])
        p_churn = float(client["P_CHURN"])
        cl_id   = int(client["CLUSTER"])
        revenue = float(client["REVENUE_AT_RISK"])
        action  = str(client["ACTION"])

        san_color   = "#00915A" if p_san   >= 0.6 else "#B7791F" if p_san   >= 0.4 else "#718096"
        churn_color = "#C0392B" if p_churn >= 0.6 else "#B7791F" if p_churn >= 0.4 else "#718096"
        cl_name, cl_color, cl_desc = CLUSTER_PROFILES.get(cl_id, ("Unknown", "#718096", ""))

        g1, g2, g3, g4 = st.columns(4, gap="medium")
        with g1:
            st.markdown(f"""<div class='gauge-wrap'>
                <div class='gauge-title'>P(Early Settlement)</div>
                <div class='gauge-value' style='color:{san_color}'>{p_san:.0%}</div>
                <div class='gauge-bar-track'>
                  <div class='gauge-bar-fill' style='width:{p_san*100:.0f}%;background:{san_color}'></div>
                </div>
                <div class='gauge-label'>{'High risk' if p_san>=0.6 else 'Moderate' if p_san>=0.4 else 'Low risk'}</div>
            </div>""", unsafe_allow_html=True)
        with g2:
            st.markdown(f"""<div class='gauge-wrap'>
                <div class='gauge-title'>P(Churn)</div>
                <div class='gauge-value' style='color:{churn_color}'>{p_churn:.0%}</div>
                <div class='gauge-bar-track'>
                  <div class='gauge-bar-fill' style='width:{p_churn*100:.0f}%;background:{churn_color}'></div>
                </div>
                <div class='gauge-label'>{'High risk' if p_churn>=0.6 else 'Moderate' if p_churn>=0.4 else 'Low risk'}</div>
            </div>""", unsafe_allow_html=True)
        with g3:
            # Intervention window for this cluster
            istart, iend, iaction, _ = CLUSTER_INTERVENTION.get(cl_id, (0, 100, "—", ""))
            st.markdown(f"""<div class='gauge-wrap'>
                <div class='gauge-title'>Segment</div>
                <div style='font-size:16px;font-weight:700;color:{cl_color};margin-bottom:6px'>#{cl_id} · {cl_name}</div>
                <div style='font-size:12px;color:#718096;line-height:1.5;margin-bottom:8px'>{cl_desc}</div>
                <div style='font-size:10px;font-weight:700;color:#718096;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px'>Intervention window</div>
                <div style='height:5px;background:#E2E8F0;border-radius:3px;overflow:hidden;position:relative'>
                    <div style='position:absolute;left:{istart}%;width:{iend-istart}%;height:100%;background:{cl_color};opacity:.8;border-radius:2px'></div>
                </div>
                <div style='font-size:11px;color:{cl_color};font-weight:600;margin-top:3px'>{istart}–{iend}% of contract · {iaction}</div>
            </div>""", unsafe_allow_html=True)
        with g4:
            st.markdown(f"""<div class='gauge-wrap'>
                <div class='gauge-title'>Revenue at risk</div>
                <div style='font-size:26px;font-weight:700;color:#B7791F;margin-bottom:5px'>
                    €{revenue:,.0f}
                </div>
                <div class='gauge-label'>Lost interest if early settlement</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Recommended action — split into WHAT + HOW ────────────────
        st.markdown("<div class='slabel'>Recommended action</div>", unsafe_allow_html=True)
        act_sty, act_col = action_style(action)

        ACTION_STEPS = {
            "Urgent restructuring":  ["Call the client within 48h", "Propose a debt renegotiation or instalment restructuring plan", "Escalate to senior relationship manager if no response"],
            "Urgent retention":      ["Prioritise for immediate outreach — within 48h", "Prepare a personalised refinancing or retention offer", "Offer a new contract before the early settlement is triggered"],
            "Debt monitoring":       ["Schedule a proactive check-in call this week", "Review recent payment history for early warning signals", "Flag for follow-up if no improvement within 30 days"],
            "Refinancing offer":     ["Contact the client before mid-contract point", "Prepare a new loan offer aligned with their financial profile", "Highlight benefits of staying vs early settlement cost"],
            "Upsell opportunity":    ["Include in next outreach cycle with a product expansion offer", "Review eligibility for credit line increase or complementary product", "Low urgency — plan for next quarterly review"],
            "Competitive proposal":  ["Contact within 2 weeks with a targeted counter-offer", "Benchmark against competitor rates the client is likely using", "Focus on consolidation benefits and simplifying their credit portfolio"],
            "Consolidation offer":   ["Identify the external credit lines the client holds", "Prepare a consolidation proposal to bring external debt to Cetelem", "Emphasise lower total monthly payments and single provider convenience"],
            "Renewal campaign":      ["Include in end-of-contract renewal campaign", "Contact 60–90 days before contract maturity", "Offer an incentive for early renewal commitment"],
            "Cross-sell":            ["Include in next product campaign cycle", "Review profile for complementary product eligibility", "Low urgency — stable client, no immediate risk"],
            "Monitor":               ["No immediate contact needed", "Include in standard monitoring dashboard", "Reassess at next quarterly review"],
            "Standard follow-up":    ["Include in standard periodic review cycle", "No immediate action required"],
        }

        steps = ACTION_STEPS.get(action, ["Review client profile and determine appropriate next step"])
        steps_html = "".join([f"<div style='display:flex;gap:10px;margin-bottom:6px;align-items:flex-start'><span style='color:{act_col};font-weight:700;font-size:14px;line-height:1.2'>→</span><span style='font-size:13px;color:#2D3748;line-height:1.5'>{s}</span></div>" for s in steps])

        st.markdown(f"""<div class='action-card {act_sty}'>
            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px'>
                <div class='action-title' style='color:{act_col};margin-bottom:0'>{action}</div>
                <span style='font-size:11px;background:{act_col}18;color:{act_col};padding:2px 10px;border-radius:20px;font-weight:700;white-space:nowrap'>Segment {cl_id} · {cl_name}</span>
            </div>
            <div class='action-desc' style='margin-bottom:12px'>{action_description(action, p_san, p_churn, cl_id)}</div>
            <div style='border-top:1px solid {act_col}22;padding-top:10px'>
                <div style='font-size:10px;font-weight:700;color:#718096;text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px'>Next steps</div>
                {steps_html}
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Client details — enriched ─────────────────────────────────
        st.markdown("<div class='slabel'>Client details</div>", unsafe_allow_html=True)

        detail_map = {
            "PRODALP":            ("Product",              lambda v: str(v)),
            "sdem_age":           ("Age",                  lambda v: f"{int(v)} yrs"),
            "N_CONTRACTS":        ("Contracts",            lambda v: str(int(v))),
            "TOTAL_MTFINO":       ("Total loan amount",    lambda v: f"€{v:,.0f}"),
            "TOTAL_MENSALIDADE":  ("Monthly instalment",   lambda v: f"€{v:,.0f}"),
            "MEDIAN_DURDEG":      ("Contract duration",    lambda v: f"{int(v)} mo."),
            "MEDIAN_RESSO":       ("Est. monthly income",  lambda v: f"€{v:,.0f}"),
            "NBENF":              ("Dependents",           lambda v: str(int(v))),
            "ALLBD_N_Dossiers__N":("Dossiers (history)",   lambda v: str(int(v))),
            "COUNT_TOTAL_MEDIAN": ("External credit lines",lambda v: str(int(v))),
        }

        available_details = [(label, fmt(client[col])) for col, (label, fmt) in detail_map.items() if col in client.index and pd.notna(client[col])]

        if available_details:
            cols_per_row = 5
            for row_start in range(0, len(available_details), cols_per_row):
                row_items = available_details[row_start:row_start + cols_per_row]
                dcols = st.columns(len(row_items))
                for di, (label, val) in enumerate(row_items):
                    with dcols[di]:
                        st.markdown(f"""<div class='mc'>
                            <div class='mc-label'>{label}</div>
                            <div style='font-size:16px;font-weight:700;color:#1A2B22'>{val}</div>
                        </div>""", unsafe_allow_html=True)

    else:
        # ── Filter bar ────────────────────────────────────────────────
        ACTION_OPTIONS = [
            "All actions", "Urgent restructuring", "Urgent retention",
            "Debt monitoring", "Refinancing offer", "Upsell opportunity",
            "Competitive proposal", "Consolidation offer",
            "Renewal campaign", "Cross-sell", "Standard follow-up", "Monitor",
        ]
        RISK_OPTIONS = [
            "All clients", "High SAN risk (≥ 60%)", "High churn risk (≥ 60%)", "Both high risk",
        ]
        SORT_MAP = {
            "Revenue at risk ↓":   ("REVENUE_AT_RISK", False),
            "P(Settlement) ↓":     ("P_SAN",           False),
            "P(Churn) ↓":          ("P_CHURN",         False),
            "Client ID ↑":         ("CONTRIB",         True),
        }

        fc1, fc2, fc3, fc4, fc5 = st.columns([2.5, 1.8, 1.8, 1.8, 1.5])
        with fc1:
            search_q = st.text_input(
                "search", placeholder="🔍  Search by Client ID…",
                label_visibility="collapsed", key="search_input",
            )
        with fc2:
            action_f = st.selectbox("Action", ACTION_OPTIONS,
                                    label_visibility="collapsed", key="action_f")
        with fc3:
            risk_f = st.selectbox("Risk", RISK_OPTIONS,
                                  label_visibility="collapsed", key="risk_f")
        with fc4:
            sort_f = st.selectbox("Sort by", list(SORT_MAP.keys()),
                                  label_visibility="collapsed", key="sort_f")
        with fc5:
            hide_unassigned = st.checkbox(
                "Hide unassigned clusters",
                value=st.session_state.hide_unassigned,
                key="hide_unassigned_cb",
            )
            st.session_state.hide_unassigned = hide_unassigned

        # ── Apply filters ─────────────────────────────────────────────
        results = df_all.copy()
        if hide_unassigned:
            results = results[results["CLUSTER"] != 0]
        if search_q:
            results = results[results["CONTRIB"].astype(str).str.contains(search_q, case=False, na=False)]
        if action_f != "All actions":
            results = results[results["ACTION"] == action_f]
        if risk_f == "High SAN risk (≥ 60%)":
            results = results[results["P_SAN"] >= 0.6]
        elif risk_f == "High churn risk (≥ 60%)":
            results = results[results["P_CHURN"] >= 0.6]
        elif risk_f == "Both high risk":
            results = results[(results["P_SAN"] >= 0.6) & (results["P_CHURN"] >= 0.6)]

        sort_col_name, sort_asc_val = SORT_MAP[sort_f]
        if sort_col_name in results.columns:
            results = results.sort_values(sort_col_name, ascending=sort_asc_val)

        # Reset to page 0 when filters change
        if "last_filter_state" not in st.session_state or \
           st.session_state.last_filter_state != (search_q, action_f, risk_f, sort_f, hide_unassigned):
            st.session_state.page_num = 0
            st.session_state.last_filter_state = (search_q, action_f, risk_f, sort_f, hide_unassigned)

        # ── Pagination config ──────────────────────────────────────────
        PAGE_SIZE   = 50
        total_rows  = len(results)
        total_pages = max(1, (total_rows + PAGE_SIZE - 1) // PAGE_SIZE)
        current_page = min(st.session_state.page_num, total_pages - 1)
        page_start   = current_page * PAGE_SIZE
        page_end     = min(page_start + PAGE_SIZE, total_rows)
        page_results = results.iloc[page_start:page_end]

        # ── Summary strip + pagination in one bar ─────────────────────
        n_res     = len(results)
        n_urgent  = int(results["ACTION"].str.startswith("Urgent").sum())
        rev_total = results["REVENUE_AT_RISK"].sum()

        sum_col, nav_col = st.columns([2, 1])
        with sum_col:
            st.markdown(f"""
            <div style='display:flex;gap:24px;align-items:center;padding:12px 0 14px;flex-wrap:wrap;
                        border-bottom:2px solid #E2E8F0'>
                <span style='font-size:13px;font-weight:700;color:#1A2B22'>{n_res:,} client(s)</span>
                <span style='width:1px;height:14px;background:#E2E8F0;display:inline-block'></span>
                <span style='font-size:12px;color:#C0392B;font-weight:600'>⚠ {n_urgent} urgent</span>
                <span style='width:1px;height:14px;background:#E2E8F0;display:inline-block'></span>
                <span style='font-size:12px;color:#718096'>Rev. at risk: <b style='color:#B7791F'>€{rev_total:,.0f}</b></span>
            </div>""", unsafe_allow_html=True)
        with nav_col:
            nc1, nc2, nc3 = st.columns([1, 2, 1])
            with nc1:
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
                if current_page > 0:
                    if st.button("←", key="prev_page", use_container_width=True):
                        st.session_state.page_num = current_page - 1
                        st.rerun()
            with nc2:
                st.markdown(f"<div style='text-align:center;padding-top:8px;font-size:12px;color:#718096'>"
                            f"Page <b style='color:#1A2B22'>{current_page+1}</b> of {total_pages} "
                            f"<span style='color:#A0AEC0'>· {page_start+1}–{page_end}</span></div>",
                            unsafe_allow_html=True)
            with nc3:
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
                if current_page < total_pages - 1:
                    if st.button("→", key="next_page", use_container_width=True):
                        st.session_state.page_num = current_page + 1
                        st.rerun()

        if results.empty:
            st.markdown("""<div class='todo' style='margin-top:1.5rem'>
                <div class='todo-t'>No clients match the current filters</div>
                <div class='todo-s'>Try adjusting the search or filter criteria above</div>
            </div>""", unsafe_allow_html=True)
        else:
            # ── Table header ──────────────────────────────────────────
            st.markdown("""
            <div style='display:grid;grid-template-columns:2fr 0.8fr 0.8fr 1.2fr 1.8fr 1.1fr 0.5fr;
                        padding:8px 4px;border-bottom:2px solid #E2E8F0;gap:0'>
                <div style='font-size:11px;font-weight:700;color:#718096;text-transform:uppercase;letter-spacing:.06em'>Client ID</div>
                <div style='font-size:11px;font-weight:700;color:#718096;text-transform:uppercase;letter-spacing:.06em'>P(SAN)</div>
                <div style='font-size:11px;font-weight:700;color:#718096;text-transform:uppercase;letter-spacing:.06em'>P(Churn)</div>
                <div style='font-size:11px;font-weight:700;color:#718096;text-transform:uppercase;letter-spacing:.06em'>Cluster</div>
                <div style='font-size:11px;font-weight:700;color:#718096;text-transform:uppercase;letter-spacing:.06em'>Recommended action</div>
                <div style='font-size:11px;font-weight:700;color:#718096;text-transform:uppercase;letter-spacing:.06em'>Rev. at risk</div>
                <div></div>
            </div>""", unsafe_allow_html=True)

            # ── Data rows ─────────────────────────────────────────────
            for idx, (_, r) in enumerate(page_results.iterrows()):
                s_col = "#C0392B" if r["P_SAN"] >= 0.6 else "#B7791F" if r["P_SAN"] >= 0.4 else "#718096"
                c_col = "#C0392B" if r["P_CHURN"] >= 0.6 else "#B7791F" if r["P_CHURN"] >= 0.4 else "#718096"
                rn, rc, _ = CLUSTER_PROFILES.get(int(r["CLUSTER"]), ("—", "#718096", ""))
                _, a_col  = action_style(r["ACTION"])
                row_bg    = "#FAFBFC" if idx % 2 == 0 else "#FFFFFF"
                contrib   = str(r['CONTRIB'])
                contrib_short = contrib[:22] + "…" if len(contrib) > 22 else contrib

                san_badge = f"<span style='background:{s_col}18;color:{s_col};padding:2px 9px;border-radius:20px;font-size:12px;font-weight:700'>{r['P_SAN']:.0%}</span>"
                chu_badge = f"<span style='background:{c_col}18;color:{c_col};padding:2px 9px;border-radius:20px;font-size:12px;font-weight:700'>{r['P_CHURN']:.0%}</span>"
                act_badge = f"<span style='background:{a_col}18;color:{a_col};padding:2px 9px;border-radius:20px;font-size:12px;font-weight:600'>{r['ACTION']}</span>"

                row_col, btn_col = st.columns([11.5, 0.5], gap="small")
                with row_col:
                    st.markdown(f"""
                    <div style='display:grid;grid-template-columns:2fr 0.8fr 0.8fr 1.2fr 1.8fr 1.1fr;
                                padding:9px 4px;background:{row_bg};border-bottom:1px solid #F0F2F5;
                                align-items:center;gap:0'>
                        <div style='font-weight:700;font-size:13px;color:#1A2B22' title='{contrib}'>{contrib_short}</div>
                        <div>{san_badge}</div>
                        <div>{chu_badge}</div>
                        <div style='font-size:12px;font-weight:600;color:{rc}'>#{int(r['CLUSTER'])} {rn}</div>
                        <div>{act_badge}</div>
                        <div style='font-weight:600;font-size:13px;color:#1A2B22'>€{r['REVENUE_AT_RISK']:,.0f}</div>
                    </div>""", unsafe_allow_html=True)
                with btn_col:
                    if st.button("→", key=f"view_{contrib}_{idx}", use_container_width=True):
                        st.session_state.selected_client = contrib
                        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════

elif page == "Clustering":
    st.markdown("<div class='page-body'>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Client Clustering</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-desc'>4 segments via multi-perspective K-Means (P1 Financial · P2 Risk · P3 History) + Ward hierarchical merging on centroids.</div>", unsafe_allow_html=True)

    # ── Segment summary cards ──────────────────────────────────────────
    cluster_counts = df_all["CLUSTER"].value_counts().sort_index()
    seg_cols = st.columns(4, gap="medium")
    for i, (cid, (cname, ccolor, cdesc)) in enumerate(CLUSTER_PROFILES.items()):
        n         = int(cluster_counts.get(cid, 0))
        pct       = n / max(len(df_all), 1) * 100
        avg_san   = df_all[df_all["CLUSTER"] == cid]["P_SAN"].mean()
        avg_churn = df_all[df_all["CLUSTER"] == cid]["P_CHURN"].mean()
        with seg_cols[i]:
            st.markdown(f"""<div class='mc' style='border-left-color:{ccolor}'>
                <div class='mc-label'>Segment {cid}</div>
                <div style='font-size:13px;font-weight:700;color:{ccolor};margin-bottom:6px'>{cname}</div>
                <div style='font-size:24px;font-weight:700;color:#1A2B22;line-height:1'>{n}</div>
                <div class='mc-sub'>{pct:.1f}% of portfolio</div>
                <div style='margin-top:10px;display:flex;gap:8px'>
                    <span style='font-size:11px;background:{ccolor}15;color:{ccolor};padding:2px 8px;border-radius:20px;font-weight:700'>SAN {avg_san:.0%}</span>
                    <span style='font-size:11px;background:#71809615;color:#718096;padding:2px 8px;border-radius:20px;font-weight:700'>Churn {avg_churn:.0%}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("<div class='slabel'>Segment profiles & intervention windows</div>", unsafe_allow_html=True)

        # Real lifecycle positions from notebook
        lifecycle_pos = {1: 53.5, 2: 36.8, 3: 48.6, 4: 56.4}

        for cid, (cname, ccolor, cdesc) in CLUSTER_PROFILES.items():
            istart, iend, iaction, idesc = CLUSTER_INTERVENTION[cid]
            lc = lifecycle_pos[cid]
            bar_width = iend - istart

            st.markdown(f"""
            <div style='margin-bottom:16px;padding:14px 16px;background:#F8FAFB;
                        border-radius:10px;border-left:3px solid {ccolor}'>
                <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px'>
                    <div>
                        <span style='font-size:12px;font-weight:700;color:{ccolor}'>Segment {cid} · {cname}</span>
                        <div style='font-size:11px;color:#718096;margin-top:2px'>{cdesc[:90]}…</div>
                    </div>
                    <span style='font-size:11px;background:{ccolor}18;color:{ccolor};padding:2px 9px;
                                 border-radius:20px;font-weight:700;white-space:nowrap;margin-left:10px'>
                        {iaction}
                    </span>
                </div>
                <div style='font-size:10px;color:#718096;margin-bottom:4px;text-transform:uppercase;letter-spacing:.06em'>
                    Intervene {istart}–{iend}% of contract · current avg {lc:.0f}%
                </div>
                <div style='position:relative;height:8px;background:#E2E8F0;border-radius:4px;overflow:hidden'>
                    <div style='position:absolute;left:{istart}%;width:{bar_width}%;height:100%;
                                background:{ccolor};opacity:.75;border-radius:2px'></div>
                    <div style='position:absolute;left:{lc}%;transform:translateX(-50%);top:0;
                                width:3px;height:100%;background:{ccolor};opacity:1'></div>
                </div>
                <div style='display:flex;justify-content:space-between;font-size:10px;color:#A0AEC0;margin-top:2px'>
                    <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
                </div>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='slabel'>Early settlement & churn by segment</div>", unsafe_allow_html=True)

        # Real rates from notebook (cells 41, 44)
        seg_data = {
            1: {"label": "Seg 1 · High Risk",          "color": "#E24B4A", "san": 25.5, "natural": 11.8, "total": 36.7},
            2: {"label": "Seg 2 · High Value",          "color": "#378ADD", "san": 46.6, "natural":  7.4, "total": 51.9},
            3: {"label": "Seg 3 · High Ext. Credit",    "color": "#1D9E75", "san": 26.9, "natural": 12.0, "total": 38.4},
            4: {"label": "Seg 4 · Base / Dormant",      "color": "#BA7517", "san": 22.8, "natural": 21.1, "total": 43.5},
        }

        for cid, d in seg_data.items():
            c = d["color"]
            st.markdown(f"""
            <div style='margin-bottom:14px'>
              <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:5px'>
                <span style='font-size:12px;font-weight:700;color:{c}'>{d["label"]}</span>
                <span style='font-size:11px;color:#718096'>Total churn <b style='color:#1A2B22'>{d["total"]:.1f}%</b></span>
              </div>
              <div style='height:10px;background:#E2E8F0;border-radius:5px;overflow:hidden;position:relative'>
                <div style='position:absolute;left:0;width:{d["san"]}%;height:100%;background:{c};opacity:.9'></div>
                <div style='position:absolute;left:{d["san"]}%;width:{d["natural"]}%;height:100%;background:{c};opacity:.35'></div>
              </div>
              <div style='display:flex;gap:14px;font-size:11px;color:#718096;margin-top:4px'>
                <span><b style='color:{c}'>■</b> Early SAN {d["san"]:.1f}%</span>
                <span><b style='color:{c};opacity:.4'>■</b> Natural SOL {d["natural"]:.1f}%</span>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='slabel'>Methodology</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:12px;color:#718096;line-height:1.75'>
            <b style='color:#2D3748'>Step 1 — 3-perspective K-Means</b> (k=3 each, silhouette-validated)<br>
            &nbsp;&nbsp;· <b>P1 Financial</b>: N_CONTRACTS, TOTAL_MTFINO, MENSALIDADE, MEDIAN_DURDEG<br>
            &nbsp;&nbsp;· <b>P2 Risk</b>: MONTVENC_LOG, RISK_EVER, RISK_RECENT, COUNT_CL, DIVIDAS<br>
            &nbsp;&nbsp;· <b>P3 History</b>: CLIENT_SENIORITY_YEARS, YEARS_SINCE_LAST_CONTRACT, N_Dossiers<br><br>
            <b style='color:#2D3748'>Step 2 — Ward hierarchical clustering</b> on combined P1_P2_P3 centroids<br>
            &nbsp;&nbsp;→ Dendrogram cut at <b>n=4</b> (deeper cuts yield segments &lt;2% of base)
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════

elif page == "Model Metrics":
    st.markdown("<div class='page-body'>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Model Performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-desc'>Cross-validation and test set evaluation for both models.</div>", unsafe_allow_html=True)

    st.markdown("<div class='slabel'>Model 1 — Early settlement (SAN vs SOL)</div>", unsafe_allow_html=True)
    st.markdown("""<div class="metrics-grid">
        <div class="mc green"><div class="mc-label">ROC-AUC</div><div class="mc-value green">—</div><div class="mc-sub">CV mean</div></div>
        <div class="mc blue"><div class="mc-label">F1-score</div><div class="mc-value blue">—</div><div class="mc-sub">CV mean</div></div>
        <div class="mc"><div class="mc-label">Precision</div><div class="mc-value">—</div><div class="mc-sub">test set</div></div>
        <div class="mc"><div class="mc-label">Recall</div><div class="mc-value">—</div><div class="mc-sub">test set</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='slabel' style='margin-top:1.25rem'>Model 2 — Churn prediction</div>", unsafe_allow_html=True)
    st.markdown("""<div class="metrics-grid">
        <div class="mc green"><div class="mc-label">ROC-AUC</div><div class="mc-value green">—</div><div class="mc-sub">CV mean</div></div>
        <div class="mc blue"><div class="mc-label">F1-score</div><div class="mc-value blue">—</div><div class="mc-sub">CV mean</div></div>
        <div class="mc"><div class="mc-label">Precision</div><div class="mc-value">—</div><div class="mc-sub">test set</div></div>
        <div class="mc"><div class="mc-label">Recall</div><div class="mc-value">—</div><div class="mc-sub">test set</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("<div class='slabel'>Confusion matrix — SAN/SOL model</div>", unsafe_allow_html=True)
        st.markdown("""<div class='todo' style='height:240px;display:flex;flex-direction:column;
                        align-items:center;justify-content:center'>
            <div class='todo-t'>Confusion matrix goes here</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='slabel'>Feature importance — top 15</div>", unsafe_allow_html=True)
        st.markdown("""<div class='todo' style='height:240px;display:flex;flex-direction:column;
                        align-items:center;justify-content:center'>
            <div class='todo-t'>Feature importance chart goes here</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)