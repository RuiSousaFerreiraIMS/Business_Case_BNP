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

/* ── Nav row (sits directly under topbar) ── */
/* We target ONLY the nav columns wrapper by giving it a custom class via markdown */
.nav-row-wrap {
    background: #F8FAFB;
    border-bottom: 1px solid #E2E8F0;
    padding: 0 2rem;
    display: flex;
    gap: 0;
}

/* Target nav buttons specifically via their data-testid key prefix */
[data-testid^="stButton-nav_"] > button {
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    border-radius: 0 !important;
    color: #718096 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 22px !important;
    height: 44px !important;
    box-shadow: none !important;
    width: 100% !important;
    font-family: 'Source Sans 3', sans-serif !important;
    transition: color .15s;
    white-space: nowrap !important;
}
[data-testid^="stButton-nav_"] > button:hover {
    color: #00915A !important;
    background: transparent !important;
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
}
[data-testid="stButton-nav_active"] > button p {
    color: #00915A !important;
    font-weight: 700 !important;
}

/* Remove Streamlit column gap/padding for nav row */
.nav-col > div { padding: 0 !important; margin: 0 !important; }

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
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def safe_get(series, key, default="—"):
    return series[key] if key in series.index else default

def action_style(action):
    return {
        "Urgent retention":   ("urgent",   "#C0392B"),
        "Offer new product":  ("offer",    "#00693E"),
        "Upsell opportunity": ("upsell",   "#0066CC"),
        "Monitor":            ("monitor",  "#92400E"),
        "Standard follow-up": ("standard", "#718096"),
    }.get(action, ("standard", "#718096"))

def action_description(action):
    return {
        "Urgent retention":   "High probability of early settlement AND churn. Immediate outreach recommended — offer a retention incentive or restructured product before the client leaves.",
        "Offer new product":  "Client is likely to settle early but shows low churn risk — they want to stay. Contact proactively with a new loan offer to retain the relationship.",
        "Upsell opportunity": "Stable client with low churn risk and low early settlement probability. Good candidate for a new product or credit line expansion.",
        "Monitor":            "Low early settlement risk but moderate churn signal. Keep in regular contact and monitor for changes in financial behaviour.",
        "Standard follow-up": "No immediate action required. Include in standard periodic review cycle.",
    }.get(action, "—")

CLUSTER_PROFILES = {
    0: ("High-value stable",     "#00915A", "Long tenure, low DTI, strong behavioral score. Low risk of churn or early settlement."),
    1: ("Early settlement risk", "#C0392B", "High repayment ratio and low remaining term. Likely to settle before contract end."),
    2: ("Churn risk",            "#B7791F", "Moderate DTI, declining risk trend. Monitor closely for signs of disengagement."),
    3: ("Renewal opportunity",   "#0066CC", "Recently settled or nearing end of contract. High propensity to take a new product."),
    4: ("New client",            "#718096", "Short contract age, limited history. Insufficient data for strong prediction."),
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
    try:
        df = pd.read_parquet("data/prepared/active_clients_scored.parquet")
        df = df.rename(columns={"Prob_SAN": "P_SAN", "Prob_Churn": "P_CHURN"})
        for col, default in [
            ("P_SAN", None), ("P_CHURN", None), ("CLUSTER", 0),
            ("ACTION", "Standard follow-up"), ("REVENUE_AT_RISK", 0),
        ]:
            if col not in df.columns:
                df[col] = default
        return df
    except FileNotFoundError:
        # ── Synthetic data with realistic spread ──
        rng = np.random.default_rng(42)
        n = 150
        cluster = rng.integers(0, 5, n)
        # Each cluster has a different base churn/SAN probability
        cluster_san_base   = {0: 0.15, 1: 0.80, 2: 0.35, 3: 0.55, 4: 0.30}
        cluster_churn_base = {0: 0.10, 1: 0.45, 2: 0.70, 3: 0.25, 4: 0.40}
        p_san   = np.clip([cluster_san_base[c]   + rng.normal(0, 0.12) for c in cluster], 0.02, 0.98)
        p_churn = np.clip([cluster_churn_base[c] + rng.normal(0, 0.12) for c in cluster], 0.02, 0.98)
        actions = []
        for s, ch in zip(p_san, p_churn):
            if s >= 0.65 and ch >= 0.55:
                actions.append("Urgent retention")
            elif s >= 0.65 and ch < 0.40:
                actions.append("Offer new product")
            elif s < 0.35 and ch < 0.35:
                actions.append("Upsell opportunity")
            elif ch >= 0.45:
                actions.append("Monitor")
            else:
                actions.append("Standard follow-up")
        return pd.DataFrame({
            "CONTRIB":               [f"C-{10000+i}" for i in range(n)],
            "TYPEPROD":              rng.choice(["CL", "CP"], n),
            "PRODALP":               rng.choice(["AUTO", "PESSOAL", "HABITACAO"], n),
            "sdem_age":              rng.integers(25, 70, n),
            "MENSALIDADE":           np.round(rng.uniform(150, 800, n), 2),
            "REMAINING_TERM_MONTHS": np.round(rng.uniform(3, 60, n), 1),
            "P_SAN":                 np.round(p_san, 3),
            "P_CHURN":               np.round(p_churn, 3),
            "CLUSTER":               cluster,
            "ACTION":                actions,
            "REVENUE_AT_RISK":       np.round(rng.uniform(500, 15000, n), 2),
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

# Nav tabs rendered as buttons — use key prefix "nav_" so CSS targets them
PAGES = ["Portfolio Overview", "Client Search", "Clustering", "Model Metrics"]
nav_cols = st.columns(len(PAGES) + 6)   # extra cols push tabs left, rest is empty
for i, pg in enumerate(PAGES):
    # Give active tab a distinct key so CSS [data-testid="stButton-nav_active"] fires
    key = "nav_active" if st.session_state.page == pg else f"nav_{i}"
    with nav_cols[i]:
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
        priority_order = ["Urgent retention", "Offer new product", "Upsell opportunity", "Monitor", "Standard follow-up"]
        action_counts["Action"] = pd.Categorical(action_counts["Action"], categories=priority_order, ordered=True)
        action_counts = action_counts.sort_values("Action")
        colors = {
            "Urgent retention": "#C0392B", "Offer new product": "#00915A",
            "Upsell opportunity": "#0066CC", "Monitor": "#B7791F", "Standard follow-up": "#718096",
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
    st.markdown("<div class='page-desc'>Search by client ID · click a column header to sort · select a row to open the full profile.</div>", unsafe_allow_html=True)

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
        cluster = int(client["CLUSTER"])
        revenue = float(client["REVENUE_AT_RISK"])
        action  = str(client["ACTION"])

        san_color   = "#00915A" if p_san   >= 0.6 else "#B7791F" if p_san   >= 0.4 else "#718096"
        churn_color = "#C0392B" if p_churn >= 0.6 else "#B7791F" if p_churn >= 0.4 else "#718096"
        cl_name, cl_color, cl_desc = CLUSTER_PROFILES.get(cluster, ("Unknown", "#718096", ""))

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
            st.markdown(f"""<div class='gauge-wrap'>
                <div class='gauge-title'>Cluster</div>
                <div style='font-size:18px;font-weight:700;color:{cl_color};margin-bottom:5px'>
                    #{cluster} · {cl_name}
                </div>
                <div class='gauge-label'>{cl_desc[:72]}…</div>
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
        st.markdown("<div class='slabel'>Recommended action</div>", unsafe_allow_html=True)
        sty, col = action_style(action)
        st.markdown(f"""<div class='action-card {sty}'>
            <div class='action-title' style='color:{col}'>{action}</div>
            <div class='action-desc'>{action_description(action)}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='slabel'>Client details</div>", unsafe_allow_html=True)
        detail_cols = ["MENSALIDADE", "REMAINING_TERM_MONTHS", "TYPEPROD", "PRODALP", "sdem_age"]
        available   = [c for c in detail_cols if c in client.index]
        if available:
            labels = {
                "MENSALIDADE": "Monthly instalment",
                "REMAINING_TERM_MONTHS": "Remaining term (mo.)",
                "TYPEPROD": "Product type", "PRODALP": "Product", "sdem_age": "Age",
            }
            dcols = st.columns(len(available))
            for i, col_name in enumerate(available):
                val   = client[col_name]
                label = labels.get(col_name, col_name)
                if isinstance(val, (float, np.floating)):
                    val = f"€{val:,.2f}" if "MENSAL" in col_name else f"{val:.1f}"
                with dcols[i]:
                    st.markdown(f"""<div class='mc'>
                        <div class='mc-label'>{label}</div>
                        <div style='font-size:17px;font-weight:700;color:#1A2B22'>{val}</div>
                    </div>""", unsafe_allow_html=True)

    else:
        # ── Search bar (always visible at top) ──────────────────────
        search_query = st.text_input(
            "search", placeholder="🔍  Search by Client ID  (e.g. C-10042)…",
            label_visibility="collapsed", key="search_input"
        )

        # Filter
        if search_query:
            mask = df_all["CONTRIB"].astype(str).str.contains(search_query, case=False, na=False)
            for extra in ["NOME", "NAME"]:
                if extra in df_all.columns:
                    mask = mask | df_all[extra].astype(str).str.contains(search_query, case=False, na=False)
            results = df_all[mask].copy()
        else:
            results = df_all.copy()

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Sort state buttons (rendered as small label above table)
        sort_col = st.session_state.sort_col
        sort_asc = st.session_state.sort_asc

        count_label = f"{len(results):,} client(s) found" if search_query else f"All {len(results):,} active clients"
        st.markdown(f"<div class='slabel'>{count_label} — click a column header to sort</div>", unsafe_allow_html=True)

        # Sort the results
        if sort_col in results.columns:
            results = results.sort_values(sort_col, ascending=sort_asc)

        # ── Sortable column header buttons ──────────────────────────
        col_keys   = list(TABLE_COLS.keys())
        col_labels = list(TABLE_COLS.values())
        # 7 columns + 1 narrow "Open" column
        hcols = st.columns([1.2, 1.1, 1.1, 1.1, 1.4, 1.2, 0.6], gap="small")
        for i, (ck, cl) in enumerate(zip(col_keys, col_labels)):
            arrow = ""
            if sort_col == ck:
                arrow = " ↑" if sort_asc else " ↓"
            with hcols[i]:
                if st.button(
                    f"{cl}{arrow}",
                    key=f"sort_{ck}",
                    use_container_width=True,
                    help=f"Sort by {cl}",
                ):
                    if st.session_state.sort_col == ck:
                        st.session_state.sort_asc = not st.session_state.sort_asc
                    else:
                        st.session_state.sort_col = ck
                        st.session_state.sort_asc = False
                    st.rerun()

        # Divider under headers
        st.markdown("<div style='height:1px;background:#E2E8F0;margin-bottom:2px'></div>", unsafe_allow_html=True)

        # ── Data rows ────────────────────────────────────────────────
        for idx, (_, r) in enumerate(results.iterrows()):
            san_col   = "#00915A" if r["P_SAN"] >= 0.6 else "#B7791F" if r["P_SAN"] >= 0.4 else "#718096"
            churn_col = "#C0392B" if r["P_CHURN"] >= 0.6 else "#B7791F" if r["P_CHURN"] >= 0.4 else "#718096"
            cl_name, cl_color, _ = CLUSTER_PROFILES.get(int(r["CLUSTER"]), ("—", "#718096", ""))
            act_sty, act_col = action_style(r["ACTION"])

            row_cols = st.columns([1.2, 1.1, 1.1, 1.1, 1.4, 1.2, 0.6], gap="small")
            with row_cols[0]:
                st.markdown(f"<div style='padding:7px 0;font-weight:700;font-size:13px'>{r['CONTRIB']}</div>", unsafe_allow_html=True)
            with row_cols[1]:
                st.markdown(f"<div style='padding:7px 0;color:{san_col};font-weight:600;font-size:13px'>{r['P_SAN']:.0%}</div>", unsafe_allow_html=True)
            with row_cols[2]:
                st.markdown(f"<div style='padding:7px 0;color:{churn_col};font-weight:600;font-size:13px'>{r['P_CHURN']:.0%}</div>", unsafe_allow_html=True)
            with row_cols[3]:
                st.markdown(f"<div style='padding:7px 0;color:{cl_color};font-weight:600;font-size:13px'>#{int(r['CLUSTER'])} {cl_name}</div>", unsafe_allow_html=True)
            with row_cols[4]:
                st.markdown(f"<div style='padding:7px 0;color:{act_col};font-weight:600;font-size:13px'>{r['ACTION']}</div>", unsafe_allow_html=True)
            with row_cols[5]:
                st.markdown(f"<div style='padding:7px 0;font-weight:600;font-size:13px'>€{r['REVENUE_AT_RISK']:,.0f}</div>", unsafe_allow_html=True)
            with row_cols[6]:
                if st.button("Open →", key=f"open_{r['CONTRIB']}_{idx}", use_container_width=True):
                    st.session_state.selected_client = r["CONTRIB"]
                    st.rerun()

            st.markdown("<div style='height:1px;background:#F0F2F5;margin:0'></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════

elif page == "Clustering":
    st.markdown("<div class='page-body'>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Client Clustering</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-desc'>Client segments based on behavioural and financial profile.</div>", unsafe_allow_html=True)

    cluster_counts = df_all["CLUSTER"].value_counts().sort_index()
    cols = st.columns(len(CLUSTER_PROFILES), gap="medium")
    for i, (cid, (cname, ccolor, cdesc)) in enumerate(CLUSTER_PROFILES.items()):
        n   = int(cluster_counts.get(cid, 0))
        pct = n / len(df_all) * 100
        avg_san   = df_all[df_all["CLUSTER"] == cid]["P_SAN"].mean()
        avg_churn = df_all[df_all["CLUSTER"] == cid]["P_CHURN"].mean()
        with cols[i]:
            st.markdown(f"""<div class='mc' style='border-left-color:{ccolor}'>
                <div class='mc-label'>Cluster #{cid}</div>
                <div style='font-size:14px;font-weight:700;color:{ccolor};margin-bottom:4px'>{cname}</div>
                <div style='font-size:22px;font-weight:700;color:#1A2B22'>{n}</div>
                <div class='mc-sub'>{pct:.1f}% · SAN {avg_san:.0%} · churn {avg_churn:.0%}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("<div class='slabel'>Cluster descriptions & recommendations</div>", unsafe_allow_html=True)
        for cid, (cname, ccolor, cdesc) in CLUSTER_PROFILES.items():
            st.markdown(f"""
            <div style='display:flex;gap:12px;margin-bottom:12px;padding:12px;
                        background:#F5F7FA;border-radius:8px;border-left:3px solid {ccolor}'>
                <div style='font-size:17px;font-weight:700;color:{ccolor};min-width:26px'>#{cid}</div>
                <div>
                    <div style='font-size:13px;font-weight:700;color:#1A2B22'>{cname}</div>
                    <div style='font-size:12px;color:#718096;margin-top:2px'>{cdesc}</div>
                </div>
            </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='slabel'>2D cluster visualisation — PCA / UMAP</div>", unsafe_allow_html=True)
        st.markdown("""<div class='todo' style='height:360px;display:flex;flex-direction:column;
                        align-items:center;justify-content:center;gap:8px'>
            <div class='todo-t'>Scatter plot goes here</div>
            <div class='todo-s'>Add after running PCA/UMAP on X_train_fs<br>
            Colour by cluster · SAN/Churn probability overlay toggle</div>
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