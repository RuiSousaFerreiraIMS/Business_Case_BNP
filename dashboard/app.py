import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Client Intelligence · BNP Paribas",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
.block-container {
    padding-top: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    padding-bottom: 3rem !important;
    max-width: 100% !important;
}

/* ── Navbar ── */
.navbar {
    width: 100%;
    background: #FFFFFF;
    border-bottom: 3px solid #00915A;
    padding: 14px 2.5rem 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.navbar-left { display: flex; align-items: center; gap: 14px; }
.navbar-icon { width: 34px; height: 34px; background: #00915A; border-radius: 6px; display: flex; align-items: center; justify-content: center; }
.navbar-icon-inner { width: 14px; height: 14px; background: white; border-radius: 2px; }
.navbar-title { font-size: 16px; font-weight: 700; color: #1A2B22; letter-spacing: -0.02em; }
.navbar-sep { width: 1px; height: 18px; background: #D1D9E0; margin: 0 2px; }
.navbar-sub { font-size: 12px; color: #718096; }
.navbar-tag { font-size: 11px; color: #718096; background: #F5F7FA; border: 1px solid #E2E8F0; padding: 3px 10px; border-radius: 20px; }

/* ── Nav tabs ── */
div[data-testid="stHorizontalBlock"] {
    background: #FAFBFC;
    border-bottom: 1px solid #E2E8F0;
    padding: 0 2.5rem !important;
    gap: 0 !important;
    margin-bottom: 0 !important;
}
div[data-testid="stHorizontalBlock"] > div { flex: 0 0 auto !important; width: auto !important; min-width: 140px !important; }
div[data-testid="stHorizontalBlock"] button {
    background: transparent !important; border: none !important;
    border-bottom: 3px solid transparent !important; border-radius: 0 !important;
    color: #718096 !important; font-size: 13px !important; font-weight: 500 !important;
    padding: 10px 20px !important; height: 44px !important;
    box-shadow: none !important; width: 100% !important;
    font-family: 'Source Sans 3', sans-serif !important;
}
div[data-testid="stHorizontalBlock"] button:hover { color: #00915A !important; background: transparent !important; }
div[data-testid="stHorizontalBlock"] button p { color: inherit !important; font-weight: inherit !important; }

/* ── Page body ── */
.page-body { padding: 2rem 2.5rem; }
.page-title { font-size: 22px; font-weight: 700; color: #1A2B22; letter-spacing: -0.02em; margin-bottom: 4px; }
.page-desc { font-size: 14px; color: #718096; margin-bottom: 1.75rem; }

/* ── Metric cards ── */
.metrics-grid { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 14px; margin-bottom: 1.75rem; }
.mc { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1.125rem 1.375rem; border-left: 4px solid #E2E8F0; }
.mc.green { border-left-color: #00915A; }
.mc.red   { border-left-color: #C0392B; }
.mc.blue  { border-left-color: #0066CC; }
.mc.amber { border-left-color: #B7791F; }
.mc-label { font-size: 11px; font-weight: 600; color: #718096; letter-spacing: 0.07em; text-transform: uppercase; margin-bottom: 8px; }
.mc-value { font-size: 28px; font-weight: 700; color: #1A2B22; line-height: 1; }
.mc-value.green { color: #00915A; }
.mc-value.red   { color: #C0392B; }
.mc-value.blue  { color: #0066CC; }
.mc-value.amber { color: #B7791F; }
.mc-sub { font-size: 12px; color: #718096; margin-top: 5px; }

/* ── Section label ── */
.slabel { font-size: 11px; font-weight: 700; color: #718096; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #E2E8F0; }

/* ── Client card ── */
.client-card { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; }
.client-id { font-size: 20px; font-weight: 700; color: #1A2B22; }
.client-meta { font-size: 13px; color: #718096; margin-top: 2px; }

/* ── Probability gauge ── */
.gauge-wrap { background: #F5F7FA; border-radius: 10px; padding: 1.25rem; }
.gauge-title { font-size: 11px; font-weight: 700; color: #718096; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 10px; }
.gauge-value { font-size: 32px; font-weight: 700; line-height: 1; margin-bottom: 6px; }
.gauge-bar-track { height: 8px; background: #E2E8F0; border-radius: 4px; overflow: hidden; margin-bottom: 6px; }
.gauge-bar-fill { height: 100%; border-radius: 4px; }
.gauge-label { font-size: 12px; color: #718096; }

/* ── Action badge ── */
.action-card { border-radius: 10px; padding: 1.25rem 1.5rem; }
.action-card.urgent  { background: #FEE2E2; border-left: 4px solid #C0392B; }
.action-card.offer   { background: #E8F5EE; border-left: 4px solid #00915A; }
.action-card.upsell  { background: #DBEAFE; border-left: 4px solid #0066CC; }
.action-card.monitor { background: #FEF9C3; border-left: 4px solid #B7791F; }
.action-card.standard{ background: #F5F7FA; border-left: 4px solid #718096; }
.action-title { font-size: 15px; font-weight: 700; margin-bottom: 4px; }
.action-desc  { font-size: 13px; line-height: 1.6; }

/* ── Cluster badge ── */
.cluster-badge { display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 13px; font-weight: 700; }

/* ── Badges ── */
.badge { display: inline-block; padding: 2px 9px; border-radius: 5px; font-size: 11px; font-weight: 700; }
.badge-san    { background: #E8F5EE; color: #00693E; }
.badge-sol    { background: #FEF3C7; color: #92400E; }
.badge-churn  { background: #FEE2E2; color: #991B1B; }
.badge-nochurn{ background: #E8F5EE; color: #00693E; }
.badge-high   { background: #DBEAFE; color: #1E40AF; }
.badge-med    { background: #FEF9C3; color: #854D0E; }
.badge-low    { background: #F3F4F6; color: #374151; }

/* ── Table ── */
.ctable { width: 100%; border-collapse: collapse; font-size: 13px; }
.ctable th { font-size: 10px; font-weight: 700; color: #718096; letter-spacing: .07em; text-transform: uppercase; padding: 0 12px 8px 0; border-bottom: 1px solid #E2E8F0; text-align: left; }
.ctable td { padding: 9px 12px 9px 0; border-bottom: 1px solid #E2E8F0; color: #1A2B22; vertical-align: middle; }
.ctable tr:last-child td { border-bottom: none; }
.ctable tr:hover td { background: #F5F7FA; }

/* ── Todo ── */
.todo { background: #F5F7FA; border: 1.5px dashed #D1D9E0; border-radius: 10px; padding: 2.5rem 2rem; text-align: center; }
.todo-t { font-size: 15px; font-weight: 600; color: #2D3748; margin-bottom: 6px; }
.todo-s { font-size: 13px; color: #718096; line-height: 1.6; }
.divider { height: 1px; background: #E2E8F0; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────
def confidence_label(p):
    if p >= 0.70: return "High", "badge-high"
    if p >= 0.50: return "Medium", "badge-med"
    return "Low", "badge-low"

def action_style(action):
    mapping = {
        "Urgent retention":  ("urgent",  "#C0392B"),
        "Offer new product": ("offer",   "#00693E"),
        "Upsell opportunity":("upsell",  "#0066CC"),
        "Monitor":           ("monitor", "#92400E"),
        "Standard follow-up":("standard","#718096"),
    }
    return mapping.get(action, ("standard", "#718096"))

def action_description(action):
    descriptions = {
        "Urgent retention":   "High probability of early settlement AND churn. Immediate outreach recommended — offer a retention incentive or restructured product before the client leaves.",
        "Offer new product":  "Client is likely to settle early but shows low churn risk — they want to stay. Contact proactively with a new loan offer to retain the relationship.",
        "Upsell opportunity": "Stable client with low churn risk and low early settlement probability. Good candidate for a new product or credit line expansion.",
        "Monitor":            "Low early settlement risk but moderate churn signal. Keep in regular contact and monitor for changes in financial behaviour.",
        "Standard follow-up": "No immediate action required. Include in standard periodic review cycle.",
    }
    return descriptions.get(action, "—")

CLUSTER_PROFILES = {
    0: ("High-value stable",    "#00915A", "Long tenure, low DTI, strong behavioral score. Low risk of churn or early settlement."),
    1: ("Early settlement risk","#C0392B", "High repayment ratio and low remaining term. Likely to settle before contract end."),
    2: ("Churn risk",           "#B7791F", "Moderate DTI, declining risk trend. Monitor closely for signs of disengagement."),
    3: ("Renewal opportunity",  "#0066CC", "Recently settled or nearing end of contract. High propensity to take a new product."),
    4: ("New client",           "#718096", "Short contract age, limited history. Insufficient data for strong prediction."),
}

# ── Load enriched dataset ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df_clients = pd.read_parquet("data/prepared/active_clients.parquet")
        df_scored  = pd.read_parquet("data/prepared/active_clients_scored.parquet")

        df = df_clients.merge(df_scored, on="CONTRIB", how="left")

        df = df.rename(columns={
            "Prob_SAN":   "P_SAN",
            "Prob_Churn": "P_CHURN",
        })

        for col, default in [
            ("P_SAN", 0.5),
            ("P_CHURN", 0.5),
            ("CLUSTER", 0),
            ("ACTION", "Standard follow-up"),
            ("REVENUE_AT_RISK", 0),
        ]:
            if col not in df.columns:
                df[col] = default

        return df

    except FileNotFoundError:
        np.random.seed(42)
        n = 120
        return pd.DataFrame({
            "CONTRIB":               [f"C-{10000+i}" for i in range(n)],
            "TYPEPROD":              np.random.choice(["CL", "CP"], n),
            "PRODALP":               np.random.choice(["AUTO", "PESSOAL", "HABITACAO"], n),
            "sdem_age":              np.random.randint(25, 70, n),
            "MENSALIDADE":           np.random.uniform(150, 800, n).round(2),
            "REMAINING_TERM_MONTHS": np.random.uniform(3, 60, n).round(1),
            "P_SAN":                 np.random.uniform(0.1, 0.95, n).round(3),
            "P_CHURN":               np.random.uniform(0.05, 0.90, n).round(3),
            "CLUSTER":               np.random.randint(0, 5, n),
            "ACTION":                np.random.choice(["Urgent retention","Offer new product","Upsell opportunity","Monitor","Standard follow-up"], n),
            "REVENUE_AT_RISK":       np.random.uniform(500, 15000, n).round(2),
        })

df_all = load_data()

# ── Session state ─────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Portfolio Overview"
if "selected_client" not in st.session_state:
    st.session_state.selected_client = None

# ── Navbar ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="navbar-left">
        <div class="navbar-icon"><div class="navbar-icon-inner"></div></div>
        <span class="navbar-title">Client Intelligence</span>
        <div class="navbar-sep"></div>
        <span class="navbar-sub">BNP Paribas · Risk Analytics</span>
    </div>
    <span class="navbar-tag">Internal · v1.0 · March 2026</span>
</div>
""", unsafe_allow_html=True)

pages = ["Portfolio Overview", "Client Search", "Clustering", "Model Metrics"]
nav_cols = st.columns(len(pages) + 4)
for i, pg in enumerate(pages):
    with nav_cols[i]:
        label = f"**{pg}**" if st.session_state.page == pg else pg
        if st.button(label, key=f"nav_{pg}", use_container_width=True):
            st.session_state.page = pg
            st.rerun()

st.markdown("<div style='height:1px;background:#E2E8F0'></div>", unsafe_allow_html=True)
page = st.session_state.page
st.markdown("<div class='page-body'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — PORTFOLIO OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == "Portfolio Overview":
    st.markdown("<div class='page-title'>Portfolio Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-desc'>Active client portfolio — early settlement risk, churn probability, and revenue impact.</div>", unsafe_allow_html=True)

    n_total      = len(df_all)
    n_san_risk   = (df_all["P_SAN"] >= 0.6).sum()
    n_churn_risk = (df_all["P_CHURN"] >= 0.6).sum()
    n_urgent     = (df_all["ACTION"] == "Urgent retention").sum()
    rev_at_risk  = df_all.loc[df_all["P_SAN"] >= 0.6, "REVENUE_AT_RISK"].sum()

    st.markdown(f"""
    <div class="metrics-grid">
        <div class="mc"><div class="mc-label">Active clients</div><div class="mc-value">{n_total:,}</div><div class="mc-sub">in current portfolio</div></div>
        <div class="mc green"><div class="mc-label">Early settlement risk</div><div class="mc-value green">{n_san_risk:,}</div><div class="mc-sub">P(SAN) ≥ 60%</div></div>
        <div class="mc red"><div class="mc-label">Churn risk</div><div class="mc-value red">{n_churn_risk:,}</div><div class="mc-sub">P(Churn) ≥ 60%</div></div>
        <div class="mc amber"><div class="mc-label">Revenue at risk</div><div class="mc-value amber">€{rev_at_risk:,.0f}</div><div class="mc-sub">from high SAN clients</div></div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.markdown("<div class='slabel'>Action priority list</div>", unsafe_allow_html=True)
        action_counts = df_all["ACTION"].value_counts().reset_index()
        action_counts.columns = ["Action", "Clients"]
        priority_order = ["Urgent retention","Offer new product","Upsell opportunity","Monitor","Standard follow-up"]
        action_counts["Action"] = pd.Categorical(action_counts["Action"], categories=priority_order, ordered=True)
        action_counts = action_counts.sort_values("Action")

        html_rows = ""
        colors = {"Urgent retention":"#C0392B","Offer new product":"#00915A","Upsell opportunity":"#0066CC","Monitor":"#B7791F","Standard follow-up":"#718096"}
        for _, row in action_counts.iterrows():
            pct = row["Clients"] / n_total * 100
            c   = colors.get(row["Action"], "#718096")
            html_rows += f"""
            <div style='margin-bottom:10px'>
              <div style='display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px'>
                <span style='font-weight:600;color:{c}'>{row["Action"]}</span>
                <span style='color:#718096'>{row["Clients"]} clients · {pct:.1f}%</span>
              </div>
              <div style='height:7px;background:#E2E8F0;border-radius:4px;overflow:hidden'>
                <div style='height:100%;width:{pct}%;background:{c};border-radius:4px'></div>
              </div>
            </div>"""
        st.markdown(html_rows, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='slabel'>Top 10 clients by revenue at risk</div>", unsafe_allow_html=True)
        top10 = df_all.nlargest(10, "REVENUE_AT_RISK")[["CONTRIB","P_SAN","P_CHURN","ACTION","REVENUE_AT_RISK"]]

        header = "<table class='ctable'><thead><tr><th>Client</th><th>P(SAN)</th><th>P(Churn)</th><th>Revenue at risk</th></tr></thead><tbody>"
        rows = ""
        for _, r in top10.iterrows():
            san_col  = "#00915A" if r["P_SAN"] >= 0.6 else "#718096"
            churn_col= "#C0392B" if r["P_CHURN"] >= 0.6 else "#718096"
            rows += f"""<tr>
                <td style='font-weight:600'>{r['CONTRIB']}</td>
                <td style='color:{san_col};font-weight:600'>{r['P_SAN']:.0%}</td>
                <td style='color:{churn_col};font-weight:600'>{r['P_CHURN']:.0%}</td>
                <td style='font-weight:600'>€{r['REVENUE_AT_RISK']:,.0f}</td>
            </tr>"""
        st.markdown(header + rows + "</tbody></table>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — CLIENT SEARCH
# ══════════════════════════════════════════════════════════════════════════
elif page == "Client Search":
    st.markdown("<div class='page-title'>Client Search</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-desc'>Search by client ID or name to view individual risk profile, predictions, and recommended actions.</div>", unsafe_allow_html=True)

    # Search bar
    col_s1, col_s2 = st.columns([3, 1], gap="medium")
    with col_s1:
        search_query = st.text_input("", placeholder="Search by Client ID (e.g. C-10042) or name...", label_visibility="collapsed")
    with col_s2:
        search_btn = st.button("Search", use_container_width=True)

    # Filter results
    if search_query:
        mask = df_all["CONTRIB"].astype(str).str.contains(search_query, case=False, na=False)
        # also search other string columns if they exist
        for col in ["NOME", "NAME"] :
            if col in df_all.columns:
                mask = mask | df_all[col].astype(str).str.contains(search_query, case=False, na=False)
        results = df_all[mask]
    else:
        results = df_all

    # If exactly one result or user selected a client → show profile
    selected_id = st.session_state.selected_client

    if selected_id and selected_id in df_all["CONTRIB"].values:
        client = df_all[df_all["CONTRIB"] == selected_id].iloc[0]

        # Back button
        if st.button("← Back to list"):
            st.session_state.selected_client = None
            st.rerun()

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Client header ──────────────────────────────────────────
        age_str  = f"Age {int(client['sdem_age'])}" if "sdem_age" in client else ""
        prod_str = client.get("PRODALP", client.get("TYPEPROD", ""))
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:16px;margin-bottom:1.5rem'>
            <div style='width:52px;height:52px;background:#E8F5EE;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:700;color:#00693E'>
                {str(client['CONTRIB'])[-2:]}
            </div>
            <div>
                <div style='font-size:20px;font-weight:700;color:#1A2B22'>{client['CONTRIB']}</div>
                <div style='font-size:13px;color:#718096'>{age_str} · {prod_str}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── 4 prediction gauges ────────────────────────────────────
        g1, g2, g3, g4 = st.columns(4, gap="medium")

        p_san   = float(client["P_SAN"])
        p_churn = float(client["P_CHURN"])
        cluster = int(client["CLUSTER"])
        revenue = float(client["REVENUE_AT_RISK"])
        action  = str(client["ACTION"])

        san_color   = "#00915A" if p_san   >= 0.6 else "#B7791F" if p_san   >= 0.4 else "#718096"
        churn_color = "#C0392B" if p_churn >= 0.6 else "#B7791F" if p_churn >= 0.4 else "#718096"
        cl_name, cl_color, _ = CLUSTER_PROFILES.get(cluster, ("Unknown", "#718096", ""))

        with g1:
            st.markdown(f"""<div class='gauge-wrap'>
                <div class='gauge-title'>P(Early Settlement)</div>
                <div class='gauge-value' style='color:{san_color}'>{p_san:.0%}</div>
                <div class='gauge-bar-track'><div class='gauge-bar-fill' style='width:{p_san*100:.0f}%;background:{san_color}'></div></div>
                <div class='gauge-label'>{'High risk' if p_san>=0.6 else 'Moderate' if p_san>=0.4 else 'Low risk'}</div>
            </div>""", unsafe_allow_html=True)

        with g2:
            st.markdown(f"""<div class='gauge-wrap'>
                <div class='gauge-title'>P(Churn)</div>
                <div class='gauge-value' style='color:{churn_color}'>{p_churn:.0%}</div>
                <div class='gauge-bar-track'><div class='gauge-bar-fill' style='width:{p_churn*100:.0f}%;background:{churn_color}'></div></div>
                <div class='gauge-label'>{'High risk' if p_churn>=0.6 else 'Moderate' if p_churn>=0.4 else 'Low risk'}</div>
            </div>""", unsafe_allow_html=True)

        with g3:
            st.markdown(f"""<div class='gauge-wrap'>
                <div class='gauge-title'>Cluster</div>
                <div style='font-size:20px;font-weight:700;color:{cl_color};margin-bottom:6px'>#{cluster} · {cl_name}</div>
                <div class='gauge-label'>{CLUSTER_PROFILES.get(cluster, ("","",""))[2][:60]}...</div>
            </div>""", unsafe_allow_html=True)

        with g4:
            st.markdown(f"""<div class='gauge-wrap'>
                <div class='gauge-title'>Revenue at risk</div>
                <div style='font-size:28px;font-weight:700;color:#B7791F;margin-bottom:6px'>€{revenue:,.0f}</div>
                <div class='gauge-label'>Lost interest if early settlement</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Recommended action ─────────────────────────────────────
        st.markdown("<div class='slabel'>Recommended action</div>", unsafe_allow_html=True)
        style, color = action_style(action)
        desc = action_description(action)
        st.markdown(f"""<div class='action-card {style}'>
            <div class='action-title' style='color:{color}'>{action}</div>
            <div class='action-desc'>{desc}</div>
        </div>""", unsafe_allow_html=True)

        # ── Client details ─────────────────────────────────────────
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='slabel'>Client details</div>", unsafe_allow_html=True)

        detail_cols = ["MENSALIDADE","REMAINING_TERM_MONTHS","TYPEPROD","PRODALP","sdem_age"]
        available   = [c for c in detail_cols if c in client.index]
        if available:
            dcols = st.columns(len(available))
            labels = {"MENSALIDADE":"Monthly instalment","REMAINING_TERM_MONTHS":"Remaining term (months)",
                      "TYPEPROD":"Product type","PRODALP":"Product","sdem_age":"Age"}
            for i, col in enumerate(available):
                val = client[col]
                label = labels.get(col, col)
                if isinstance(val, float):
                    val = f"€{val:,.2f}" if "MENSAL" in col or "REVENUE" in col else f"{val:.1f}"
                with dcols[i]:
                    st.markdown(f"""<div class='mc'>
                        <div class='mc-label'>{label}</div>
                        <div style='font-size:18px;font-weight:700;color:#1A2B22'>{val}</div>
                    </div>""", unsafe_allow_html=True)

    else:
        # ── Results list ───────────────────────────────────────────
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        count_label = f"{len(results)} client(s) found" if search_query else f"All {len(results)} active clients"
        st.markdown(f"<div class='slabel'>{count_label}</div>", unsafe_allow_html=True)

        display_df = results.head(50)[["CONTRIB","P_SAN","P_CHURN","CLUSTER","ACTION","REVENUE_AT_RISK"]].copy()

        header = "<table class='ctable'><thead><tr><th>Client ID</th><th>P(Early Settlement)</th><th>P(Churn)</th><th>Cluster</th><th>Action</th><th>Revenue at risk</th><th></th></tr></thead><tbody>"
        rows = ""
        for _, r in display_df.iterrows():
            san_col   = "#00915A" if r["P_SAN"]>=0.6 else "#B7791F" if r["P_SAN"]>=0.4 else "#718096"
            churn_col = "#C0392B" if r["P_CHURN"]>=0.6 else "#B7791F" if r["P_CHURN"]>=0.4 else "#718096"
            cl_name, cl_color, _ = CLUSTER_PROFILES.get(int(r["CLUSTER"]), ("—","#718096",""))
            act_sty, act_col = action_style(r["ACTION"])
            rows += f"""<tr>
                <td style='font-weight:700'>{r['CONTRIB']}</td>
                <td style='color:{san_col};font-weight:600'>{r['P_SAN']:.0%}</td>
                <td style='color:{churn_col};font-weight:600'>{r['P_CHURN']:.0%}</td>
                <td><span style='color:{cl_color};font-weight:600'>#{int(r['CLUSTER'])} {cl_name}</span></td>
                <td style='color:{act_col};font-weight:600'>{r['ACTION']}</td>
                <td style='font-weight:600'>€{r['REVENUE_AT_RISK']:,.0f}</td>
                <td></td>
            </tr>"""
        st.markdown(header + rows + "</tbody></table>", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
        if len(results) > 0:
            client_ids = results["CONTRIB"].tolist()
            selected   = st.selectbox("Open client profile:", ["— select —"] + client_ids[:50], label_visibility="collapsed")
            if selected != "— select —":
                st.session_state.selected_client = selected
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════
elif page == "Clustering":
    st.markdown("<div class='page-title'>Client Clustering</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-desc'>Client segments based on behavioural and financial profile.</div>", unsafe_allow_html=True)

    # Cluster summary cards
    cluster_counts = df_all["CLUSTER"].value_counts().sort_index()
    cols = st.columns(len(CLUSTER_PROFILES), gap="medium")
    for i, (cid, (cname, ccolor, cdesc)) in enumerate(CLUSTER_PROFILES.items()):
        n = cluster_counts.get(cid, 0)
        pct = n / len(df_all) * 100
        with cols[i]:
            avg_san   = df_all[df_all["CLUSTER"]==cid]["P_SAN"].mean()
            avg_churn = df_all[df_all["CLUSTER"]==cid]["P_CHURN"].mean()
            st.markdown(f"""<div class='mc' style='border-left-color:{ccolor}'>
                <div class='mc-label'>Cluster #{cid}</div>
                <div style='font-size:15px;font-weight:700;color:{ccolor};margin-bottom:4px'>{cname}</div>
                <div style='font-size:22px;font-weight:700;color:#1A2B22'>{n}</div>
                <div class='mc-sub'>{pct:.1f}% · avg SAN {avg_san:.0%} · churn {avg_churn:.0%}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("<div class='slabel'>Cluster descriptions</div>", unsafe_allow_html=True)
        for cid, (cname, ccolor, cdesc) in CLUSTER_PROFILES.items():
            st.markdown(f"""<div style='display:flex;gap:12px;margin-bottom:12px;padding:12px;background:#F5F7FA;border-radius:8px;border-left:3px solid {ccolor}'>
                <div style='font-size:18px;font-weight:700;color:{ccolor};min-width:28px'>#{cid}</div>
                <div>
                    <div style='font-size:13px;font-weight:700;color:#1A2B22'>{cname}</div>
                    <div style='font-size:12px;color:#718096;margin-top:2px'>{cdesc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='slabel'>2D cluster visualisation — PCA / UMAP</div>", unsafe_allow_html=True)
        st.markdown("<div class='todo' style='height:360px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px'><div class='todo-t'>Scatter plot goes here</div><div class='todo-s'>Add after running PCA/UMAP on X_train_fs<br>Colour by cluster · overlay SAN/Churn toggle</div></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════
elif page == "Model Metrics":
    st.markdown("<div class='page-title'>Model Performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-desc'>Cross-validation and test set evaluation for both models.</div>", unsafe_allow_html=True)

    st.markdown("<div class='slabel'>Model 1 — Early settlement (SAN vs SOL)</div>", unsafe_allow_html=True)
    st.markdown("""<div class="metrics-grid">
        <div class="mc green"><div class="mc-label">ROC-AUC</div><div class="mc-value green">—</div><div class="mc-sub">CV mean</div></div>
        <div class="mc blue"><div class="mc-label">F1-score</div><div class="mc-value blue">—</div><div class="mc-sub">CV mean</div></div>
        <div class="mc"><div class="mc-label">Precision</div><div class="mc-value">—</div><div class="mc-sub">test set</div></div>
        <div class="mc"><div class="mc-label">Recall</div><div class="mc-value">—</div><div class="mc-sub">test set</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='slabel' style='margin-top:1rem'>Model 2 — Churn prediction</div>", unsafe_allow_html=True)
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
        st.markdown("<div class='todo' style='height:240px;display:flex;flex-direction:column;align-items:center;justify-content:center'><div class='todo-t'>Confusion matrix goes here</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='slabel'>Feature importance — top 15</div>", unsafe_allow_html=True)
        st.markdown("<div class='todo' style='height:240px;display:flex;flex-direction:column;align-items:center;justify-content:center'><div class='todo-t'>Feature importance goes here</div></div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)