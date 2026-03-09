"""
eda_insights.py  —  Cetelem Churn EDA
======================================
Functions are fully separated by target.

PART 1 — IS_EARLY_SETTLER
--------------------------
san_1_distribution()     → class balance, profile, rate over time
san_2_temporal()         → cohort, lifecycle stage, seasonality
san_3_financial()        → loan size, income, DTI, stress indicators
san_4_risk()             → LAST_RISK, MAX_RISKA, restructurings
san_5_demographics()     → CSP, dependents, client history

PART 2 — IS_CHURN
------------------
churn_1_distribution()   → class balance, profile, rate over time
churn_2_temporal()       → cohort, lifecycle stage, seasonality
churn_3_financial()      → loan size, income, DTI, stress indicators
churn_4_risk()           → LAST_RISK, MAX_RISKA, restructurings
churn_5_demographics()   → CSP, dependents, client history

PART 3 — COMBINED OVERVIEW
----------------------------
overview_bridge()        → 2x2 matrix + volume + profile comparison

Usage
-----
    import src.code.eda_insights as f
    importlib.reload(f)

    # Part 1
    f.san_1_distribution(abt_data)
    f.san_2_temporal(abt_data)
    f.san_3_financial(abt_data)
    f.san_4_risk(abt_data)
    f.san_5_demographics(abt_data)

    # Part 2
    f.churn_1_distribution(abt_data)
    f.churn_2_temporal(abt_data)
    f.churn_3_financial(abt_data)
    f.churn_4_risk(abt_data)
    f.churn_5_demographics(abt_data)

    # Part 3
    f.overview_bridge(abt_data)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── constants ────────────────────────────────────────────────
TARGET_E = "IS_EARLY_SETTLER"
TARGET_C = "IS_CHURN"
C_SAT    = "#2E7D32"   # early settler colour (green)
C_CHR    = "#C62828"   # churn colour (red)
C_NO     = "#B0BEC5"   # "no" bar colour

CSP_LABELS = {
    10: "Merchants",
    15: "Service Providers",
    20: "Managing Partners",
    25: "Pension Income",
    30: "Private Directors",
    31: "Private Dept Heads",
    32: "Health Professional",
    33: "Nurse",
    34: "Doctor",
    35: "Teacher",
    40: "Public Directors",
    41: "Public Dept Heads",
    60: "Office Employees",
    70: "Public Employees",
    74: "Military/Police",
    80: "Workers",
    81: "Driver-Security",
    86: "Cleaning Staff",
    90: "Retired Private",
    91: "Retired Public",
    92: "Unemployed",
    96: "Fixed-term Contract",
    99: "Indeterminate",
}


def _fmt_pct(ax):
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y)}"))

def _section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def _kde(ax, s, color, label):
    """Plot a KDE on ax for series s."""
    s = s.dropna()
    if len(s) < 20:
        return
    s = s.clip(*s.quantile([0.01, 0.99]))
    kde = stats.gaussian_kde(s)
    xs  = np.linspace(s.min(), s.max(), 300)
    ax.fill_between(xs, kde(xs), alpha=0.25, color=color)
    ax.plot(xs, kde(xs), color=color, lw=2,
            label=f"{label}  (mean={s.mean():,.0f})")

def _rate_bar(ax, series, target_series, title, top_n=10, highlight_color=C_SAT):
    """Horizontal bar: target rate per category, sorted descending."""
    grp = (pd.DataFrame({"cat": series, "t": target_series})
             .groupby("cat")["t"]
             .agg(["mean", "count"])
             .query("count >= 20")
             .sort_values("mean", ascending=True)
             .tail(top_n))
    grp["mean"] *= 100
    median = grp["mean"].median()
    colors = [highlight_color if v > median else "#78909C" for v in grp["mean"]]
    # map numeric CSP codes to readable labels
    def _lbl(x):
        try:
            return CSP_LABELS.get(int(float(str(x))), str(x))
        except:
            return str(x)
    y_labels = [_lbl(x) for x in grp.index]
    ax.barh(y_labels, grp["mean"], color=colors, edgecolor="white")
    for i, val in enumerate(grp["mean"]):
        ax.text(val + 0.2, i, f"{val:.1f}%", va="center", fontsize=8)
    _fmt_pct(ax)
    ax.set_title(title, fontsize=9)


# ╔═════════════════════════════════════════════════════════════╗
# ║  PART 1 — IS_EARLY_SETTLER                                  ║
# ╚═════════════════════════════════════════════════════════════╝

def san_1_distribution(df):
    """
    1.1 — IS_EARLY_SETTLER distribution.
    How many clients settle early? What does their basic profile look like?
    """
    T   = TARGET_E
    COL = C_SAT
    N   = len(df)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("1.1  IS_EARLY_SETTLER — Target Distribution",
                 fontsize=13, fontweight="bold")

    # ── class balance ────────────────────────────────────────
    ax = axes[0]
    vc = df[T].value_counts().sort_index()
    ax.bar(["No (0)\nGoes to maturity", "Yes (1)\nSettles early"],
           vc.values, color=[C_NO, COL], edgecolor="white", width=0.5)
    for i, v in enumerate(vc.values):
        ax.text(i, v + N * 0.005,
                f"{v/N*100:.1f}%", ha="center", fontsize=11)
    ax.set_ylabel("number of clients")
    ax.set_title("Class Balance", fontsize=10)

    # ── mean profile: settler vs non-settler ─────────────────
    ax = axes[1]
    cols = [c for c in ["MEDIAN_RESSO", "TOTAL_MTFINO",
                         "MEDIAN_DURDEG", "N_CONTRACTS"] if c in df.columns]
    means = df.groupby(T)[cols].mean()
    short = [c.replace("TOTAL_","").replace("MEDIAN_","") for c in cols]
    x = np.arange(len(cols))
    w = 0.35
    for i, (val, color, lbl) in enumerate([
        (0, C_NO,  "Settler=0 (maturity)"),
        (1, COL,   "Settler=1 (early)"),
    ]):
        if val not in means.index: continue
        norm = means.loc[val] / means.max()
        ax.bar(x + (i - 0.5) * w, norm.values, w * 0.9,
               color=color, alpha=0.85, edgecolor="white", label=lbl)
        for j, (raw, nv) in enumerate(zip(means.loc[val].values, norm.values)):
            ax.text(x[j] + (i - 0.5) * w, nv + 0.02,
                    f"{raw:,.0f}", ha="center", fontsize=7, rotation=40)
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=9)
    ax.set_ylim(0, 1.4)
    ax.set_ylabel("Normalised mean", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title("Mean Profile: Early Settler vs Not", fontsize=10)

    # ── rate over time ────────────────────────────────────────
    ax = axes[2]
    tmp = df.copy()
    tmp["obs_m"] = pd.to_datetime(
        tmp.get("LAST_DPOS", pd.Series(dtype="object")),
        errors="coerce", dayfirst=True).dt.to_period("M").astype(str)
    grp = (tmp.groupby("obs_m")[T]
              .agg(["mean", "count"])
              .query("count >= 20")
              .sort_index())
    grp["mean"] *= 100
    ax2 = ax.twinx()
    ax2.bar(range(len(grp)), grp["count"], color="#CFD8DC",
            alpha=0.4, width=0.9, zorder=1)
    ax2.set_ylabel("number of clients", fontsize=8)
    ax.plot(range(len(grp)), grp["mean"], "o-",
            color=COL, lw=2, ms=4, zorder=2)
    ax.fill_between(range(len(grp)), grp["mean"], alpha=0.15, color=COL)
    step = max(1, len(grp) // 8)
    ax.set_xticks(range(0, len(grp), step))
    ax.set_xticklabels(grp.index[::step], rotation=45, ha="right", fontsize=7)
    _fmt_pct(ax)
    ax.set_ylabel("% IS_EARLY_SETTLER", fontsize=8)
    ax.set_title("Early Settlement Rate Over Time", fontsize=10)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    plt.tight_layout()
    plt.show()


def san_2_temporal(df):
    """
    1.2 — WHEN do clients settle early?
    Cohort analysis, lifecycle stage, seasonality, contract age at exit.
    """
    T   = TARGET_E
    COL = C_SAT

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("1.2  IS_EARLY_SETTLER — When Do They Settle?",
                 fontsize=13, fontweight="bold")

    tmp = df.copy()
    tmp["FIRST_DCREAT"] = pd.to_datetime(tmp["FIRST_DCREAT"], errors="coerce", dayfirst=True)
    tmp["LAST_DPOS"]    = pd.to_datetime(tmp["LAST_DPOS"],    errors="coerce", dayfirst=True)
    tmp["COHORT_Q"]     = tmp["FIRST_DCREAT"].dt.to_period("Q").astype(str)
    tmp["OBS_MONTH"]    = tmp["LAST_DPOS"].dt.month
    tmp["CONTRACT_AGE_M"] = ((tmp["LAST_DPOS"] - tmp["FIRST_DCREAT"])
                              .dt.days / 30.44).clip(0)
    tmp["LIFECYCLE_PCT"] = (tmp["CONTRACT_AGE_M"] /
                             tmp["MEDIAN_DURDEG"].replace(0, np.nan)).clip(0, 1) * 100

    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]

    # ── 1. Origination cohort ─────────────────────────────────
    ax = axes[0, 0]
    valid = tmp.groupby("COHORT_Q").filter(lambda x: len(x) >= 30)
    grp   = valid.groupby("COHORT_Q")[T].agg(["mean", "count"]).sort_index()
    grp["mean"] *= 100
    ax2 = ax.twinx()
    ax2.bar(range(len(grp)), grp["count"], color="#C8E6C9",
            alpha=0.5, width=0.9, zorder=1)
    ax2.set_ylabel("number of clients", fontsize=8)
    ax.plot(range(len(grp)), grp["mean"], "o-",
            color=COL, lw=2.5, ms=5, zorder=2)
    ax.fill_between(range(len(grp)), grp["mean"], alpha=0.15, color=COL)
    step = max(1, len(grp) // 8)
    ax.set_xticks(range(0, len(grp), step))
    ax.set_xticklabels(grp.index[::step], rotation=45, ha="right", fontsize=7)
    _fmt_pct(ax)
    ax.set_ylabel("% Early Settler", fontsize=8)
    ax.set_title("Early Settlement Rate by Origination Cohort\n"
                 "→ Are certain cohorts more prone to early repayment?", fontsize=9)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)

    # ── 2. Lifecycle stage ────────────────────────────────────
    ax = axes[0, 1]
    tmp["stage"] = pd.cut(tmp["LIFECYCLE_PCT"],
                           bins=[0, 25, 50, 75, 100],
                           labels=["0-25%\n(early)", "25-50%\n(mid)",
                                   "50-75%\n(late)", "75-100%\n(near end)"],
                           include_lowest=True)
    grp = tmp.groupby("stage", observed=False)[T].agg(["mean", "count"])
    grp["mean"] *= 100
    alphas = np.linspace(0.4, 1.0, len(grp))
    bars = ax.bar(grp.index.astype(str), grp["mean"],
                  color=[(*plt.cm.colors.to_rgb(COL), a) for a in alphas],
                  edgecolor="white", width=0.6)
    for bar, (_, row) in zip(bars, grp.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%",
                ha="center", fontsize=8)
    _fmt_pct(ax)
    ax.set_title("Early Settlement Rate by Lifecycle Stage\n"
                 "→ At what point in the contract do they settle?", fontsize=9)

    # ── 3. Seasonality ────────────────────────────────────────
    ax = axes[1, 0]
    grp = tmp.groupby("OBS_MONTH")[T].agg(["mean", "count"])
    grp = grp[grp["count"] >= 20]
    grp["mean"] *= 100
    ax.plot(grp.index, grp["mean"], "o-",
            color=COL, lw=2.5, ms=7)
    ax.fill_between(grp.index, grp["mean"], alpha=0.15, color=COL)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTHS, fontsize=8)
    _fmt_pct(ax)
    ax.set_title("Seasonality: Early Settlement Rate by Month\n"
                 "→ January bonus effect? End-of-year peaks?", fontsize=9)

    # ── 4. Contract age at exit ───────────────────────────────
    ax = axes[1, 1]
    _kde(ax, tmp.loc[tmp[T] == 1, "CONTRACT_AGE_M"], COL, "Settler=1")
    _kde(ax, tmp.loc[tmp[T] == 0, "CONTRACT_AGE_M"], C_NO, "Settler=0")
    ax.legend(fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("Contract age at observation (months)")
    ax.set_title("Contract Age Distribution\n"
                 "→ How old is the contract when they settle?", fontsize=9)

    plt.tight_layout()
    plt.show()


def san_3_financial(df):
    """
    1.3 — FINANCIAL profile of early settlers.
    Loan size, income, DTI, loan-to-income.
    """
    T   = TARGET_E
    COL = C_SAT

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("1.3  IS_EARLY_SETTLER — Financial Profile",
                 fontsize=13, fontweight="bold")

    tmp = df.copy()
    tmp["DTI"]  = (tmp["TOTAL_MENSALIDADE"] /
                   tmp["MEDIAN_RESSO"].replace(0, np.nan)).clip(0, 2)
    tmp["LTI"]  = (tmp["TOTAL_MTFINO"] /
                   tmp["MEDIAN_RESSO"].replace(0, np.nan)).clip(0, 50)
    tmp["INCOME_RANGE"] = tmp["MAX_RESSO"] - tmp["MIN_RESSO"]

    metrics = [
        ("TOTAL_MTFINO",   "Original Loan Amount (EUR)"),
        ("MEDIAN_RESSO",   "Median Monthly Income (EUR)"),
        ("DTI",            "Approx. DTI (instalment / income)"),
        ("LTI",            "Loan-to-Income Ratio"),
    ]

    for ax, (col, label) in zip(axes.flatten(), metrics):
        _kde(ax, tmp.loc[tmp[T] == 0, col], C_NO,  "Settler=0")
        _kde(ax, tmp.loc[tmp[T] == 1, col], COL,   "Settler=1")
        ax.legend(fontsize=8)
        ax.set_yticks([])
        ax.set_xlabel(label, fontsize=9)
        ax.set_title(label, fontsize=9)

    plt.tight_layout()
    plt.show()


def san_4_risk(df):
    """
    1.4 — RISK profile of early settlers.
    LAST_RISK, MAX_RISKA, restructuring history.
    """
    T   = TARGET_E
    COL = C_SAT

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("1.4  IS_EARLY_SETTLER — Risk Profile",
                 fontsize=13, fontweight="bold")

    # ── 1. MAX_RISKA ──────────────────────────────────────────
    ax = axes[0, 0]
    grp = df.groupby("MAX_RISKA")[T].agg(["mean", "count"])
    grp = grp[grp["count"] >= 20].sort_index()
    grp["mean"] *= 100
    grp.index = grp.index.map(lambda x: str(int(float(x))) if str(x) != 'nan' else x)
    ax.bar(grp.index, grp["mean"],
           color=COL, alpha=0.8, edgecolor="white")
    for i, val in enumerate(grp["mean"]):
        ax.text(i, val + 0.3, f"{val:.1f}%",
                ha="center", fontsize=8)
    _fmt_pct(ax)
    ax.set_title("Early Settlement Rate by MAX_RISKA", fontsize=9)

    # ── 2. Last risk code ─────────────────────────────────────
    ax = axes[0, 1]
    tmp = df.copy()
    tmp["last_risk"] = tmp["LAST_RISK"].astype(str).str[-1]
    _rate_bar(ax, tmp["last_risk"], df[T],
              "Early Settlement Rate by Last Risk Code")

    # ── 3. EVER_RBT ───────────────────────────────────────────
    ax = axes[1, 0]
    def _get(grp, k):
        for key in [k, int(k), bool(k)]:
            if key in grp.index: return grp[key]
        return 0
    grp = df.groupby("EVER_RBT")[T].mean() * 100
    ax.bar(["Never restructured", "Ever restructured"],
           [_get(grp, 0), _get(grp, 1)],
           color=[C_NO, COL], edgecolor="white", width=0.5)
    _fmt_pct(ax)
    ax.set_title("Early Settlement Rate: Restructured vs Never", fontsize=9)

    # ── 4. N_RBT distribution ─────────────────────────────────
    ax = axes[1, 1]
    for val, color, label in [
        (0, C_NO, "Settler=0"), (1, COL, "Settler=1")
    ]:
        vc = (df[df[T] == val]["N_RBT"]
                .value_counts().sort_index().head(3))
        vc = vc / vc.sum() * 100
        ax.plot(vc.index.astype(str), vc.values, "o-",
                color=color, lw=2, ms=5, label=label)
    ax.legend(fontsize=8)
    _fmt_pct(ax)
    ax.set_xlabel("N_RBT (number of restructurings)")
    ax.set_title("Restructuring Count Distribution by Target", fontsize=9)

    plt.tight_layout()
    plt.show()


def san_5_demographics(df):
    """
    1.5 — DEMOGRAPHICS & CLIENT HISTORY of early settlers.
    CSP, dependents, contract count, loyalty profile.
    """
    T   = TARGET_E
    COL = C_SAT

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("1.5  IS_EARLY_SETTLER — Demographics & Client History",
                 fontsize=13, fontweight="bold")

    top_csp = df["CSP"].value_counts().head(10).index
    tmp = df[df["CSP"].isin(top_csp)].copy()

    # ── 1. CSP rate ───────────────────────────────────────────
    _rate_bar(axes[0, 0], tmp["CSP"], tmp[T],
              "Early Settlement Rate by CSP (top 10)\n"
              "→ Which professions settle early most?")

    # ── 2. NBENF ──────────────────────────────────────────────
    ax = axes[0, 1]
    grp = df.groupby("NBENF")[T].agg(["mean", "count"])
    grp = grp[grp["count"] >= 20]
    grp["mean"] *= 100
    grp.index = grp.index.map(lambda x: str(int(float(x))) if str(x) != 'nan' else x)
    ax.bar(grp.index, grp["mean"],
           color=COL, alpha=0.8, edgecolor="white")
    _fmt_pct(ax)
    ax.set_xlabel("NBENF (dependents)")
    ax.set_title("Early Settlement Rate by Number of Dependents", fontsize=9)

    # ── 3. N_CONTRACTS ────────────────────────────────────────
    ax = axes[1, 0]
    grp = df.groupby("N_CONTRACTS")[T].agg(["mean", "count"])
    grp = grp[grp["count"] >= 20]
    grp["mean"] *= 100
    ax.bar(grp.index.astype(str), grp["mean"],
           color=COL, alpha=0.8, edgecolor="white")
    _fmt_pct(ax)
    ax.set_xlabel("N_CONTRACTS")
    ax.set_title("Early Settlement Rate by Number of Contracts\n"
                 "→ More contracts = more or less likely to settle early?", fontsize=9)

    # ── 4. History profile ────────────────────────────────────
    ax = axes[1, 1]
    def _m(col, pos=True):
        s = df[col]
        return (s == 1) if pos else (s == 0)
    combos = {
        "Only SOL":  (_m("EVER_SOL")  & _m("EVER_SAN",False) & _m("EVER_RBT",False)),
        "Only SAN":  (_m("EVER_SOL",False) & _m("EVER_SAN") & _m("EVER_RBT",False)),
        "SOL+SAN":   (_m("EVER_SOL")  & _m("EVER_SAN")),
        "Had RBT":   _m("EVER_RBT"),
        "None":      (_m("EVER_SOL",False) & _m("EVER_SAN",False) & _m("EVER_RBT",False)),
    }
    rates = {k: df.loc[v, T].mean() * 100 for k, v in combos.items() if v.sum() >= 20}
    ns    = {k: v.sum() for k, v in combos.items()}
    vals  = list(rates.values())
    ax.barh(list(rates.keys()), vals,
            color=[COL if v > np.mean(vals) else C_NO for v in vals],
            edgecolor="white")
    for i, (v, lbl) in enumerate(zip(vals, rates.keys())):
        ax.text(v + 0.2, i, f"{v:.1f}%",
                va="center", fontsize=8)
    _fmt_pct(ax)
    ax.set_title("Early Settlement Rate by Contract History Profile", fontsize=9)

    plt.tight_layout()
    plt.show()


# ╔═════════════════════════════════════════════════════════════╗
# ║  PART 2 — IS_CHURN                                          ║
# ╚═════════════════════════════════════════════════════════════╝

def churn_1_distribution(df):
    """
    2.1 — IS_CHURN distribution.
    How many clients churn? What does their basic profile look like?
    """
    T   = TARGET_C
    COL = C_CHR
    N   = len(df)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("2.1  IS_CHURN — Target Distribution",
                 fontsize=13, fontweight="bold")

    # ── class balance ─────────────────────────────────────────
    ax = axes[0]
    vc = df[T].value_counts().sort_index()
    ax.bar(["No (0)\nRenewed contract", "Yes (1)\nDid not renew"],
           vc.values, color=[C_NO, COL], edgecolor="white", width=0.5)
    for i, v in enumerate(vc.values):
        ax.text(i, v + N * 0.005,
                f"{v/N*100:.1f}%", ha="center", fontsize=11)
    ax.set_ylabel("number of clients")
    ax.set_title("Class Balance", fontsize=10)

    # ── mean profile ──────────────────────────────────────────
    ax = axes[1]
    cols = [c for c in ["MEDIAN_RESSO", "TOTAL_MTFINO",
                         "MEDIAN_DURDEG", "N_CONTRACTS"] if c in df.columns]
    means = df.groupby(T)[cols].mean()
    short = [c.replace("TOTAL_","").replace("MEDIAN_","") for c in cols]
    x = np.arange(len(cols))
    w = 0.35
    for i, (val, color, lbl) in enumerate([
        (0, C_NO,  "Churn=0 (renewed)"),
        (1, COL,   "Churn=1 (left)"),
    ]):
        if val not in means.index: continue
        norm = means.loc[val] / means.max()
        ax.bar(x + (i - 0.5) * w, norm.values, w * 0.9,
               color=color, alpha=0.85, edgecolor="white", label=lbl)
        for j, (raw, nv) in enumerate(zip(means.loc[val].values, norm.values)):
            ax.text(x[j] + (i - 0.5) * w, nv + 0.02,
                    f"{raw:,.0f}", ha="center", fontsize=7, rotation=40)
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=9)
    ax.set_ylim(0, 1.4)
    ax.set_ylabel("Normalised mean", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title("Mean Profile: Churned vs Retained", fontsize=10)

    # ── rate over time ────────────────────────────────────────
    ax = axes[2]
    tmp = df.copy()
    tmp["obs_m"] = pd.to_datetime(
        tmp.get("LAST_DPOS", pd.Series(dtype="object")),
        errors="coerce", dayfirst=True).dt.to_period("M").astype(str)
    grp = (tmp.groupby("obs_m")[T]
              .agg(["mean", "count"])
              .query("count >= 20")
              .sort_index())
    grp["mean"] *= 100
    ax2 = ax.twinx()
    ax2.bar(range(len(grp)), grp["count"], color="#FFCDD2",
            alpha=0.4, width=0.9, zorder=1)
    ax2.set_ylabel("number of clients", fontsize=8)
    ax.plot(range(len(grp)), grp["mean"], "o-",
            color=COL, lw=2, ms=4, zorder=2)
    ax.fill_between(range(len(grp)), grp["mean"], alpha=0.15, color=COL)
    step = max(1, len(grp) // 8)
    ax.set_xticks(range(0, len(grp), step))
    ax.set_xticklabels(grp.index[::step], rotation=45, ha="right", fontsize=7)
    _fmt_pct(ax)
    ax.set_ylabel("% IS_CHURN", fontsize=8)
    ax.set_title("Churn Rate Over Time", fontsize=10)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    plt.tight_layout()
    plt.show()


def churn_2_temporal(df):
    """
    2.2 — WHEN do clients churn?
    Cohort analysis, lifecycle stage, seasonality, contract age at exit.
    """
    T   = TARGET_C
    COL = C_CHR

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("2.2  IS_CHURN — When Do They Churn?",
                 fontsize=13, fontweight="bold")

    tmp = df.copy()
    tmp["FIRST_DCREAT"] = pd.to_datetime(tmp["FIRST_DCREAT"], errors="coerce", dayfirst=True)
    tmp["LAST_DPOS"]    = pd.to_datetime(tmp["LAST_DPOS"],    errors="coerce", dayfirst=True)
    tmp["COHORT_Q"]     = tmp["FIRST_DCREAT"].dt.to_period("Q").astype(str)
    tmp["OBS_MONTH"]    = tmp["LAST_DPOS"].dt.month
    tmp["CONTRACT_AGE_M"] = ((tmp["LAST_DPOS"] - tmp["FIRST_DCREAT"])
                              .dt.days / 30.44).clip(0)
    tmp["LIFECYCLE_PCT"] = (tmp["CONTRACT_AGE_M"] /
                             tmp["MEDIAN_DURDEG"].replace(0, np.nan)).clip(0, 1) * 100

    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]

    # ── 1. Origination cohort ─────────────────────────────────
    ax = axes[0, 0]
    valid = tmp.groupby("COHORT_Q").filter(lambda x: len(x) >= 30)
    grp   = valid.groupby("COHORT_Q")[T].agg(["mean", "count"]).sort_index()
    grp["mean"] *= 100
    ax2 = ax.twinx()
    ax2.bar(range(len(grp)), grp["count"], color="#FFCDD2",
            alpha=0.5, width=0.9, zorder=1)
    ax2.set_ylabel("number of clients", fontsize=8)
    ax.plot(range(len(grp)), grp["mean"], "o-",
            color=COL, lw=2.5, ms=5, zorder=2)
    ax.fill_between(range(len(grp)), grp["mean"], alpha=0.15, color=COL)
    step = max(1, len(grp) // 8)
    ax.set_xticks(range(0, len(grp), step))
    ax.set_xticklabels(grp.index[::step], rotation=45, ha="right", fontsize=7)
    _fmt_pct(ax)
    ax.set_ylabel("% Churn", fontsize=8)
    ax.set_title("Churn Rate by Origination Cohort\n"
                 "→ Is the book deteriorating?", fontsize=9)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)

    # ── 2. Lifecycle stage ────────────────────────────────────
    ax = axes[0, 1]
    tmp["stage"] = pd.cut(tmp["LIFECYCLE_PCT"],
                           bins=[0, 25, 50, 75, 100],
                           labels=["0-25%\n(early)", "25-50%\n(mid)",
                                   "50-75%\n(late)", "75-100%\n(near end)"],
                           include_lowest=True)
    grp = tmp.groupby("stage", observed=False)[T].agg(["mean", "count"])
    grp["mean"] *= 100
    stage_colors = ["#E53935", "#EF6C00", "#F9A825", "#43A047"]
    bars = ax.bar(grp.index.astype(str), grp["mean"],
                  color=stage_colors, edgecolor="white", width=0.6)
    for bar, (_, row) in zip(bars, grp.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%",
                ha="center", fontsize=8)
    _fmt_pct(ax)
    ax.set_title("Churn Rate by Lifecycle Stage\n"
                 "→ When should we intervene?", fontsize=9)

    # ── 3. Seasonality ────────────────────────────────────────
    ax = axes[1, 0]
    grp = tmp.groupby("OBS_MONTH")[T].agg(["mean", "count"])
    grp = grp[grp["count"] >= 20]
    grp["mean"] *= 100
    ax.plot(grp.index, grp["mean"], "o-",
            color=COL, lw=2.5, ms=7)
    ax.fill_between(grp.index, grp["mean"], alpha=0.15, color=COL)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTHS, fontsize=8)
    _fmt_pct(ax)
    ax.set_title("Seasonality: Churn Rate by Month\n"
                 "→ When during the year does churn peak?", fontsize=9)

    # ── 4. Contract age at exit ───────────────────────────────
    ax = axes[1, 1]
    _kde(ax, tmp.loc[tmp[T] == 1, "CONTRACT_AGE_M"], COL, "Churn=1")
    _kde(ax, tmp.loc[tmp[T] == 0, "CONTRACT_AGE_M"], C_NO, "Churn=0")
    ax.legend(fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("Contract age at observation (months)")
    ax.set_title("Contract Age Distribution\n"
                 "→ How old is the contract when they churn?", fontsize=9)

    plt.tight_layout()
    plt.show()


def churn_3_financial(df):
    """
    2.3 — FINANCIAL profile of churners.
    Loan size, income, DTI, loan-to-income.
    """
    T   = TARGET_C
    COL = C_CHR

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("2.3  IS_CHURN — Financial Profile",
                 fontsize=13, fontweight="bold")

    tmp = df.copy()
    tmp["DTI"] = (tmp["TOTAL_MENSALIDADE"] /
                  tmp["MEDIAN_RESSO"].replace(0, np.nan)).clip(0, 2)
    tmp["LTI"] = (tmp["TOTAL_MTFINO"] /
                  tmp["MEDIAN_RESSO"].replace(0, np.nan)).clip(0, 50)

    metrics = [
        ("TOTAL_MTFINO",  "Original Loan Amount (EUR)"),
        ("MEDIAN_RESSO",  "Median Monthly Income (EUR)"),
        ("DTI",           "Approx. DTI (instalment / income)"),
        ("LTI",           "Loan-to-Income Ratio"),
    ]

    for ax, (col, label) in zip(axes.flatten(), metrics):
        _kde(ax, tmp.loc[tmp[T] == 0, col], C_NO,  "Churn=0")
        _kde(ax, tmp.loc[tmp[T] == 1, col], COL,   "Churn=1")
        ax.legend(fontsize=8)
        ax.set_yticks([])
        ax.set_xlabel(label, fontsize=9)
        ax.set_title(label, fontsize=9)

    plt.tight_layout()
    plt.show()


def churn_4_risk(df):
    """
    2.4 — RISK profile of churners.
    LAST_RISK, MAX_RISKA, restructuring history.
    """
    T   = TARGET_C
    COL = C_CHR

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("2.4  IS_CHURN — Risk Profile",
                 fontsize=13, fontweight="bold")

    # ── 1. MAX_RISKA ──────────────────────────────────────────
    ax = axes[0, 0]
    grp = df.groupby("MAX_RISKA")[T].agg(["mean", "count"])
    grp = grp[grp["count"] >= 20].sort_index()
    grp["mean"] *= 100
    ax.bar(grp.index.astype(str), grp["mean"],
           color=COL, alpha=0.8, edgecolor="white")
    for i, (val, n) in enumerate(zip(grp["mean"], grp["count"])):
        ax.text(i, val + 0.3, f"{val:.1f}%",
                ha="center", fontsize=8)
    _fmt_pct(ax)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.set_title("Churn Rate by MAX_RISKA", fontsize=9)

    # ── 2. Last risk code ─────────────────────────────────────
    ax = axes[0, 1]
    tmp = df.copy()
    tmp["last_risk"] = tmp["LAST_RISK"].astype(str).str[-1]
    _rate_bar(ax, tmp["last_risk"], df[T],
              "Churn Rate by Last Risk Code", highlight_color=C_CHR)

    # ── 3. EVER_RBT ───────────────────────────────────────────
    ax = axes[1, 0]
    def _get(grp, k):
        for key in [k, int(k), bool(k)]:
            if key in grp.index: return grp[key]
        return 0
    grp = df.groupby("EVER_RBT")[T].mean() * 100
    ax.bar(["Never restructured", "Ever restructured"],
           [_get(grp, 0), _get(grp, 1)],
           color=[C_NO, COL], edgecolor="white", width=0.5)
    _fmt_pct(ax)
    ax.set_title("Churn Rate: Restructured vs Never", fontsize=9)

    # ── 4. N_RBT distribution ─────────────────────────────────
    ax = axes[1, 1]
    for val, color, label in [
        (0, C_NO, "Churn=0"), (1, COL, "Churn=1")
    ]:
        vc = (df[df[T] == val]["N_RBT"]
                .value_counts().sort_index().head(3))
        vc = vc / vc.sum() * 100
        ax.plot(vc.index.astype(str), vc.values, "o-",
                color=color, lw=2, ms=5, label=label)
    ax.legend(fontsize=8)
    _fmt_pct(ax)
    ax.set_xlabel("N_RBT (number of restructurings)")
    ax.set_title("Restructuring Count Distribution by Churn Status", fontsize=9)

    plt.tight_layout()
    plt.show()


def churn_5_demographics(df):
    """
    2.5 — DEMOGRAPHICS & CLIENT HISTORY of churners.
    CSP, dependents, contract count, loyalty profile.
    """
    T   = TARGET_C
    COL = C_CHR

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("2.5  IS_CHURN — Demographics & Client History",
                 fontsize=13, fontweight="bold")

    top_csp = df["CSP"].value_counts().head(10).index
    tmp = df[df["CSP"].isin(top_csp)].copy()

    # ── 1. CSP churn rate ─────────────────────────────────────
    _rate_bar(axes[0, 0], tmp["CSP"], tmp[T],
              "Churn Rate by CSP (top 10)\n"
              "→ Which professions churn most?", highlight_color=C_CHR)

    # ── 2. NBENF ──────────────────────────────────────────────
    ax = axes[0, 1]
    grp = df.groupby("NBENF")[T].agg(["mean", "count"])
    grp = grp[grp["count"] >= 20]
    grp["mean"] *= 100
    grp.index = grp.index.map(lambda x: str(int(float(x))) if str(x) != 'nan' else x)
    ax.bar(grp.index, grp["mean"],
           color=COL, alpha=0.8, edgecolor="white")
    _fmt_pct(ax)
    ax.set_xlabel("NBENF (dependents)")
    ax.set_title("Churn Rate by Number of Dependents", fontsize=9)

    # ── 3. N_CONTRACTS ────────────────────────────────────────
    ax = axes[1, 0]
    grp = df.groupby("N_CONTRACTS")[T].agg(["mean", "count"])
    grp = grp[grp["count"] >= 20]
    grp["mean"] *= 100
    ax.bar(grp.index.astype(str), grp["mean"],
           color=COL, alpha=0.8, edgecolor="white")
    _fmt_pct(ax)
    ax.set_xlabel("N_CONTRACTS")
    ax.set_title("Churn Rate by Number of Contracts\n"
                 "→ Multi-contract clients: more loyal or higher risk?", fontsize=9)

    # ── 4. History profile × churn ────────────────────────────
    ax = axes[1, 1]
    def _m(col, pos=True):
        s = df[col]
        return (s == 1) if pos else (s == 0)
    combos = {
        "Only SOL":  (_m("EVER_SOL")  & _m("EVER_SAN",False) & _m("EVER_RBT",False)),
        "Only SAN":  (_m("EVER_SOL",False) & _m("EVER_SAN") & _m("EVER_RBT",False)),
        "SOL+SAN":   (_m("EVER_SOL")  & _m("EVER_SAN")),
        "Had RBT":   _m("EVER_RBT"),
        "None":      (_m("EVER_SOL",False) & _m("EVER_SAN",False) & _m("EVER_RBT",False)),
    }
    rates = {k: df.loc[v, T].mean() * 100 for k, v in combos.items() if v.sum() >= 20}
    ns    = {k: v.sum() for k, v in combos.items()}
    vals  = list(rates.values())
    ax.barh(list(rates.keys()), vals,
            color=[COL if v > np.mean(vals) else C_NO for v in vals],
            edgecolor="white")
    for i, (v, lbl) in enumerate(zip(vals, rates.keys())):
        ax.text(v + 0.2, i, f"{v:.1f}%",
                va="center", fontsize=8)
    _fmt_pct(ax)
    ax.set_title("Churn Rate by Contract History Profile", fontsize=9)

    plt.tight_layout()
    plt.show()


# ╔═════════════════════════════════════════════════════════════╗
# ║  PART 3 — COMBINED OVERVIEW                                  ║
# ╚═════════════════════════════════════════════════════════════╝

def overview_bridge(df):
    """
    3.1 — The Business Case Bridge.
    IS_EARLY_SETTLER × IS_CHURN: 2x2 matrix, volume, profile comparison.
    Directly maps to slide 9 of the brief.
    """
    sub = df[[TARGET_E, TARGET_C]].dropna().astype(int)
    N   = len(sub)

    q = {
        "san_renewed": ((sub[TARGET_E]==1) & (sub[TARGET_C]==0)).sum(),
        "san_churned": ((sub[TARGET_E]==1) & (sub[TARGET_C]==1)).sum(),
        "mat_renewed": ((sub[TARGET_E]==0) & (sub[TARGET_C]==0)).sum(),
        "mat_churned": ((sub[TARGET_E]==0) & (sub[TARGET_C]==1)).sum(),
    }
    QUAD = {
        "san_renewed": ("#1B5E20", "SAN=1 / Churn=0\nBEST CASE"),
        "san_churned": ("#B71C1C", "SAN=1 / Churn=1\nCRITICAL LOSS"),
        "mat_renewed": ("#1565C0", "SAN=0 / Churn=0\nSTABLE"),
        "mat_churned": ("#E65100", "SAN=0 / Churn=1\nNATURAL END"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle("3.1  Business Case Bridge: Early Settlement × Churn\n"
                 "Connecting Objective 1 and Objective 2",
                 fontsize=13, fontweight="bold")

    # ── volume bars ───────────────────────────────────────────
    ax = axes[0]
    labels = [QUAD[k][1] for k in q]
    values = list(q.values())
    colors = [QUAD[k][0] for k in q]
    bars   = ax.bar(labels, values, color=colors, edgecolor="white", width=0.6)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + N*0.003,
                f"{v/N*100:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("# clients")
    ax.set_title("Volume per Outcome Quadrant", fontsize=10)
    plt.setp(ax.get_xticklabels(), fontsize=8)

    # ── 2x2 heatmap ───────────────────────────────────────────
    ax = axes[1]
    matrix = np.array([[q["san_renewed"], q["san_churned"]],
                        [q["mat_renewed"], q["mat_churned"]]])
    annot  = np.array([
        [f"{q['san_renewed']:,}\n({q['san_renewed']/N*100:.1f}%)",
         f"{q['san_churned']:,}\n({q['san_churned']/N*100:.1f}%)"],
        [f"{q['mat_renewed']:,}\n({q['mat_renewed']/N*100:.1f}%)",
         f"{q['mat_churned']:,}\n({q['mat_churned']/N*100:.1f}%)"],
    ])
    ax.imshow(matrix, cmap="RdYlGn", vmin=0, aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, annot[i, j], ha="center", va="center",
                    fontsize=12, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="black", alpha=0.3, ec="none"))
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Churn=0\n(Renewed)", "Churn=1\n(Left)"])
    ax.set_yticklabels(["SAN=1\n(Early Settler)", "SAN=0\n(Maturity)"])
    ax.set_title("2×2 Outcome Matrix", fontsize=10)

    # ── profile comparison (normalised) ───────────────────────
    ax = axes[2]
    profile_cols = [c for c in ["MEDIAN_RESSO", "TOTAL_MTFINO",
                                  "TOTAL_MENSALIDADE", "MEDIAN_DURDEG",
                                  "MAX_RISKA", "N_CONTRACTS"] if c in df.columns]
    quad_map = {(1,0): "SAN+Renewed", (1,1): "SAN+Churned",
                (0,0): "Mat+Renewed", (0,1): "Mat+Churned"}
    quad_colors = {"SAN+Renewed": "#1B5E20", "SAN+Churned": "#B71C1C",
                   "Mat+Renewed": "#1565C0", "Mat+Churned": "#E65100"}
    tmp2 = df[[TARGET_E, TARGET_C] + profile_cols].dropna(
        subset=[TARGET_E, TARGET_C]).copy()
    tmp2["quadrant"] = [quad_map.get((int(e), int(c)), None)
                        for e, c in zip(tmp2[TARGET_E], tmp2[TARGET_C])]
    tmp2 = tmp2.dropna(subset=["quadrant"])
    for col in profile_cols:
        mn, mx = tmp2[col].min(), tmp2[col].max()
        tmp2[f"{col}_n"] = (tmp2[col]-mn)/(mx-mn) if mx > mn else 0.5
    means = tmp2.groupby("quadrant")[[f"{c}_n" for c in profile_cols]].mean()
    means.columns = [c.replace("TOTAL_","").replace("MEDIAN_","")
                     for c in profile_cols]
    x = np.arange(len(profile_cols))
    w = 0.2
    for i, (quad, color) in enumerate(quad_colors.items()):
        if quad not in means.index: continue
        ax.bar(x + (i-1.5)*w, means.loc[quad].values, w*0.9,
               color=color, alpha=0.85, edgecolor="white", label=quad)
    ax.set_xticks(x)
    ax.set_xticklabels(means.columns, fontsize=8, rotation=20, ha="right")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Normalised mean (0=low, 1=high)", fontsize=8)
    ax.legend(fontsize=7.5)
    ax.set_title("Client Profile per Quadrant", fontsize=10)

    # ── summary printout ──────────────────────────────────────
    print(f"\n{'─'*52}\n  BRIDGE SUMMARY\n{'─'*52}")
    print(f"  Total: {N:,} clients\n")
    for key, (_, label) in QUAD.items():
        v = q[key]
        print(f"  {label.replace(chr(10),' '): <30}  {v:>6,}  ({v/N*100:.1f}%)")
    n_san = q["san_renewed"] + q["san_churned"]
    n_mat = q["mat_renewed"] + q["mat_churned"]
    print()
    if n_san > 0:
        print(f"  Of early settlers   → {q['san_churned']/n_san*100:.1f}% did NOT renew")
    if n_mat > 0:
        print(f"  Of maturity clients → {q['mat_churned']/n_mat*100:.1f}% did NOT renew")
    print(f"{'─'*52}")

    plt.tight_layout()
    plt.show()


def overview_compare(df):
    """
    3.2 — Side-by-side comparison: Early Settler vs Churn.
    Four charts that directly compare the two targets across
    key dimensions: CSP, financial stress, risk, and contract history.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("3.2  Early Settler vs Churn — Direct Comparison",
                 fontsize=13, fontweight="bold")

    # ── 1. CSP: both rates side by side ──────────────────────
    ax = axes[0, 0]
    top_csp = df["CSP"].value_counts().head(10).index
    tmp = df[df["CSP"].isin(top_csp)].copy()

    def _lbl(x):
        try: return CSP_LABELS.get(int(float(str(x))), str(x))
        except: return str(x)

    csp_san   = tmp.groupby("CSP")[TARGET_E].mean() * 100
    csp_churn = tmp.groupby("CSP")[TARGET_C].mean() * 100
    order     = csp_san.sort_values(ascending=True).index
    y_labels  = [_lbl(x) for x in order]
    x         = np.arange(len(order))
    w         = 0.38
    ax.barh(x - w/2, csp_san.reindex(order).values,  w, color=C_SAT, alpha=0.85,
            edgecolor="white", label="Early Settler")
    ax.barh(x + w/2, csp_churn.reindex(order).values, w, color=C_CHR, alpha=0.85,
            edgecolor="white", label="Churn")
    ax.set_yticks(x)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.legend(fontsize=8)
    _fmt_pct(ax)
    ax.set_title("Early Settlement vs Churn Rate by CSP\n"
                 "→ Same profession, different behaviours?", fontsize=9)

    # ── 2. Financial stress KDE: DTI comparison ───────────────
    ax = axes[0, 1]
    tmp2 = df.copy()
    tmp2["DTI"] = (tmp2["TOTAL_MENSALIDADE"] /
                   tmp2["MEDIAN_RESSO"].replace(0, np.nan)).clip(0, 2)
    _kde(ax, tmp2.loc[tmp2[TARGET_E] == 1, "DTI"], C_SAT, "Early Settler=1")
    _kde(ax, tmp2.loc[tmp2[TARGET_C] == 1, "DTI"], C_CHR, "Churn=1")
    _kde(ax, tmp2.loc[(tmp2[TARGET_E]==0) & (tmp2[TARGET_C]==0), "DTI"],
         C_NO, "Neither (control)")
    ax.legend(fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("Approx. DTI (instalment / income)")
    ax.set_title("DTI Distribution: Settler vs Churner vs Neither\n"
                 "→ Who is most financially stretched?", fontsize=9)

    # ── 3. MAX_RISKA: grouped bar both targets ─────────────────
    ax = axes[1, 0]
    grp_san   = df.groupby("MAX_RISKA")[TARGET_E].mean() * 100
    grp_churn = df.groupby("MAX_RISKA")[TARGET_C].mean() * 100
    all_risk  = sorted(set(grp_san.index) | set(grp_churn.index))
    int_risk  = [str(int(float(r))) if str(r) != 'nan' else str(r) for r in all_risk]
    x         = np.arange(len(all_risk))
    w         = 0.38
    ax.bar(x - w/2, [grp_san.get(r, 0)   for r in all_risk], w,
           color=C_SAT, alpha=0.85, edgecolor="white", label="Early Settler")
    ax.bar(x + w/2, [grp_churn.get(r, 0) for r in all_risk], w,
           color=C_CHR, alpha=0.85, edgecolor="white", label="Churn")
    ax.set_xticks(x)
    ax.set_xticklabels(int_risk, fontsize=9)
    ax.legend(fontsize=8)
    _fmt_pct(ax)
    ax.set_xlabel("MAX_RISKA")
    ax.set_title("Target Rate by MAX_RISKA\n"
                 "→ Does risk score predict churn more than early settlement?", fontsize=9)

    # ── 3.1 Contract history: grouped bar both targets ──────────
    ax = axes[1, 1]
    def _m(col, pos=True):
        s = df[col]; return (s == 1) if pos else (s == 0)
    combos = {
        "Only SOL": (_m("EVER_SOL")  & _m("EVER_SAN", False) & _m("EVER_RBT", False)),
        "Only SAN": (_m("EVER_SOL", False) & _m("EVER_SAN") & _m("EVER_RBT", False)),
        "SOL+SAN":  (_m("EVER_SOL")  & _m("EVER_SAN")),
        "Had RBT":  _m("EVER_RBT"),
        "None":     (_m("EVER_SOL", False) & _m("EVER_SAN", False) & _m("EVER_RBT", False)),
    }
    labels_h = list(combos.keys())
    rates_san   = [df.loc[v, TARGET_E].mean() * 100 for v in combos.values()]
    rates_churn = [df.loc[v, TARGET_C].mean() * 100 for v in combos.values()]
    x = np.arange(len(labels_h))
    w = 0.38
    ax.bar(x - w/2, rates_san,   w, color=C_SAT, alpha=0.85,
           edgecolor="white", label="Early Settler")
    ax.bar(x + w/2, rates_churn, w, color=C_CHR, alpha=0.85,
           edgecolor="white", label="Churn")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_h, fontsize=9)
    ax.legend(fontsize=8)
    _fmt_pct(ax)
    ax.set_title("Target Rate by Contract History\n"
                 "→ Clients with SOL history: loyal renewers or churners?", fontsize=9)

    plt.tight_layout()
    plt.show()

