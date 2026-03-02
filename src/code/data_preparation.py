import os
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")


# ─────────────────────────────────────────────────────────────────────────────
# DATA UNDERSTANDING (for use in 1.Data_Understanding.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

def data_understanding_summary(df, dataset_name="Dataset"):
    rows, cols = df.shape

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    datetime_cols = df.select_dtypes(include=["datetime", "datetime64[ns]"]).columns

    total_missing = df.isnull().sum().sum()
    total_cells = rows * cols if rows * cols > 0 else 1
    missing_pct = (total_missing / total_cells) * 100

    duplicates = df.duplicated().sum()

    summary_text = f"""
============================================================
DATA UNDERSTANDING SUMMARY - {dataset_name}
============================================================

Structure
---------
Rows:                     {rows:,}
Columns:                  {cols:,}

Variable Types
--------------
Numeric variables:        {len(numeric_cols)}
Categorical variables:    {len(categorical_cols)}
Datetime variables:       {len(datetime_cols)}

Data Quality
------------
Total missing values:     {total_missing:,}
Missing percentage:       {missing_pct:.2f}%
Duplicate rows:           {duplicates:,}

============================================================
"""

    return summary_text


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION HELPERS (for use in 2.Data_Preparation.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

# -------------- BDOSS --------------------------------------------------------

def _encode_risk(risk_series: pd.Series) -> pd.Series:
    """
    The RISK column is a 24-character string where each character represents
    a month in the past 24 months, and any digit != '0' signals a delinquency.
    
    Returns: 1 if at least one non-zero character exists, 0 otherwise.

    Implementation: fully vectorized via regex - avoids slow row-by-row .apply().
    """
    # Cast to string, strip whitespace, then flag any string that contains
    # a character other than '0' (handles mixed int/str representations)
    s = risk_series.astype(str).str.strip()
    # '0' or '0.0' (integer zero read as float) -> no risk
    # any string with a non-zero digit -> risk
    has_nonzero = s.str.contains(r'[1-9]', regex=True, na=False)
    return has_nonzero.astype(int)


def clean_bdoss(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the BDOSS loan-level dataset.

    Steps:
    1. Parse datetime columns.
    2. Encode the RISK target variable as binary (0 / 1).
    3. Impute missing values in numeric columns.
    4. Encode nominal categorical columns with one-hot encoding.
    5. Derive time-based features (loan age, remaining months).
    6. Drop raw datetime columns and high-leakage columns.

    Returns:
    Cleaned DataFrame with CONTRIB and OBS_DATE retained as join keys.
    """
    df = df.copy()

    # 1. PARSE DATETIMES
    date_cols = ["OBS_DATE", "DCREAT", "DATFIN", "D1FIN", "DPOS", "DCSP"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 2. ENCODE RISK (binary)
    df["RISK"] = _encode_risk(df["RISK"])

    # 3. DERIVED TIME FEATURES
    # pandas does not support timedelta64 with unit 'M' (months are ambiguous).
    # we'll use year/month arithmetic instead for an exact month count.
    if "DCREAT" in df.columns and "OBS_DATE" in df.columns:
        obs_ym = df["OBS_DATE"].dt.year * 12 + df["OBS_DATE"].dt.month
        cre_ym = df["DCREAT"].dt.year * 12 + df["DCREAT"].dt.month
        df["loan_age_months"] = (obs_ym - cre_ym).astype("Int64")

    if "loan_age_months" in df.columns and "DURDEG" in df.columns:
        df["remaining_months"] = (df["DURDEG"] - df["loan_age_months"]).astype("Int64")

    # 4. NUMERIC IMPUTATION (median)
    num_impute_cols = [
        "DURDEG", "RANGPRO", "RANGCLI", "MENSALIDADE_CORR",
        "RESSO", "NBENF", "BICONTRATO", "RISKA", "AGFIN", "SREC",
        "ACTIVIDADE_GLOBAL", "RN", "RD", "CSP",
    ]
    for col in num_impute_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # MENSALIDADE_CORR: fill with MENSALIDADE where available
    if "MENSALIDADE_CORR" in df.columns and "MENSALIDADE" in df.columns:
        df["MENSALIDADE_CORR"] = df["MENSALIDADE_CORR"].fillna(df["MENSALIDADE"])

    # 5. CATEGORICAL ENCODING
    # OHE only low-cardinality columns (< ~20 unique values).
    # PTT is a 4-digit postal code with 763 unique values - OHE would create 763 new columns × 2.6M rows -> kernel crash!
    # We will cast to int instead.
    # TYPEPROD has only 1 unique value -> useless, will be dropped.
    if "PTT" in df.columns:
        df["PTT"] = pd.to_numeric(df["PTT"], errors="coerce").astype("Int64")

    ohe_cols = ["POS", "PAGAMENTO", "NATIO", "MODCONTACTO",
                "POLE", "TYPEPROD", "PRODALP"]
    for col in ohe_cols:
        if col in df.columns:
            # Cast to 'category' dtype first to reduce memory before expansion
            df[col] = df[col].astype("category")
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=np.int8)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)

    # 6. DROP RAW DATE COLS, ID COLS, EMPTY COLS AND CONSTANT COLS
    # TYPEPROD has 1 unique value -> constant -> no predictive value
    # DOSSIER is loan-ID, not a feature
    # ACTIVIDADE_GLOBAL is 100% null in the source data -> drop
    drop_cols = ["DCREAT", "DATFIN", "D1FIN", "DPOS", "DCSP",
                 "DOSSIER", "ACTIVIDADE_GLOBAL"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Also drop any remaining columns that are 100% null
    all_null_cols = df.columns[df.isnull().mean() == 1.0].tolist()
    if all_null_cols:
        print(f"[clean_bdoss] Dropping {len(all_null_cols)} fully-null cols: {all_null_cols}")
        df.drop(columns=all_null_cols, inplace=True)

    print(f"[clean_bdoss] shape: {df.shape} | "
          f"RISK distribution:\n{df['RISK'].value_counts().to_dict()}")
    return df


# -------------- CRC ----------------------------------------------------------

def clean_crc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the CRC (credit bureau) dataset.

    - Parses OBS_DATE.
    - Fills COUNT_* and monetary columns with 0 (no record = no exposure).
    - Imputes MT_MENSAL (monthly instalment total) with the global median.
    """
    df = df.copy()

    if "OBS_DATE" in df.columns:
        df["OBS_DATE"] = pd.to_datetime(df["OBS_DATE"], errors="coerce")

    # count and monetary cols: NaN means 'no product' -> 0
    zero_fill_cols = [c for c in df.columns
                      if c.startswith("COUNT_") or
                      c.startswith("MONTVENC_") or
                      c.startswith("MONTABATV_") or
                      c.startswith("DIVIDAS_")]
    df[zero_fill_cols] = df[zero_fill_cols].fillna(0)

    # MT_MENSAL: total monthly credit obligations – impute with median
    if "MT_MENSAL" in df.columns:
        df["MT_MENSAL"] = df["MT_MENSAL"].fillna(df["MT_MENSAL"].median())

    # Derived: total overdue amount across product types
    montvenc_cols = [c for c in df.columns if c.startswith("MONTVENC_")]
    if montvenc_cols:
        df["crc_total_overdue"] = df[montvenc_cols].sum(axis=1)

    dividas_cols = [c for c in df.columns if c.startswith("DIVIDAS_")]
    if dividas_cols:
        df["crc_total_debt"] = df[dividas_cols].sum(axis=1)

    print(f"[clean_crc] shape: {df.shape}")
    return df


# -------------- CREDSCORE ----------------------------------------------------

_KP_SQE_ORDER = ["A", "B", "C", "D", "E", "F", "G", "H"]  # best -> worst

def clean_credscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the credit-score dataset.

    - Parses sys_data_procura.
    - Ordinal-encodes kp_sqe (A=0 … H=7, lower is better).
    - Keeps only the most-recent record per CONTRIB.
    """
    df = df.copy()

    if "sys_data_procura" in df.columns:
        df["sys_data_procura"] = pd.to_datetime(df["sys_data_procura"], errors="coerce")

    # Ordinal encode kp_sqe
    if "kp_sqe" in df.columns:
        kp_map = {v: i for i, v in enumerate(_KP_SQE_ORDER)}
        df["kp_sqe_enc"] = (
            df["kp_sqe"].str.strip().str.upper().map(kp_map)
        )
        df.drop(columns=["kp_sqe"], inplace=True)

    # Keep most-recent record per contributor
    if "sys_data_procura" in df.columns:
        df = (df.sort_values("sys_data_procura", ascending=False)
                .drop_duplicates(subset=["CONTRIB"])
                .reset_index(drop=True))

    # Drop date column - no longer needed for modelling
    df.drop(columns=["sys_data_procura"], inplace=True, errors="ignore")
    df.drop(columns=["sys_numero_submissao"], inplace=True, errors="ignore")

    print(f"[clean_credscore] shape: {df.shape}")
    return df


# -------------- FAMA ---------------------------------------------------------

def clean_fama(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the FAMA customer-aggregated feature dataset.

    - Parses Date_Obs.
    - One-hot encodes sdem_SITFAM and sdem_HABITAT.
    - Keeps most-recent record per CONTRIB.
    """
    df = df.copy()

    if "Date_Obs" in df.columns:
        df["Date_Obs"] = pd.to_datetime(df["Date_Obs"], errors="coerce")

    # Keep latest record per contributor
    if "Date_Obs" in df.columns:
        df = (df.sort_values("Date_Obs", ascending=False)
                .drop_duplicates(subset=["CONTRIB"])
                .reset_index(drop=True))

    # One-hot encode socio-demographic categoricals
    ohe_cols = ["sdem_SITFAM", "sdem_HABITAT"]
    for col in ohe_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=int)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)

    # Drop date column
    df.drop(columns=["Date_Obs"], inplace=True, errors="ignore")

    # Numeric imputation for FAMA feature cols (fill with median)
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    print(f"[clean_fama] shape: {df.shape}")
    return df



# -------------- MERGE --------------------------------------------------------

def merge_datasets(
    bdoss: pd.DataFrame,
    crc: pd.DataFrame,
    credscore: pd.DataFrame,
    fama: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all 4 cleaned datasets into a single Analytical Base Table (ABT).

    Join strategy:
    - bdoss      <- (left) base table                 (one row per loan × observation month)
    - crc        <- left-join on [CONTRIB, OBS_DATE]  (bureau data for same month)
    - credscore  <- left-join on [CONTRIB]            (latest credit score)
    - fama       <- left-join on [CONTRIB]            (latest customer features)

    Returns:
    Merged DataFrame with the RISK target column preserved.
    """
    print("[merge] Starting merge...")

    # Rename CRC OBS_DATE to avoid column collision
    crc_merge = crc.copy()

    abt = bdoss.merge(crc_merge, on=["CONTRIB", "OBS_DATE"], how="left",
                      suffixes=("", "_crc"))

    abt = abt.merge(credscore, on="CONTRIB", how="left",
                    suffixes=("", "_credscore"))

    abt = abt.merge(fama, on="CONTRIB", how="left",
                    suffixes=("", "_fama"))

    # Summary
    null_pct = (abt.isnull().sum().sum() / abt.size * 100).round(2)
    print(f"[merge] Final ABT shape: {abt.shape}")
    print(f"[merge] Overall missing: {null_pct}%")
    print(f"[merge] RISK distribution:\n{abt['RISK'].value_counts().to_dict()}")
    return abt



# -------------- SAVE ---------------------------------------------------------

def save_prepared(df: pd.DataFrame, path: str = "../data/prepared/abt.parquet"):
    """
    Save the Analytical Base Table (ABT) to disk.
    Creates the directory if it does not exist.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"[save] ABT saved -> {path}  |  shape: {df.shape}")




