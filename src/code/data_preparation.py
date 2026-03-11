import os
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")


# ─────────────────────────────────────────────────────────────────────────────
# DATA UNDERSTANDING (to use in 1.Data_Understanding.ipynb)
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


def visualize_by_variable(df, max_cat=10, dataset_name="Dataset"):
    """
    Visualizations per variable: each variable gets a row with 3 plots
    - Numeric: histogram | boxplot | missing %
    - Categorical: countplot | top N categories | missing %
    - Datetime: count per year | count per month | count per weekday
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    datetime_cols = df.select_dtypes(include=['datetime', 'datetime64[ns]']).columns
    all_cols = df.columns

    n_rows = len(all_cols)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 3))

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, col in enumerate(all_cols):
        col_data = df[col]

        # ------------------------
        # Numeric columns
        # ------------------------
        if col in numeric_cols:
            non_null = col_data.dropna()
            # Histogram
            sns.histplot(non_null, bins=30, kde=True, ax=axes[i, 0])
            axes[i, 0].set_title(f"{col} histogram")

            # Boxplot
            if not non_null.empty:
                sns.boxplot(x=non_null, ax=axes[i, 1])
                axes[i, 1].set_title(f"{col} boxplot")

            # Missing %
            missing_pct = col_data.isnull().mean() * 100
            axes[i, 2].bar(0, missing_pct)
            axes[i, 2].set_title(f"{col} missing %")
            axes[i, 2].set_xticks([])
            axes[i, 2].set_ylim(0, 100)

        # ------------------------
        # Categorical columns
        # ------------------------
        elif col in categorical_cols:
            if col_data.nunique() <= max_cat:
                sns.countplot(y=col, data=df, order=col_data.value_counts().index, ax=axes[i, 0])
                axes[i, 0].set_title(f"{col} countplot")

                top_counts = col_data.value_counts().head(max_cat)
                axes[i, 1].barh(top_counts.index, top_counts.values)
                axes[i, 1].set_title(f"{col} top {max_cat}")
            else:
                axes[i, 0].text(0.5, 0.5, "Too many categories", ha='center')
                axes[i, 0].set_title(f"{col} skipped")
                axes[i, 1].axis('off')

            missing_pct = col_data.isnull().mean() * 100
            axes[i, 2].bar(0, missing_pct)
            axes[i, 2].set_title(f"{col} missing %")
            axes[i, 2].set_xticks([])
            axes[i, 2].set_ylim(0, 100)

        # ------------------------
        # Datetime columns
        # ------------------------
        elif col in datetime_cols:
            non_null = col_data.dropna()
            if not non_null.empty:
                df_temp = non_null.to_frame()
                df_temp['year'] = df_temp[col].dt.year
                df_temp['month'] = df_temp[col].dt.month
                df_temp['weekday'] = df_temp[col].dt.day_name()

                sns.countplot(x='year', data=df_temp, ax=axes[i, 0])
                axes[i, 0].set_title(f"{col} by year")

                sns.countplot(x='month', data=df_temp, ax=axes[i, 1])
                axes[i, 1].set_title(f"{col} by month")

                sns.countplot(x='weekday', data=df_temp, ax=axes[i, 2],
                              order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
                axes[i, 2].set_title(f"{col} by weekday")
            else:
                axes[i, 0].text(0.5, 0.5, "No data", ha='center')
                axes[i, 0].set_title(f"{col} empty")
                axes[i, 1].axis('off')
                axes[i, 2].axis('off')

        # ------------------------
        # Other columns
        # ------------------------
        else:
            axes[i, 0].text(0.5, 0.5, "Plot skipped", ha='center')
            axes[i, 0].set_title(f"{col}")
            axes[i, 1].axis('off')
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()



# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL DATA PREPARATION HELPERS (to use in 2.Data_Preparation.ipynb)

# This section contains general-purpose functions for data cleaning and 
# preparation that can be applied to any dataset without the risk of data leakage. 
# They are designed to be flexible and configurable via parameters, including 
# detailed printouts of the actions taken for transparency.
# ─────────────────────────────────────────────────────────────────────────────

def initial_preparation(
    df: pd.DataFrame,
    dataset_name: str = "Dataset",
    datetime_cols: list[str] | None = None,
    cols_to_remove: list[str] | None = None,
    datetime_format: str | None = None,
) -> pd.DataFrame:
    """
    Step 1 - Initial preparation for any dataset

    Args:
        df              : Input DataFrame.
        dataset_name    : Name to display in the summary (optional).
        datetime_cols   : Columns to convert to datetime (optional).
        cols_to_remove  : Columns to drop (optional).
        datetime_format : Datetime format string e.g. '%Y-%m-%d' (optional).
                          If None, pandas will infer the format automatically.

    Returns:
        Cleaned and prepared DataFrame.
    """
    df = df.copy()
    original_cols = set(df.columns)

    # Convert to datetime 
    converted = []
    if datetime_cols:
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format=datetime_format, errors="coerce")
                converted.append(col)

    # Remove columns 
    removed = []
    if cols_to_remove:
        existing = [col for col in cols_to_remove if col in df.columns]
        df = df.drop(columns=existing)
        removed = existing

    # Summary
    final_cols = list(df.columns)

    print(f"[{dataset_name}]")
    print(f"  Removed   ({len(removed)})  : {removed if removed else '—'}")
    print(f"  Datetime  ({len(converted)}) : {converted if converted else '—'}")
    print(f"  Remaining ({len(final_cols)}): {final_cols}")
    print()

    return df

def handle_missing_values(
    df: pd.DataFrame,
    dataset_name: str = "Dataset",
    numeric_strategy: str = "fixed",  # "fixed" only (mean/median removed to prevent global leakage)
    numeric_fill_value: float | None = 0.0,
    datetime_strategy: str = "ffill",  # "ffill" | "bfill" | "drop"
    drop_row_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Step 2 - Handle missing values.

    Args:
        df                  : Input DataFrame.
        dataset_name        : Name to display in the summary.
        numeric_strategy    : Strategy for numeric columns ('fixed' only).
                              NOTE: 'mean' and 'median' were removed to prevent global data leakage.
                              Rely on the ML Pipeline (ClientImputer) for statistical imputation.
        numeric_fill_value  : Value to use when numeric_strategy='fixed' (default 0.0).
        datetime_strategy   : Strategy for datetime columns — 'ffill', 'bfill', or 'drop'.
        drop_row_threshold  : Drop rows where the fraction of missing values exceeds this (0-1).
                              Applied before column-level imputation.

    Returns:
        DataFrame with missing values handled.
    """
    df = df.copy()

    # Initial diagnostic
    missing_before = df.isnull().sum()
    total_missing_before = missing_before.sum()

    # Drop rows with too many missing values
    rows_before = len(df)
    df = df.dropna(thresh=int(df.shape[1] * (1 - drop_row_threshold)))
    rows_dropped = rows_before - len(df)

    # Separate columns per type
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Numeric imputation
    numeric_imputed = []
    for col in numeric_cols:
        if df[col].isnull().any():
            if numeric_strategy == "fixed":
                df[col] = df[col].fillna(numeric_fill_value)
            numeric_imputed.append(col)

    # Dates imputation
    datetime_imputed = []
    for col in datetime_cols:
        if df[col].isnull().any():
            if datetime_strategy == "ffill":
                df[col] = df[col].ffill()
            elif datetime_strategy == "bfill":
                df[col] = df[col].bfill()
            elif datetime_strategy == "drop":
                df = df.dropna(subset=[col])
            datetime_imputed.append(col)

    # Categorical imputation
    categorical_imputed = []
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna("Unknown")
            categorical_imputed.append(col)

    # SUMMARY
    missing_after = df.isnull().sum().sum()

    print(f"[{dataset_name}]")
    print(f"  Missing before     : {total_missing_before}")
    print(f"  Rows dropped       : {rows_dropped}  (threshold: >{int(drop_row_threshold*100)}% missing)")
    print(f"  Numeric imputed  ({len(numeric_imputed)}) [{numeric_strategy}] : {numeric_imputed if numeric_imputed else '—'}")
    print(f"  Datetime imputed ({len(datetime_imputed)}) [{datetime_strategy}]  : {datetime_imputed if datetime_imputed else '—'}")
    print(f"  Categ.  imputed  ({len(categorical_imputed)}) [Unknown]  : {categorical_imputed if categorical_imputed else '—'}")
    print(f"  Missing after      : {missing_after}")
    print()

    return df

def handle_duplicates(
    df: pd.DataFrame,
    dataset_name: str = "Dataset",
    key_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Step 3 - Handle duplicates.

    Args:
        df          : Input DataFrame.
        dataset_name: Name to display in the summary.
        key_cols    : Columns that define a unique record (optional).
                      If provided, duplicate keys are resolved by keeping
                      the row with the fewest missing values.

    Returns:
        DataFrame with duplicates removed.
    """
    df = df.copy()
    rows_before = len(df)

    # Exact duplicates
    exact_before = len(df)
    df = df.drop_duplicates()
    exact_dropped = exact_before - len(df)

    # Key duplicates
    key_dropped = 0
    if key_cols:
        existing_keys = [col for col in key_cols if col in df.columns]
        if existing_keys:
            # For each group of duplicate keys, keep the row with the fewest missing values
            df["_missing_count"] = df.isnull().sum(axis=1)
            df = (
                df.sort_values("_missing_count")
                  .drop_duplicates(subset=existing_keys, keep="first")
                  .drop(columns="_missing_count")
            )
            key_dropped = (exact_before - exact_dropped) - len(df)

    # SUMMARY
    total_dropped = rows_before - len(df)

    print(f"[{dataset_name}]")
    print(f"  Rows before        : {rows_before}")
    print(f"  Exact duplicates   : {exact_dropped} dropped")
    print(f"  Key duplicates     : {key_dropped} dropped  (key: {key_cols if key_cols else '—'})")
    print(f"  Rows after         : {len(df)}  (total dropped: {total_dropped})")
    print()

    return df




def detect_outliers(
    df: pd.DataFrame,
    dataset_name: str = "Dataset",
    iqr_threshold: float = 1.5,
    zscore_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Step 4 - Detect outliers in numeric columns (no data is modified).

    Args:
        df               : Input DataFrame.
        dataset_name     : Name to display in the summary.
        iqr_threshold    : IQR multiplier to define outlier bounds (default: 1.5).
        zscore_threshold : Z-score threshold to flag outliers (default: 3.0).

    Returns:
        None — outliers are only reported, no data is modified.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    print(f"[{dataset_name}] Outlier Detection")
    print(f"  {'Column':<25} {'IQR outliers':>14} {'Z-score outliers':>17} {'% rows':>8}")
    print(f"  {'-'*25} {'-'*14} {'-'*17} {'-'*8}")

    total_iqr = 0
    total_z = 0

    for col in numeric_cols:
        series = df[col].dropna()

        # IQR
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        iqr_mask = (df[col] < q1 - iqr_threshold * iqr) | (df[col] > q3 + iqr_threshold * iqr)
        iqr_count = iqr_mask.sum()

        # Z-score
        z_scores = (df[col] - series.mean()) / series.std()
        z_mask = z_scores.abs() > zscore_threshold
        z_count = z_mask.sum()

        pct = round(max(iqr_count, z_count) / len(df) * 100, 1)
        total_iqr += iqr_count
        total_z += z_count

        print(f"  {col:<25} {iqr_count:>14} {z_count:>17} {pct:>7}%")

    print(f"  {'-'*25} {'-'*14} {'-'*17} {'-'*8}")
    print(f"  {'TOTAL':<25} {total_iqr:>14} {total_z:>17}")
    print()



# SPECIFIC PER DATASET FUNCTIONS
# -------------- BDOSS --------------------------------------------------------

def _encode_risk_ever(risk_series: pd.Series) -> pd.Series:
    """
    The RISK column is a 24-character string where each character represents
    a month in the past 24 months, and any digit != '0' signals a delinquency.

    Returns 1 if at least one non-zero character exists in the 24-char string, 0 otherwise.
    """
    s = risk_series.astype(str).str.strip()
    has_nonzero = s.str.contains(r'[1-9]', regex=True, na=False)
    return has_nonzero.astype(int)

def _encode_risk_recent(risk_series: pd.Series) -> pd.Series:
    """
    Returns 1 if there's any non-zero character in the last 6 digits of the string (last 6 months), 0 otherwise.
    """
    s = risk_series.astype(str).str.strip()
    s_last_6 = s.str[-6:]
    has_nonzero = s_last_6.str.contains(r'[1-9]', regex=True, na=False)
    return has_nonzero.astype(int)


def clean_bdoss(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the BDOSS loan-level dataset.

    Steps:
    1. Parse datetime columns.
    2. Encode the RISK target variable as binary (0 / 1).
    3. Drop raw datetime, ID, constant, and fully null columns.

    Returns:
    Cleaned DataFrame with CONTRIB and OBS_DATE retained as join keys.
    """
    df = df.copy()

    # 1. PARSE DATETIMES
    date_cols = ["OBS_DATE", "DCREAT", "DATFIN", "D1FIN", "DPOS", "DCSP"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 2. ENCODE RISK VARIABLES
    df["RISK_EVER"] = _encode_risk_ever(df["RISK"])
    df["RISK_RECENT"] = _encode_risk_recent(df["RISK"])
    df["RISK"] = pd.to_numeric(df["RISK"], errors="coerce").fillna(0).astype(int)


    # 3. DROP RAW DATE COLS, ID COLS, EMPTY COLS AND CONSTANT COLS
    # TYPEPROD has 1 unique value -> constant -> no predictive value
    # ACTIVIDADE_GLOBAL is 100% null in the source data -> drop

    # TODO: These are not the Columns in the Excel - check again!
    drop_cols = ["ACTIVIDADE_GLOBAL"]
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
    - Note: Missing values in MT_MENSAL are NOT imputed here to avoid data leakage;
      they are intentionally left as NaN so the downstream Pipeline (ClientImputer)
      can safely impute them per-fold.
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

    # MT_MENSAL: total monthly credit obligations
    # Missing values will flow to the ABT for safely imputed per fold by ClientImputer.
    if "MT_MENSAL" in df.columns:
        pass

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

    # Sort by date but DO NOT drop duplicates to preserve history for point-in-time merges
    if "sys_data_procura" in df.columns:
        df = (df.sort_values(["CONTRIB", "sys_data_procura"], ascending=[True, False])
                .reset_index(drop=True))

    # Do not drop date column - needed for point-in-time merge downstream!
    # df.drop(columns=["sys_data_procura"], inplace=True, errors="ignore")
    df.drop(columns=["sys_numero_submissao"], inplace=True, errors="ignore")

    print(f"[clean_credscore] shape: {df.shape}")
    return df


# -------------- FAMA ---------------------------------------------------------

def clean_fama(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the FAMA customer-aggregated feature dataset.

    - Parses Date_Obs.
    - Keeps most-recent record per CONTRIB.
    - Note: Numeric features are intentionally left with `NaN`s to avoid data
      leakage. The downstream Pipeline (ClientImputer) will safely handle them.
    """
    df = df.copy()

    if "Date_Obs" in df.columns:
        df["Date_Obs"] = pd.to_datetime(df["Date_Obs"], errors="coerce")

    # Sort by date but DO NOT drop duplicates to preserve history for point-in-time merges
    if "Date_Obs" in df.columns:
        df = (df.sort_values(["CONTRIB", "Date_Obs"], ascending=[True, False])
                .reset_index(drop=True))

    # NOTE: One-Hot Encoding for 'sdem_SITFAM' and 'sdem_HABITAT' must be handled
    # by the ML Pipeline (ClientDataCleaner) to avoid feature mismatch during inference.

    # Do not drop date column - needed for point-in-time merge downstream!
    # df.drop(columns=["Date_Obs"], inplace=True, errors="ignore")

    # Numeric cols pass through. Missing values will be imputed by ClientImputer per CV fold
    # to avoid data leakage.
    pass

    print(f"[clean_fama] shape: {df.shape}")
    return df







# I THINK WE CAN ERASE EVERYTHING FROM HERE, BUT PLEASE CONFIRM
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



