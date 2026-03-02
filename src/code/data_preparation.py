
import numpy as np
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

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
DATA UNDERSTANDING SUMMARY — {dataset_name}
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

