import os
import pandas as pd

# Resolve paths relative to this file's location (src/code/),
# so they work regardless of where Jupyter's working directory is set.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src/code
_PROJECT_ROOT = os.path.abspath(os.path.join(_SRC_DIR, "..", ".."))  # project root

DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "converted")   # original data
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "data")               # data outputs

def data_path(file_name: str) -> str:
    """Return full path for converted data."""
    return os.path.join(DATA_DIR, file_name)

def output_path(file_name: str) -> str:
    """Return full path for processed/cleaned output data."""
    return os.path.join(OUTPUT_DIR, file_name)

def load(file_path: str, parse_dates=None) -> pd.DataFrame:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path, parse_dates=parse_dates)
    elif ext in [".parquet", ".pq"]:
        df = pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet")
    print(f"[LOAD] {file_path} | shape: {df.shape}")
    return df

def _sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Cast object columns containing oversized Python ints to float64 so PyArrow doesn't overflow."""
    df = df.copy()
    int64_max = 2**63 - 1
    int64_min = -(2**63)
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna()
        if sample.empty:
            continue
        if all(isinstance(v, int) for v in sample.iloc[:100]):
            col_max = sample.max()
            col_min = sample.min()
            if col_max > int64_max or col_min < int64_min:
                df[col] = df[col].astype("float64")
    return df


def save(df: pd.DataFrame, file_path: str, index=False):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df.to_csv(file_path, index=index)
    elif ext in [".parquet", ".pq"]:
        df = _sanitize_for_parquet(df)
        df.to_parquet(file_path, index=index)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet")
    print(f"[SAVE] {file_path} | shape: {df.shape}")

def read_or_load(file_path: str, parse_dates=None) -> pd.DataFrame:
    return load(file_path, parse_dates=parse_dates)