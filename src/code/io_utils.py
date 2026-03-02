import os
import pandas as pd

# Default folders
DATA_DIR = "../data/converted/"  # original data
OUTPUT_DIR = "../data/"          # data outputs

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

def save(df: pd.DataFrame, file_path: str, index=False):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df.to_csv(file_path, index=index)
    elif ext in [".parquet", ".pq"]:
        df.to_parquet(file_path, index=index)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet")
    print(f"[SAVE] {file_path} | shape: {df.shape}")

def read_or_load(file_path: str, parse_dates=None) -> pd.DataFrame:
    return load(file_path, parse_dates=parse_dates)