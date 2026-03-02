from src.code.converter_functions import convert_parquet_to_csv

if __name__ == "__main__":
    path_csv = "../../data/converted"
    import os

    print("Running from:", os.getcwd())
    convert_parquet_to_csv("../../data/raw/bdoss.parquet", path_csv)
    convert_parquet_to_csv("../../data/raw/crc.parquet", path_csv)
    convert_parquet_to_csv("../../data/raw/credscore.parquet", path_csv)
    convert_parquet_to_csv("../../data/raw/fama.parquet", path_csv)