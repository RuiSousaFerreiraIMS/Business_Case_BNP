import pandas as pd
import os
from pathlib import Path


def convert_parquet_to_csv(parquet_path, output_path=None):
    """
    Converte um ficheiro Parquet para CSV.

    :param parquet_path: Caminho do ficheiro .parquet
    :param output_path: Caminho do ficheiro .csv (opcional)
    """

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Ficheiro não encontrado: {parquet_path}")

    # Se não for passado output_path, cria automaticamente com o mesmo nome

    output_path = Path(output_path)
    if output_path.is_dir():  # se for uma pasta, cria o ficheiro com o mesmo nome do parquet
        output_path = output_path / (Path(parquet_path).stem + ".csv")

    print(f"A converter: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    df.to_csv(output_path, index=False)

    print(f"Conversão concluída: {output_path}")