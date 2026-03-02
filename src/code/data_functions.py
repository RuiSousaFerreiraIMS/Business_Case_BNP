import pandas as pd
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

def load(file_name: str, parse_dates=None) -> pd.DataFrame:
    df = pd.read_csv(file_name, parse_dates=parse_dates)
    return df