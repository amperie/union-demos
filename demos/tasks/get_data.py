from datasets import load_dataset
import pandas as pd
from sys import platform


def get_data_hf() -> pd.DataFrame:
    ds = load_dataset('AnguloM/loan_data')
    df = ds['train'].to_pandas()
    # Fix column names to not include '.'
    df.columns = list(map(lambda col: col.replace('.', '_'), df.columns))
    df = df.reindex(sorted(df.columns), axis=1)
    return df


def get_data_databricks(profile_file) -> pd.DataFrame:

    import delta_sharing
    if platform != "linux":
        profile_file = "config.delta.share"
    table_url = profile_file + "#angulo.demo.angulo_m_loan_data"
    df = delta_sharing.load_as_pandas(table_url)
    df.columns = list(map(lambda col: col.replace('.', '_'), df.columns))
    df = df.reindex(sorted(df.columns), axis=1)
    return df
