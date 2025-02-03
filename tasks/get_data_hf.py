from datasets import load_dataset
import pandas as pd


def get_data_hf() -> pd.DataFrame:
    ds = load_dataset('AnguloM/loan_data')
    df = ds['train'].to_pandas()
    return df
