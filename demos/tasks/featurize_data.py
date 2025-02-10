import pandas as pd


def featurize(df: pd.DataFrame) -> pd.DataFrame:

    df_encoded = pd.get_dummies(df, columns=['purpose'])
    return df_encoded
