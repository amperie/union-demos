from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple


def get_training_split(cfg: dict, df: pd.DataFrame)\
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "test_size" in cfg:
        test_size = cfg['test_size']
    else:
        test_size = .3
    target_column = cfg['target_column']

    y = df[target_column]
    X = df.drop([target_column], axis=1)

    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=test_size)
    return (X_train, X_test, y_train, y_test)
