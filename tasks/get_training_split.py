from sklearn.model_selection import train_test_split
import pandas as pd
from dataclass_defs import DataFrameDict


def get_training_split(cfg: dict, df: pd.DataFrame)\
        -> DataFrameDict:
    if "test_size" in cfg:
        test_size = cfg['test_size']
    else:
        test_size = .3
    target_column = cfg['target_column']

    y = df[target_column]
    X = df.drop([target_column], axis=1)

    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=test_size)
    retVal = DataFrameDict()
    retVal['X_train'] = X_train
    retVal['X_test'] = X_test
    retVal['y_train'] = y_train
    retVal['y_test'] = y_test
    return retVal
