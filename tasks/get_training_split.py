from sklearn.model_selection import train_test_split
import pandas as pd
from dataclass_defs import DataSplits
from flytekit.types.structured import StructuredDataset


def get_training_split(cfg: dict, df: pd.DataFrame)\
        -> DataSplits:
    if "test_size" in cfg:
        test_size = cfg['test_size']
    else:
        test_size = .3
    target_column = cfg['target_column']

    y = df[target_column]
    X = df.drop([target_column], axis=1)

    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=test_size)
    retVal = DataSplits(
        StructuredDataset(dataframe=X_train),
        StructuredDataset(dataframe=X_test),
        StructuredDataset(dataframe=y_train), 
        StructuredDataset(dataframe=y_test))
    return retVal
