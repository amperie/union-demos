from sklearn.model_selection import train_test_split
import pandas as pd
from demos.tasks.dataclass_defs import DataFrameDict
# from flytekit.types.structured import StructuredDataset


def get_training_split(df: pd.DataFrame)\
        -> DataFrameDict:
    test_size = .3
    target_column = "credit_policy"

    y = df[target_column]
    X = df.drop([target_column], axis=1)

    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=test_size, random_state=42)
    retVal = DataFrameDict()
    retVal.add("X_train", X_train)
    retVal.add("X_test", X_test)
    retVal.add("y_train", y_train)
    retVal.add("y_test", y_test)
    return retVal
