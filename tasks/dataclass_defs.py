from sklearn.base import BaseEstimator
from dataclasses import dataclass
from flytekit import FlyteFile
from joblib import dump
from flytekit.types.structured import StructuredDataset
import pandas as pd
from mashumaro.mixins.json import DataClassJSONMixin


@dataclass
class DataFrameDict(DataClassJSONMixin):
    _dataframes = {}

    def __getitem__(self, key):
        # return self._dataframes[key].open(pd.DataFrame).all()
        return self._dataframes[key].dataframe

    def __setitem__(self, key, value):
        if not isinstance(value, pd.DataFrame)\
                and not isinstance(value, pd.Series):
            raise TypeError("Item must be a pandas dataframe/series")
        print(value)
        sd = StructuredDataset(dataframe=value)
        self._dataframes[key] = sd


@dataclass
class Hyperparameters:
    max_depth: int
    max_leaf_nodes: int
    n_estimators: int


@dataclass
class SearchSpace:
    max_depth: list[int]
    max_leaf_nodes: list[int]
    n_estimators: list[int]


@dataclass
class HpoResults:
    hp: Hyperparameters
    acc: float
    _model: FlyteFile = None

    @property
    def model(self) -> BaseEstimator:
        return self._model

    @model.setter
    def model(self, model: BaseEstimator):
        if model is None:
            return
        dump(model, 'model.joblib')
        f = FlyteFile('model.joblib')
        self._model = f
