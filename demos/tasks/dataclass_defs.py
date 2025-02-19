from sklearn.base import BaseEstimator
from dataclasses import dataclass
from flytekit import FlyteFile, FlyteDirectory
from flytekit.types.structured import StructuredDataset
import pandas as pd
import pickle
from joblib import dump, load
import os


@dataclass
class DataFrameDict():
    _dataframes = {}

    def get(self, key):
        # return self._dataframes[key].open(pd.DataFrame).all()
        return self._dataframes[key].dataframe

    def add(self, key, value):
        if not isinstance(value, pd.DataFrame)\
                and not isinstance(value, pd.Series):
            raise TypeError("Item must be a pandas dataframe/series")
        print(value)
        sd = StructuredDataset(dataframe=value)
        self._dataframes[key] = sd


@dataclass
class DataSplits():
    X_train: StructuredDataset
    X_test: StructuredDataset
    y_train: pd.Series
    y_test: pd.Series


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
    _data: FlyteFile = None
    _model: FlyteFile = None

    @property
    def model(self) -> BaseEstimator:
        return self._model

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    def serialize(self):
        filename = "pkld_model.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        return FlyteFile(filename)

    def deserialize(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    @model.setter
    def model(self, model: BaseEstimator):
        if model is None:
            return
        dump(model, 'model.joblib')
        f = FlyteFile('model.joblib')
        self._model = f

    @model.getter
    def model(self):
        if self._model is None:
            return None
        model = load(self._model)
        return model

    @data.getter
    def data(self):
        if self._data is None:
            return None
        return pd.read_parquet(self._data)

    @data.setter
    def data(self, data: pd.DataFrame):
        data.to_parquet('data.parquet')
        self._data = FlyteFile('data.parquet')

    def to_flytedir(self) -> FlyteDirectory:
        folder = "tmpStorage"
        if not os.path.exists(folder):
            os.makedirs(folder)

        vars = {
            "acc": self.acc,
            "hp": self.hp
        }

        with open(os.path.join(folder, 'vars.pkl'), 'wb') as handle:
            pickle.dump(vars, handle, protocol=pickle.HIGHEST_PROTOCOL)

        model = self.model
        if model is not None:
            dump(model, os.path.join(folder, 'model.joblib'))

        data = self.data
        if data is not None:
            data.to_parquet(os.path.join(folder, 'data.parquet'))

        return FlyteDirectory(folder)

    @staticmethod
    def from_flytedir(flytedir: FlyteDirectory):
        with open(os.path.join(flytedir, 'vars.pkl'), 'rb') as handle:
            vars = pickle.load(handle)

        acc = vars['acc']
        hp = vars['hp']
        model = None
        if os.path.exists(os.path.join(flytedir, 'model.joblib')):
            model = load(os.path.join(flytedir, 'model.joblib'))

        data = None
        if os.path.exists(os.path.join(flytedir, 'data.parquet')):
            data = pd.read_parquet(os.path.join(flytedir, 'data.parquet'))
        retVal = HpoResults(hp, acc)
        retVal.model = model
        retVal.data = data
        return retVal
