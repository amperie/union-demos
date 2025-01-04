from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from typing import Tuple
from sklearn.base import BaseEstimator
import pandas as pd
from dataclasses import dataclass
from itertools import product
from flytekit import FlyteFile


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
    model: FlyteFile

    @property
    def model(self) -> BaseEstimator:
        pass
        # deserialize the model from flytefile

    @model.setter
    def model(self, model: BaseEstimator):
        pass
        # serialize the model and put it in flytefile


def create_search_grid(searchspace: SearchSpace) -> list[Hyperparameters]:

    keys = vars(searchspace).keys()
    values = [getattr(searchspace, key) for key in keys]

    grid = [Hyperparameters(**dict(zip(keys, combination)))
            for combination in product(*values)]

    return grid


def train_classifier_hpo(
        cfg: dict,
        hp: Hyperparameters,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series) -> Tuple[BaseEstimator, float]:

    clf = RandomForestClassifier(
        max_depth=hp.max_depth,
        max_leaf_nodes=hp.max_leaf_nodes,
        n_estimators=hp.n_estimators)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print("ACCURACY OF THE MODEL:", acc)
    return HpoResults(hp, acc, clf)
