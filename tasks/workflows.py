import union
import pandas as pd
from typing import Tuple
from functools import partial
from sklearn.base import BaseEstimator
from .get_data_hf import get_data_hf
from .featurize_data import featurize
from .get_training_split import get_training_split
from .train_model import train_classifier
from .train_model_hpo import train_classifier_hpo, SearchSpace
from .train_model_hpo import create_search_grid, Hyperparameters
from .train_model_hpo import HpoResults

image = union.ImageSpec(
    name="data",
    packages=["datasets", "pandas"]
)

image_sklearn = union.ImageSpec(
    name="data",
    packages=["scikit-learn", "pandas"]
)


@union.task(
    container_image=image,
    cache=False,
    cache_version="1",
)
def tsk_get_data_hf(cfg: dict) -> pd.DataFrame:
    return get_data_hf(cfg)


@union.task(
    container_image=image,
    cache=False,
    cache_version="1",
)
def tsk_featurize(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    return featurize(cfg, df)


@union.task(
    container_image=image_sklearn,
    cache=False,
    cache_version="1",
)
def tsk_get_training_split(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    retVal = get_training_split(cfg, df)
    print(retVal)
    return retVal


@union.task(
    container_image=image_sklearn,
    cache=False,
    cache_version="1",
)
def tsk_train_model(cfg: dict,
                    X_train: pd.DataFrame,
                    X_test: pd.DataFrame,
                    y_train: pd.Series,
                    y_test: pd.Series) -> BaseEstimator:
    retVal = train_classifier(cfg, X_train, X_test, y_train, y_test)
    return retVal


@union.task(
    container_image=image_sklearn,
    cache=False,
    cache_version="1",
)
def tsk_train_model_hpo(cfg: dict,
                        hp: Hyperparameters,
                        X_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        y_train: pd.Series,
                        y_test: pd.Series) -> HpoResults:
    results = train_classifier_hpo(
        cfg, hp, X_train, X_test, y_train, y_test)
    return results


@union.workflow
def wf():
    cfg = {"target_column": "credit.policy"}

    df = get_data_hf(cfg)
    fdf = featurize(cfg, df)
    splits = get_training_split(cfg, fdf)
    X_train, X_test, y_train, y_test = splits
    # model = tsk_train_model(cfg, X_train, X_test, y_train, y_test)

    ss = SearchSpace(
        max_depth=[2, 10, 20],
        max_leaf_nodes=[2, 10, 20],
        n_estimators=[10, 20, 30])
    grid = create_search_grid(ss)

    pf = partial(
        tsk_train_model_hpo,
        cfg=cfg,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )
    results = union.map_task(pf)(hp=grid)
    print(results)


if __name__ == "__main__":
    wf()
