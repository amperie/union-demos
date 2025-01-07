import union
import pandas as pd
from functools import partial
from sklearn.base import BaseEstimator
from get_data_hf import get_data_hf
from featurize_data import featurize
from get_training_split import get_training_split
from train_model import train_classifier
from train_model_hpo import train_classifier_hpo
from train_model_hpo import create_search_grid
from dataclass_defs import DataFrameDict, HpoResults
from dataclass_defs import Hyperparameters, SearchSpace


# Configuration Parameters
enable_data_cache = False

image = union.ImageSpec(
    name="data",
    registry="pablounionai",
    packages=["scikit-learn", "datasets", "pandas", "union"]
)


@union.task(
    container_image=image,
    cache=enable_data_cache,
    cache_version="1",
)
def tsk_get_data_hf(cfg: dict) -> pd.DataFrame:
    return get_data_hf(cfg)


@union.task(
    container_image=image,
    cache=enable_data_cache,
    cache_version="1",
)
def tsk_featurize(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    return featurize(cfg, df)


@union.task(
    container_image=image,
    cache=enable_data_cache,
    cache_version="1",
)
def tsk_get_training_split(cfg: dict, df: pd.DataFrame) -> DataFrameDict:
    retVal = get_training_split(cfg, df)
    return retVal


@union.task(
    container_image=image,
    cache=False,
    cache_version="1",
)
def tsk_train_model(cfg: dict,
                    splits: DataFrameDict) -> BaseEstimator:
    X_train = splits.X_train
    X_test = splits.X_test
    y_train = splits.y_train
    y_test = splits.y_test
    retVal = train_classifier(cfg, X_train, X_test, y_train, y_test)
    return retVal


@union.task(
    container_image=image,
    cache=False,
    cache_version="1",
)
def tsk_train_model_hpo(cfg: dict,
                        hp: Hyperparameters,
                        splits: DataFrameDict) -> HpoResults:
    results = train_classifier_hpo(
        cfg, hp, splits)
    return results


@union.task(
    container_image=image,
    cache=False,
    cache_version="1",
)
def tsk_train_model_hpo_df(
        cfg: dict,
        hp: Hyperparameters,
        df: pd.DataFrame) -> HpoResults:
    splits = get_training_split(cfg, df)
    results = train_classifier_hpo(
        cfg, hp, splits)
    return results


@union.workflow
def pablo_wf():
    cfg = {"target_column": "credit.policy"}

    df = tsk_get_data_hf(cfg)
    fdf = tsk_featurize(cfg, df)
    splits = tsk_get_training_split(cfg, fdf)

    ss = SearchSpace(
        max_depth=[2, 10, 20],
        max_leaf_nodes=[2, 10, 20],
        n_estimators=[10, 20, 30])
    grid = create_search_grid(ss)

    pf = partial(
        tsk_train_model_hpo_df,
        cfg=cfg,
        df=fdf
    )
    results = union.map_task(pf)(hp=grid)
    print(results)


if __name__ == "__main__":
    pablo_wf()
