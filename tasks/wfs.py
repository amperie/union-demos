import union
from union import Resources
from union.actor import ActorEnvironment
import pandas as pd
from get_data_hf import get_data_hf
from featurize_data import featurize
from get_training_split import get_training_split
from train_model_hpo import train_classifier_hpo
from train_model_hpo import create_search_grid
from dataclass_defs import HpoResults
from dataclass_defs import Hyperparameters, SearchSpace


# Configuration Parameters
enable_data_cache = True
cfg = {"target_column": "credit.policy"}
cfg = {}

image = union.ImageSpec(
    base_image="ghcr.io/unionai-oss/union:py3.11-0.1.121",
    name="data",
    registry="pablounionai",
    packages=["scikit-learn", "datasets", "pandas", "union"]
)

hpo_actor = ActorEnvironment(
    name="hpo-actor",
    replica_count=3,
    ttl_seconds=30,
    requests=Resources(
        cpu="2",
        mem="300Mi",
    ),
)


@union.task(
    container_image=image,
    cache=enable_data_cache,
    cache_version="1",
)
def tsk_get_data_hf() -> pd.DataFrame:
    return get_data_hf()


@union.task(
    container_image=image,
    cache=enable_data_cache,
    cache_version="1",
)
def tsk_featurize(df: pd.DataFrame) -> pd.DataFrame:
    return featurize(df)


# @union.task()()
    # cache=False,
    # cache_version="1",
@hpo_actor.task
def tsk_train_model_hpo_df(
        hp: Hyperparameters,
        df: pd.DataFrame) -> HpoResults:
    splits = get_training_split(df)
    results = train_classifier_hpo(
        hp, splits)
    return results


@union.dynamic(
    container_image=image,
    cache=False,
    cache_version="1")
def tsk_hyperparameter_optimization(
        grid: list[Hyperparameters],
        df: pd.DataFrame) -> list[HpoResults]:

    models = []
    for hp in grid:
        res = tsk_train_model_hpo_df(hp, df)
        models.append(res)
    return models


@union.workflow
def pablo_wf():

    df = tsk_get_data_hf()
    fdf = tsk_featurize(df)

    ss = SearchSpace(
        max_depth=[10, 20],
        max_leaf_nodes=[10, 20],
        n_estimators=[10, 20])

    grid = create_search_grid(ss)
    models = tsk_hyperparameter_optimization(grid, fdf)

    print(models)


if __name__ == "__main__":
    pablo_wf()
