import union
from union import Resources
from union.actor import ActorEnvironment
import pandas as pd
from demos.tasks.get_data_hf import get_data_hf
from demos.tasks.featurize_data import featurize
from demos.tasks.get_training_split import get_training_split
from demos.tasks.train_model_hpo import train_classifier_hpo
from demos.tasks.train_model_hpo import create_search_grid
from demos.tasks.get_best import get_best
from demos.tasks.dataclass_defs import HpoResults
from demos.tasks.dataclass_defs import Hyperparameters, SearchSpace
from union import Artifact
from union.artifacts import OnArtifact
from typing_extensions import Annotated
from demos.tasks.automation_wfs import model_automation_wf


# Configuration Parameters
enable_data_cache = True
enable_model_cache = False
cfg = {"target_column": "credit.policy"}
cfg = {}

ClsModelResults = Artifact(
    name="pablo_classifier_model_results"
)

image = union.ImageSpec(
    base_image="ghcr.io/unionai-oss/union:py3.10-latest",
    name="data",
    registry="pablounionai",
    packages=["scikit-learn", "datasets", "pandas", "union"]
)

hpo_actor = ActorEnvironment(
    name="hpo-actor",
    replica_count=3,
    ttl_seconds=30,
    container_image=image,
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
    cache=enable_model_cache,
    cache_version="1")
def tsk_hyperparameter_optimization(
        grid: list[Hyperparameters],
        df: pd.DataFrame) -> list[HpoResults]:

    models = []
    for hp in grid:
        res = tsk_train_model_hpo_df(hp, df)
        models.append(res)
    return models


@hpo_actor.task
def tsk_get_best(results: list[HpoResults]) -> HpoResults:
    return get_best(results)


@hpo_actor.task
def tsk_register_artifact(results: HpoResults)\
        -> Annotated[HpoResults, ClsModelResults]:
    return ClsModelResults.create_from(results.serialize())


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

    best = tsk_get_best(models)
    logged_artifact = tsk_register_artifact(best)

    print(logged_artifact)


on_model_results = OnArtifact(
    trigger_on=ClsModelResults,
)

on_model_results_triggered = union.LaunchPlan.create(
    "pablo_model_automation_wf",
    model_automation_wf,
    trigger=on_model_results
)

if __name__ == "__main__":
    pablo_wf()
