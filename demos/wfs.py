import union
from union import Resources
from union.actor import ActorEnvironment
import pandas as pd
from demos.tasks.get_data import get_data_hf, get_data_databricks
from demos.tasks.featurize_data import featurize
from demos.tasks.get_training_split import get_training_split
from demos.tasks.train_model_hpo import train_classifier_hpo
from demos.tasks.train_model_hpo import create_search_grid
from demos.tasks.get_best import get_best
from demos.tasks.dataclass_defs import HpoResults
from demos.tasks.dataclass_defs import Hyperparameters, SearchSpace
from demos.tasks.featurize_data import featurize_spark
from demos.tasks.featurize_data import spark_task_config, spark_image_spec
from demos.tasks.featurize_data import databricks_task_config
from demos.tasks.featurize_data import databricks_image_spec
from union import Artifact, Secret
from typing_extensions import Annotated
from flytekit import FlyteDirectory


# Configuration Parameters
enable_data_cache = False
enable_model_cache = False
cache_version = "3"
cfg = {"target_column": "credit_policy"}
cfg = {}

ClsModelResults = Artifact(
    name="pablo_classifier_model_results_fd"
)

image = union.ImageSpec(
    builder="envd",
    base_image="ghcr.io/unionai-oss/union:py3.10-latest",
    name="data",
    registry="pablounionai",
    packages=["scikit-learn", "datasets", "pandas",
              "union", "flytekitplugins-spark", "delta-sharing"],
)

hpo_actor = ActorEnvironment(
    name="hpo-actor",
    replica_count=3,
    ttl_seconds=30,
    container_image=image,
    requests=Resources(
        cpu="2",
        mem="1Gi",
    ),
)


@union.task(
    container_image=image,
    cache=enable_data_cache,
    cache_version=cache_version,
)
def tsk_get_data_hf() -> pd.DataFrame:
    return get_data_hf()


@union.task(
    container_image=image,
    cache=enable_data_cache,
    cache_version=cache_version,
    secret_requests=[
        Secret(
            key="delta_sharing_creds",
            mount_requirement=Secret.MountType.FILE),
    ]
)
def tsk_get_data_databricks() -> pd.DataFrame:
    creds_file_path =\
        union.current_context().secrets.get_secrets_file("delta_sharing_creds")

    return get_data_databricks(creds_file_path)


@union.task(
    container_image=image,
    cache=enable_data_cache,
    cache_version=cache_version,
)
def tsk_featurize(df: pd.DataFrame) -> pd.DataFrame:
    return featurize(df)


@union.task(
    task_config=spark_task_config,
    container_image=spark_image_spec,
)
def tsk_featurize_spark(df: pd.DataFrame) -> pd.DataFrame:
    return featurize_spark(df)


@union.task(
    task_config=databricks_task_config,
    container_image=databricks_image_spec,
)
def tsk_featurize_databricks(df: pd.DataFrame) -> pd.DataFrame:
    return featurize_spark(df)


@union.task(
    container_image=image,
    requests=Resources(mem="6Gi")
)
def tsk_failure(df: pd.DataFrame, fd: FlyteDirectory) -> None:
    fail = True
    if fail:
        raise Exception("Failure")


@hpo_actor.task
def tsk_train_model_hpo_df(
        hp: Hyperparameters,
        df: pd.DataFrame) -> HpoResults:
    splits = get_training_split(df)
    results = train_classifier_hpo(
        hp, splits)
    results.data = df
    return results


@union.dynamic(
    container_image=image,
    cache=enable_model_cache,
    cache_version=cache_version)
def tsk_hyperparameter_optimization(
        grid: list[Hyperparameters],
        df: pd.DataFrame) -> list[HpoResults]:

    models = []
    for hp in grid:
        res = tsk_train_model_hpo_df(hp, df)
        models.append(res)
    return models


@hpo_actor.task(
    cache=enable_model_cache,
    cache_version=cache_version)
def tsk_get_best(results: list[HpoResults]) -> HpoResults:
    return get_best(results)


@hpo_actor.task
def tsk_register_fd_artifact(results: HpoResults)\
        -> Annotated[FlyteDirectory, ClsModelResults]:
    return ClsModelResults.create_from(results.to_flytedir())


@union.workflow
def pablo_wf():

    df = tsk_get_data_hf()
    # fdf = tsk_featurize_databricks(df)
    # fdf = tsk_featurize_spark(df)
    fdf = tsk_featurize(df)

    ss = SearchSpace(
        max_depth=[10, 20],
        max_leaf_nodes=[10, 20],
        n_estimators=[10, 20])

    grid = create_search_grid(ss)
    models = tsk_hyperparameter_optimization(grid, fdf)

    best = tsk_get_best(models)
    logged_artifact = tsk_register_fd_artifact(best)
    # tsk_failure(fdf, logged_artifact)
    print(logged_artifact)

    return None


if __name__ == "__main__":
    pablo_wf()
