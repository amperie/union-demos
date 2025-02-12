import union
from union import Artifact
from demos.tasks.dataclass_defs import HpoResults


# Configuration Parameters
enable_data_cache = False
enable_model_cache = True
cfg = {"target_column": "credit.policy"}
cfg = {}


ClsModelResults = Artifact(
    name="pablo_classifier_model_results"
)

query = ClsModelResults.query()

image = union.ImageSpec(
    base_image="ghcr.io/unionai-oss/union:py3.10-latest",
    name="data",
    registry="pablounionai",
    packages=["scikit-learn", "datasets", "pandas", "union"]
)


@union.task(
    container_image=image,
    cache=enable_data_cache,
    cache_version="1",)
def get_artifact(art: HpoResults) -> HpoResults:

    # art = remote.get_artifact(query=art_query.to_flyte_idl())
    print(art.acc)
    print(type(art.model))
    print(art.model)
    return art


@union.workflow
def pablo_launch_app_wf(art_query: HpoResults = query):
    art = get_artifact(art_query)
    print(art.model)
    return art
