import union
from union import Artifact
from flytekit import FlyteDirectory
from demos.tasks.dataclass_defs import HpoResults


# Configuration Parameters
enable_data_cache = False
enable_model_cache = True
cfg = {"target_column": "credit.policy"}
cfg = {}


ClsModelResults = Artifact(
    name="pablo_classifier_model_results_fd"
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
def get_artifact(art: FlyteDirectory) -> HpoResults:

    # art = remote.get_artifact(query=art_query.to_flyte_idl())
    art = HpoResults.from_flytedir(art)
    print(art.acc)
    print(art.hp)
    print(art.model)
    return art


@union.workflow
def pablo_launch_app_wf(art_query: FlyteDirectory = query):
    print(query)
    print(art_query)
    art = get_artifact(art_query)
    print(art.model)
    return art
