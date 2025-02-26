import union
from union.artifacts import Artifact
from demos.tasks.dataclass_defs import HpoResults
from app_threshold_definition import app_threshold
from union import Secret
from union.remote import UnionRemote

image = union.ImageSpec(
    name="app-wf",
    packages=[
        "scikit-learn", "pandas", "union>=0.1.145", "flytekit",
        ],
    registry="pablounionai",
)

ClsModelResults = Artifact(
    name="pablo_classifier_model_results"
)
query = ClsModelResults.query()


@union.task(
    container_image=image,
    secret_requets=[
        Secret(
            key="pablo-api-key", env_var="UNION_API_KEY",
            mount_requirement=Secret.MountType.ENV_VAR)]
)
def tsk_deploy_app_threshold():
    print("Creating AppRemote")
    # app_remote = AppRemote(project="flytesnacks", domain="development")
    print("Creating App")
    # app_remote.create_or_update(app_threshold)
    remote = UnionRemote()
    remote.deploy_app(app_threshold)


@union.workflow
def wf_deploy_app_threshold(art_query: HpoResults = query):
    tsk_deploy_app_threshold()
