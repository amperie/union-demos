from copy import deepcopy
import union
from union.artifacts import Artifact

from app_threshold_definition import app_threshold
from flytekit import Secret, task, FlyteDirectory

image = union.ImageSpec(
    name="app-wf",
    packages=[
        "scikit-learn",
        "datasets",
        "pandas",
        "union>=0.1.146",
        "flytekit",
    ],
    builder="union",
)

ClsModelResults = Artifact(name="pablo_classifier_model_results_fd")


@task(
    container_image=image,
    secret_requests=[
        Secret(
            key="pablo-api-key",
            mount_requirement=Secret.MountType.ENV_VAR,
            env_var="UNION_API_KEY",
        )
    ],
)
def tsk_deploy_app_threshold(art_results: FlyteDirectory) -> str:
    app_remote = union.UnionRemote(
        default_project="flytesnacks", default_domain="development"
    )
    print("Creating App")
    app_threshold_copy = deepcopy(app_threshold)
    app_threshold_copy.env = {"CLS_MODEL_RESULTS": art_results.remote_source}
    app_idl = app_remote.deploy_app(
        app_threshold_copy, project="flytesnacks", domain="development"
    )
    return app_idl.status.ingress.public_url


@union.workflow
def wf_deploy_app_threshold(art_results: FlyteDirectory = ClsModelResults.query()):
    tsk_deploy_app_threshold(art_results=art_results)
