import union
from union import Resources
from union.remote._app_remote import AppRemote

ClsModelResults = union.Artifact(name="pablo_classifier_model_results_fd")

image = union.ImageSpec(
    name="streamlit-app",
    packages=[
        "union-runtime>=0.1.10", "streamlit==1.41.1",
        "scikit-learn", "datasets", "pandas", "union", "flytekit",
        "seaborn", "matplotlib", "ing-theme-matplotlib"
        ],
    registry="pablounionai",
)

app_threshold = union.app.App(
    name="pablo-threshold-app",
    inputs=[
        union.app.Input(
            name="pablo_classifier_model_results_fd",
            value=ClsModelResults.query(),
            download=True,
            env_var="CLS_MODEL_RESULTS",
        ),
    ],
    container_image=image,
    args="streamlit run app_threshold.py --server.port 8080",
    port=8080,
    include=["."],
    limits=Resources(cpu="1", mem="1Gi"),
)


@union.task(
    container_image=image,
)
def tsk_deploy_app_threshold():
    print("Creating AppRemote")
    app_remote = AppRemote(project="flytesnacks", domain="development")
    print("Creating App")
    deployed_app = app_remote.create_or_update(app_threshold)
    print(f"Deployed app: {deployed_app}")


@union.workflow
def wf_deploy_app_threshold():
    tsk_deploy_app_threshold()
