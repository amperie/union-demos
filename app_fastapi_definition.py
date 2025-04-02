import union
from union import Resources

ClsModelResults = union.Artifact(name="pablo_classifier_model_results_fd")

image = union.ImageSpec(
    name="fastapi-app",
    packages=[
        "union-runtime>=0.1.10", "fastapi[standard]", "flytekitplugins-spark",
        "scikit-learn", "datasets", "pandas", "union>=0.1.145",
        "flytekit>=1.15.0",
        ],
    registry="pablounionai",
)

app_threshold = union.app.App(
    name="pablo-fastapi-app",
    inputs=[
        union.app.Input(
            name="pablo_classifier_model_results_fd",
            value=ClsModelResults.query(),
            download=True,
            env_var="CLS_MODEL_RESULTS",
        ),
    ],
    container_image=image,
    args="fastapi run app_fastapi.py --port 8080",
    port=8080,
    include=["."],
    limits=Resources(cpu="1", mem="1Gi"),
)
