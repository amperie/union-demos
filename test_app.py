import union
from union import Resources

ClsModelResults = union.Artifact(name="pablo_classifier_model_results_fd")

image = union.ImageSpec(
    name="streamlit-app",
    packages=[
        "union-runtime>=0.1.10", "streamlit==1.41.1",
        "scikit-learn", "datasets", "pandas", "union", "flytekit"
        ],
    registry="pablounionai",
)

app = union.app.App(
    name="streamlit-test-app",
    inputs=[
        union.app.Input(
            name="pablo_classifier_model_results",
            value=ClsModelResults.query(),
            download=True,
            env_var="CLS_MODEL_RESULTS",
        ),
    ],
    container_image=image,
    args="streamlit run test_app_main.py --server.port 8080",
    port=8080,
    include=["."],
    limits=Resources(cpu="1", mem="1Gi"),
)
