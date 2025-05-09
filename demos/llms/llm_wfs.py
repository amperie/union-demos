import union
from functions import get_rag_data_databricks
from functions import build_vector_db
from functions import evaluate_rag_parameters
import pandas as pd
from sys import platform
from typing_extensions import Annotated
from langchain_core.prompts.prompt import PromptTemplate

enable_data_cache = True
data_cache_version = "2"

image = union.ImageSpec(
    builder="envd",
    base_image="ghcr.io/unionai-oss/union:py3.11-latest",
    name="llm",
    registry="pablounionai",
    packages=["chromadb", "pandas", "vllm", "langchain",
              "union", "delta-sharing", "tqdm"],
)


vdb_artifact = union.Artifact(
    name="pablo_chroma_vector_db",
)


@union.task(
    container_image=image,
    cache=enable_data_cache,
    cache_version=data_cache_version,
    secret_requests=[
        union.Secret(
            key="delta_sharing_creds",
            mount_requirement=union.Secret.MountType.FILE),
    ]
)
def tsk_get_data() -> pd.DataFrame:

    if platform != "linux":
        profile_file = "../../config.delta.share"
    else:
        profile_file =\
            union.current_context().secrets\
            .get_secrets_file("delta_sharing_creds")
    return get_rag_data_databricks(profile_file)


@union.task(
    container_image=image,
    cache=enable_data_cache,
    cache_version=data_cache_version,
    requests=union.Resources(mem="4Gi")
)
def tsk_build_vector_db(df: pd.DataFrame) -> union.FlyteDirectory:
    return build_vector_db(df)


@union.task(
    container_image=image,
    cache=enable_data_cache,
    cache_version=data_cache_version,
)
def tsk_register_vdb_artifact(vdb_dir: union.FlyteDirectory)\
        -> Annotated[union.FlyteDirectory, vdb_artifact]:
    return vdb_artifact.create_from(vdb_dir)


@union.task(
    container_image=image,
    requests=union.Resources(mem="2Gi")
)
def tsk_evaluate_rag_parameters(
        queries: list[str],
        vdb_dir: union.FlyteDirectory,
        llm_name: str,
        params: dict,
        prompt_template: PromptTemplate)\
            -> float:

    return evaluate_rag_parameters(
        queries, vdb_dir, llm_name, params, prompt_template)


@union.dynamic(
    container_image=image,
    enable_deck=True,
    requests=union.Resources(mem="2Gi")
)
def tsk_evaluate_rag(queries: list[str],
                     vdb_dir: union.FlyteDirectory,
                     llm_name: str,
                     params_list: list[dict],
                     prompt_template: PromptTemplate)\
        -> list[float]:
    results = []
    for params in params_list:
        res = tsk_evaluate_rag_parameters(
            queries, vdb_dir, llm_name, params, prompt_template
        )
        results.append(res)
    union.Deck("RAG Evaluation", "Best Result: " + str(results))
    return results


@union.task(
    container_image=image,
    requests=union.Resources(mem="16Gi", cpu="4", gpu="1"),
    enable_deck=True
)
def tsk_evaluate_rag_parameter(
        queries: list[str],
        vdb_dir: union.FlyteDirectory)\
            -> float:

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="Answer the question using the context.\n"
        "Question: {question}\nContext: {context}"
    )
    params = {"temperature": 0.1, "top_p": 0.9}

    made_up_metric = evaluate_rag_parameters(
        queries, vdb_dir,
        "facebook/opt-125m",
        params,
        prompt_template)

    union.Deck("RAG Evaluation", "Best Result: " + str(made_up_metric))
    return made_up_metric


@union.workflow
def pablo_rag_vdb_wf() -> union.FlyteDirectory:

    # Get data
    df = tsk_get_data()

    # Build vector db from data and register as an artifact
    vdb = tsk_build_vector_db(df)
    vdb_artifact = tsk_register_vdb_artifact(vdb)

    # Evaluate single RAG parameters
    result_single = tsk_evaluate_rag_parameter(
        queries=["start webex?", "restart computer?"],
        vdb_dir=vdb_artifact
    )

    return vdb
