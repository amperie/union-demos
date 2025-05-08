import union
import os
from functions import init_environment, make_pc_records
from functions import upsert_to_pinecone, parse_pdf
from functions import evaluate_query, get_s3_files
from pinecone import Pinecone

enable_data_cache = False
data_cache_version = "2"
# pc_key = "pcsk_R82ES_9iKGuZ1Mqe2HEowfpMqoNJ1pbjDNnqctbwDyumQBLopKGU1riMCCPWRXkwqxfNH"
index_name = "pdfs"

image = union.ImageSpec(
    builder="envd",
    base_image="ghcr.io/unionai-oss/union:py3.11-latest",
    name="llm",
    registry="pablounionai",
    packages=["langchain_community", "pypdf",
              "nltk", "pinecone", "union"],
)


@union.task(
    container_image=image,
)
def tsk_pre():
    print(os.listdir("Data"))


@union.task(
    container_image=image,
    cache=enable_data_cache,
    cache_version=data_cache_version,
)
def tsk_get_files() -> union.FlyteDirectory:
    retVal = union.FlyteDirectory("Data")
    return retVal


@union.task(
    container_image=image,
    cache=enable_data_cache,
    cache_version=data_cache_version,
)
def tsk_get_s3_files(bucket_name: str) -> union.FlyteDirectory:
    get_s3_files(bucket_name, download_dir="temp_data")
    return union.FlyteDirectory("temp_data")


@union.task(
    container_image=image,
    cache=enable_data_cache,
    cache_version=data_cache_version,
    secret_requests=[union.Secret(key="pablo_pc_key")]
)
def tsk_build_index(
        files: union.FlyteDirectory, version: str, index_name: str) -> bool:

    pc_key = union.current_context().secrets.get(key="pablo_pc_key")
    pc = init_environment(pc_key, index_name)

    for filename in os.listdir(files):
        print(f"Checking file: {filename}")
        if filename.endswith(".pdf"):
            filepath = os.path.join(files, filename)
            sentences = parse_pdf(filepath)
            records = make_pc_records(sentences, filepath)
            print(f"Upserting {len(records)} from {filepath}")
            upsert_to_pinecone(pc, index_name, records, version)

    return True


@union.task(
    container_image=image,
    cache=enable_data_cache,
    cache_version=data_cache_version,
    secret_requests=[union.Secret(key="pablo_pc_key")]
)
def tsk_evaluate(
        version: str, retVal: bool, index_name: str):

    eval_dataset = [
        "Explain a tariff",
        "What is the tariff on metals?",
        "What is the tariff on cultured pearls?",
        "what is the tariff on agricultural products?",
        "what is the tariff on chemicals?",
        "what is the tariff on metaldehyde?",
        "are tariffs good?",
        "are tariffs bad?"
    ]

    pc_key = union.current_context().secrets.get(key="pablo_pc_key")
    pc = Pinecone(api_key=pc_key)
    dense_index = pc.Index(index_name)
    eval_counts = {"TN": 1, "FN": 1, "FP": 1, "TP": 1}
    for q in eval_dataset:
        r = evaluate_query(dense_index, version, q)
        eval_counts[r] += 1

    print(eval_counts)
    precision = eval_counts["TP"]/(eval_counts["TP"]+eval_counts["FP"])
    recall = eval_counts["TP"]/(eval_counts["TP"]+eval_counts["FN"])
    f1 = (2*precision*recall)/(precision+recall)

    print(f"f1: {f1}\nprecision: {precision}\nrecall: {recall}")


@union.workflow
def pablo_fc_wf(version: str, bucket_name: str):
    tsk_pre()
    files = tsk_get_files()
    retVal = tsk_build_index(files, version, index_name)
    tsk_evaluate(version, retVal, index_name)


# pablo_fc_wf(version="test")
