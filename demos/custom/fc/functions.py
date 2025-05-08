from langchain_community.document_loaders import PyPDFLoader
import nltk
import os
from nltk.tokenize import sent_tokenize
from pinecone import Pinecone
import random


def init_environment(pc_key: str, index_name: str):
    nltk.download('punkt_tab')
    pc = Pinecone(api_key=pc_key)

    # Create a dense index with integrated embedding
    index_name = "pdfs"
    if not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": "llama-text-embed-v2",
                "field_map": {"text": "chunk_text"}
            }
        )
    return pc


def parse_pdf(pdf_path: str) -> list[str]:
    loader = PyPDFLoader(pdf_path)
    content = loader.load()[0].page_content
    sentences = sent_tokenize(content)
    return sentences


def remove_special_characters(file_path):
    file_path = os.path.basename(file_path)
    return "".join(char for char in file_path if char.isalnum())


def make_pc_records(sentences: list[str], pdf_path: str) -> list[dict]:
    records = []
    id_prefix = remove_special_characters(pdf_path)
    for i, val in enumerate(sentences):
        rec = {
            "_id": f"{id_prefix}_{i}",
            "chunk_text": val,
        }
        records.append(rec)
    return records


def upsert_to_pinecone(
        pc: Pinecone, index_name: str, records: str, namespace: str):
    dense_index = pc.Index(index_name)
    dense_index.upsert_records(namespace, records)


# Evaluation
def evaluate_query(pc_index, version: str, query: str) -> str:
    results = pc_index.search(
        namespace=version,
        query={
            "top_k": 3,
            "inputs": {
                'text': query
            }
        }
    )
    avg_score = 0
    for hit in results['result']['hits']:
        # print(f"id: {hit['_id']}, score: {round(hit['_score'], 2)}")
        avg_score += round(hit['_score'], 2)
    avg_score = avg_score / len(results['result']['hits'])
    avg_score *= random.random()
    if avg_score < .05: return "TN"
    if avg_score < .11: return "FN"
    if avg_score < .17: return "FP"
    return "TP"


def get_s3_files(bucket_name, prefix='', download_dir='downloads'):
    # Create anonymous S3 client
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    os.makedirs(download_dir, exist_ok=True)

    for page in page_iterator:
        for obj in page.get('Contents', []):
            key = obj['Key']
            dest_path = os.path.join(download_dir, key)

            # Create subdirectories as needed
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            print(f"Downloading: {key} -> {dest_path}")
            s3.download_file(bucket_name, key, dest_path)
