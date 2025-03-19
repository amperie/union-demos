import pandas as pd
import union
import random
import chromadb
from vllm import LLM, SamplingParams
from langchain_core.prompts.prompt import PromptTemplate


def get_rag_data_databricks(profile_file) -> pd.DataFrame:

    import delta_sharing
    table_url = profile_file + "#angulo.demo.synthetic_knowledge_items"
    return delta_sharing.load_as_pandas(table_url)


def build_vector_db(df: pd.DataFrame) -> union.FlyteDirectory:

    vdb = chromadb.PersistentClient(path="vector_db")
    collection = vdb.get_or_create_collection("rag_data")
    ids = [str(elm) for elm in df.index.tolist()]
    collection.add(documents=df['ki_text'].tolist(), ids=ids)
    return union.FlyteDirectory("vector_db")


def evaluate_rag_parameters(
        queries: list[str],
        vdb_dir: union.FlyteDirectory,
        llm_name: str,
        params: dict,
        prompt_template: str)\
            -> float:

    # Setup Vector DB from Chroma
    vdb = chromadb.PersistentClient(path=vdb_dir.path)
    collection = vdb.get_or_create_collection("rag_data")
    # Setup LLM to test
    llm = LLM(model=llm_name)
    sample_params = SamplingParams(**params)

    # Make prompts
    prompts = []
    for query in queries:
        context = collection.query(
            query_texts=query, n_results=1)
        context_text = context["documents"][0]
        prompt = prompt_template.format(context=context_text, question=query)
        prompts.append(prompt)

    # Run LLM
    outputs = llm.generate(prompts, sample_params)

    # Do evaluation of outputs
    # Just making up a metric here
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    made_up_metric = random.uniform(.7, .9)

    return made_up_metric
