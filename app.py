import json
from dotenv import load_dotenv
from pydantic import BaseModel
import streamlit as st
import pandas as pd  # type: ignore
import tiktoken
from config import MATCH_TEMPLATE, RETRIEVE_CHAIN_TEMPLATE, Configuration
from datetime import datetime

from evaluate import evaluate


class ValidationSample(BaseModel):
    query: str
    answer: str


class ValidationList(BaseModel):
    validations: list[ValidationSample]


def run_app():
    # data
    documents = None
    val_list = None

    st.title("LLM-chain-eval")

    # sidebar
    st.sidebar.header("Configuration")

    doc_upload = st.sidebar.file_uploader("Source text file.", type="txt")
    val_upload = st.sidebar.file_uploader("Validation json file", type="json")

    st.sidebar.subheader("Parameters")
    llm_model = st.sidebar.selectbox(
        "LLM Model",
        [
            Configuration.LlmModel.gpt_3_turbo.value,
            Configuration.LlmModel.google_flan_t5_large.value,
            Configuration.LlmModel.google_flan_t5_xl.value,
            Configuration.LlmModel.dolly_v2_3b.value,
        ],
        index=0
    )
    chunk_size = st.sidebar.slider("Chunk Size", 100, 3000, 1000, 100)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 1000, 0, 1)
    chain_type = st.sidebar.selectbox(
        "Chain Type",
        ["stuff", "map_reduce", "refine", "map_rerank"],
        index=0
    )
    emb_type = st.sidebar.selectbox(
        "Embedding Model",
        [
            Configuration.EmbeddingModel.gpt_embedding.value,
            Configuration.EmbeddingModel.huggingface_embedding.value,
        ],
        index=0
    )
    top_k_chunk = st.sidebar.slider("Top K Chunk", 1, 10, 2, 1)
    retrieve_chain_template = st.sidebar.text_area("Retrieve Chain template", RETRIEVE_CHAIN_TEMPLATE)
    match_template = st.sidebar.text_area("Match template", MATCH_TEMPLATE)

    config = Configuration(
        llm_model=llm_model,
        embedding_model=emb_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chain_type=chain_type,
        top_k_chunk=top_k_chunk,
        retrieve_chain_template=retrieve_chain_template,
        match_template=match_template,
    )

    if doc_upload is not None:
        documents = load_documents(doc_upload)
        st.subheader("Source documents")
        st.caption(documents[0].page_content[:200] + "...")
        char_count = sum([len(doc.page_content) for doc in documents])
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        token_count = sum([len(encoding.encode(doc.page_content)) for doc in documents])
        st.text(f"{char_count} characters, {token_count} tokens")
    if val_upload is not None:
        val_list = ValidationList.parse_raw(val_upload.read()).validations

    if val_list is not None:
        st.subheader("Validation datasets")
        st.dataframe(pd.DataFrame([v.dict() for v in val_list]))

    if documents is not None and val_list is not None:
        evaluate_button = st.button("Run Evaluate", type="primary")

        if evaluate_button and documents is not None:
            with st.spinner(text="Evaluating..."):
                eval_results = evaluate(documents, val_list, config)

            score = 100 * sum([1 if res["correct"] else 0 for res in eval_results]) / len(eval_results)
            st.divider()
            st.header(f"Accuracy: {round(score, 2)}%")
            st.subheader("Samples")
            for i, res in enumerate(eval_results):
                result_md = f"""
                #### Validation {i}
                ###### {res["query"]}
                ###### AI Response:  {res["response"]}
                ###### Ground Truth: {res["answer"]}
                ###### Correct: :{'green' if res["correct"] else 'red'}[{res["correct"]}]
                """
                st.markdown(result_md)

            # save results and config
            with open(f"eval_{datetime.now().strftime('%Y%m%d%H%M%S')}.json", "w") as save_file:
                res = {
                    "score": score,
                    "eval_results": eval_results,
                    "config": config.dict(),
                }
                json.dump(res, save_file)


def load_documents(uploaded_file):
    from langchain.document_loaders import TextLoader
    documents = TextLoader(uploaded_file.name).load()
    return documents


if __name__ == "__main__":
    load_dotenv()
    run_app()
