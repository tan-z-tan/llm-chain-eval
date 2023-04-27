import json
from dotenv import load_dotenv
from pydantic import BaseModel
import streamlit as st
import pandas as pd  # type: ignore
from langchain.chains.question_answering import stuff_prompt, map_reduce_prompt, refine_prompts, map_rerank_prompt
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
    val_list = [
        ValidationSample(
            query="What is the capital of Japan?",
            answer="Tokyo",
        )
    ]

    st.set_page_config(
        page_title="LLM-chain-eval",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("LLM-chain-eval")

    # sidebar
    st.sidebar.title("Configuration")

    st.sidebar.subheader("LLM Model")
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
    st.sidebar.subheader("Chunk size")
    st.sidebar.markdown("The number of characters per chunk.")
    chunk_size = st.sidebar.slider("chunk size", 100, 3000, 1000, 100, label_visibility="collapsed")
    st.sidebar.subheader("Chunk Overlap")
    st.sidebar.caption("The number of characters to overlap between chunks.")
    chunk_overlap = st.sidebar.slider("chunk overlap", 0, 1000, 0, 1, label_visibility="collapsed")

    st.sidebar.subheader("Chain type")
    st.sidebar.caption("The way to retrieve the answer.")
    chain_type = st.sidebar.selectbox(
        "chain type",
        ["stuff", "map_reduce", "refine", "map_rerank"],
        index=0,
        label_visibility="collapsed",
    )
    st.sidebar.subheader("Embedding Model")
    st.sidebar.caption("Model to calculate embedding.")
    emb_type = st.sidebar.selectbox(
        "Embedding Model",
        [
            Configuration.EmbeddingModel.gpt_embedding.value,
            Configuration.EmbeddingModel.huggingface_embedding.value,
        ],
        index=0,
        label_visibility="collapsed",
    )
    st.sidebar.subheader("Top K Chunk")
    st.sidebar.caption("The number of chunks to retrieve.")
    top_k_chunk = st.sidebar.slider("top k chunk", 1, 10, 2, 1, label_visibility="collapsed")

    if chain_type == "stuff":
        st.sidebar.subheader("Prompt template (currently only for stuff)")
        st.sidebar.caption("Template to retrieve chain.")
        retrieve_chain_template = st.sidebar.text_area("Retrieve Chain template", stuff_prompt.prompt_template, label_visibility="collapsed")
    else:
        retrieve_chain_template = None
    st.sidebar.subheader("Match template")
    st.sidebar.caption("Template used to match the answer.")
    match_template = st.sidebar.text_area("Match template", MATCH_TEMPLATE, label_visibility="collapsed")

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

    # source file
    st.subheader("Source documents")
    doc_upload = st.file_uploader("Text file you want to evaluate", type=["txt", "pdf"])
    if doc_upload is not None:
        documents = load_documents(doc_upload)
        st.caption(documents[0].page_content[:200] + "...")
        char_count = sum([len(doc.page_content) for doc in documents])
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        token_count = sum([len(encoding.encode(doc.page_content)) for doc in documents])
        st.markdown(f":green[{char_count}] characters, :green[{token_count}] tokens (gpt-3.5-turbo)")

    # validation file (optional)
    st.subheader("Validation datasets")
    val_upload = st.file_uploader("Validation json file. You can input query and answer directly.", type=["json"])
    if val_upload is not None:
        val_list = ValidationList.parse_raw(val_upload.read()).validations

    val_data = pd.DataFrame([v.dict() for v in val_list], columns=["query", "answer"])

    edited_val_df = st.experimental_data_editor(val_data, num_rows="dynamic", use_container_width=True)
    val_list = [ValidationSample(**v) for v in edited_val_df.to_dict("records") if v["query"] is not None and v["answer"] is not None]

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
                json.dump(res, save_file, ensure_ascii=False, indent=2)


def load_documents(uploaded_file):
    from langchain.document_loaders import TextLoader
    documents = TextLoader(uploaded_file.name).load()
    return documents


if __name__ == "__main__":
    load_dotenv()
    run_app()
