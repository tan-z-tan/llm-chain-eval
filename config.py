from enum import Enum
from pydantic import BaseModel


MATCH_TEMPLATE = """Judge if the two answers to the question are the same, ignoring small differences in expression. Return only "OK" if they are the same, otherwise return only "NG".
===
1: {response}
===
2: {answer}
===
"""

RETRIEVE_CHAIN_TEMPLATE = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer in Japanese:"""


class Configuration(BaseModel):
    class LlmModel(str, Enum):
        gpt_3_turbo = "gpt-3.5-turbo"
        google_flan_t5_xl = "google/flan-t5-xl"

    llm_model: LlmModel = LlmModel.gpt_3_turbo
    chunk_size: int = 1000
    chunk_overlap: int = 100
    top_k_chunk: int = 2
    chain_type: str = "stuff"
    retrieve_chain_template: str = RETRIEVE_CHAIN_TEMPLATE
    match_template: str = MATCH_TEMPLATE
