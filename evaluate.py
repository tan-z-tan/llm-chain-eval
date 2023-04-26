from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from config import Configuration


def evaluate(documents, eval_datasets, config: Configuration):
    results = []
    llm = load_llm(config)
    emb_model = load_emb_model(config)

    for dataset in eval_datasets:
        query = dataset.query
        answer = dataset.answer

        text_splitter = CharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separator="\n")

        texts = text_splitter.split_documents(documents)
        retriever = FAISS.from_documents(
            texts,
            emb_model,
            ).as_retriever(search_kwargs={"k": config.top_k_chunk})

        prompt = PromptTemplate(
            # template=prompt_template, input_variables=["context", "question"]
            template=config.retrieve_chain_template,
            input_variables=["context", "question"],
        )

        chain_type_kwargs = {"prompt": prompt}
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=config.chain_type,
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs,
            verbose=True)

        response = qa(query)['result']
        results.append({
            "query": query,
            "answer": answer,
            "response": response,
            "correct": match_response(response, answer, config),
        })
    return results


def match_response(response, answer, config: Configuration):
    llm = ChatOpenAI(temperature=0)
    match_query = config.match_template.format(response=response, answer=answer)
    prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True)

    res = chain.run(match_query)

    return res.strip() == "OK"


def load_llm(config: Configuration):
    if config.llm_model == Configuration.LlmModel.gpt_3_turbo:
        return ChatOpenAI(temperature=0)
    elif config.llm_model == Configuration.LlmModel.google_flan_t5_xl:
        from langchain import HuggingFaceHub
        repo_id = "google/flan-t5-xl"
        return HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={"temperature": 0, "max_length": 2000}
        )
    elif config.llm_model == Configuration.LlmModel.google_flan_t5_large:
        from langchain import HuggingFaceHub
        repo_id = "google/flan-t5-large"
        return HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={"temperature": 0, "max_length": 2000}
        )
    elif config.llm_model == Configuration.LlmModel.dolly_v2_3b:
        import torch
        from transformers import pipeline
        model = pipeline(
            model="databricks/dolly-v2-3b",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",)
        # TODO: wrap in a class

        return model
    else:
        raise ValueError(f"Unknown LLM model {config.llm_model}")


def load_emb_model(config: Configuration):
    if config.embedding_model == Configuration.EmbeddingModel.gpt_embedding:
        return OpenAIEmbeddings()
    elif config.embedding_model == Configuration.EmbeddingModel.flan_embedding:
        raise NotImplementedError()
    elif config.embedding_model == Configuration.EmbeddingModel.huggingface_embedding:
        return HuggingFaceEmbeddings()
    else:
        raise ValueError(f"Unknown embedding model {config.embedding_model}")
