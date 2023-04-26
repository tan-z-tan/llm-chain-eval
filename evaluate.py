from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from config import Configuration


def evaluate(documents, eval_datasets, config: Configuration):
    results = []
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
            OpenAIEmbeddings(),
            ).as_retriever(search_kwargs={"k": config.top_k_chunk})

        prompt = PromptTemplate(
            # template=prompt_template, input_variables=["context", "question"]
            template=config.retrieve_chain_template,
            input_variables=["context", "question"],
        )

        chain_type_kwargs = {"prompt": prompt}
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0),
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