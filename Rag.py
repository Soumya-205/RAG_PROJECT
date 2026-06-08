from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

load_dotenv()

def ingest_document(file_path):
    loader=PyPDFLoader(file_path)
    docs=loader.load()
    splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    chunks=splitter.split_documents(docs)

    #creating embedding objects
    embeddings=OllamaEmbeddings(model="nomic-embed-text",base_url=os.getenv("OLLAMA_BASE_URL"))

    #store it into vector database
    vectorstore=Chroma.from_documents(documents=chunks,embedding=embeddings,
                                  persist_directory=os.getenv("VECTORSTORE_DIR"))
    return vectorstore

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def query_rag(question):
    #creating embedding objects
    embeddings=OllamaEmbeddings(model="nomic-embed-text",base_url=os.getenv("OLLAMA_BASE_URL"))
    #load vector database
    vectorstore=Chroma(persist_directory=os.getenv("VECTORSTORE_DIR"),embedding_function=embeddings)
    #creating retriever from vectorstore
    retriever=vectorstore.as_retriever(search_kwargs={"k":3})

    llm=ChatOllama(model=os.getenv("OLLAMA_MODEL"),base_url=os.getenv("OLLAMA_BASE_URL"),num_gpu=5,num_ctx=4096)

    system_prompt=(
        "You are an assistant for question-answering only."
        "Use only the retrieved chunks as context to answer the question."
        "If you don't know the answer, say that I do not know."
        "\n\nContext:\n{context}"
    )
    prompt=ChatPromptTemplate.from_messages([("system",system_prompt),("human","{input}"),])
    rag_chain=({"context":retriever | format_docs,"input":RunnablePassthrough()}
              |prompt
              |llm
              |StrOutputParser())
    answer = rag_chain.invoke(question)
    return answer


        