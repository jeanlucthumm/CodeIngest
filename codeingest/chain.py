import os

from typing import Any, List

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader

PERSIST_DIRECTORY = ".chromadb"


def format_docs(docs: List[Document]):
    return "\n\n".join(doc.page_content for doc in docs)


def clear_persist():
    if os.path.exists(PERSIST_DIRECTORY):
        for file in os.listdir(PERSIST_DIRECTORY):
            os.remove(os.path.join(PERSIST_DIRECTORY, file))


def load_docs() -> VectorStoreRetriever:
    """Load the documents from the current directory and return a retriever."""
    loader = DirectoryLoader(".", exclude=["poetry.lock"])
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    return vectorstore.as_retriever()


def make_chain() -> RunnableSerializable[Any, str]:
    # Before every run, delete everything in the persist directory.
    # Eventually we will live update based on the file
    clear_persist()
    retriever = load_docs()

    template = """
    Use the following context to answer questions: {context}

    Question: {question}
    AI:
    """

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template(template)
        | ChatOpenAI(model="gpt-3.5-turbo")
        | StrOutputParser()
    )
