import sys
from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
import os

PERSIST_DIRECTORY = ".chromadb"

# Before every run, delete everything in the persist directory
if os.path.exists(PERSIST_DIRECTORY):
    for file in os.listdir(PERSIST_DIRECTORY):
        os.remove(os.path.join(PERSIST_DIRECTORY, file))

# Load the OpenAI key
with open(os.path.expanduser("~/.config/openai.key")) as f:
    key = f.read().strip()
os.environ["OPENAI_API_KEY"] = key


# Load documents from the current directory
loader = DirectoryLoader(".", exclude=["poetry.lock"])
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()


def format_docs(docs: List[Document]):
    return "\n\n".join(doc.page_content for doc in docs)


template = """
Use the following context to answer questions: {context}

Question: {question}
AI:
"""


llm = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | ChatPromptTemplate.from_template(template)
    | ChatOpenAI(model="gpt-3.5-turbo")
    | StrOutputParser()
)


def get_question():
    if len(sys.argv) > 1:
        return sys.argv[1]
    else:
        return input("Enter a question: ")


print(llm.invoke(get_question()))
