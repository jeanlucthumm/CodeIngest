from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
import os

with open(os.path.expanduser("~/.config/openai.key")) as f:
    key = f.read().strip()
os.environ["OPENAI_API_KEY"] = key

template = """User: {question}"""


llm = (
    {"question": RunnablePassthrough()}
    | ChatPromptTemplate.from_template(template)
    | ChatOpenAI(model="gpt-3.5-turbo")
    | StrOutputParser()
)

print(llm.invoke("Hello there"))
