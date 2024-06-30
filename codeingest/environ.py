import os


def langsmith():
    """Set up the LangSmith environment variables."""
    with open("./langsmith.key") as f:
        key = f.read().strip()

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = key
    os.environ["LANGCHAIN_PROJECT"] = "CodeIngest"


def openai():
    """Set up the OpenAI environment variables."""
    with open("./openai.key") as f:
        key = f.read().strip()
    os.environ["OPENAI_API_KEY"] = key
