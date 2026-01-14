# %%
from langchain.tools import tool
from langchain_ollama import ChatOllama


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


llm = ChatOllama(model="tinyllama:latest", temperature=0)

llm_with_tools = llm.bind_tools([multiply])

response = llm_with_tools.invoke("What is 6 times 7? Use the tool.")
print(response.content)

# %%
