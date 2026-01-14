# %%
from langchain_community.tools import tool, DuckDuckGoSearchResults
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage


# @tool
# def multiply(a: int, b: int) -> int:
#     """Multiply two integers."""
#     return a * b

# %%
web_search_tool = DuckDuckGoSearchResults(output_format="json")
# %%
llm = ChatOllama(model="llama3.2:latest")
# %%
llm_tool = llm.bind_tools([web_search_tool])
messages = []
query = HumanMessage("today's temperature in Siegen Germany")
messages.append(query)
print(messages)
# %%
result = llm_tool.invoke(messages)
messages.append(result)
print("\033[91mBefore using tool the LLm response is \n\n\033[0m")
print(messages)
# print(result.tool_calls)

# %%
result_with_tool = web_search_tool.invoke(result.tool_calls[0])
messages.append(result_with_tool)
# %%
print(messages)
# %%
final_result = llm_tool.invoke(messages)
