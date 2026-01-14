"""Extract the real time information form internet using DuckDuckGo web search tool and generate response from LLM"""

# %% import dependencies
from langchain_ollama.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage

# %% Create tool
web_search_tool = DuckDuckGoSearchResults(output_format="json")
# %% llm object
llm = ChatOllama(model="llama3.2:latest")
# %% llm object with tool
llm_tool = llm.bind_tools([web_search_tool])
# storing different messages history for complete context to LLM to generate grounded response
messages = []
query = HumanMessage("Who won the odi between todays India and New Zealand ODI match?")
messages.append(query)
result = llm_tool.invoke(messages)
# print(result.content)
messages.append(result)
# print(messages)
# %% get the result by invoking tool call
tool_result = web_search_tool.invoke(result.tool_calls[0])
messages.append(tool_result)
print(messages)
# %%
final_output = llm_tool.invoke(messages)
print(final_output.content)
