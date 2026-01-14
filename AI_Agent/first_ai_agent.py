# %% import dependencies
from langchain_ollama.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic import hub

# %% Create tool
web_search_tool = DuckDuckGoSearchResults(output_format="json")
# %% llm object
llm = ChatOllama(model="llama3.2:latest")

# %% pull the ReAct prompt from langchain

prompt = hub.pull("hwchase17/react")

# %% creating a ReAct agent using the prompt pull from langchain hub
agent = create_react_agent(llm=llm, tools=[web_search_tool], prompt=prompt)

# %% wrap the agent with AgentExecutor
agent_executor = AgentExecutor(
    name="Mukul's first AI Agent",
    agent=agent,
    tools=[web_search_tool],
    verbose=True,
    max_iterations=2,
)

# %% invoke the agent with query
response = agent_executor.invoke({"input": "How is Siegen's weather today?"})
print(response)

# %%
