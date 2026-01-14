# %% import dependencies
from langchain_ollama.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic import hub
from pydantic import BaseModel, Field
import requests
import json
from langchain_core.tools import tool

# %% Create tool
web_search_tool = DuckDuckGoSearchResults(output_format="json")


class weather_validation(BaseModel):
    city_name: str = Field(..., description="Name of the city")


@tool(args_schema=weather_validation)
def weather(city: str):
    """
    Fetches the current weather information for a given city using the Weatherstack API.

    This function calls the Weatherstack `current` endpoint to retrieve real-time
    weather data for the specified city. It is designed to be used as a tool
    within an AI agent or LangChain-based workflow.

    Args:
        city (str): Name of the city for which current weather data is requested.

    Returns:
        dict: JSON response containing current weather details such as temperature,
              weather conditions, humidity, wind speed, and location metadata.
    """
    url = f"http://api.weatherstack.com/current?access_key=API_Key&query=New York{city}"
    response = requests.get(url)
    return response.json()


# %% llm object
llm = ChatOllama(model="llama3.2:latest")

# %% pull the ReAct prompt from langchain

prompt = hub.pull("hwchase17/react")

# %% creating a ReAct agent using the prompt pull from langchain hub
agent = create_react_agent(llm=llm, tools=[web_search_tool, weather], prompt=prompt)

# %% wrap the agent with AgentExecutor
agent_executor = AgentExecutor(
    name="Weather AI Agent",
    agent=agent,
    tools=[web_search_tool, weather],
    verbose=True,
    # max_iterations=2,
)

# %% invoke the agent with query
response = agent_executor.invoke(
    {"input": "current weather condition in Siegen, Germany?"}
)
print(response)
