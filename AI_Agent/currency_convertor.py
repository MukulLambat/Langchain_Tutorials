# %% import dependencies
from langchain_ollama.chat_models import ChatOllama
from langchain_classic.tools import tool, StructuredTool
import json
import requests
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic import hub
from pydantic import BaseModel, Field


# %% Define input schemas
class CurrencyFactorInput(BaseModel):
    Base_Currency: str = Field(description="The base currency code (e.g., 'USD')")
    Target_Currency: str = Field(description="The target currency code (e.g., 'INR')")


class ConvertedAmountInput(BaseModel):
    Base_Amount: float = Field(description="The amount to convert")
    Conversion_Rate: float = Field(description="The conversion rate")


# %% Define 2 tools
@tool(args_schema=CurrencyFactorInput)
def currency_factor(Base_Currency: str, Target_Currency: str):
    """
    Fetch conversion factor between two currencies (e.g., USD -> INR).

    Args:
        Base_Currency (str): The code of the base currency (e.g., "USD").
        Target_Currency (str): The code of the target currency (e.g., "INR").

    Returns:
        dict: A JSON response containing the conversion factor and other related data.
    """
    url = f"https://v6.exchangerate-api.com/v6/c6797d8fe07eb737a651ffac/pair/{Base_Currency}/{Target_Currency}"
    response = requests.get(url)
    return json(response)


@tool(args_schema=ConvertedAmountInput)
def converted_amount(Base_Amount: float, Conversion_Rate: float) -> float:
    """
    Convert amount using conversion rate.

    Args:
        Base_Amount (float): The amount in the base currency to be converted.
        Conversion_Rate (float): The conversion rate from the base currency to the target currency.

    Returns:
        float: The converted amount in the target currency.
    """
    return Base_Amount * Conversion_Rate


# %% Create LLM object

llm = ChatOllama(model="llama3.2:latest")
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=[currency_factor, converted_amount],
    prompt=prompt,
)

agent_extractor = AgentExecutor(
    name="Currency_Convertor",
    agent=agent,
    verbose=True,
    tools=[currency_factor, converted_amount],
    handle_parsing_errors=True,  # Added to handle parsing errors
)

response = agent_extractor.invoke({"input": "what is 10 USD to INR ?"})
print(response)

# %%
