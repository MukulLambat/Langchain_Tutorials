# %%
from langchain_community.tools import DuckDuckGoSearchResults, ShellTool
import os

# %% To use Duck Duck go tool provided by Langchain
search_tool = DuckDuckGoSearchResults(output_format="json")
query = search_tool.invoke(
    "provide me the list of schedule of India vs New Zealand schedule 2025 ?"
)

# %% To use ShellTool provided by langchain
shell_tool = ShellTool()
os.chdir("/Users/mukul/Desktop/Generativ_Ai")
command = shell_tool.invoke("pwd")
print(command)

# %%
