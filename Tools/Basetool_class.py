# %%
from pydantic import BaseModel, Field
from langchain_community.tools import BaseTool


class AddInput(BaseModel):
    a: int = Field(required=True, description="First integer")
    b: int = Field(required=True, description="Second integer")


class AddTool(BaseTool):
    name: str = "add"
    description: str = "Adds two integers and returns the result."
    args_schema: type[BaseModel] = AddInput

    def _run(self, a: int, b: int) -> int:
        return a + b


add_tool = AddTool()
result = add_tool.invoke({"a": 4, "b": 5})
print(result)

# %%
