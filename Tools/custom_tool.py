# %%
from pydantic import BaseModel, Field
from langchain_community.tools import StructuredTool


# Define a Pydantic model for structured input
class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="First integer")
    b: int = Field(required=True, description="Second integer")


def multiply_structured(a, b):
    """Multiply two integers and return the result."""
    return a * b


multiply_tool = StructuredTool.from_function(
    func=multiply_structured,
    name="multiply",
    args_schema=MultiplyInput,
)


# Example usage
result_structured = multiply_tool.invoke({"a": 5, "b": 3})
print(result_structured)

# %%
