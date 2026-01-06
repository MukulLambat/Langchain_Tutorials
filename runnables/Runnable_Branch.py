"""
1. What is RunnableBranch?

Instead of “always run everything” (Parallel) or “always run in order” (Sequence), Branch decides which path to take at runtime based on the input.

RunnableBranch allows you to route input to different branches (Runnables) based on conditions, much like an if-elif-else structure.

            Input  ----------------> |  RunnableBranch  |
                                    +--------+---------+
                                            |
                ------------------------------------------------
                |                       |                      |
            [Condition A]           [Condition B]         [Default]
                |                       |                      |
                v                       v                      v
            +------------+       +--------------+       +------------------+
            | Runnable A |       | Runnable B   |       | Runnable Default |
            +------------+       +--------------+       +------------------+
                |                       |                      |
                v                       v                      v
            Output A              Output B              Default Output
 
## Simple example (hello world)

from langchain_core.runnables import RunnableLambda

uppercase = RunnableLambda(lambda x: x.upper())

uppercase.invoke("hello")

"Wraps any Python function or lambda as a 'Runnable' so it can be chained using the | operator, supporting async, batch, and streaming by default."

"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_ollama import ChatOllama

llm = ChatOllama(model="tinyllama:latest", temperature=0.2)

short_prompt = PromptTemplate(
    template="Answer briefly:\n{question}",
    input_variables=["question"],
)

long_prompt = PromptTemplate(
    template="Answer in detail with examples:\n{question}",
    input_variables=["question"],
)

short_chain = short_prompt | llm | StrOutputParser()
long_chain = long_prompt | llm | StrOutputParser()

branch = RunnableBranch(
    (lambda x: len(x["question"]) < 80, short_chain),
    (long_chain),  # default
)

out = branch.invoke({"question": "Explain RunnableBranch in LangChain."})
print(out)
