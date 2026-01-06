"""
1. What is RunnableLambda?

RunnableLambda wraps a Python function or lambda expression and executes it. It's useful for transforming data on-the-fly within a chain.

RunnableLambda lets you turn plain Python logic into a first-class runnable that composes cleanly with LangChain pipelines.

            +-------------------+
            Input  ---> |  RunnableLambda   | ---> Output
            +-------------------+
                    |
         (e.g., lambda x: x.upper())
         
## Simple example (hello world)

from langchain_core.runnables import RunnableLambda

uppercase = RunnableLambda(lambda x: x.upper())

uppercase.invoke("hello")

"Wraps any Python function or lambda as a 'Runnable' so it can be chained using the | operator, supporting async, batch, and streaming by default."

"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

llm_model = ChatOllama(model='tinyllama:latest',
                       temperature=0.2,
                       num_ctx=256,)

prompt = PromptTemplate(
    template="Rewrite in a polite tone:\n{text}",
    input_variables=["text"],
)

strip_whitespace = RunnableLambda(lambda x: x.upper())

chain = prompt | llm_model | StrOutputParser() | strip_whitespace

out = chain.invoke({"text": "  send the file now  "})

print(out)

