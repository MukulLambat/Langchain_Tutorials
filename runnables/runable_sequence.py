"""
RunnableSequence is LangChain’s way (in LCEL) to run multiple steps in order, where the output of one step becomes the input to the next.

Think: pipeline / assembly line
Step 1 → Step 2 → Step 3 →    

How you build a RunnableSequence

1) Using the pipe operator (|)

chain = prompt | llm | parser
That is a RunnableSequence under the hood.

2) Explicitly with RunnableSequence([...])

Useful when you want to build dynamically:

from langchain_core.runnables import RunnableSequence

chain = RunnableSequence([prompt, llm, parser])

Same idea: execute in order.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence 
from langchain_ollama import ChatOllama


llm = ChatOllama(model="tinyllama:latest",
                 temperature=0.2,)

prompt = PromptTemplate(
    template="Rewrite this text in a professional tone:\n{text}",
    input_variables=["text"],
)

# prompt2 = PromptTemplate(
#     template="Rewrite this text in a professional tone:\n{text}",
#     input_variables=["text"],
# )
parser = StrOutputParser()
chain = prompt | llm | StrOutputParser()
#chain = RunnableSequence(prompt, llm, parser) # This way also you can run simple runables.
#chain = prompt | llm | StrOutputParser() | prompt2 | llm | StrOutputParser() # This way two simple chain are connected.

out = chain.invoke({"text": "hey can u send me the file asap?"})
print(out)