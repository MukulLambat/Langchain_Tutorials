"""
1. What is RunnablePassthrough?

RunnablePassthrough is a utility runnable that (as the name says) mostly passes the input through unchanged, but it’s super useful for building pipelines where you want to keep the original input while adding new keys.


        +------------------------+
        |  RunnablePassthrough  |
        +------------------------+
                  |
           (any input)
                  |
                  v
        +------------------------+
        |      Same Output       |
        +------------------------+
Input:  {"text": "Hello, world!"}
Output: {"text": "Hello, world!"}

"""
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm_model = ChatOllama(model='tinyllama:latest',
                       temperature=0.2,
                       num_ctx=256,)

joke_creation = PromptTemplate(
                        template = 'act as a joke writing assistant\n'
                               'write a short and laughable joke on {topic}.\n',   
                        input_variables = ['topic'],
                        )
joke_explanation = PromptTemplate(
                        template = 'act as a joke explanation assistant\n'
                               'explain the provided joke {text}.\n',
                        input_variables = ['text'],
                        )

parser = StrOutputParser()

joke_gen_chain = joke_creation | llm_model | parser

joke_explanation = RunnableParallel({
                                    "generated_joke": RunnablePassthrough(),
                                    "joke_explanation": RunnableSequence(joke_explanation, llm_model, parser),                         
                                    })

final_chain = RunnableSequence(joke_gen_chain,joke_explanation)

joke = final_chain.invoke({'topic':'cricket'})

print("\n--- Parallel final_chain ---")
print("\nSummary:\n", joke["generated_joke"])
print("\nKeywords:\n", joke["joke_explanation"])
