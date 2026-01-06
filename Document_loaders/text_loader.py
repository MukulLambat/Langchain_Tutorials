from langchain.document_loaders import TextLoader
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

llm_model = ChatOllama(
                        model='tinyllama:latest',
                        temperature=0.2
                      ) 

prompt = PromptTemplate(
                         template='Write a poem on {topic}',
                         input_variables=['topic']
    
                        )

parser = StrOutputParser()


loader = TextLoader('/Users/mukul/Desktop/Generativ_Ai/Document_loaders/logs.txt')

text_file = loader.load()

# print(text_file)
# print(type(text_file))
# print(len(text_file))
# print(text_file[0].page_content)
poem_chain = prompt | llm_model | parser

poem = poem_chain.invoke({'topic':text_file[0].page_content})

print(poem)
