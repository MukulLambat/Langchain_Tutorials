from langchain.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

loader = PyPDFLoader('/Users/mukul/Desktop/Generativ_Ai/Document_loaders/MiniRocket.pdf')

text_file = loader.load()

# print(text_file)
# print(type(text_file))
# print(len(text_file))
print(text_file[0].page_content)
#poem_chain = prompt | llm_model | parser

#poem = poem_chain.invoke({'topic':text_file[0].page_content})

#print(poem)
