"""
What is a Directory Loader?

A DirectoryLoader is a meta-loader.

It walks through a directory, finds files, and uses another document loader to load each file.

Think of it as:

Folder
 ├─ file1.pdf  → PDFLoader
 ├─ file2.txt  → TextLoader
 ├─ file3.md   → MarkdownLoader
 └─ ...
        ↓
     List[Document]


It does not parse files itself.
It delegates parsing to a loader you specify.

2. Why DirectoryLoader exists

Real projects rarely have one document.

You usually have:

a docs/ folder

many files

mixed formats

DirectoryLoader:

automates ingestion

standardizes metadata

scales easily

"""
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

loader = DirectoryLoader(path='/Users/mukul/Desktop/Generativ_Ai/Document_loaders/',
                         glob='*.pdf',
                         loader_cls=PyPDFLoader)

text_file = loader.load()

# print(text_file)
# print(type(text_file))
print(len(text_file))
#print(text_file[0].page_content)
#poem_chain = prompt | llm_model | parser

#poem = poem_chain.invoke({'topic':text_file[0].page_content})

#print(poem)
