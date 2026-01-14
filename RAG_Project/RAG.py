# It is complete pipeline for the RAG

# %% step 1:Import all the dependencies
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)

# %% step 2:Load the pdf using Langchain
pdf_loader = PyPDFLoader(file_path="./MiniRocket.pdf")

pdf = pdf_loader.load()
# %% step 3:Chunk the pdf in smaller chunks using Langchain
pdf_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

chunks = pdf_splitter.split_documents(pdf)
# %% step 4: Create the vector store using langchain
# step 4.1: Create the embeddings of the chunks
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="/Users/mukul/Desktop/Generativ_Ai/RAG_Project",
)
# %% step 5: Create a retriever
retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.25}
)
# %% step 6: Create a prompt with chunks as context
prompt = PromptTemplate(
    template="you are RAG Based Assistant answer the question from the given context.\n\n\
Question:\n {user_query}\n\n\
answer the above question using the context given below.\n\n\
context:\n\n {context_chunks}",
    input_variables=["user_question", "retrieved_chunk"],
)
# %% step 7: Search the relevant chunks in vector store using retriever
user_query = "what is mini-rocket?"
context_chunks = retriever.invoke(user_query)
prompt = prompt.invoke({"user_query": "user_query", "context_chunks": "context_chunks"})
# print(query.page_content)
# %% step 8: Initialize LLM from Ollama
llm_model = ChatOllama(model="tinyllama:latest", temperature=0.5)
# %% step 9: Send the created prompt to LLM to get the answer for the query
answer = llm_model.invoke(prompt)
# %% step 10 : Print the output of the user query
print(answer.content)


# %% step 11: To Create a chain
def format_doc(retrieved_chunks):
    context_chunks = "\n\n".join(chunk.page_content for chunk in retrieved_chunks)
    return context_chunks


query_content_chain = RunnableParallel(
    {
        "user_query": RunnablePassthrough(),
        "context_chunks": retriever | RunnableLambda(format_doc),
    }
)
parser = StrOutputParser()

final_chain = query_content_chain | prompt | llm_model | parser

answer = final_chain.invoke("what is mini-rocket?")

# %%
print(answer)
# %%
