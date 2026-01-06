"""

Exercise 1: Build the smallest working vector store (the “hello retrieval”)

Goal: Understand what gets stored and what comes back.

What to do

Create 8–12 short documents (1–3 paragraphs each) across 3 topics (e.g., bats, LangChain, Kafka).

Split into chunks (choose a chunking strategy and record it).

Create a vector store and add the chunks.

Run similarity search for 5 queries.

Must test

A query that should match strongly

A query that should match weakly

A query that has synonyms (same meaning, different words)

Methods/features you should use

add_documents / from_documents

similarity_search

similarity_search_with_score (or equivalent scoring variant)

Deliverable

A short note describing what “score” means in your store and whether higher or lower is better.

"""
#%% import dependencies
from langchain.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama,OllamaEmbeddings
from langchain_community.vectorstores import Chroma
#%% step 1: Load the text file 
text_loader = TextLoader (
                    file_path='/Users/mukul/Desktop/Generativ_Ai/vector_store/text.txt',
                  )
text = text_loader.load() # Here we get the langchain's document object with page_content and metadata

print(text[0].page_content)
print(text[0].metadata)
#%% step 2: Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
                                                chunk_size=100,
                                                chunk_overlap=10,
                                              )     

chunks = text_splitter.split_documents(text)
#%% step 3: Create a vector store and pass the embedding model to it
embedding_model = OllamaEmbeddings(
                                    model='nomic-embed-text:latest'
                                  )
vector_store = Chroma.from_documents(
                                    documents=chunks,
                                    embedding=embedding_model,
                                    persist_directory="/Users/mukul/Desktop/Generativ_Ai/vector_store/chroma_db_emd/exercise_1",   # saves locally
                                    collection_name='exercise_1'
                                    )

#%% step 4: Do similarity search for a query on vactor store
query = 'what is langchain?'

similarity_search = vector_store.similarity_search(query=query, k=2)

print("\nTop results:")
for i, d in enumerate(similarity_search, 1):
    print(f"\n{i}) {d.page_content}")
    print("   metadata:", d.metadata)
