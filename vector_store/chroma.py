from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core import documents

# Sample documents as langchain Document Object
docs = [
    documents(
        page_content="TinyLlama is a small language model that can run locally using Ollama.",
        metadata={"source": "notes.txt", "topic": "llm"},
    ),
    documents(
        page_content="Vector stores store embeddings and allow similarity search for retrieval-augmented generation (RAG).",
        metadata={"source": "notes.txt", "topic": "rag"},
    ),
    documents(
        page_content="RecursiveCharacterTextSplitter splits text into chunks with overlap to preserve context.",
        metadata={"source": "notes.txt", "topic": "splitting"},
    ),
]

splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
chunks = splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 4) Build / persist a Chroma vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="/Users/mukul/Desktop/Generativ_Ai/vector_store/chroma_db_emd",  # saves locally
    collection_name="my_knowledge_base",
)

query = "What is a vector store used for?"
results = vectorstore.similarity_search(query, k=2)

print("\nTop results:")
for i, d in enumerate(results, 1):
    print(f"\n{i}) {d.page_content}")
    print("   metadata:", d.metadata)
