"""

2. What is a length-based text splitter?

A length-based text splitter splits text based on size, not meaning.

“Length” usually means:

number of characters

or number of tokens

Not:

sentences

paragraphs

semantics

2. What is a length-based text splitter?

A length-based text splitter splits text based on size, not meaning.

“Length” usually means:

number of characters

or number of tokens

Not:

sentences

paragraphs

semantics


"""

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
                        file_path='/Users/mukul/Desktop/Generativ_Ai/Document_loaders/MiniRocket.pdf',
                        
                    )
document = loader.load()

splitter = CharacterTextSplitter(
                                    chunk_size=100,
                                    chunk_overlap=0,
                                    separator=''
    
                                )
result = splitter.split_documents(document)
print(result[1].page_content)