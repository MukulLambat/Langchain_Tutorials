"""
1. What is a structure-based text splitter?

A structure-based text splitter splits text using document structure, not raw length.

“Structure” can mean:

>> paragraphs

>> headings

>> sections

>> sentences

>> markdown elements

>> code blocks

>> HTML tags

Instead of saying:

>>  “Cut every 1000 characters”

It says:

“Respect how the document is written”

"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """Heading

This is paragraph one. It has multiple sentences.

This is paragraph two. More content here.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=60,
    chunk_overlap=10,
)

chunks = splitter.split_text(text)
for i, c in enumerate(chunks, 1):
    print(i, repr(c)) ## Here 'repr' is just a way to show a string with its hidden characters visible
    #print(i, c)
