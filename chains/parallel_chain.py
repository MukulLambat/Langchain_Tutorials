""" 
What are Parallel Chains?

A Parallel Chain runs multiple chains at the same time on the same input, then merges their outputs into a single structured result.

In LangChain (LCEL), this is done using:

RunnableParallel 

One input → many independent LLM tasks → one combined output

"""
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_ollama import ChatOllama

# 1) Model
llm = ChatOllama(
    model="tinyllama:latest",
    temperature=0.2,
)

# 2) Prompts
summary_prompt = PromptTemplate(
    template=(
        "You are a concise technical writer.\n"
        "Summarize the text in 3 bullet points.\n\n"
        "Text:\n{text}\n"
    ),
    input_variables=["text"],
)

keywords_prompt = PromptTemplate(
    template=(
        "Extract exactly 7 keywords from the text.\n"
        "Return them as a comma-separated list (no extra text).\n\n"
        "Text:\n{text}\n"
    ),
    input_variables=["text"],
)

sentiment_prompt = PromptTemplate(
    template=(
        "Classify the sentiment of the text as one of: positive, neutral, negative.\n"
        "Return ONLY the label (no extra text).\n\n"
        "Text:\n{text}\n"
    ),
    input_variables=["text"],
)

# 3) String output parser
parser = StrOutputParser()

# 3) Chains (Prompt -> LLM -> String)
summary_chain = summary_prompt | llm | parser
keywords_chain = keywords_prompt | llm | parser
sentiment_chain = sentiment_prompt | llm | parser

# 4) Parallel composition (same input dict is sent to every branch)
parallel_chain = RunnableParallel(
    summary=summary_chain,
    keywords=keywords_chain,
    sentiment=sentiment_chain,
)
document = (
        "Wind energy expansion is essential for climate goals, but it can harm biodiversity. "
        "Bats and birds may collide with turbine blades, especially at night or in fog. "
        "A combined radar + acoustic monitoring approach can detect activity and enable "
        "smart shutdowns only when risk is high, reducing unnecessary downtime."
    )

result = parallel_chain.invoke({"text": document})

print("\n--- Parallel chain output ---")
print("\nSummary:\n", result["summary"])
print("\nKeywords:\n", result["keywords"])
print("\nSentiment:\n", result["sentiment"])
