# %%
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

# %%
llm = ChatOllama(model="tinyllama", num_ctx=312, temperature=0.2)

chathistory = [SystemMessage(content="You are helpful AI Assistant")]

while True:
    query = input("Write Question:")
    if query.lower() == "exit":
        break
    chathistory.append(HumanMessage(content=query))
    # for chunk in llm.stream(query):
    #    print(chunk.content, end="",flush=True)
    reply = llm.invoke(query)
    print("ChatBot:", reply.content)
    chathistory.append(AIMessage(content=reply.content))

print(chathistory)
