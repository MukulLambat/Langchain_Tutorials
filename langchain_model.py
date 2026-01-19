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

# %%
from langchain.chat_models import init_chat_model


model = init_chat_model(model="llama3.2:latest", model_provider="ollama")
response = model.invoke("Hello")

# %% Load model using LLAMACPP
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8080",
    api_key="not-needed",  # llama-server doesn't enforce it unless you configure one
    model="local-model",  # any string is fine for llama-server
    temperature=0.5,
)
response = llm.invoke("Say Hello")
print(response.content)

# %%
