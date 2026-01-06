from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model='tinyllama:latest')

# prompt 1 -> detailed report on topic
prompt1 = PromptTemplate(

                        template='Write a detailed report on the {topic}',
                        input_variables=['topic']
                        
                        )
#prompt1 = prompt1.invoke({'topic':'GenAI'})
print("Detailed Report:\n")

# detailed_report = ''
# for chunk in llm.stream(prompt1):
#     print(chunk.content, end='',flush=True)
#     detailed_report = detailed_report + chunk.content

# print()
# print('#' * 20)
# print('#' * 20)
# print('Summary of the topic')

# prompt 2 -> summary in 5 line
prompt2 = PromptTemplate(

                        template='Write a brief summary of the text below \n \'{content}\'',
                        input_variables=['content']
                        
                        )
# prompt2 = prompt2.invoke({'content':detailed_report})
# for chunk in llm.stream(prompt2):
#     print(chunk.content, end='',flush=True)
# print('\n')

parser = StrOutputParser()

chain = prompt1 | llm | parser | prompt2 | llm | parser 
output = chain.invoke({'topic':'GenAI'})
print(output)