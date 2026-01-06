from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
prompt_template1 = PromptTemplate(
                                    template='Generate a detailed report on the topic given below, \n \'{topic}\'',
                                    input_variables=['topic']
                                )
prompt_template2 = PromptTemplate(
                                    template='Generate short summary of the content given below about the topic, \n \'{detailed_report}\'',
                                    input_variables=['detailed_report']
                                )
model = ChatOllama(model='tinyllama:latest')
parser = StrOutputParser()
chain = prompt_template1 | model | parser |prompt_template2 | model | parser
topic = str(input('Enter the topic to generate the summary:'))
output = chain.invoke({'topic':topic})
print(output)
chain.get_graph().print_ascii()

# prompt = prompt_template.format(topic='Machine Learning')
# output = model.invoke(prompt) 
# print(output.content)