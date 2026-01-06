from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = ChatOllama(model='tinyllama:latest')
parser = JsonOutputParser()
prompt1 = PromptTemplate(

                        template='give the name,age,city of {topic} \n {format_instruction}',
                        input_variables=['topic'],
                        partial_variables={"format_instruction":parser.get_format_instructions()} # partial_variables are variables that are filled in once when you define the prompt, not at runtime.
                        
                        # When using output parsers, partial_variables are most commonly used to inject parser instructions into the prompt.
                        
                        # Parser → provides format instructions → injected via partial_variables
                        )
# prompt1 = prompt1.invoke({"topic":"Mukul Lambat"})

# output = llm.invoke(prompt1)
# final_output = parser.parse(output.content)
# print(final_output)
# print(type(final_output))

chain = prompt1 | llm | parser
output = chain.invoke({'topic':'mukul lambat'})
print(output["name"])