#%%
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
#%%
llm = ChatOllama(model='tinyllama:latest')

schema = [
    ResponseSchema(name="fact_1", description='fact 1 about topic'),
    ResponseSchema(name="fact_2", description='fact 2 about topic'),
    ResponseSchema(name="fact_3", description='fact 3 about topic'),
]
parser = StructuredOutputParser.from_response_schemas(schema)


prompt1 = PromptTemplate(

                        template='Provide me 3 facts about {topic}. \n {format_instruction}\n\nRespond ONLY with valid JSON, no extra text.',
                        input_variables=['topic'],
                        partial_variables={"format_instruction":parser.get_format_instructions()}
                        
                        )

# prompt = prompt1.invoke({'topic':'Machine Learning'})
# output = llm.invoke(prompt)
# output = parser.parse(output.content)
# print(output)

chain = prompt1 | llm | parser

result = chain.invoke({'topic': 'machine learning'})
print(result)

