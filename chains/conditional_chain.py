from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal

model = ChatOllama(model='tinyllama:latest')

parser = StrOutputParser()

class Feedback(BaseModel):
    
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt = PromptTemplate(
                            template='Classify this review sentiment as positive or negative.\nReview:\n {feedback} \n {format_instruction}',
                            input_variables=['feedback'],
                            partial_variables={'format_instruction':parser2.get_format_instructions()}
                        )
#chain = prompt | model | parser2

prompt2 = PromptTemplate(
                            template='Write an appropriate response for positive response:\n {feedback}',
                            input_variables=['feedback'],
                        )

prompt3 = PromptTemplate(
                            template='Write an appropriate response for negative response:\n {feedback}',
                            input_variables=['feedback'],
                        )

branch_chain = RunnableBranch(
                                (lambda x: x['sentiment'] == 'positive', prompt2 | model | parser),
                                (lambda x: x['sentiment'] == 'negative', prompt3 | model | parser),
                                RunnableLambda(lambda x: 'could not find sentiment'),
                             )
chain = prompt | model | parser2 | branch_chain
print(chain.invoke({'feedback':'This is a wonderful iphone'}))