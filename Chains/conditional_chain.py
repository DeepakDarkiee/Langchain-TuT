from langchain_groq import ChatGroq
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

parser = StrOutputParser()

class FeedbackSentiment(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the feedback")

pydantic_parser = PydanticOutputParser(pydantic_object=FeedbackSentiment)

prompt1 = PromptTemplate(
    template="""
Classify the sentiment of the following feedback as positive or negative.

Feedback:
{feedback}

{format_instruction}
""",
    input_variables=["feedback"],
    partial_variables={
        "format_instruction": pydantic_parser.get_format_instructions()
    }
)

classifier_chain = prompt1 | model | pydantic_parser

prompt2 = PromptTemplate(
  template ="Write an appropriate response to the positive feedback: {feedback}",
  input_variables=["feedback"],
)

prompt3 = PromptTemplate(
  template ="Write an appropriate response to the negative feedback: {feedback}",
  input_variables=["feedback"],
)


branch_chain = RunnableBranch(
(lambda x:x.sentiment=="positive",prompt2 | model | parser),

(lambda x:x.sentiment=="negative",prompt3 | model | parser),
  RunnableLambda(
        lambda x: "Sorry, I couldn't classify the sentiment of the feedback."
    )
)



chain = classifier_chain | branch_chain

result = chain.invoke({"feedback": "I hate the new design of your website!"})

print(result)