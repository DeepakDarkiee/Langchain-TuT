from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.1,   # lower temperature for structured output
    max_new_tokens=100
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(gt=18, description="The age of the person")
    city: str = Field(description="The city of the person")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="""
Generate details of a fictional {Place} person.

{format_instructions}

Return ONLY valid JSON.
""",
    input_variables=["Place"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

chain = template | model | parser

result = chain.invoke({"Place": "Indian"})

print(result)