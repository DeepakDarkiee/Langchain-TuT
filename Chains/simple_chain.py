from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template="Generate 5 Intresting facts about {Topic}",
    input_variables=["Topic"]
)

model = ChatGroq(model="llama-3.3-70b-versatile")

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"Topic": "Space Exploration"})

print(result)

chain.get_graph().print_ascii()