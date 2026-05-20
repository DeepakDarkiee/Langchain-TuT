from langchain_groq import ChatGroq
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")


prompt = PromptTemplate(
  template="Write a joke about {topic}",
  input_variables=["topic"],
)

parser = StrOutputParser()

chain  = RunnableSequence(prompt , model , parser)

result = chain.invoke({"topic":"chickens"})
print(result)