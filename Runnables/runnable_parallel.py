from langchain_groq import ChatGroq
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence
load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

prompt1 = PromptTemplate(
  template= "Generate Linkedin posts for the following topic: {topic}",
  input_variables=["topic"],
)

prompt2 = PromptTemplate(
  template= "Generate Twitter posts for the following topic: {topic}",
  input_variables=["topic"],
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
  'tweet': RunnableSequence(prompt2, model, parser),
  'linkedin': RunnableSequence(prompt1, model, parser)
})

result = parallel_chain.invoke({"topic":"AI in healthcare"})
print(result)
