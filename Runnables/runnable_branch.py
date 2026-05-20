from langchain_groq import ChatGroq
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableBranch
load_dotenv()

def word_count(input: str) -> int:
    return len(input.split())

model = ChatGroq(model="llama-3.3-70b-versatile")

prompt1 = PromptTemplate(
    template="Write a detailed description of {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="summarize {topic}",
    input_variables=["topic"],
)

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
  (lambda x: len(x.split()) > 100, RunnableSequence(prompt2, model, parser)),
  RunnablePassthrough()
)

final_chain = RunnableSequence(
    report_gen_chain,
    branch_chain
)

result = final_chain.invoke({"topic": "AI in healthcare"})  

print(result)
