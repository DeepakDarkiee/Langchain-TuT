from langchain_groq import ChatGroq
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnableSequence,
    RunnablePassthrough
)

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Write a fun fact about {topic}",
    input_variables=["topic"],
)

parser = StrOutputParser()

chain1 = RunnableSequence(prompt1, model, parser)
chain2 = RunnableSequence(prompt2, model, parser)

# Preserve original input + generate outputs
parallel_chain1 = RunnableParallel({
    "topic": RunnablePassthrough(),
    "joke": chain1,
})

parallel_chain2 = RunnableParallel({
    "joke": lambda x: x["joke"],
    "fun_fact": lambda x: chain2.invoke(x["topic"])
})

final_chain = RunnableSequence(
    parallel_chain1,
    parallel_chain2
)

result = final_chain.invoke({"topic": "chickens"})

print(result)