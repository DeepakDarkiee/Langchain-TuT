from langchain_groq import ChatGroq
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnableLambda, RunnablePassthrough
load_dotenv()

def word_count(input: str) -> int:
    return len(input.split())

model = ChatGroq(model="llama-3.3-70b-versatile")

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"],
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "word_count": RunnableLambda(word_count),
})

final_chain = RunnableSequence(
    joke_gen_chain,
    parallel_chain
)

result = final_chain.invoke({"topic": "chickens"})

print(result)
