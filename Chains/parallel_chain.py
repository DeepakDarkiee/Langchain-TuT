from langchain_groq import ChatGroq
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

parser = StrOutputParser()

# Prompt 1
prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text:\n{text}",
    input_variables=["text"]
)

# Prompt 2
prompt2 = PromptTemplate(
    template="Generate a quiz with 5 questions from the following notes:\n{notes}",
    input_variables=["notes"]
)

# Prompt 3
prompt3 = PromptTemplate(
    template="""
Merge the following notes and quiz into a single output.

Notes:
{notes}

Quiz:
{quiz}
""",
    input_variables=["notes", "quiz"]
)

# Step 1: Generate notes
notes_chain = prompt1 | model | parser

# Step 2: Parallel execution
parallel_chain = RunnableParallel(
    {
        "notes": RunnablePassthrough(),
        "quiz": prompt2 | model | parser
    }
)

# Full chain
chain = (
    {"notes": notes_chain}
    | parallel_chain
    | prompt3
    | model
    | parser
)

result = chain.invoke({
    "text": """
Black holes are regions in space where gravity is so strong
that nothing, not even light, can escape.
"""
})

print(result)