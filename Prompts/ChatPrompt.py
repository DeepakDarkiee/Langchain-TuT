from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant that explains concepts in {domain}."),
    ("human", "Can you explain the concept of {concept}?")
])

prompt = chat_template.invoke({
    "domain": "machine learning",
    "concept": "overfitting"
})

print(prompt)