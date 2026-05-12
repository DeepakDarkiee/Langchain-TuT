from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful support agent."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{query}?"),
])

chat_history = []
with open("./chat_history.txt", "r") as f:
    chat_history.extend(f.read().splitlines())
    


print(chat_template.invoke({
    "history": chat_history,
    "query": "Can you help me with my order"
}))