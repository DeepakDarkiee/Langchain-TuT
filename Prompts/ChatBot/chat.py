from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")
chat_history = [
    SystemMessage(content="You are a helpful assistant."),
]

while True:
  user_input = input("You: ")
  chat_history.append(HumanMessage(content=user_input))
  if user_input.lower() in ["exit", "quit"]:
    print("Exiting chat...")
    break
  result= model.invoke(
    chat_history
  )
  chat_history.append(AIMessage(content=result.content))
  print("ChatAI: " + result.content)
print("Chat history:")