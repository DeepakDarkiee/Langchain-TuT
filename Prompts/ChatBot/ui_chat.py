import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)

# Load environment variables
load_dotenv()

# Initialize model
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    streaming=True
)

# Page config
st.set_page_config(
    page_title="Groq Chat Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Groq Chat Assistant")

# Initialize chat history
if "chat_history" not in st.session_state:

    st.session_state.chat_history = [
        SystemMessage(
            content="You are a helpful assistant."
        )
    ]

# Display messages
for message in st.session_state.chat_history:

    # Skip system message from UI
    if isinstance(message, SystemMessage):
        continue

    role = (
        "user"
        if isinstance(message, HumanMessage)
        else "assistant"
    )

    with st.chat_message(role):
        st.markdown(message.content)

# Chat input
user_input = st.chat_input(
    "Type your message..."
)

# When user sends message
if user_input:

    # Add user message to history
    st.session_state.chat_history.append(
        HumanMessage(content=user_input)
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):

        response_stream = model.stream(
            st.session_state.chat_history
        )

        response = st.write_stream(
            chunk.content
            for chunk in response_stream
        )

    # Save AI response
    st.session_state.chat_history.append(
        AIMessage(content=response)
    )