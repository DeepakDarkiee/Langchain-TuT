from langchain_groq import ChatGroq
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

st.title("Research Paper Summarizer")

st.subheader("ChatGroq LLM Demo")

paper_input = st.selectbox("Select Research Paper Name",["Select...","Attention Is All You Need","BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding","GPT-3: Language Models are Few-Shot Learners"])

style_input = st.selectbox("Select Response Style",["Select...","Formal","Informal","Technical","Simple"])

length_input = st.selectbox("Select Response Length",["Select...","Short","Medium","Long"])

template = load_prompt("Prompts/prompt_template/research_summarizer.json")

prompt = template.invoke(
  {'paper_input': paper_input, 'style_input': style_input, 'length_input': length_input}
  )

if st.button("Summarize Paper"):

    if (
        paper_input != "Select..."
        and style_input != "Select..."
        and length_input != "Select..."
    ):

        st.subheader("LLM Response:")

        response = model.stream(prompt)
        with st.spinner("Generating summary..."):
            st.write_stream(
                chunk.content for chunk in response
            )

    else:
        st.warning("Please select all options.")