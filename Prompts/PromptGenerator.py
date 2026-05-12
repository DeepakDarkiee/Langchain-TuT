
from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["paper_input", "style_input", "length_input"],
    template="You are an expert in summarizing research papers. Please provide a {length_input} summary of the paper '{paper_input}' in a {style_input} style.",
    validate_template=True
) 

template.save("Prompts/prompt_template/research_summarizer.json")