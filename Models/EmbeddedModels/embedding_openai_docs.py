from langchain.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

documents = ["What is the capital of France?", "What is the largest mammal?", "What is the tallest mountain?"]
results = embeddings.embed_documents(documents)

print(str(results))