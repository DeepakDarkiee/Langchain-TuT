from langchain.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

result = embeddings.embed_query("What is the capital of France?")

print(str(result))