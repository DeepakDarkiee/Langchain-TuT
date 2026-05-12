from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

texts = ["What is the capital of France?", "What is the largest mammal?", "What is the tallest mountain?"]
results = embeddings.embed_documents(texts)

print(str(results))