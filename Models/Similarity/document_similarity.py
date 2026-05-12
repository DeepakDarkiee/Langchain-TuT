from langchain_huggingface import HuggingFaceEmbeddings

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# Initialize the Hugging Face embeddings model
embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

# Sample documents to compare
documents = [
    "virat kohli is the best batsman in the world",
    "narendra modi is the prime minister of india",
    "mount everest is the tallest mountain in the world",
    "the great wall of china is a famous landmark",
    "the amazon rainforest is the largest tropical rainforest"
]

# Compute embeddings for the documents

query = "who is narendra modi"

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

# Compute cosine similarity between the document embeddings
similarity_matrix = cosine_similarity([query_embedding], doc_embeddings)

# Print the similarity matrix
print("Document Similarity Matrix:", similarity_matrix)
print(np.array2string(similarity_matrix, precision=2))

print("Most similar document:", documents[np.argmax(similarity_matrix)])