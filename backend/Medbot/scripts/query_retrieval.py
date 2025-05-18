import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# File paths
FAISS_INDEX_PATH = "data_clean/index/faiss_index"
EMBEDDINGS_PATH = "data_clean/index/embeddings.pkl"

# Load FAISS index
index = faiss.read_index(FAISS_INDEX_PATH)

# Load metadata
with open(EMBEDDINGS_PATH, "rb") as f:
    metadata = pickle.load(f)

texts = metadata["texts"]
sources = metadata["sources"]

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def retrieve_relevant_chunks(query, top_k=5):
    """Retrieve top_k most similar text chunks from FAISS index"""
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Direct FAISS search (efficient)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        if idx >= len(texts):  # Prevent out-of-bounds errors
            continue

        results.append({
            "source": sources[idx],
            "text": texts[idx],
            "score": distances[0][i]  # Lower is better
        })

    return results

# Example query
query = "What are the symptoms of diabetes?"
retrieved_chunks = retrieve_relevant_chunks(query)

# Print retrieved results
for i, res in enumerate(retrieved_chunks):
    print(f"Rank {i+1}: (Score: {res['score']})\nSource: {res['source']}\nText: {res['text']}\n")
