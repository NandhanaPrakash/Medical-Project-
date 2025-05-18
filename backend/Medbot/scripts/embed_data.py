import os
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# Paths
CSV_FILE = "data_clean/processed/medqa_cleaned.csv"
FAISS_INDEX_PATH = "data_clean/index/faiss_index"
EMBEDDINGS_PATH = "data_clean/index/embeddings.pkl"

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load processed data
df = pd.read_csv(CSV_FILE)

# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(df["text_chunk"].tolist(), convert_to_numpy=True)

# Create FAISS index
d = embeddings.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# Save FAISS index
os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
faiss.write_index(index, FAISS_INDEX_PATH)

# Save embeddings & metadata (for retrieval)
with open(EMBEDDINGS_PATH, "wb") as f:
    pickle.dump({"index": FAISS_INDEX_PATH, "texts": df["text_chunk"].tolist(), "sources": df["source"].tolist()}, f)

print(f"FAISS index saved at: {FAISS_INDEX_PATH}")
