import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os

# STEP 1: Load CSV
csv_path = "/Users/s_lokesh/Medbot/data_clean/processed/medqa_cleaned.csv"
df = pd.read_csv(csv_path)

# STEP 2: Use the correct text column
if 'text_chunk' in df.columns:
    texts = df['text_chunk'].astype(str).tolist()
else:
    raise ValueError(
        "Column 'text_chunk' not found. Please verify your CSV structure.")

# STEP 3: Convert to LangChain Documents (optionally add metadata if needed)
documents = [Document(page_content=text) for text in texts]

# STEP 4: Define Embedding Model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

# STEP 5: Create FAISS Index
vectorstore = FAISS.from_documents(documents, embedding)

# STEP 6: Save the index in a separate directory
save_dir = "/Users/s_lokesh/Medbot/data_clean/index_new"
os.makedirs(save_dir, exist_ok=True)

vectorstore.save_local(save_dir)

print(f"✅ FAISS index created and saved to: {save_dir}")
