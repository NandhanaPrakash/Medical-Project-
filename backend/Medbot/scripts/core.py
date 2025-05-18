import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import faiss
import pandas as pd
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers

# ===== CONFIGURATION =====
DB_FAISS_PATH = "C:/Users/Dell/Downloads/medical_project/medical_project/Medbot/data_clean/index_new"
DATA_CSV_PATH = "C:/Users/Dell/Downloads/medical_project/medical_project/Medbot/data_clean/processed/medqa_cleaned.csv"
MODEL_DEVICE = "cpu"  # Force CPU usage

# Disable parallel tokenizers and limit threads
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
faiss.omp_set_num_threads(1)

# ===== LOAD DATA =====
try:
    df = pd.read_csv(DATA_CSV_PATH)
    corpus = df["text_chunk"].tolist()
    print(f"✅ Loaded corpus with {len(corpus)} documents")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load corpus: {str(e)}")

# ===== INITIALIZE COMPONENTS =====
# 1. Load Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': MODEL_DEVICE}
)

# 2. Load FAISS Index
try:
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    print("✅ FAISS Index Loaded Successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Error loading FAISS index: {e}")

# 3. Load LLM (CTransformers)
try:
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        config={"max_new_tokens": 256, "temperature": 0.5}
    )
    print("✅ LLM Model Loaded Successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Error loading LLM model: {e}")

# 4. Load Summarizer
try:
    summarizer = pipeline(
        "summarization", 
        model="facebook/bart-large-cnn", 
        device=-1  # CPU only
    )
    print("✅ Summarizer Loaded Successfully!")
except Exception as e:
    summarizer = None
    print(f"⚠️ Warning: Summarizer not available: {e}")

# ===== GAN MODEL DEFINITION =====
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize GAN components
embedding_dim = 384
hidden_dim = 256
device = torch.device("cpu")

generator = Generator(embedding_dim, hidden_dim).to(device)
discriminator = Discriminator(embedding_dim, hidden_dim).to(device)

criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0005)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0005)

# ===== CORE FUNCTIONS =====
def retrieve_with_gan(query, top_k=3):
    """Retrieve documents with GAN-enhanced filtering"""
    print(f"\nSearching for: {query}")
    
    query_embedding = np.array(embeddings.embed_query(query)).astype(np.float32)
    distances, indices = db.index.search(np.expand_dims(query_embedding, axis=0), top_k)
    print(f"Found indices: {indices}")

    retrieved_texts = [corpus[idx] for idx in indices[0] if 0 <= idx < len(corpus)]
    if not retrieved_texts:
        return "No relevant answer found."

    retrieved_embeddings = np.array(embeddings.embed_documents(retrieved_texts)).astype(np.float32)
    generated_embeddings = generator(torch.randn_like(
        torch.tensor(retrieved_embeddings, dtype=torch.float32).to(device)))

    similarities = torch.cosine_similarity(
        torch.tensor(retrieved_embeddings, dtype=torch.float32).to(device), 
        generated_embeddings.detach()
    ).cpu().numpy()

    best_index = np.argmax(similarities)
    return retrieved_texts[best_index] if best_index < len(retrieved_texts) else "No relevant answer found."

def process_response(response):
    """Process and optionally summarize the response"""
    print(f"\nProcessing response: {response[:100]}...")
    
    if not response:
        return "No relevant answer found."
    
    if summarizer:
        try:
            summarized = summarizer(
                response, 
                max_length=150, 
                min_length=80, 
                do_sample=False
            )
            return summarized[0]["summary_text"]
        except Exception as e:
            print(f"Summarization failed: {e}")
    
    return response

def generate_final_response(query, retrieved_text):
    """Generate final response using LLM"""
    print(f"\nGenerating response for:\nQ: {query}\nA: {retrieved_text[:200]}...")
    
    prompt = f"User Query: {query}\n\nRetrieved Medical Info: {retrieved_text}\n\nProvide a well-structured medical response:"
    response = llm.invoke(prompt)
    print(f"Raw LLM response: {response}")
    
    return response.strip() if response else "No relevant answer found."