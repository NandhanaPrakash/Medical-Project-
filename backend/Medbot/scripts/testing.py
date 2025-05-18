import os
import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import chainlit as cl
import requests

# -------------------- CONFIGURATION --------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
faiss.omp_set_num_threads(1)  # Prevents multithreading issues

# Use relative paths or environment variables for deployment
DB_FAISS_PATH = os.getenv("DB_FAISS_PATH", "data_clean/index_new")
DATA_CSV_PATH = os.getenv("DATA_CSV_PATH", "data_clean/processed/medqa_cleaned.csv")
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cpu")  # Force CPU usage

# -------------------- LOAD DATA --------------------
try:
    df = pd.read_csv(DATA_CSV_PATH)
    corpus = df["text_chunk"].tolist()
except Exception as e:
    raise RuntimeError(f"❌ Error loading data: {e}")

# -------------------- LOAD EMBEDDINGS --------------------
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': MODEL_DEVICE}
    )
    print("✅ Embeddings Loaded Successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Error loading embeddings: {e}")

# -------------------- LOAD FAISS INDEX --------------------
try:
    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(f"FAISS index file not found at {DB_FAISS_PATH}")
    
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    print("✅ FAISS Index Loaded Successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Error loading FAISS index: {e}")

# -------------------- LOAD SUMMARIZER --------------------
try:
    summarizer = pipeline(
        "summarization", 
        model="facebook/bart-large-cnn", 
        device=-1  # CPU Only
    )
    print("✅ Summarizer Loaded Successfully!")
except Exception as e:
    summarizer = None
    print(f"⚠️ Warning: Summarizer not available: {e}")

# -------------------- LOAD LLM (CTransformers) --------------------
try:
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_file="llama-2-7b-chat.ggmlv3.q4_0.bin",  # Specify model file
        model_type="llama",
        config={
            "max_new_tokens": 256, 
            "temperature": 0.5,
            "context_length": 2048  # Added context length
        }
    )
    print("✅ LLM Model Loaded Successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Error loading LLM model: {e}")

# -------------------- GAN MODEL FOR RETRIEVAL OPTIMIZATION --------------------
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

embedding_dim = 384
hidden_dim = 256
device = torch.device("cpu")

generator = Generator(embedding_dim, hidden_dim).to(device)
discriminator = Discriminator(embedding_dim, hidden_dim).to(device)

# -------------------- RETRIEVAL AND RESPONSE FUNCTIONS --------------------
def retrieve_with_gan(query, top_k=3):
    try:
        query_embedding = np.array(embeddings.embed_query(query)).astype(np.float32)
        distances, indices = db.index.search(np.expand_dims(query_embedding, axis=0), top_k)

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
    except Exception as e:
        print(f"Error in retrieval: {e}")
        return "Error retrieving information."

def process_response(response):
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
            print(f"Summarization error: {e}")
    return response

def generate_final_response(query, retrieved_text):
    if not retrieved_text:
        return "No relevant answer found."

    prompt_template = """Use the following medical information to answer the user's question. 
    Be precise and professional in your response.

    Question: {query}
    Medical Context: {context}

    Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["query", "context"]
    )
    
    try:
        response = llm(prompt.format(query=query, context=retrieved_text))
        return response.strip() if response else "No relevant answer found."
    except Exception as e:
        print(f"LLM generation error: {e}")
        return "Error generating response."

# -------------------- CHAINLIT INTEGRATION --------------------
@cl.on_chat_start
async def start():
    await cl.Message(content="Hello! I'm your medical assistant. How can I help you today?").send()

@cl.on_message
async def main(message: cl.Message):
    try:
        user_input = message.content
        
        # Step 1: Retrieve relevant information
        retrieved_text = retrieve_with_gan(user_input)
        
        # Step 2: Process the response
        processed_text = process_response(retrieved_text)
        
        # Step 3: Generate final response
        final_response = generate_final_response(user_input, processed_text)
        
        # Send the response
        await cl.Message(content=final_response).send()
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an error processing your request: {str(e)}"
        await cl.Message(content=error_msg).send()