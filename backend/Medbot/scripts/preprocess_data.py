import os
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize

# Download NLTK tokenization models
nltk.download('punkt_tab')

# Set path to MedQA textbooks folder
TEXTBOOKS_FOLDER = "/Users/s_lokesh/Medbot/data_clean/textbooks/en"  # Change this to your actual path
OUTPUT_CSV = "/Users/s_lokesh/Medbot/data_clean/processed/medqa_cleaned.csv"

# Chunking parameters
CHUNK_SIZE = 512  # Number of tokens per chunk
OVERLAP_RATIO = 0.2  # 20% overlap

def clean_text(text):
    """Remove extra spaces and normalize text."""
    text = text.replace("\n", " ").strip()
    return " ".join(text.split())

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap_ratio=OVERLAP_RATIO):
    """Split text into overlapping chunks."""
    words = word_tokenize(text)
    chunks = []
    step = int(chunk_size * (1 - overlap_ratio))
    
    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
    
    return chunks

def process_textbooks(folder_path):
    """Read all .txt files, clean, chunk, and store in a DataFrame."""
    data = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            clean_text_data = clean_text(text)
            chunks = chunk_text(clean_text_data)
            
            for chunk in chunks:
                data.append({"source": filename, "text_chunk": chunk})
    
    return pd.DataFrame(data)

# Run processing
df = process_textbooks(TEXTBOOKS_FOLDER)

# Save to CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"Processed data saved to {OUTPUT_CSV}")
