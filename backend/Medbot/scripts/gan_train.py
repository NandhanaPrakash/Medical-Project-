import torch
import torch.nn as nn
import torch.optim as optim
import faiss
import numpy as np
import pickle

# File paths
FAISS_INDEX_PATH = "data_clean/index/faiss_refined_index.bin"
EMBEDDINGS_PATH = "data_clean/refined_embeddings.pkl"

# Load Original FAISS Index
index_path = "data_clean/index/faiss_index"  # Your original FAISS index
index = faiss.read_index(index_path)

# Extract stored embeddings
D = index.d  # Dimension of embeddings
faiss_embeddings = np.zeros((index.ntotal, D), dtype=np.float32)

for i in range(index.ntotal):
    faiss_embeddings[i] = index.reconstruct(i)  # Extract embeddings

# Convert to PyTorch Tensor
real_embeddings = torch.tensor(faiss_embeddings, dtype=torch.float32)

# Define Generator Model
class Generator(nn.Module):
    def __init__(self, embedding_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x):
        return self.model(x)

# Define Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, embedding_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Train GAN on FAISS embeddings
def train_gan(real_embeddings, epochs=1000, batch_size=32):
    embedding_dim = real_embeddings.shape[1]
    generator = Generator(embedding_dim)
    discriminator = Discriminator(embedding_dim)

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0001)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001)

    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        for i in range(0, len(real_embeddings), batch_size):
            real_batch = real_embeddings[i:i+batch_size]

            # Generate fake embeddings
            noise = torch.randn_like(real_batch)
            fake_embeddings = generator(noise)

            # Train Discriminator
            real_labels = torch.ones((real_batch.size(0), 1))
            fake_labels = torch.zeros((real_batch.size(0), 1))

            optimizer_D.zero_grad()
            real_loss = loss_fn(discriminator(real_batch), real_labels)
            fake_loss = loss_fn(discriminator(fake_embeddings.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            g_loss = loss_fn(discriminator(fake_embeddings), real_labels)
            g_loss.backward()
            optimizer_G.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: D Loss {d_loss.item()}, G Loss {g_loss.item()}")

    return generator

# Train the model
generator = train_gan(real_embeddings)

# Generate refined embeddings
def generate_refined_embeddings(generator, embeddings):
    refined_embeddings = generator(embeddings).detach().numpy()
    return refined_embeddings

refined_embeddings = generate_refined_embeddings(generator, real_embeddings)

# Load metadata (texts and sources)
with open("data_clean/index/embeddings.pkl", "rb") as f:
    metadata = pickle.load(f)

texts = metadata["texts"]
sources = metadata["sources"]

# Ensure metadata matches refined embeddings
num_embeddings = len(refined_embeddings)
texts = texts[:num_embeddings]
sources = sources[:num_embeddings]

# Save refined FAISS index
index = faiss.IndexFlatL2(D)
index.add(refined_embeddings)
faiss.write_index(index, FAISS_INDEX_PATH)

# Save refined metadata and embeddings
metadata = {
    "texts": texts,
    "sources": sources,
    "embeddings": refined_embeddings.tolist()
}

with open(EMBEDDINGS_PATH, "wb") as f:
    pickle.dump(metadata, f)

print(f"✅ Refined FAISS index and embeddings saved!")
