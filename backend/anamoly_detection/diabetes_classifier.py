import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Load dataset
df = pd.read_excel("anamoly.xlsx",header=2)

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace(r"[()]", "", regex=True)

# Rename the label column properly
df.rename(columns={"Diabetic/NonDiabetic_D/N": "Diabetic"}, inplace=True)

# Check the column was renamed
if "Diabetic" not in df.columns:
    raise ValueError("Diabetic column not found. Check your column names!")

# Convert label to numeric
df['Diabetic'] = df['Diabetic'].map({'D': 1, 'N': 0})

# Features and labels
X = df.drop(columns=['Diabetic'])
y = df['Diabetic']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Define the neural network
class DiabetesNet(nn.Module):
    def __init__(self, input_dim):
        super(DiabetesNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Model setup
input_dim = X_train.shape[1]
model = DiabetesNet(input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_labels = (predictions >= 0.5).float()
    accuracy = (predicted_labels == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f"\n✅ Model Accuracy on Test Set: {accuracy * 100:.2f}%")

# Save model
torch.save(model.state_dict(), "diabetes_classifier_model.pth")
print("📦 Model saved to diabetes_classifier_model.pth")
