import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib  # For saving scalers

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
data = pd.read_csv("bus.csv")

# Sort and Preprocess
data = data.sort_values(by=["route_id", "vehicle_id", "unixtimestamp_scheduled"])
features = ["stop_id", "day", "day_of_year", "scheduled_time"]
target = "delay"

# Normalize Features
scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])
data[target] = target_scaler.fit_transform(data[[target]])

# Save scalers for later use
joblib.dump(scaler, "feature_scaler.pkl")
joblib.dump(target_scaler, "target_scaler.pkl")

# Convert data to tensors
data_tensor = torch.tensor(data[features].values, dtype=torch.float32, device=device)
target_tensor = torch.tensor(data[target].values, dtype=torch.float32, device=device)

# Create Sequences
class BusScheduleDataset(Dataset):
    def __init__(self, data_tensor, target_tensor, seq_length):
        self.seq_length = seq_length
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __len__(self):
        return len(self.data_tensor) - self.seq_length

    def __getitem__(self, idx):
        X = self.data_tensor[idx:idx + self.seq_length]
        y = self.target_tensor[idx + self.seq_length]
        return X, y

seq_length = 20
dataset = BusScheduleDataset(data_tensor, target_tensor, seq_length)

# Split Data into Train/Test
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)

# Define Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2).to(device)
        self.fc = nn.Linear(hidden_dim, output_dim).to(device)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

input_dim = len(features)
hidden_dim = 128
output_dim = 1  # Predicting delay
num_layers = 3

model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(device)

criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



# Training Loop
num_epochs = 200
import tqdm 
bar = tqdm.tqdm(range(num_epochs))
losses = []
best_model = None
best_loss = float('inf')
for epoch in bar:
    model.train()
    train_loss = 0.0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        # Forward Pass
        outputs = model(X)
        loss = criterion(outputs.squeeze(), y)

        # Backward Pass and Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    if train_loss < best_loss:
        best_loss = train_loss
        best_model = model.state_dict()
        print(f"Best model found at epoch {epoch+1}, Loss: {best_loss:.4f}")
    losses.append(train_loss)
    bar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "model.pth")
print("Model saved to model.pth")
pd.DataFrame(losses).to_csv("losses.csv", index=False)
# save graph of los
plt.plot(range(num_epochs), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("loss.png")