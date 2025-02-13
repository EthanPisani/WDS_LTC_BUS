import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import joblib
import matplotlib.pyplot as plt
import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
data = pd.read_csv("bus.csv")

# Handle missing values
data.dropna(inplace=True)

# Preprocess and Feature Engineering
features = ["stop_id", "day", "day_of_year", "scheduled_time"]
target = "delay"

# Convert to datetime and extract time features
data['datetime'] = pd.to_datetime(data['unixtimestamp_scheduled'], unit='s')
data['hour'] = data['datetime'].dt.hour
data['minute'] = data['datetime'].dt.minute
features += ['hour', 'minute']

# Sort and Preprocess
data = data.sort_values(by=["route_id", "vehicle_id", "unixtimestamp_scheduled"])

# Normalize Features
scaler = StandardScaler()
target_scaler = StandardScaler()
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

# Data Loaders with Dynamic Batch Sizes
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# Define Model with Improvements
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2, bidirectional=True).to(device)
        self.fc = nn.Linear(hidden_dim * 2, output_dim).to(device)  # Multiply by 2 for bidirectional
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out, (hidden, _) = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # Take the last hidden state
        return self.fc(out)

input_dim = len(features)
print(features)
hidden_dim = 128
output_dim = 1
num_layers = 3

model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(device)

# Define optimizer and loss function
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Early stopping
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, current_loss):
        if current_loss < (self.best_loss - self.min_delta):
            self.best_loss = current_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            else:
                return False

early_stopping = EarlyStopping(patience=10, min_delta=0.001)

# TensorBoard for visualization
writer = SummaryWriter("logs")

# Training Loop
num_epochs = 200
bar = tqdm.tqdm(range(num_epochs))
best_model = None
best_train_loss = float('inf')
best_val_loss = float('inf')

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs.squeeze(), y)
            val_loss += loss.item()

    val_loss /= len(test_loader)

    # Update learning rate scheduler
    scheduler.step(val_loss)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model.state_dict()
        torch.save(best_model, "best_model.pth")

    # Early stopping
    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch+1}")
        break

    # TensorBoard logging
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)

    # Update progress bar
    bar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save final model
torch.save(model.state_dict(), "model.pth")

# Close TensorBoard writer
writer.close()

# Evaluate the model
model.load_state_dict(best_model)
model.eval()
predicted = []
actual = []
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        predicted.extend(outputs.squeeze().cpu().numpy())
        actual.extend(y.cpu().numpy())

# Calculate additional metrics
mse = mean_squared_error(actual, predicted)
mae = mean_absolute_error(actual, predicted)
print(f"Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")

# Plot predicted vs actual
plt.figure(figsize=(10, 6))
plt.plot(actual[:100], label='Actual')
plt.plot(predicted[:100], label='Predicted')
plt.xlabel('Time')
plt.ylabel('Delay')
plt.title('Actual vs Predicted Delay')
plt.legend()
plt.grid(True)
plt.savefig('actual_vs_predicted.png')
