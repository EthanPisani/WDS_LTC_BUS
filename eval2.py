import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
# import umap
import shap
import joblib
import os

# Ensure the output directory exists
os.makedirs("analysis_results", exist_ok=True)
os.makedirs("analysis_results/epoch200", exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the LSTM model architecture
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2).to(device)
        self.fc = nn.Linear(hidden_dim, output_dim).to(device)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])
# Load the saved model
input_dim = 4  # Number of features
hidden_dim = 128
output_dim = 1  # Predicting delay
num_layers = 3
model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
scaler = joblib.load("feature_scaler.pkl")
target_scaler = joblib.load("target_scaler.pkl")

# Load and preprocess data
data = pd.read_csv("bus.csv")
features = ["stop_id", "day", "day_of_year", "scheduled_time"]
target = "delay"
data[features] = scaler.transform(data[features])
data[target] = target_scaler.transform(data[[target]])

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

# Evaluation
predictions = []
true_values = []

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        predictions.extend(outputs.squeeze().cpu().numpy())
        true_values.extend(y.cpu().numpy())

# Inverse transform predictions and true values
predictions = target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
true_values = target_scaler.inverse_transform(np.array(true_values).reshape(-1, 1))

# Save predictions and true values
results_df = pd.DataFrame({"True Values": true_values.flatten(), "Predictions": predictions.flatten()})
results_df.to_csv("analysis_results/epoch200/predictions_and_true_values.csv", index=False)

# Performance Metrics
mae = mean_absolute_error(true_values, predictions)
rmse = np.sqrt(mean_squared_error(true_values, predictions))
r2 = r2_score(true_values, predictions)

metrics = {
    "Mean Absolute Error (MAE)": mae,
    "Root Mean Squared Error (RMSE)": rmse,
    "R-squared (R²)": r2
}

# Save metrics to a text file
with open("analysis_results/epoch200/performance_metrics.txt", "w") as f:
    for metric, value in metrics.items():
        f.write(f"{metric}: {value:.4f}\n")

# Plot Actual vs. Predicted
plt.figure(figsize=(12, 6))
plt.plot(true_values[:100], label="True Values")
plt.plot(predictions[:100], label="Predictions")
plt.title("Actual vs. Predicted Delays (First 100 Samples)")
plt.xlabel("Time Steps")
plt.ylabel("Delay (Seconds)")
plt.legend()
plt.grid(True)
plt.savefig("analysis_results/epoch200/actual_vs_predicted.png")
plt.show()

# Error Distribution Plot
errors = true_values - predictions
# clamp to -500, 500
errors = np.clip(errors, -1000, 1000)

plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True, bins=30)
plt.title("Error Distribution (True Values - Predictions)")
plt.xlabel("Error (Seconds)")
plt.ylabel("Frequency")
plt.savefig("analysis_results/epoch200/error_distribution.png")
plt.show()

# Calculate absolute errors
absolute_errors = np.abs(errors)
absolute_errors /= 60  # Convert seconds to minutes
# Define error bins
bins = [0, 1, 2, 3, 4, 5, 10, np.inf]
labels = ["0-1 min", "1-2 min", "2-3 min", "3-4 min", "4-5 min", "5-10 min", "10+ min"]
# colors are green to red, bright green for 0-1 min, bright red for 10+ min
colors = ["#00FF00", "#99FF00", "#66FF00", "#99FF00", "#FFCC00", "#FF9900", "#FF1100"]
# Categorize errors into bins
error_categories = pd.cut(absolute_errors.flatten(), bins=bins, labels=labels, right=False)

# Count the number of errors in each category
error_counts = error_categories.value_counts().sort_index()

# Calculate percentages
error_percentages = error_counts / error_counts.sum() * 100

# Plot pie chart
plt.figure(figsize=(8, 8))
plt.pie(error_percentages, labels=error_percentages.index, autopct="%1.1f%%", colors=colors)
plt.title("Percentage of Errors by Range")
plt.savefig("analysis_results/epoch200/error_pie_chart.png")
plt.show()
# Feature Importance using Permutation Importance
X_test = []
y_test = []
for X, y in test_loader:
    X_test.append(X.cpu().numpy())
    y_test.append(y.cpu().numpy())

X_test = np.vstack(X_test)  # Shape: (num_samples, seq_length, input_dim)
y_test = np.hstack(y_test)  # Shape: (num_samples,)

# Reshape X_test to 2D by averaging over the sequence dimension
X_test_2d = X_test.mean(axis=1)  # Shape: (num_samples, input_dim)

# # Calculate permutation importance
# result = permutation_importance(
#     model, X_test_2d, y_test, n_repeats=10, random_state=42, n_jobs=-1
# )

# # Plot Feature Importance
# feature_importance = result.importances_mean
# plt.figure(figsize=(10, 6))
# plt.bar(range(len(features)), feature_importance, tick_label=features)
# plt.title("Feature Importance (Permutation Importance)")
# plt.xlabel("Features")
# plt.ylabel("Importance")
# plt.savefig("analysis_results/epoch200/feature_importance.png")
# plt.show()

# Latent Space Visualization (t-SNE)
latent_representations = []
with torch.no_grad():
    for X, _ in test_loader:
        latent = model.lstm(X.to(device))[0]  # Extract LSTM hidden states
        latent_representations.append(latent.cpu().numpy())
latent_representations = np.vstack(latent_representations)

# # Apply t-SNE
# tsne = TSNE(n_components=2)
# latent_2d = tsne.fit_transform(latent_representations)

# # Plot t-SNE
# plt.figure(figsize=(10, 6))
# plt.scatter(latent_2d[:, 0], latent_2d[:, 1])
# plt.title("t-SNE of Latent Space")
# plt.savefig("analysis_results/epoch200/latent_space_tsne.png")
# plt.show()

# SHAP Analysis (for interpretability)
# Convert X_test to a PyTorch tensor
# X_test_tensor = torch.tensor(X_test[:100], dtype=torch.float32, device=device)

# # Temporarily switch the model to training mode
# model.train()

# # Initialize SHAP DeepExplainer
# explainer = shap.DeepExplainer(model, X_test_tensor)

# # Calculate SHAP values
# shap_values = explainer.shap_values(X_test_tensor)

# # Switch the model back to evaluation mode
# model.eval()

# # Plot SHAP summary
# shap.summary_plot(shap_values, X_test[:100], feature_names=features, show=False)
# plt.savefig("analysis_results/epoch200/shap_summary.png")
# plt.show()

from torchviz import make_dot
x = torch.randn(1, seq_length, input_dim).to(device)
y = model(x)
make_dot(y, params=dict(model.named_parameters())).render("model_architecture", format="png")

# from torchsummary import summary
# size = pd.DataFrame(data = [seq_length, input_dim])
# summary(model, size)
# for feature_idx in range(input_dim):
#     print(X_test)
#     X_perturbed = X_test[435]
#     print(X_perturbed)

#     X_perturbed[:, :, feature_idx] += 0.1  # Perturb the feature
#     y_perturbed = model(X_perturbed)
#     print(f"Change in predictions for feature {feature_idx}: {torch.mean(y_perturbed - y_test[435])}")


import optuna
# import torch
# import tqdm
# def objective(trial):
#     lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
#     hidden_dim = trial.suggest_int("hidden_dim", 64, 256)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     # Train and evaluate model
#     validation_loss = 0
#     num_epochs = 10
#     criterion = nn.MSELoss()
#     for epoch in tqdm.tqdm(range(num_epochs)):
#         model.train()
#         for X, y in train_loader:
#             optimizer.zero_grad()
#             outputs = model(X)
#             loss = criterion(outputs, y)
#             loss.backward()
#             optimizer.step()
#     return validation_loss

# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=50)