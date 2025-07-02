import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from preprocessing import load_and_prepare

# Set random seed
torch.manual_seed(42)
np.random.seed(42)


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])  # Take output from last time step
        return out


def prepare_sequences(X, y, time_steps=10):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)


def train_lstm_pytorch(X, y):
    os.makedirs('results', exist_ok=True)

    # Normalize
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    # Create sequences
    time_steps = 10
    X_seq, y_seq = prepare_sequences(X_scaled, y_scaled, time_steps)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # DataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    # Model, loss, optimizer
    model = LSTMRegressor(input_size=X.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    train_losses = []
    for epoch in range(50):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

    # Predict
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).numpy()
    y_pred = scaler_y.inverse_transform(y_pred)
    y_test = scaler_y.inverse_transform(y_test.numpy())

    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)

    # Save metrics
    metrics_df = pd.DataFrame([{
        'Model': 'LSTM_PyTorch',
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae
    }])
    metrics_df.to_csv('results/lstm_pytorch_metrics.csv', index=False)

    # Save loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('PyTorch LSTM Training Loss Curve')
    plt.tight_layout()
    plt.savefig('results/lstm_pytorch_loss_curve.png')
    plt.close()

    print("✅ LSTM (PyTorch) training complete. Metrics and plot saved in results/.")


if __name__ == "__main__":
    df_min, X, y, _ = load_and_prepare()
    train_lstm_pytorch(X, y)
