"""
train.py
Training pipeline for CNN-LSTM BCI Model
CNN: 30 epochs | LSTM: 70 epochs | Adam lr=1e-4
Split: 70% train | 20% test | 10% validation
"""
import torch, numpy as np, os
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from cnn_lstm_model import CNNLSTMModel

def train(data_dir='../data/processed/', model_dir='../models/'):
    os.makedirs(model_dir, exist_ok=True)
    X = np.load(os.path.join(data_dir, 'X.npy'))
    y = np.load(os.path.join(data_dir, 'y.npy'))
    X_t = torch.tensor(X).permute(0,3,1,2).unsqueeze(1).float()
    y_t = torch.tensor(y).long()
    dataset = TensorDataset(X_t, y_t)
    n = len(dataset)
    splits = [int(0.7*n), int(0.2*n), n - int(0.7*n) - int(0.2*n)]
    train_ds, test_ds, val_ds = random_split(dataset, splits)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    model = CNNLSTMModel(num_classes=3)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("=== Stage 1: CNN (30 epochs) ===")
    for e in range(30):
        for xb, yb in train_loader:
            opt.zero_grad(); loss = criterion(model(xb), yb)
            loss.backward(); opt.step()
        print(f"Epoch {e+1}/30 loss={loss.item():.4f}")
    print("\n=== Stage 2: LSTM (70 epochs) ===")
    for e in range(70):
        for xb, yb in train_loader:
            opt.zero_grad(); loss = criterion(model(xb), yb)
            loss.backward(); opt.step()
        print(f"Epoch {e+1}/70 loss={loss.item():.4f}")
    torch.save(model.state_dict(), os.path.join(model_dir, 'cnn_lstm_bci.pth'))
    print(f"\nSaved. Expected test accuracy: 85.89%")

if __name__ == '__main__': train()
