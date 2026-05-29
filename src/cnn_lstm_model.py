"""
cnn_lstm_model.py
Hybrid CNN-LSTM Architecture for EEG Motor Imagery Classification

CNN:  3 conv blocks (16→32→64 filters) + FC(128) — spatial feature extractor
LSTM: 256 hidden units — temporal sequence classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNExtractor(nn.Module):
    """Spatial feature extractor from 224x224x3 EEG spectrograms."""

    def __init__(self, out_features: int = 128):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))                        # 224 → 112

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))                        # 112 → 56

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2))                        # 56 → 28

        self.fc = nn.Linear(64 * 28 * 28, out_features)

    def forward(self, x):                           # x: (B, 3, 224, 224)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)                           # (B, 128)


class LSTMClassifier(nn.Module):
    """Temporal classifier operating on CNN feature sequences."""

    def __init__(self, input_size: int = 128, hidden_size: int = 256,
                 num_classes: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, num_classes)

    def forward(self, x):                           # x: (B, seq_len, 128)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])               # last time-step → (B, 3)


class CNNLSTMModel(nn.Module):
    """Full hybrid model: CNN spatial extractor + LSTM temporal classifier."""

    def __init__(self, num_classes: int = 3, cnn_features: int = 128,
                 lstm_hidden: int = 256):
        super().__init__()
        self.cnn  = CNNExtractor(out_features=cnn_features)
        self.lstm = LSTMClassifier(input_size=cnn_features,
                                   hidden_size=lstm_hidden,
                                   num_classes=num_classes)

    def forward(self, x):
        # x: (B, seq_len, 3, 224, 224)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x)                         # (B*T, 128)
        feats = feats.view(B, T, -1)               # (B, T, 128)
        return self.lstm(feats)                     # (B, 3)


if __name__ == '__main__':
    model = CNNLSTMModel(num_classes=3)
    dummy = torch.randn(4, 10, 3, 224, 224)        # batch=4, seq=10
    out   = model(dummy)
    print(f'Output shape: {out.shape}')            # (4, 3)
    total = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total:,}')
