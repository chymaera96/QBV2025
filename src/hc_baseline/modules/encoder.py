import torch
import torch.nn as nn

class ControlFeatureBiLSTMEncoder(nn.Module):
    def __init__(self, input_dim=3, conv_dim=64, hidden_dim=128, num_layers=2, emb_dim=512):
        super().__init__()

        self.frontend = nn.Sequential(
            nn.Conv1d(input_dim, conv_dim, kernel_size=5, stride=1, padding=2),  # [B, 64, T]
            nn.BatchNorm1d(conv_dim),
            nn.ReLU(),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU()
        )

        self.rnn1 = nn.LSTM(
            input_size=conv_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.rnn2 = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, x):  # x: [B, 3, T]
        x = self.frontend(x)         # [B, conv_dim, T]
        x = x.permute(0, 2, 1)       # [B, T, conv_dim]

        rnn_out1, _ = self.rnn1(x)     # [B, T, 2H]
        rnn_out2, _ = self.rnn2(rnn_out1)
        pooled = rnn_out2.mean(dim=1)  # [B, 2H]

        return self.projector(pooled)  # [B, emb_dim]
