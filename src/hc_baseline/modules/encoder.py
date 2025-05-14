import torch
import torch.nn as nn

class ControlFeatureBiLSTMEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=2, emb_dim=512):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, emb_dim),  # 2 for bidirectional
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, x): 
        x = x.permute(0, 2, 1)  # [B, T, 3]

        rnn_out, _ = self.rnn(x)  # [B, T, 2H]
        pooled = rnn_out.mean(dim=1)  # [B, 2H] mean over time

        return self.projector(pooled)  # [B, emb_dim]
