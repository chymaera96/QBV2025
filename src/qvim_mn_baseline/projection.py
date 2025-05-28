import torch
import torch.nn as nn
from hear21passt.base import get_basic_model


class PaSSTWithProjection(nn.Module):
    def __init__(self, projection_dim=512, freeze_backbone=True):
        super().__init__()
        self.backbone = get_basic_model(mode="embed_only")  # exclude classifier

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # PaSST outputs [batch, tokens, 768], so we pool over tokens
        self.projector = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, projection_dim)
        )

    def forward(self, x):
        features = self.backbone(x)  # [B, N, 768]
        print(f"Features shape: {features.shape}")  # Debugging lineW
        pooled = features.mean(dim=1)  # global average pooling over tokens
        return self.projector(pooled)
