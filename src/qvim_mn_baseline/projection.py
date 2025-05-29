import torch
import torch.nn as nn
from hear21passt.base import get_basic_model
import torch
import torch.nn as nn
from hear21passt.base import get_basic_model


class PaSSTSelectiveFineTune(nn.Module):
    def __init__(self, projection_dim=512):
        super().__init__()
        self.backbone = get_basic_model(mode="all")
        self.encoder = self.backbone.net  # This is the PaSST module
        self.mel = self.backbone.mel

        # Freeze all but the last transformer block
        for i, block in enumerate(self.encoder.blocks):
            for param in block.parameters():
                param.requires_grad = (i == len(self.encoder.blocks) - 1)

        # Freeze patch embedding
        for param in self.encoder.patch_embed.parameters():
            param.requires_grad = False

        # Final LayerNorm (keep trainable or not)
        for param in self.encoder.norm.parameters():
            param.requires_grad = True  # You can toggle this

        # Remove classifier heads
        self.backbone.head = nn.Identity()
        self.backbone.head_dist = nn.Identity()
        self.backbone.pre_logits = nn.Identity()

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, projection_dim)
        )


    def forward(self, x):
        assert x.shape[1] == 320000, f"Expected input shape [B, 320000], got {x.shape}"
        X = self.mel(x).unsqueeze(1)
        logits, features = self.encoder(X)          # [B, N, 768]
        # pooled = features.mean(dim=1)       # global avg pooling
        return self.projector(features)       # [B, projection_dim]
