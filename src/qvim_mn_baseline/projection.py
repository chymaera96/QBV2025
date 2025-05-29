import torch
import torch.nn as nn
from hear21passt.base import get_basic_model


class PaSSTSelectiveFineTune(nn.Module):
    def __init__(self, projection_dim=512):
        super().__init__()
        self.backbone = get_basic_model(mode="all")  # full model w/ head

        # Access internal PaSST encoder
        self.encoder = self.backbone.net

        # Freeze all transformer blocks except the last
        for i, block in enumerate(self.encoder.blocks):
            for param in block.parameters():
                param.requires_grad = (i == len(self.encoder.blocks) - 1)

        # Freeze patch embed, cls token, and positional embeddings
        for param in self.encoder.patch_embed.parameters():
            param.requires_grad = False
        self.encoder.cls_token.requires_grad = False
        self.encoder.pos_embed.requires_grad = False

        # (Optional) freeze the final normalization layer
        for param in self.encoder.norm.parameters():
            param.requires_grad = True  # set to False to freeze it

        # Remove classification head
        self.backbone.head = nn.Identity()
        self.backbone.head_dist = nn.Identity()
        self.backbone.pre_logits = nn.Identity()

        # Add projection head
        self.projector = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, projection_dim)
        )

    def forward(self, x):
        assert x.shape[1] == 320000, f"Expected input shape [B, 320000], got {x.shape}"
        features = self.encoder(x)          # [B, N, 768]
        # pooled = features.mean(dim=1)       # global avg pooling
        return self.projector(features)       # [B, projection_dim]
