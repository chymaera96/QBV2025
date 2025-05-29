import torch
import torch.nn as nn
from hear21passt.base import get_basic_model
import torch
import torch.nn as nn
from hear21passt.base import get_basic_model


class PaSSTSelectiveFineTune(nn.Module):
    def __init__(self, projection_dim=512):
        super().__init__()
        self.backbone = get_basic_model(mode="embed_only")
        # self.mel = self.backbone.mel

        # Freeze all parameters first
        for param in self.backbone.net.parameters():
            param.requires_grad = False

        # Unfreeze the last transformer block
        for param in self.backbone.net.blocks[-1].parameters():
            param.requires_grad = True


        # Final LayerNorm (keep trainable or not)
        for param in self.backbone.net.norm.parameters():
            # param.requires_grad = True  # You can toggle this
            param.requires_grad = True

        # Remove classifier heads
        self.backbone.net.head = nn.Identity()
        self.backbone.net.head_dist = nn.Identity()
        self.backbone.net.pre_logits = nn.Identity()

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, projection_dim)
        )


    def forward(self, x):
        assert x.shape[1] == 320000, f"Expected input shape [B, 320000], got {x.shape}"
        # X = self.mel(x).unsqueeze(1)
        # logits, features = self.encoder(X)    
        features = self.backbone(x)      
        # pooled = features.mean(dim=1)       # global avg pooling
        return self.projector(features)       # [B, projection_dim]
