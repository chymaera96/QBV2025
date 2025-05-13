import torch
import torch.nn as nn
from mn.model import get_model

class ContextEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()

        # Load pretrained MobileNetV3-Large with AudioSet weights
        self.backbone = get_model(
            num_classes=527,
            pretrained_name="mn10_as",
            head_type="mlp"
        )

        # Remove classifier head (MLP) and use the pooled feature output (dim=1280)
        self.backbone.classifier = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        _, features = self.backbone(x)         # features: [B, 1280]
        return self.projector(features)        # [B, out_dim]
