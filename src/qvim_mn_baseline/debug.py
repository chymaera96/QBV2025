# Script to check shape of features from the PaSST model
import torch
from qvim_mn_baseline.projection import PaSSTWithProjection

model = PaSSTWithProjection()
x = torch.randn(3, 320000)
y = model.backbone(x)  # [B, N, 768]
print(f"Output shape from backbone: {y.shape}")  # Should be [3, N, 768]