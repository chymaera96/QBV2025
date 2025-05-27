import torch
import torch.nn as nn
from qvim_mn_baseline.mn.model import get_model

# Simulate input Mel spectrogram: batch size 2, 1 channel, 128 mel bands, 1000 time frames
x = torch.randn(2, 1, 128, 1000)

# Get model with MLP head, width multiplier = 1.0
model = get_model(
    pretrained_name="mn10_as",  # or None if you want random weights
    head_type="mlp",
    width_mult=1.0,
    # input_dim_f=128,
    # input_dim_t=1000
)

head = nn.Linear(960, 1024)

# Only use the convolutional feature extractor
# features_module = model.features

# Run input through feature extractor
with torch.no_grad():
    _, out = model(x)

print("Feature shape:", out.shape)
# Run features through MLP head
print("MLP head shape:", head(out).shape)
