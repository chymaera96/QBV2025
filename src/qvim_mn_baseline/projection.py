import torch
import os
import torch.nn as nn
from hear21passt.base import get_basic_model
import torch
import torch.nn as nn
from hear21passt.base import get_basic_model

from qvim_mn_baseline.mn.model import get_model
from qvim_mn_baseline.utils import NAME_TO_WIDTH



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
    


class MobileNetWithProjection(nn.Module):
    def __init__(self, pretrained_name="mn10_as", projection_dim=128):
        super().__init__()

        if pretrained_name not in NAME_TO_WIDTH:
            width_mult = 1.0
            ckpt_path = pretrained_name
        else:
            width_mult = NAME_TO_WIDTH(pretrained_name)
            ckpt_path = None

        self.encoder = get_model(
            pretrained_name=pretrained_name,
            head_type="mlp",
            width_mult=width_mult,
        )


        if ckpt_path is not None:
            print(f"Loading pretrained encoder from {ckpt_path}")
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu')
                state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

                # Strict loading to ensure all keys match
                self.encoder.load_state_dict({
                    k.replace('encoder.', ''): v
                    for k, v in state_dict.items()
                    if k.startswith('encoder.')
                }, strict=True)

                print("Pretrained weights loaded successfully.")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print(f"Current working directory: {os.getcwd()}")
                raise RuntimeError("Failed to load pretrained weights. Exiting.")
        else:
            raise ValueError("Pretrained checkpoint path not provided. Exiting.")
        
        for param in self.encoder.parameters():
            param.requires_grad = False

       # Grab the MLP's penultimate output (embedding size is 960 for width_mult=1.0)
        self.projection = nn.Sequential(
            nn.Linear(960, 1024),
            nn.ReLU(),
            nn.Linear(1024, projection_dim)
        )
        # self.head = nn.Linear(960, 1024)

    def forward(self, x):
        _, features = self.encoder(x)  # returns (logits, features)
        return self.projection(features)
