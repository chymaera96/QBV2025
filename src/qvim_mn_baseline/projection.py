import torch
import torch.nn as nn
import laion_clap
from qvim_mn_baseline.mn.model import get_model
from qvim_mn_baseline.utils import NAME_TO_WIDTH

class CLAPWithProjection(nn.Module):
    def __init__(self, model_id=1, projection_dim=512, hidden_dim=1024):
        super(CLAPWithProjection, self).__init__()

        # Initialize and freeze CLAP
        enable_fusion = model_id in [2, 3]
        self.model = laion_clap.CLAP_Module(enable_fusion=enable_fusion).to("cuda")
        self.model.load_ckpt(model_id=model_id)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(512, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x):
        with torch.no_grad():  # CLAP stays frozen
            features = self.model.get_audio_embedding_from_filelist(x=x, use_tensor=True)
        
        projected = self.projection(features)
        return projected

class MobileNetWithProjection(nn.Module):
    def __init__(self, pretrained_name="mn10_as", projection_dim=512):
        super().__init__()
        self.backbone = get_model(
            pretrained_name=pretrained_name,
            head_type="mlp",
            width_mult=NAME_TO_WIDTH(pretrained_name),
        )
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        # Grab the MLP's penultimate output (embedding size is 960 for width_mult=1.0)
        self.projection = nn.Sequential(
            nn.Linear(960, 1024),
            nn.ReLU(),
            nn.Linear(1024, projection_dim)
        )

    def forward(self, x):
        _, features = self.backbone(x)  # returns (logits, features)
        return self.projection(features)
