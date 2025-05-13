import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from modules.encoder import ContextEncoder  
from dataset import VimiSketchDataset       
from tqdm import tqdm
import torch.nn.functional as F


# --- Predictor MLP (JEPA-style) ---
class Predictor(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.net(x)

# --- JEPA training function for one epoch ---
def train_jepa(encoder, predictor, dataloader, optimizer, loss_fn, device):
    encoder.train()
    predictor.train()
    total_loss = 0

    for x_vocal, z_clap in tqdm(dataloader, desc="Training"):
        x_vocal = x_vocal.to(device)       # [B, 1, T, 64]
        z_clap = z_clap.to(device)         # [B, 512]

        z_context = encoder(x_vocal)       # [B, 512]
        z_hat = predictor(z_context)       # [B, 512]

        # Normalize
        z_hat = nn.functional.normalize(z_hat, dim=-1)
        z_clap = nn.functional.normalize(z_clap, dim=-1)

        # Cosine loss
        loss = loss_fn(z_hat, z_clap, torch.ones(z_hat.size(0)).to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def train_clr(encoder, dataloader, optimizer, loss_fn, device, temperature=0.07):
    encoder.train()
    total_loss = 0

    for x_vocal, z_clap in tqdm(dataloader, desc="Contrastive Training"):
        x_vocal = x_vocal.to(device)        # [B, 1, T, 64]
        z_clap = z_clap.to(device)          # [B, 512]

        z_vocal = encoder(x_vocal)          # [B, 512]
        
        # Normalize embeddings
        z_vocal = F.normalize(z_vocal, dim=-1)
        z_clap = F.normalize(z_clap, dim=-1)

        # Compute cosine similarity matrix
        logits = torch.matmul(z_vocal, z_clap.T) / temperature  # [B, B]
        labels = torch.arange(z_vocal.size(0)).to(device)       # positives on the diagonal

        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# --- Main training loop ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = ContextEncoder().to(device)
    predictor = Predictor().to(device)

    params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = optim.Adam(params, lr=1e-4, weight_decay=1e-5)
    loss_fn = nn.CosineEmbeddingLoss()

    dataset = VimiSketchDataset(
        imitations_dir="data/vocal/",
        targets_clap_dir="data/clap_targets.npz"
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    for epoch in range(1, 51):
        avg_loss = train_jepa(encoder, predictor, dataloader, optimizer, loss_fn, device)
        print(f"[Epoch {epoch}] JEPA Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()
