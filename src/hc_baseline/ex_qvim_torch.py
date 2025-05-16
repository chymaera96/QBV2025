import argparse
import os
import math
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from hc_baseline.dataset import VimSketchDataset, AESAIMLA_DEV
from qvim_mn_baseline.download import download_vimsketch_dataset, download_qvim_dev_dataset
from hc_baseline.modules.encoder import ControlFeatureBiLSTMEncoder
from hc_baseline.metrics import compute_mrr, compute_ndcg


def cosine_similarity(a, b):
    return torch.matmul(a, b.T)


class QVIMModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = ControlFeatureBiLSTMEncoder(
            input_dim=3, hidden_dim=128, num_layers=2, emb_dim=512
        )
        self.tau = torch.nn.Parameter(torch.tensor([config.initial_tau]), requires_grad=config.tau_trainable)

    def forward(self, x):
        z = self.encoder(x)
        return torch.nn.functional.normalize(z, dim=-1)


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    download_vimsketch_dataset(config.dataset_path)
    download_qvim_dev_dataset(config.dataset_path)

    train_ds = VimSketchDataset(
        os.path.join(config.dataset_path, 'Vim_Sketch_Dataset'),
        sample_rate=config.sample_rate,
        duration=config.duration
    )
    eval_ds = AESAIMLA_DEV(
        os.path.join(config.dataset_path, 'qvim-dev'),
        sample_rate=config.sample_rate,
        duration=config.duration
    )

    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    eval_dl = DataLoader(eval_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = QVIMModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.max_lr, weight_decay=config.weight_decay)

    for epoch in range(config.n_epochs):
        model.train()
        total_loss = 0
        for batch in train_dl:
            optimizer.zero_grad()

            z_i = model(batch["imitation"].to(device))
            z_r = model(batch["reference"].to(device))

            print(f"[DEBUG] ||z_i||: {z_i.norm(dim=1).mean():.4f}, ||z_r||: {z_r.norm(dim=1).mean():.4f}")

            logits = torch.matmul(z_i, z_r.T) / torch.abs(model.tau)

            print(f"[DEBUG] logits: min={logits.min().item():.2f}, max={logits.max().item():.2f}")

            targets = torch.tensor([hash(p) for p in batch["imitation_filename"]])
            mask = torch.tensor(targets[None, :] == targets[:, None], dtype=torch.bool, device=device)
            print(f"[DEBUG] mask.sum(): {mask.sum().item()} (of {mask.numel()})")


            log_probs = torch.log_softmax(logits, dim=1)
            loss = -log_probs[mask].mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Train Loss: {total_loss / len(train_dl):.4f} | Tau: {model.tau.item():.4f}")

        # Validation
        model.eval()
        validation_output = []

        with torch.no_grad():
            for batch in eval_dl:
                z_i = model(batch["imitation"].to(device))
                z_r = model(batch["reference"].to(device))

                validation_output.append({
                    'imitation': z_i.cpu().numpy(),
                    'reference': z_r.cpu().numpy(),
                    'imitation_filename': batch['imitation_filename'],
                    'reference_filename': batch['reference_filename'],
                    'imitation_class': batch['imitation_class'],
                    'reference_class': batch['reference_class']
                })

        # === Same as on_validation_epoch_end ===
        imitations = np.concatenate([b['imitation'] for b in validation_output])
        reference = np.concatenate([b['reference'] for b in validation_output])

        imitation_filenames = sum([b['imitation_filename'] for b in validation_output], [])
        reference_filenames = sum([b['reference_filename'] for b in validation_output], [])
        imitation_classes = sum([b['imitation_class'] for b in validation_output], [])
        reference_classes = sum([b['reference_class'] for b in validation_output], [])

        ground_truth_mrr = {fi: rf for fi, rf in zip(imitation_filenames, reference_filenames)}

        _, unique_indices = np.unique(reference_filenames, return_index=True)
        reference = reference[unique_indices]
        reference_filenames = [reference_filenames[i] for i in unique_indices.tolist()]
        reference_classes = [reference_classes[i] for i in unique_indices.tolist()]

        ground_truth_classes = {
            ifn: [rfn for rfn, rfc in zip(reference_filenames, reference_classes) if rfc == ifc]
            for ifn, ifc in zip(imitation_filenames, imitation_classes)
        }

        scores_matrix = np.dot(imitations, reference.T)
        similarity_df = pd.DataFrame(scores_matrix, index=imitation_filenames, columns=reference_filenames)

        mrr = compute_mrr(similarity_df, ground_truth_mrr)
        ndcg = compute_ndcg(similarity_df, ground_truth_classes)

        print(f"[VAL] MRR: {mrr:.4f}, NDCG: {ndcg:.4f}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default="qvim")
    parser.add_argument('--id', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--model_save_path', type=str, default=None)
    parser.add_argument('--dataset_path', type=str, default='data')
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=15)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--max_lr', type=float, default=0.0003)
    parser.add_argument('--min_lr', type=float, default=0.0001)
    parser.add_argument('--warmup_percent', type=float, default=0.05)
    parser.add_argument('--rampdown_percent', type=float, default=0.8)
    parser.add_argument('--initial_tau', type=float, default=0.07)
    parser.add_argument('--tau_trainable', default=False, action='store_true')
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=800)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--freqm', type=int, default=2)
    parser.add_argument('--timem', type=int, default=200)
    parser.add_argument('--fmin', type=int, default=0)
    parser.add_argument('--fmax', type=int, default=None)
    args = parser.parse_args()

    if args.random_seed:
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)

    train(args)
