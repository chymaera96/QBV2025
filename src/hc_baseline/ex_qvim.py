import argparse
import os
import math
import copy

from copy import deepcopy
import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

# from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from hc_baseline.dataset import VimSketchDataset, AESAIMLA_DEV
from qvim_mn_baseline.download import download_vimsketch_dataset, download_qvim_dev_dataset
from hc_baseline.metrics import compute_mrr, compute_ndcg
from hc_baseline.modules.encoder import ControlFeatureBiLSTMEncoder


class QVIMModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = ControlFeatureBiLSTMEncoder(
            input_dim=3, hidden_dim=128, num_layers=2, emb_dim=512
        )

        initial_tau = torch.zeros((1,)) + config.initial_tau
        self.tau = torch.nn.Parameter(initial_tau, requires_grad=config.tau_trainable)

        self.validation_output = []

    def forward(self, queries, items):
        return self.forward_imitation(queries), self.forward_reference(items)

    def forward_imitation(self, imitations):
        y_imitation = self.encoder(imitations)
        y_imitation = torch.nn.functional.normalize(y_imitation, dim=1)
        return y_imitation

    def forward_reference(self, items):
        y_reference = self.encoder(items)
        y_reference = torch.nn.functional.normalize(y_reference, dim=1)
        return y_reference

    def training_step(self, batch, batch_idx):
        self.lr_scheduler_step(batch_idx)

        y_imitation, y_reference = self.forward(batch['imitation'], batch['reference'])
        C = torch.matmul(y_imitation, y_reference.T)
        C = C / torch.abs(self.tau)
        C_text = torch.log_softmax(C, dim=1)

        paths = np.array([hash(p) for p in batch['imitation_filename']])
        I = torch.tensor(paths[None, :] == paths[:, None])

        loss = - C_text[torch.where(I)].mean()

        self.log('train/loss', loss, prog_bar=True)
        self.log('train/tau', self.tau)
        # print(f"[TRAIN] Loss: {loss.item():.4f}, Tau: {self.tau.item():.4f}")

        return loss

    def validation_step(self, batch, batch_idx):
        y_imitation, y_reference = self.forward(batch['imitation'], batch['reference'])

        C = torch.matmul(y_imitation, y_reference.T)
        C = C / torch.abs(self.tau)
        C_text = torch.log_softmax(C, dim=1)

        paths = np.array([hash(p) for p in batch['imitation_filename']])
        I = torch.tensor(paths[None, :] == paths[:, None])

        loss = - C_text[torch.where(I)].mean()

        self.log('val/loss', loss)
        self.log('val/tau', self.tau)

        self.validation_output.append({
            'imitation': copy.deepcopy(y_imitation.detach().cpu().numpy()),
            'reference': copy.deepcopy(y_reference.detach().cpu().numpy()),
            'imitation_filename': batch['imitation_filename'],
            'reference_filename': batch['reference_filename'],
            'imitation_class': batch['imitation_class'],
            'reference_class': batch['reference_class']
        })

    def on_validation_epoch_end(self):
        validation_output = self.validation_output
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

        self.log('val/mrr', mrr, prog_bar=True)
        self.log('val/ndcg', ndcg, prog_bar=True)
        # print(f"[VAL] MRR: {mrr:.4f}, NDCG: {ndcg:.4f}")

        self.validation_output = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.config.weight_decay,
            amsgrad=False
        )
        return optimizer

    def lr_scheduler_step(self, batch_idx):
        steps_per_epoch = self.trainer.num_training_batches
        min_lr = self.config.min_lr
        max_lr = self.config.max_lr
        current_step = self.current_epoch * steps_per_epoch + batch_idx

        warmup_epochs = self.config.n_epochs * self.config.warmup_percent
        rampdown_epochs = self.config.n_epochs * self.config.rampdown_percent
        warmup_steps = int(warmup_epochs * steps_per_epoch)
        total_steps = int((warmup_epochs + rampdown_epochs) * steps_per_epoch)
        decay_steps = total_steps - warmup_steps

        if current_step < warmup_steps:
            lr = min_lr + (max_lr - min_lr) * (current_step / warmup_steps)
        elif current_step < total_steps:
            decay_progress = (current_step - warmup_steps) / decay_steps
            lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * decay_progress))
        else:
            lr = min_lr

        for param_group in self.optimizers(use_pl_optimizer=False).param_groups:
            param_group['lr'] = lr

        self.log('train/lr', lr, sync_dist=True)


def train(config):
    download_vimsketch_dataset(config.dataset_path)
    download_qvim_dev_dataset(config.dataset_path)

    # wandb_logger = WandbLogger(
    #     project=config.project,
    #     id=config.id,
    #     config=config
    # )
    logger = CSVLogger(save_dir="logs", name="qvim_logs")


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

    train_dl = DataLoader(
        dataset=train_ds,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=True
    )

    eval_dl = DataLoader(
        dataset=eval_ds,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False
    )

    pl_module = QVIMModule(config)

    callbacks = []
    if config.model_save_path:
        callbacks.append(
            ModelCheckpoint(
                dirpath=os.path.join(config.model_save_path, "qvim_checkpoints"),
                filename="best-checkpoint",
                monitor="val/mrr",
                mode="min",
                save_top_k=1,
                save_last=True
            )
        )

    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        logger=logger,
        num_sanity_val_steps=0,
        accelerator='auto',
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=True),
        sync_batchnorm=config.sync_batchnorm,
        precision=config.precision,
        accumulate_grad_batches=config.acc_grad
    )

    trainer.validate(pl_module, dataloaders=eval_dl)
    trainer.fit(pl_module, train_dataloaders=train_dl, val_dataloaders=eval_dl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Argument parser for training the QVIM model.")

    parser.add_argument('--project', type=str, default="qvim")
    parser.add_argument('--id', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_gpus', type=int, default=1)
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
    parser.add_argument('--precision', default="bf16-mixed")
    parser.add_argument('--sync_batchnorm', action='store_true')
    parser.add_argument('--acc_grad', type=int, default=1)

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
        pl.seed_everything(args.random_seed)

    train(args)
