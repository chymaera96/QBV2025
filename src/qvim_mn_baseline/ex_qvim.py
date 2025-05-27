import argparse
import os
import math
import copy

from copy import deepcopy
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from wandb import Settings

from qvim_mn_baseline.dataset import VimSketchDataset, AESAIMLA_DEV
from qvim_mn_baseline.download import download_vimsketch_dataset, download_qvim_dev_dataset
from qvim_mn_baseline.mn.preprocess import AugmentMelSTFT
from qvim_mn_baseline.mn.model import get_model
from qvim_mn_baseline.metrics import compute_mrr, compute_ndcg

class QVIMModule(pl.LightningModule):
    """
    Pytorch Lightning Module for the QVIM Model
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.mel = AugmentMelSTFT(
            n_mels=config.n_mels,
            sr=config.sample_rate,
            win_length=config.window_size,
            hopsize=config.hop_size,
            n_fft=config.n_fft,
            freqm=config.freqm,
            timem=config.timem,
            fmin=config.fmin,
            fmax=config.fmax,
            fmin_aug_range=config.fmin_aug_range,
            fmax_aug_range=config.fmax_aug_range
        )

        self.mask_ratio = config.mask_ratio

        def build_encoder():
            base = get_model(pretrained_name="mn10_as", head_type="mlp", width_mult=1.0)
            projector = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(start_dim=1),
                nn.Linear(960, config.projection_dim)
            )
            return nn.Sequential(base.features, projector)

        self.context_encoder = build_encoder()

        self.target_encoder = deepcopy(self.context_encoder)  # Momentum encoder
        self.predictor = nn.Sequential(
            nn.Conv2d(960, 960, 1),
            nn.ReLU(),
            nn.Conv2d(960, 960, 1)
)
        
        self.global_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(960, config.projection_dim),
        )

        initial_tau = torch.zeros((1,)) + config.initial_tau
        self.tau = torch.nn.Parameter(initial_tau, requires_grad=config.tau_trainable)

        self.validation_output = []

    @torch.no_grad()
    def momentum_update(self, m=0.999):
        for p_q, p_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_k.data = m * p_k.data + (1. - m) * p_q.data

    def mask_features(self, f_map):
        B, C, F, T = f_map.shape
        num_patches = F * T
        num_mask = int(self.mask_ratio * num_patches)

        mask = torch.ones(B, F, T, device=f_map.device)
        for i in range(B):
            idx = torch.randperm(num_patches)[:num_mask]
            f_idx = idx // T
            t_idx = idx % T
            mask[i, f_idx, t_idx] = 0

        return f_map * mask.unsqueeze(1), mask

    def forward(self, x):
        x = self.mel(x).unsqueeze(1)

        with torch.no_grad():
            target_f = self.target_encoder[0](x)                     # [B, 960, 4, 32]
            target_projected = self.target_encoder[1](target_f).detach()  # [B, proj_dim]

        context_f = self.context_encoder[0](x)                       # [B, 960, 4, 32]
        masked_f, mask = self.mask_features(context_f)              
        predicted_f = self.predictor(masked_f)                      # [B, 960, 4, 32]
        projected = self.context_encoder[1](predicted_f)            # [B, proj_dim]

        return projected, target_projected, mask

    def forward_imitation(self, x):
        with torch.no_grad():
            x = self.mel(x).unsqueeze(1)
            z = self.context_encoder(x)
        return F.normalize(z, dim=1)

    def forward_reference(self, x):
        with torch.no_grad():
            x = self.mel(x).unsqueeze(1)
            z = self.context_encoder(x)
        return F.normalize(z, dim=1)

    def training_step(self, batch, batch_idx):
        self.mel.eval()
        self.lr_scheduler_step(batch_idx)
        self.momentum_update()

        x = batch['imitation']
        predicted_f, target_f, mask = self.forward(x)

        loss = F.mse_loss(predicted_f[mask == 0], target_f[mask == 0])
        self.log('train/loss', loss, on_step=True, on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        self.mel.eval()
        y_imit = self.forward_imitation(batch['imitation'])
        y_ref = self.forward_reference(batch['reference'])

        C = torch.matmul(y_imit, y_ref.T)
        C = C / torch.abs(self.tau)

        C_text = torch.log_softmax(C, dim=1)

        paths = np.array([hash(p) for i, p in enumerate(batch['imitation_filename'])])
        I = torch.tensor(paths[None, :] == paths[:, None])


        loss = - C_text[torch.where(I)].mean()

        self.log('val/loss', loss, )
        self.log('val/tau', self.tau)


        self.validation_output.extend([
            {
                'imitation': copy.deepcopy(y_imit.detach().cpu().numpy()),
                'reference': copy.deepcopy(y_ref.detach().cpu().numpy()),
                'imitation_filename': batch['imitation_filename'],
                'reference_filename': batch['reference_filename'],
                'imitation_class': batch['imitation_class'],
                'reference_class': batch['reference_class']
            }
        ])

    def on_validation_epoch_end(self):
        validation_output = self.validation_output

        # Concatenate imitation and reference arrays
        imitations = np.concatenate([b['imitation'] for b in validation_output])
        reference = np.concatenate([b['reference'] for b in validation_output])

        # Flatten filenames lists
        imitation_filenames = sum([b['imitation_filename'] for b in validation_output], [])
        reference_filenames = sum([b['reference_filename'] for b in validation_output], [])

        # Compute new ground truth based on classes
        imitation_classes = sum([b['imitation_class'] for b in validation_output], [])
        reference_classes = sum([b['reference_class'] for b in validation_output], [])

        # Generate ground truth mapping
        ground_truth_mrr = {fi: rf for fi, rf in zip(imitation_filenames, reference_filenames)}

        # Compute similarity scores using matrix multiplication
        # Remove duplicates in reference vectors and filenames
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

        self.log('val/mrr', mrr)
        self.log('val/ndcg', ndcg)

        # clear the cached outputs
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
 
        # modified to calculate step based on epoch proportion... 
        warmup_epochs = self.config.n_epochs * self.config.warmup_percent
        rampdown_epochs = self.config.n_epochs * self.config.rampdown_percent
        warmup_steps = int(warmup_epochs * steps_per_epoch)
        total_steps = int((warmup_epochs + rampdown_epochs) * steps_per_epoch)
        decay_steps = total_steps - warmup_steps

        if current_step < warmup_steps:
            # Linear warmup
            lr = min_lr + (max_lr - min_lr) * (current_step / warmup_steps)
        elif current_step < total_steps:
            # Cosine decay
            decay_progress = (current_step - warmup_steps) / decay_steps
            lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * decay_progress))
        else:
            # Constant learning rate
            lr = min_lr

        for param_group in self.optimizers(use_pl_optimizer=False).param_groups:
            param_group['lr'] = lr

        self.log('train/lr', lr, sync_dist=True)


def train(config):
    # Train dual encoder for QBV

    # download the data set if the folder does not exist
    download_vimsketch_dataset(config.dataset_path)
    download_qvim_dev_dataset(config.dataset_path)

    wandb_logger = WandbLogger(
        project=config.project,
        id=config.id, # added
        settings=Settings(init_timeout=300),
        config=config
    )
    wandb_logger.experiment.config.update(vars(config))

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
            dirpath=os.path.join(config.model_save_path, wandb_logger.experiment.name),  # Directory to save checkpoints
            filename="best-checkpoint",
            monitor="val/mrr",  # Metric to monitor for best model
            mode="min",  # Save model with lowest val_loss
            save_top_k=1,  # Only keep the best checkpoint
            save_last=True  # Always save the last checkpoint
            )
        )
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        logger=wandb_logger,
        accelerator='auto',
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=True), # fix for multi-GPU support
        sync_batchnorm=config.sync_batchnorm, # fix for multi-GPU support (default: False for quicker experiment)
        precision=config.precision,
        accumulate_grad_batches=config.acc_grad
    )

    trainer.validate(
        pl_module,
        dataloaders=eval_dl
    )

    trainer.fit(
        pl_module,
        train_dataloaders=train_dl,
        val_dataloaders=eval_dl
    )

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Argument parser for training the QVIM model.")

    # General
    parser.add_argument('--project', type=str, default="qvim",
                        help="Project name in wandb.")
    parser.add_argument('--id', type=str, default=None,
                        help="WandB run_id. If not specified, randomly generated")  # added
    parser.add_argument('--num_workers', type=int, default=8,
                        help="Number of data loader workers. Set to 0 for no multiprocessing.")
    parser.add_argument('--num_gpus', type=int, default=1,
                        help="Number of GPUs to use for training.")
    parser.add_argument('--model_save_path', type=str, default=None,
                        help="Path to store the checkpoints. Use None to disable saving.")
    parser.add_argument('--dataset_path', type=str, default='data',
                        help="Path to the data sets.")

    # Encoder architecture
    parser.add_argument('--pretrained_name', type=str, default="mn10_as",
                        help="Pretrained model name for transfer learning.")
    parser.add_argument('--projection_dim', type=int, default=512,
                        help="Dimension of the projection space for the model.")

    # Training
    parser.add_argument('--random_seed', type=int, default=None,
                        help="A seed to make the experiment reproducible. Set to None to disable.")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Number of samples per batch.")
    parser.add_argument('--n_epochs', type=int, default=15,
                        help="Total number of training epochs.")
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help="L2 weight regularization to prevent overfitting.")
    parser.add_argument('--max_lr', type=float, default=1.0e-4,
                        help="Maximum learning rate.")
    parser.add_argument('--min_lr', type=float, default=1.0e-5,
                        help="Final learning rate at the end of training.")
    # parser.add_argument('--warmup_epochs', type=int, default=1,
    #                    help="Number of warm-up epochs where learning rate increases gradually.")
    # parser.add_argument('--rampdown_epochs', type=int, default=7,
    #                    help="Duration (in epochs) for learning rate ramp-down.")
    parser.add_argument('--warmup_percent', type=float, default=0.05, # modified to use percent
                    help="Fraction of total epochs for warmup (e.g., default = 0.1 = 10%)")
    parser.add_argument('--rampdown_percent', type=float, default=0.8, # modified to use percent
                    help="Fraction of total epochs for cosine rampdown (e.g., default = 0.8 = 80%)")
    parser.add_argument('--initial_tau', type=float, default=0.1,
                        help="Temperature parameter for the loss function.")
    parser.add_argument('--tau_trainable', default=False, action='store_true',
                        help="make tau trainable or not.")
    parser.add_argument('--precision', type=str, default="bf16-mixed", # fix for mixed-precision support
                        help="precision (default='bf16-mixed') {32, 16, bf16, bf16-mixed}")
    parser.add_argument('--sync_batchnorm', action='store_true',
                        help="Enable synchronized batch normalization across GPUs. Default is False.")
    parser.add_argument('--acc_grad', type=int, default=1,
                        help="Accumulate gradient batches, default=1")
    parser.add_argument('--mask_ratio', type=float, default=0.3,
                        help="Mask ratio for the masked feature prediction task. Default is 0.3.")

    # Preprocessing
    parser.add_argument('--duration', type=float, default=10.0,
                        help="Duration of audio clips in seconds.")

    # Spectrogram Parameters
    parser.add_argument('--sample_rate', type=int, default=32000,
                        help="Target sampling rate for audio resampling.")
    parser.add_argument('--window_size', type=int, default=800,
                        help="Size of the window for STFT in samples.")
    parser.add_argument('--hop_size', type=int, default=320,
                        help="Hop length for STFT in samples.")
    parser.add_argument('--n_fft', type=int, default=1024,
                        help="Number of FFT bins for spectral analysis.")
    parser.add_argument('--n_mels', type=int, default=32,
                        help="Number of mel filter banks for Mel spectrogram conversion.")
    parser.add_argument('--freqm', type=int, default=2,
                        help="Frequency masking parameter for spectrogram augmentation.")
    parser.add_argument('--timem', type=int, default=200,
                        help="Time masking parameter for spectrogram augmentation.")
    parser.add_argument('--fmin', type=int, default=0,
                        help="Minimum frequency cutoff for Mel spectrogram.")
    parser.add_argument('--fmax', type=int, default=None,
                        help="Maximum frequency cutoff for Mel spectrogram (None means use Nyquist frequency).")
    parser.add_argument('--fmin_aug_range', type=int, default=10,
                        help="Variation range for fmin augmentation.")
    parser.add_argument('--fmax_aug_range', type=int, default=2000,
                        help="Variation range for fmax augmentation.")

    args = parser.parse_args()

    if args.random_seed:
        pl.seed_everything(args.random_seed)

    train(args)