import argparse
import os
import math
import copy
import warnings
warnings.filterwarnings("ignore")

from copy import deepcopy
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from qvim_mn_baseline.dataset import VimSketchDataset, AESAIMLA_DEV, VocalSketchDataset, TripletBatchDataset
from qvim_mn_baseline.download import download_vimsketch_dataset, download_qvim_dev_dataset
from qvim_mn_baseline.mn.preprocess import AugmentMelSTFT
from qvim_mn_baseline.mn.model import get_model as get_mobilenet
from qvim_mn_baseline.projection import MobileNetWithProjection
from qvim_mn_baseline.utils import NAME_TO_WIDTH
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

        # encoder = get_mobilenet(
        #     width_mult=NAME_TO_WIDTH(config.pretrained_name),
        #     pretrained_name=None
        # )

        # if config.pretrained_ckpt_path:
        #     print(f"Loading pretrained encoder from {config.pretrained_ckpt_path}")
        #     try:
        #         ckpt = torch.load(config.pretrained_ckpt_path, map_location='cpu')
        #         state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

        #         # Strict loading to ensure all keys match
        #         encoder.load_state_dict({
        #             k.replace('encoder.', ''): v
        #             for k, v in state_dict.items()
        #             if k.startswith('encoder.')
        #         }, strict=True)

        #         print("Pretrained weights loaded successfully.")
        #     except Exception as e:
        #         print(f"Error loading checkpoint: {e}")
        #         print(f"Current working directory: {os.getcwd()}")
        #         raise RuntimeError("Failed to load pretrained weights. Exiting.")
        # else:
        #     raise ValueError("Pretrained checkpoint path not provided. Exiting.")

        self.encoder = get_mobilenet(
            width_mult=NAME_TO_WIDTH(config.pretrained_name),
            pretrained_name=config.pretrained_name,
        )
    
        initial_tau = torch.zeros((1,)) + config.initial_tau
        self.tau = torch.nn.Parameter(initial_tau, requires_grad=config.tau_trainable)

        self.validation_output = []

    def forward_pass(self, x):
        x = self.mel(x).unsqueeze(1)
        out = self.encoder(x)
        if isinstance(out, tuple):
            out = out[1]
        return torch.nn.functional.normalize(out, dim=1)
        # return out



    def training_step(self, batch, batch_idx):
        self.lr_scheduler_step(batch_idx)

        # ===== Embeddings =====
        anchor = self.forward_pass(batch['anchor'])        # (B, D)
        positive = self.forward_pass(batch['positive'])    # (B, D)
        negative = self.forward_pass(batch['negative'])    # (B, D)

        # === Contrastive loss ===
        C = torch.matmul(anchor, positive.T)  # (B, B)
        C = C / torch.abs(self.tau)
        C_text = torch.log_softmax(C, dim=1)

        paths = np.array([hash(p) for i, p in enumerate(batch['positive_filename'])])
        I = torch.tensor(paths[None, :] == paths[:, None])


        contrastive_loss = - C_text[torch.where(I)].mean()

        # === Triplet loss ===
        d_ap = F.pairwise_distance(anchor, positive)
        d_an = F.pairwise_distance(anchor, negative)
        triplet_margin = self.config.margin
        triplet_loss = F.triplet_margin_loss(anchor, positive, negative, margin=triplet_margin, p=2)

        # === Dynamic weighting ===
        with torch.no_grad():
            active = (d_ap + triplet_margin > d_an).sum()
            active_ratio = active.item() / anchor.size(0)
            self.log('train/active_triplets', active.item())

        # Start with small weight if collapse is likely
        alpha = min(1.0, max(0.3, active_ratio))  # Clamp between 0.1 and 1.0
        total_loss = contrastive_loss + alpha * triplet_loss

        # Logging
        self.log('train/loss', total_loss)
        self.log('train/loss_contrastive', contrastive_loss)
        self.log('train/loss_triplet', triplet_loss)
        self.log('train/alpha_triplet', alpha)

        return total_loss



    def validation_step(self, batch, batch_idx):

        with torch.no_grad():
            # y_imitation = F.normalize(self.forward_pass(batch['imitation']), dim=1)
            # y_reference = F.normalize(self.forward_pass(batch['reference']), dim=1)
            y_imitation = self.forward_pass(batch['imitation'])
            y_reference = self.forward_pass(batch['reference'])

        self.validation_output.extend([
            {
                'imitation': copy.deepcopy(y_imitation.detach().cpu().numpy()),
                'reference': copy.deepcopy(y_reference.detach().cpu().numpy()),
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
        warmup_steps = self.config.warmup_epochs * steps_per_epoch
        total_steps = (self.config.warmup_epochs + self.config.rampdown_epochs) * steps_per_epoch
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
    # download_vimsketch_dataset(config.dataset_path)
    # download_qvim_dev_dataset(config.dataset_path)

    wandb_logger = WandbLogger(
        project=config.project,
        id=config.id, # added
        config=config
    )

    ap_ds = VimSketchDataset(
        os.path.join(config.dataset_path, 'Vim_Sketch_Dataset'),
        sample_rate=config.sample_rate,
        duration=config.duration
    )

    neg1_ds = VocalSketchDataset(
        os.path.join(config.neg_path, 'vocal_imitations'),
        sample_rate=config.sample_rate,
        duration=config.duration
    )

    neg2_ds = VocalSketchDataset(
        os.path.join(config.neg_path, 'vocal_imitations_set2'),
        sample_rate=config.sample_rate,
        duration=config.duration
    )

    neg_ds = torch.utils.data.ConcatDataset([neg1_ds, neg2_ds])

    train_ds = TripletBatchDataset(ap_ds, neg_ds)
    train_dl = DataLoader(
        dataset=train_ds,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=True
    )


    eval_ds = AESAIMLA_DEV(
        os.path.join(config.dataset_path, 'qvim-dev'),
        sample_rate=config.sample_rate,
        duration=config.duration
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
            filename="best-mrr-{epoch:02d}",
            monitor="val/mrr",  # Metric to monitor for best model
            mode="max",  
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
        precision=config.precision
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
    parser.add_argument('--model_save_path', type=str, default="/data/home/acw723/QBV2025/",
                        help="Path to store the checkpoints. Use None to disable saving.")
    parser.add_argument('--dataset_path', type=str, default='data',
                        help="Path to the data sets.")
    parser.add_argument('--neg_path', type=str, default='/data/scratch/acw723/VocalSketchDataset',
                        help="Path to the first negative dataset (vocal imitations).")

    # Encoder architecture
    parser.add_argument('--pretrained_name', type=str, default="mn10_as",
                        help="Pretrained model name for transfer learning.")
    parser.add_argument('--pretrained_ckpt_path', type=str, default=None,
                    help="Path to a pretraining checkpoint to initialize the encoder weights.")

    # Training
    parser.add_argument('--random_seed', type=int, default=None,
                        help="A seed to make the experiment reproducible. Set to None to disable.")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Number of samples per batch.")
    parser.add_argument('--n_epochs', type=int, default=15,
                        help="Total number of training epochs.")
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help="L2 weight regularization to prevent overfitting.")
    parser.add_argument('--max_lr', type=float, default=0.0002,
                        help="Maximum learning rate.")
    parser.add_argument('--min_lr', type=float, default=5.e-5,
                        help="Final learning rate at the end of training.")
    parser.add_argument('--warmup_epochs', type=int, default=1,
                        help="Number of warm-up epochs where learning rate increases gradually.")
    parser.add_argument('--rampdown_epochs', type=int, default=25,
                        help="Duration (in epochs) for learning rate ramp-down.")
    parser.add_argument('--initial_tau', type=float, default=0.07,
                        help="Temperature parameter for the loss function.")
    parser.add_argument('--tau_trainable', default=False, action='store_true',
                        help="make tau trainable or not.")
    parser.add_argument('--precision', type=str, default="bf16-mixed", # fix for mixed-precision support
                        help="precision (default='bf16-mixed') {32, 16, bf16, bf16-mixed}")
    parser.add_argument('--sync_batchnorm', action='store_true',
                        help="Enable synchronized batch normalization across GPUs. Default is False.")
    parser.add_argument('--margin', type=float, default=0.2,
                        help="Margin for the triplet loss function.")
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
    parser.add_argument('--n_mels', type=int, default=128,
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