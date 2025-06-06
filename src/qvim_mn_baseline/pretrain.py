import argparse
import os
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from qvim_mn_baseline.dataset import VimSketchDataset
from qvim_mn_baseline.download import download_vimsketch_dataset
from qvim_mn_baseline.mn.preprocess import AugmentMelSTFT
from qvim_mn_baseline.mn.model import get_model as get_mobilenet
from qvim_mn_baseline.utils import NAME_TO_WIDTH
from hc_baseline.modules.augmentations import Augment


class QVIMPretrainModule(pl.LightningModule):
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

        self.encoder = get_mobilenet(
            width_mult=NAME_TO_WIDTH(config.pretrained_name),
            pretrained_name=config.pretrained_name
        )

        self.tau = torch.nn.Parameter(torch.tensor([config.initial_tau]), requires_grad=config.tau_trainable)

    def forward(self, x):
        x = self.mel(x).unsqueeze(1)
        y = self.encoder(x)[1]
        return torch.nn.functional.normalize(y, dim=1)

    def training_step(self, batch, batch_idx):
        z1 = self(batch['reference_1'])
        z2 = self(batch['reference_2'])

        C = torch.matmul(z1, z2.T)
        C = C / torch.abs(self.tau)

        C_text = torch.log_softmax(C, dim=1)

        paths = np.array([hash(p) for i, p in enumerate(batch['imitation_filename'])])
        I = torch.tensor(paths[None, :] == paths[:, None])


        loss = - C_text[torch.where(I)].mean()

        self.log('train/loss', loss, )
        self.log('train/tau', self.tau)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config.max_lr,
            weight_decay=self.config.weight_decay
        )


class ContrastivePretrainDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, augment_fn):
        self.dataset = dataset
        self.augment = augment_fn

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        ref = sample['reference']
        return {
            'reference_1': self.augment(ref.clone()),
            'reference_2': self.augment(ref.clone()),
        }

    def __len__(self):
        return len(self.dataset)


def train(config):
    download_vimsketch_dataset(config.dataset_path)

    raw_dataset = VimSketchDataset(
        os.path.join(config.dataset_path, 'Vim_Sketch_Dataset'),
        sample_rate=config.sample_rate,
        duration=config.duration
    )

    contrastive_dataset = ContrastivePretrainDataset(raw_dataset, augment_fn=Augment(sample_rate=config.sample_rate))

    train_dl = DataLoader(
        dataset=contrastive_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    pl_module = QVIMPretrainModule(config)

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(config.model_save_path, config.id or "pretrain"),
            filename="checkpoint-{epoch}",
            every_n_epochs=5,
            save_top_k=-1  # save every 5 epochs
        )
    ]

    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        logger=WandbLogger(project=config.project, id=config.id, config=config),
        accelerator='auto',
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=True),
        sync_batchnorm=config.sync_batchnorm,
        precision=config.precision
    )

    trainer.fit(pl_module, train_dl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pretraining QVIM model with contrastive learning.")

    # General
    parser.add_argument('--project', type=str, default="qvim-pretrain")
    parser.add_argument('--id', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--dataset_path', type=str, default='data')

    # Architecture
    parser.add_argument('--pretrained_name', type=str, default="mn10_as")

    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--max_lr', type=float, default=0.0003)
    parser.add_argument('--initial_tau', type=float, default=0.07)
    parser.add_argument('--tau_trainable', action='store_true')
    parser.add_argument('--precision', type=str, default="bf16-mixed")
    parser.add_argument('--sync_batchnorm', action='store_true')

    # Preprocessing
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
    parser.add_argument('--fmin_aug_range', type=int, default=10)
    parser.add_argument('--fmax_aug_range', type=int, default=2000)

    args = parser.parse_args()

    if args.id is None:
        import uuid
        args.id = str(uuid.uuid4())

    pl.seed_everything(42)
    train(args)
