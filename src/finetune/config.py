"""Configuration for VimSketch CLAP fine-tuning"""

import os
from pathlib import Path

# Update these paths according to your setup
PATHS = {
    "vim_dataset": "/home/chris/dev/QBV2025/data/Vim_Sketch",
    "webdataset_output": "/home/chris/dev/QBV2025/data/vim_webdataset",
    "clap_checkpoint": "/home/chris/dev/QBV2025/models/music_speech_audioset_epoch_15_esc_89.98.pt",
    "logs": "/home/chris/dev/QBV2025/logs/vim_clap_finetune",
    "experiment_name": "vim_sketch_finetune_v2",
}

# Training configuration
TRAINING_CONFIG = {
    "epochs": 20,
    "batch_size": 2,
    "learning_rate": 1e-5,
    "weight_decay": 0.1,
    "warmup_steps": 1000,
    "save_frequency": 5,
}

# Model configuration
MODEL_CONFIG = {
    "audio_model": "HTSAT-base",  # or "PANN-14"
    "text_model": "roberta",
    "freeze_text": False,  # Full fine-tuning
    "precision": "fp32",  # Mixed precision
}
