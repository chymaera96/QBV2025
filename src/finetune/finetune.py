#!/usr/bin/env python3
import logging
import os
import sys
from pathlib import Path

# Add paths
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
clap_src = project_root / "CLAP" / "src" / "laion_clap"
sys.path.insert(0, str(clap_src))

from .config import MODEL_CONFIG, PATHS, TRAINING_CONFIG


def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create command line arguments for CLAP training
    clap_args = [
        "python",
        "-m",
        "training.main",
        # Dataset args - Use VimSketch dataset type
        "--dataset-type",
        "vim",
        "--vim-dataset-path",
        PATHS["vim_dataset"],
        # Model args
        "--amodel",
        MODEL_CONFIG["audio_model"],
        "--tmodel",
        MODEL_CONFIG["text_model"],
        "--precision",
        MODEL_CONFIG["precision"],
        "--pretrained",
        PATHS["clap_checkpoint"],
        # Training args - Use safer values
        "--epochs",
        str(TRAINING_CONFIG["epochs"]),
        "--batch-size",
        str(TRAINING_CONFIG["batch_size"]),
        "--lr",
        str(TRAINING_CONFIG["learning_rate"]),
        "--wd",
        str(TRAINING_CONFIG["weight_decay"]),
        "--warmup",
        str(TRAINING_CONFIG["warmup_steps"]),
        # Fine-tuning specific - Much lower learning rates
        "--split-opt",
        "--lr-pretrained",
        "1e-8",  # Very small for pretrained
        "--lr-new",
        "1e-6",  # Small for new parameters
        # Data processing
        "--max-len",
        "480000",
        "--data-filling",
        "repeatpad",
        "--data-truncating",
        "rand_trunc",
        # Checkpointing
        "--save-frequency",
        str(TRAINING_CONFIG["save_frequency"]),
        "--save-most-recent",
        "--save-top-performance",
        "3",
        "--logs",
        PATHS["logs"],
        "--name",
        PATHS.get("experiment_name", "vim_sketch_finetune"),
        # Evaluation - Enable it but start from epoch 1
        "--val-frequency",
        str(TRAINING_CONFIG.get("val_frequency", 2)),
        # System
        "--workers",
        "2",  # Reduced workers
        "--seed",
        "42",
        # Skip initial evaluation to avoid the error
        # Enable wandb logging
        "--wandb",
    ]

    # Run training
    print("Starting CLAP fine-tuning with VimSketch dataset...")
    print(f"Command: {' '.join(clap_args)}")

    os.chdir(clap_src)
    os.system(" ".join(clap_args))


if __name__ == "__main__":
    main()
