import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from .dataset import VimSketch
from .evaluate import evaluate_qvim_system
from .features import CLAPFeatureExtractor


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.2):
        super(MLP, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer for binary classification
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_epoch(
    model, dataloader, criterion, optimizer, device, epoch=0, use_wandb=False
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        query_features = batch["query_item"].to(device)
        reference_features = batch["reference_item"].to(device)

        combined_features = torch.cat([query_features, reference_features], dim=1)
        labels = batch["is_match"].float().to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(combined_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        running_loss += batch_loss * query_features.size(0)

        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Log per-batch metrics to wandb
        if use_wandb:
            wandb.log(
                {
                    "batch/loss": batch_loss,
                }
            )

    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total

    return epoch_loss, accuracy


def validate_with_evaluate(mlp_model, clap_feature_extractor, data_path, device):
    mlp_model.eval()

    def compute_similarities(items, queries):
        """Custom similarity function that uses our MLP model"""
        results = {}

        # Extract features for all items
        item_features = {}
        for item_id, file_path in tqdm(items.items(), desc="Processing items"):
            with torch.no_grad():
                # Need to use file loading since we can't modify the function signature
                if clap_feature_extractor.is_afclap:
                    feature = (
                        clap_feature_extractor.model.get_audio_embedding_from_filelist(
                            x=[file_path], sr=16000, use_tensor=True
                        )
                    )
                else:
                    feature = (
                        clap_feature_extractor.model.get_audio_embedding_from_filelist(
                            x=[file_path], use_tensor=True
                        )
                    )
                item_features[item_id] = feature.cpu()

        # Extract features for all queries
        query_features = {}
        for query_id, file_path in tqdm(queries.items(), desc="Processing queries"):
            with torch.no_grad():
                if clap_feature_extractor.is_afclap:
                    feature = (
                        clap_feature_extractor.model.get_audio_embedding_from_filelist(
                            x=[file_path], sr=16000, use_tensor=True
                        )
                    )
                else:
                    feature = (
                        clap_feature_extractor.model.get_audio_embedding_from_filelist(
                            x=[file_path], use_tensor=True
                        )
                    )
                query_features[query_id] = feature.cpu()

        # Compute similarities using our MLP model
        for query_id, query_feature in tqdm(
            query_features.items(), desc="Computing similarities"
        ):
            results[query_id] = {}

            for item_id, item_feature in item_features.items():
                # Combine features
                combined_features = torch.cat([query_feature, item_feature], dim=1).to(
                    device
                )

                # Get similarity score from MLP
                with torch.no_grad():
                    similarity = mlp_model(combined_features).item()

                results[query_id][item_id] = similarity

        return results

    metrics = evaluate_qvim_system(compute_similarities, data_path=data_path)

    return metrics


def main(args):
    # Initialize wandb with environment variables
    wandb_entity = os.environ.get("WANDB_ENTITY", None)
    wandb_project = os.environ.get("WANDB_PROJECT", "qbv2025-clap-mlp")
    save_path = os.path.join(
        "models",
        f"CF_clap-{args.clap_model_id}_aug-{args.augment}_neg-{args.negative_ratio}.pt",
    )

    # Initialize wandb only if entity is provided
    if args.use_wandb:
        if wandb_entity is None:
            print("Warning: WANDB_ENTITY not set. Using anonymous logging.")

        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=args.run_name,
            config=vars(args),
        )
        print(f"W&B initialized with project: {wandb_project}, entity: {wandb_entity}")

    # Set device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Using device: {device}")

    # Create the CLAP feature extractor
    print("Initializing CLAP feature extractor...")
    feature_extractor = CLAPFeatureExtractor(
        model_id=int(args.clap_model_id) if args.clap_model_id != "AF" else "AF",
        device=device,
    )
    feature_dim = 512 if args.clap_model_id != "AF" else 2048

    train_dataset = VimSketch(
        root_dir=args.data_dir,
        sample_rate=16000,
        feature_extractor=feature_extractor,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
        augment=args.augment,
        max_transforms=args.max_transforms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    print("Creating MLP model...")
    model = MLP(
        input_dim=feature_dim * 2,  # Concatenated features
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout_rate,
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,  # Initial restart interval
        T_mult=2,  # Multiplicative factor for restart interval
        eta_min=1e-6,  # Minimum learning rate
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Training loop
    print("Starting training...")
    best_metric = 0.0
    patience = args.patience
    patience_counter = 0

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            use_wandb=args.use_wandb,
        )
        print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # Log training metrics to wandb
        if args.use_wandb:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        # Call scheduler.step() each epoch instead of after validation
        scheduler.step()

        # Evaluate model
        if (epoch + 1) % args.eval_every == 0:
            print("Evaluating model...")
            metrics = validate_with_evaluate(
                model, feature_extractor, args.eval_data_dir, device
            )

            print(f"MRR: {metrics['mrr']:.4f}")
            print(f"Class-wise MRR: {metrics['class_wise_mrr']:.4f}")
            print(f"NDCG: {metrics['ndcg']:.4f}")

            # Log evaluation metrics to wandb
            if args.use_wandb:
                wandb.log(
                    {
                        "mrr": metrics["mrr"],
                        "class_wise_mrr": metrics["class_wise_mrr"],
                        "ndcg": metrics["ndcg"],
                    }
                )

            # Check if model improved
            current_metric = metrics["mrr"]
            if current_metric > best_metric:
                best_metric = current_metric
                patience_counter = 0  # Reset patience counter
                print(f"New best model with MRR: {best_metric:.4f}")

                # Save model checkpoint
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metrics": metrics,
                    },
                    save_path,
                )
                print(f"Model saved to {save_path}")

                # Log best model to wandb as a proper artifact
                if args.use_wandb:
                    artifact = wandb.Artifact(
                        name=f"model-{wandb.run.id}",
                        type="model",
                        description=f"Best model from run {wandb.run.name}",
                    )
                    artifact.add_file(save_path)
                    wandb.log_artifact(artifact)

                    # Also update summary metrics
                    wandb.run.summary["best_mrr"] = best_metric
                    wandb.run.summary["best_class_wise_mrr"] = metrics["class_wise_mrr"]
                    wandb.run.summary["best_ndcg"] = metrics["ndcg"]
                    wandb.run.summary["best_epoch"] = epoch + 1
            else:
                patience_counter += 1
                print(
                    f"No improvement for {patience_counter} evaluations (patience: {patience})"
                )

                # Early stopping check
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    if args.use_wandb:
                        wandb.run.summary["early_stopped"] = True
                        wandb.run.summary["stopped_epoch"] = epoch + 1
                    break

    print("Training complete!")

    # Close wandb run
    if args.use_wandb:
        wandb.finish()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP for audio pair matching")

    # Data parameters
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/Vim_Sketch",
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--eval_data_dir",
        type=str,
        default="./data/DEV",
        help="Path to the evaluation dataset",
    )
    parser.add_argument(
        "--negative_ratio",
        type=float,
        default=1.0,
        help="Ratio of negative to positive examples",
    )

    # Model parameters
    parser.add_argument(
        "--clap_model_id",
        type=str,
        default="1",
        help="CLAP model ID (1, 2, 3, or 'AF')",
    )
    parser.add_argument(
        "--hidden_dims",
        nargs="+",
        type=int,
        default=[512, 256],
        help="Dimensions of hidden layers in MLP",
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.2, help="Dropout rate for MLP layers"
    )

    # Training parameters
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=25, help="Number of training epochs"
    )
    parser.add_argument(
        "--eval_every", type=int, default=1, help="Evaluate every N epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (epochs without improvement)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Disable CUDA even if available"
    )

    # Augmentation parameters
    parser.add_argument(
        "--augment",
        type=str,
        default=None,
        choices=[None, "query", "reference", "both"],
        help="Which audio to augment (None, 'query', 'reference', or 'both')",
    )
    parser.add_argument(
        "--max_transforms",
        type=int,
        default=1,
        help="Maximum number of transforms to apply during augmentation",
    )

    parser.add_argument(
        "--use_wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="Name for the wandb run"
    )

    args = parser.parse_args()

    # If run_name is not provided, create one based on model parameters
    if args.use_wandb and args.run_name is None:
        args.run_name = (
            f"CF_clap-{args.clap_model_id}_aug-{args.augment}_neg-{args.negative_ratio}"
        )

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    main(args)
