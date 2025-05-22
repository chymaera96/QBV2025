import argparse
import os
import random

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


class ProjectionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
        super(ProjectionMLP, self).__init__()
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)  # L2
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


def train_epoch(
    model, dataloader, criterion, optimizer, device, epoch=0, use_wandb=False
):
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1} Training")
    for batch_idx, batch in enumerate(progress_bar):
        anchor_features = batch["anchor_features"].to(device)
        positive_features = batch["positive_features"].to(device)
        negative_features = batch["negative_features"].to(device)

        optimizer.zero_grad()

        # Get embeddings from the projection MLP
        anchor_emb = model(anchor_features)
        positive_emb = model(positive_features)
        negative_emb = model(negative_features)

        loss = criterion(anchor_emb, positive_emb, negative_emb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * anchor_features.size(0)
        progress_bar.set_postfix(loss=loss.item())

        # Log per-batch metrics to wandb (matching classification format)
        if use_wandb:
            wandb.log(
                {
                    "batch/loss": loss.item(),
                }
            )

    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Epoch {epoch + 1} Training Loss: {epoch_loss:.4f}")

    return epoch_loss


def validate_with_evaluate(projection_mlp, clap_feature_extractor, data_path, device):
    projection_mlp.eval()
    clap_feature_extractor.model.eval()

    def compute_similarities(items, queries):
        results = {}

        # Extract features for all items
        item_embeddings = {}
        for item_id, file_path in tqdm(items.items(), desc="Processing items"):
            with torch.no_grad():
                # Get CLAP embeddings
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

                # Project through trained MLP
                feature = feature.to(device)
                embedding = projection_mlp(feature)

                # Store projected embedding
                item_embeddings[item_id] = embedding.squeeze().cpu()

        # Extract features for all queries
        query_embeddings = {}
        for query_id, file_path in tqdm(queries.items(), desc="Processing queries"):
            with torch.no_grad():
                # Get CLAP embeddings
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

                # Project through trained MLP
                feature = feature.to(device)
                embedding = projection_mlp(feature)

                # Store projected embedding
                query_embeddings[query_id] = embedding.squeeze().cpu()

        # Compute cosine similarities between all query-item pairs
        import torch.nn.functional as F

        print("Computing similarities...")
        for query_id, query_emb in tqdm(
            query_embeddings.items(), desc="Computing similarities"
        ):
            similarities = {}
            for item_id, item_emb in item_embeddings.items():
                # Normalize vectors for proper cosine similarity
                query_norm = F.normalize(query_emb.unsqueeze(0), p=2, dim=1)
                item_norm = F.normalize(item_emb.unsqueeze(0), p=2, dim=1)

                # Compute cosine similarity
                similarity = torch.mm(query_norm, item_norm.t()).item()
                similarities[item_id] = similarity

            results[query_id] = similarities

        return results

    # Call the evaluation function with our similarity function
    metrics = evaluate_qvim_system(compute_similarities, data_path=data_path)

    return metrics


def main(args):
    wandb_entity = os.environ.get("WANDB_ENTITY", None)
    wandb_project = os.environ.get(
        "WANDB_PROJECT", "qvim_contrastive"
    )  # Changed project name slightly

    # Create a more descriptive save path for the contrastive model
    model_name_parts = [
        "SC",
        f"clap-{args.clap_model_id}",
        f"aug-{args.augment}",
        f"proj-{args.projection_output_dim}",
    ]
    if args.hidden_dims:
        model_name_parts.append(f"hidden-{'-'.join(map(str, args.hidden_dims))}")

    run_name_prefix = "_".join(model_name_parts)
    save_filename = run_name_prefix + ".pt"
    save_path = os.path.join("models", save_filename)

    # Generate run_name if not provided
    if args.use_wandb and args.run_name is None:
        args.run_name = run_name_prefix
    elif args.use_wandb and args.run_name:
        # If a run_name is provided, use it for the save_path as well for consistency
        save_filename = args.run_name + ".pt"
        save_path = os.path.join("models", save_filename)

    # Initialize wandb only if use_wandb is True
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
    else:
        print("Wandb disabled (use_wandb is False).")

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Using device: {device}")

    print("Initializing CLAP feature extractor...")
    clap_sample_rate = 48000 if args.clap_model_id != "AF" else 16000
    feature_extractor = CLAPFeatureExtractor(
        model_id=int(args.clap_model_id) if args.clap_model_id != "AF" else "AF",
        device=device,
    )
    # CLAP features are frozen, so set to eval mode
    feature_extractor.model.eval()

    clap_feature_dim = 512
    if args.clap_model_id == "AF":
        clap_feature_dim = 2048

    train_dataset = VimSketch(
        root_dir=args.data_dir,
        sample_rate=clap_sample_rate,
        feature_extractor=feature_extractor,
        seed=args.seed,
        augment=args.augment,
        max_transforms=args.max_transforms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    print("Creating Projection MLP model...")
    model = ProjectionMLP(
        input_dim=clap_feature_dim,
        hidden_dims=args.hidden_dims,
        output_dim=args.projection_output_dim,
        dropout_rate=args.dropout_rate,
    ).to(device)

    criterion = TripletLoss(margin=args.triplet_margin)
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=args.scheduler_patience, factor=0.5, verbose=True
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("Starting training...")
    best_val_metric = 0.0
    patience_counter = 0

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        epoch_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.use_wandb
        )
        print(f"Training Loss: {epoch_loss:.4f}")

        # Log training metrics to wandb (matching classification format)
        if args.use_wandb:
            wandb.log(
                {
                    "train_loss": epoch_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        if scheduler and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_loss)
        elif scheduler:
            scheduler.step()  # For other schedulers like CosineAnnealing

        if (epoch + 1) % args.eval_every == 0:
            print(f"Epoch {epoch + 1}: Validating...")
            # Pass the MLP (projection head) and the CLAP feature extractor
            val_metrics = validate_with_evaluate(
                model, feature_extractor, args.eval_data_dir, device
            )
            print(f"MRR: {val_metrics['mrr']:.4f}")
            print(f"Class-wise MRR: {val_metrics['class_wise_mrr']:.4f}")
            print(f"NDCG: {val_metrics['ndcg']:.4f}")

            # Log evaluation metrics to wandb (matching classification format)
            if args.use_wandb:
                wandb.log(
                    {
                        "mrr": val_metrics["mrr"],
                        "class_wise_mrr": val_metrics["class_wise_mrr"],
                        "ndcg": val_metrics["ndcg"],
                    }
                )

            current_metric = val_metrics.get(args.tracking_metric, 0.0)
            if current_metric > best_val_metric:
                best_val_metric = current_metric
                patience_counter = 0
                print(
                    f"New best {args.tracking_metric}: {best_val_metric:.4f}. Saving model to {save_path}"
                )

                # Save model checkpoint (matching classification format)
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metrics": val_metrics,
                    },
                    save_path,
                )
                print(f"Model saved to {save_path}")

                # Log best model to wandb as a proper artifact (matching classification)
                if args.use_wandb:
                    artifact = wandb.Artifact(
                        name=f"model-{wandb.run.id}",
                        type="model",
                        description=f"Best model from run {wandb.run.name}",
                    )
                    artifact.add_file(save_path)
                    wandb.log_artifact(artifact)

                    # Also update summary metrics (matching classification)
                    wandb.run.summary["best_mrr"] = val_metrics["mrr"]
                    wandb.run.summary["best_class_wise_mrr"] = val_metrics[
                        "class_wise_mrr"
                    ]
                    wandb.run.summary["best_ndcg"] = val_metrics["ndcg"]
                    wandb.run.summary["best_epoch"] = epoch + 1
            else:
                patience_counter += 1
                print(
                    f"No improvement for {patience_counter} evaluations (patience: {args.patience})"
                )

                # Early stopping check (matching classification)
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    if args.use_wandb:
                        wandb.run.summary["early_stopped"] = True
                        wandb.run.summary["stopped_epoch"] = epoch + 1
                    break

    print("Training complete!")

    # Close wandb run (matching classification)
    if args.use_wandb:
        wandb.finish()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MLP for audio contrastive learning"
    )

    # Data parameters
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/Vim_Sketch",
        help="Path to the training dataset directory (VocalSketch)",
    )
    parser.add_argument(
        "--eval_data_dir",
        type=str,
        default="./data/DEV",
        help="Path to the evaluation dataset (AIMLA DEV)",
    )

    # Model parameters
    parser.add_argument(
        "--clap_model_id",
        type=str,
        default="1",
        help="CLAP model ID (e.g., '1', '2', '3', or 'AF' for AFCLAP)",
    )
    parser.add_argument(
        "--hidden_dims",
        nargs="+",
        type=int,
        default=[],
        help="Dimensions of hidden layers in Projection MLP",
    )
    parser.add_argument(
        "--projection_output_dim",
        type=int,
        default=512,
        help="Output dimension of the Projection MLP (embedding size)",
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.2, help="Dropout rate for MLP layers"
    )

    # Training parameters
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--eval_every", type=int, default=1, help="Evaluate every N epochs"
    )
    parser.add_argument(
        "--triplet_margin", type=float, default=0.2, help="Margin for Triplet Loss"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Patience for early stopping"
    )
    parser.add_argument(
        "--tracking_metric",
        type=str,
        default="mrr",
        choices=["mrr", "class_wise_mrr", "ndcg"],
        help="Metric to track for early stopping and best model saving",
    )

    # Augmentation parameters
    parser.add_argument(
        "--augment",
        type=str,
        default=None,
        choices=["query", "reference", "both", None],
        help="Augmentation mode: 'query', 'reference', 'both', or None",
    )
    parser.add_argument(
        "--max_transforms",
        type=int,
        default=1,
        help="Maximum number of augmentations to apply if augmentation is enabled",
    )

    # System parameters
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes for DataLoader",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Force use CPU even if CUDA is available"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for logging (requires WANDB_ENTITY to be set)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Custom name for the W&B run. If None, a name is generated from hyperparams.",
    )
    parser.add_argument(
        "--use_scheduler",
        action="store_true",
        help="Use a learning rate scheduler (ReduceLROnPlateau by default)",
    )
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=5,
        help="Patience for ReduceLROnPlateau scheduler (if use_scheduler is True)",
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    main(args)
