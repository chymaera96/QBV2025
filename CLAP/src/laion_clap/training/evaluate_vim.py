import logging
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add the project root to the path to import the evaluation script
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(project_root, "src"))

try:
    from contrastive.evaluate import evaluate_qvim_system

    EVALUATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import evaluation system: {e}")
    evaluate_qvim_system = None
    EVALUATION_AVAILABLE = False


def validate_clap_with_evaluate(model, data_path, device, args):
    """
    Validate CLAP model using the VimSketch evaluation system
    """
    if not EVALUATION_AVAILABLE or evaluate_qvim_system is None:
        logging.warning("Evaluation system not available, skipping validation")
        return {"mrr": 0.0, "class_wise_mrr": 0.0, "ndcg": 0.0}

    model.eval()

    def compute_similarities(items, queries):
        """
        Compute similarities using the fine-tuned CLAP model
        """
        results = {}

        # Extract embeddings for all items
        item_embeddings = {}
        logging.info("Processing reference items...")
        for item_id, file_path in tqdm(items.items(), desc="Processing items"):
            with torch.no_grad():
                try:
                    # Get CLAP audio embedding
                    embedding = model.get_audio_embedding_from_filelist(
                        x=[file_path], use_tensor=True
                    )
                    if embedding is not None:
                        item_embeddings[item_id] = F.normalize(
                            embedding.squeeze(), dim=0
                        ).cpu()
                except Exception as e:
                    logging.warning(f"Failed to process item {item_id}: {e}")
                    continue

        # Extract embeddings for all queries
        query_embeddings = {}
        logging.info("Processing query items...")
        for query_id, file_path in tqdm(queries.items(), desc="Processing queries"):
            with torch.no_grad():
                try:
                    # Get CLAP audio embedding
                    embedding = model.get_audio_embedding_from_filelist(
                        x=[file_path], use_tensor=True
                    )
                    if embedding is not None:
                        query_embeddings[query_id] = F.normalize(
                            embedding.squeeze(), dim=0
                        ).cpu()
                except Exception as e:
                    logging.warning(f"Failed to process query {query_id}: {e}")
                    continue

        # Compute cosine similarities between all query-item pairs
        logging.info("Computing similarities...")
        for query_id, query_emb in tqdm(
            query_embeddings.items(), desc="Computing similarities"
        ):
            similarities = {}
            for item_id, item_emb in item_embeddings.items():
                # Compute cosine similarity
                similarity = torch.cosine_similarity(
                    query_emb.unsqueeze(0), item_emb.unsqueeze(0), dim=1
                ).item()
                similarities[item_id] = similarity

            results[query_id] = similarities

        return results

    try:
        # Call the evaluation function
        metrics = evaluate_qvim_system(compute_similarities, data_path=data_path)
        return metrics
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        return {"mrr": 0.0, "class_wise_mrr": 0.0, "ndcg": 0.0}


def evaluate_vim_clap(model, data, epoch, args, tb_writer=None):
    """
    Modified evaluation function for VimSketch CLAP fine-tuning
    """
    if args.dataset_type != "vim":
        logging.warning("VimSketch evaluation called for non-vim dataset")
        return {}

    # Path to evaluation data (DEV set)
    eval_data_path = os.path.join(os.path.dirname(args.vim_dataset_path), "DEV")
    if not os.path.exists(eval_data_path):
        logging.warning(f"Evaluation data not found at {eval_data_path}")
        return {"mrr": 0.0, "class_wise_mrr": 0.0, "ndcg": 0.0}

    device = torch.device(args.device)

    logging.info(f"Running VimSketch evaluation for epoch {epoch}")

    # Get the CLAP model from the wrapper
    from .train import unwrap_model

    clap_model = unwrap_model(model)

    # Run evaluation
    metrics = validate_clap_with_evaluate(clap_model, eval_data_path, device, args)

    if metrics:
        logging.info(f"Evaluation results for epoch {epoch}:")
        logging.info(f"  MRR: {metrics.get('mrr', 0):.4f}")
        logging.info(f"  Class-wise MRR: {metrics.get('class_wise_mrr', 0):.4f}")
        logging.info(f"  NDCG: {metrics.get('ndcg', 0):.4f}")

        # Log to tensorboard if available
        if tb_writer is not None:
            tb_writer.add_scalar("val/mrr", metrics.get("mrr", 0), epoch)
            tb_writer.add_scalar(
                "val/class_wise_mrr", metrics.get("class_wise_mrr", 0), epoch
            )
            tb_writer.add_scalar("val/ndcg", metrics.get("ndcg", 0), epoch)

        # Log to wandb if available
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {
                        "val/mrr": metrics.get("mrr", 0),
                        "val/class_wise_mrr": metrics.get("class_wise_mrr", 0),
                        "val/ndcg": metrics.get("ndcg", 0),
                        "val/epoch": epoch,
                    }
                )
        except:
            pass

    return metrics
