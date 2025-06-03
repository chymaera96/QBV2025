import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse

import librosa
import torch
from ced_model.feature_extraction_ced import (
    CedFeatureExtractor as HFCedFeatureExtractor,
)
from ced_model.modeling_ced import CedForAudioClassification
from torch import nn
from tqdm import tqdm

from src.retrieval.evaluate import evaluate_qvim_system


class CED(nn.Module):
    def __init__(self, model_name="mispeech/ced-base"):
        super(CED, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 16000  # CED expects 16kHz audio
        self.model_name = model_name

        # Load the feature extractor and model
        self.feature_extractor = HFCedFeatureExtractor.from_pretrained(model_name)
        self.model = CedForAudioClassification.from_pretrained(model_name).to(
            self.device
        )

        # Set model to eval mode and freeze parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Extract CED embeddings from audio file paths.

        Args:
            x: List of audio file paths

        Returns:
            torch.Tensor: Embeddings for each audio file
        """
        embeddings = []

        for file_path in x:
            try:
                audio, orig_sr = librosa.load(file_path, sr=16000)

                # Use feature extractor with proper parameters
                inputs = self.feature_extractor(
                    audio,
                    sampling_rate=self.feature_extractor.sampling_rate,
                    return_tensors="pt",
                )

                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get model outputs (we want hidden states, not classification logits)
                with torch.no_grad():
                    encoder_outputs = self.model.encoder(**inputs)

                    # it's not actually logits
                    last_hidden_state = encoder_outputs["logits"][-1]

                    # Global average pooling over the sequence dimension
                    pooled_features = last_hidden_state.mean(dim=0).squeeze(0)

                    embeddings.append(pooled_features)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

                # if an embedding is already computed, we'll be able to get its shape
                # to return an empty embedding
                # if this happened at the start, there's possibly a bug that needs fixing
                zero_embedding = torch.zeros(
                    pooled_features.shape[0], device=self.device
                )
                embeddings.append(zero_embedding)

        return torch.stack(embeddings)

    def compute_similarities(self, items, queries):
        """
        Compute similarities between queries and items.
        Extract features on GPU, but store and compute similarities on CPU.

        Args:
            items (dict): Dictionary of item file paths.
            queries (dict): Dictionary of query file paths.

        Returns:
            dict: Dictionary containing similarities for each query.
        """

        # Infer embedding paths based on common directory structure
        def get_embedding_path(file_path):
            # Check if it's already an embedding file
            if file_path.endswith(".pt"):
                return file_path

            # Try to find corresponding embedding
            base_dir = os.path.dirname(os.path.dirname(file_path))
            # Adjust path replacement for CED embeddings
            emb_dir = base_dir.replace("/Items", "/embeddings/CED/Items").replace(
                "/Queries", "/embeddings/CED/Queries"
            )

            relative_path = os.path.relpath(file_path, base_dir)
            embedding_path = os.path.join(
                emb_dir, os.path.splitext(relative_path)[0] + ".pt"
            )
            if os.path.exists(embedding_path):
                return embedding_path

            # If no embedding found, return None to indicate need for extraction
            return None

        # Extract or load features for items
        item_features = {}
        for item_id, file_path in tqdm(
            items.items(), total=len(items), desc="Processing items"
        ):
            embedding_path = get_embedding_path(file_path)

            if embedding_path and os.path.exists(embedding_path):
                item_features[item_id] = torch.load(embedding_path, map_location="cpu")
            else:
                # Extract features using CED
                feature = self.forward([file_path])
                # feature is already processed to be 1D per file
                if feature.ndim > 1:
                    processed_feature = feature.squeeze(0)  # Remove batch dimension
                else:
                    processed_feature = feature

                item_features[item_id] = processed_feature.detach().cpu()

            torch.cuda.empty_cache()

        # Extract or load features for queries
        query_features = {}
        for query_id, file_path in tqdm(
            queries.items(), total=len(queries), desc="Processing queries"
        ):
            embedding_path = get_embedding_path(file_path)

            if embedding_path and os.path.exists(embedding_path):
                query_features[query_id] = torch.load(
                    embedding_path, map_location="cpu"
                )
            else:
                feature = self.forward([file_path])
                if feature.ndim > 1:
                    processed_feature = feature.squeeze(0)  # Remove batch dimension
                else:
                    processed_feature = feature

                query_features[query_id] = processed_feature.detach().cpu()

            torch.cuda.empty_cache()

        results = {}
        for query_id, query_feature in tqdm(
            query_features.items(), desc="Computing similarities"
        ):
            results[query_id] = {}
            # Normalize query feature for cosine similarity
            query_feature_normalized = query_feature / query_feature.norm(
                dim=-1, keepdim=True
            )

            for item_id, item_feature in item_features.items():
                # Normalize item feature for cosine similarity
                item_feature_normalized = item_feature / item_feature.norm(
                    dim=-1, keepdim=True
                )

                similarity = torch.matmul(
                    query_feature_normalized, item_feature_normalized.T
                ).item()
                results[query_id][item_id] = similarity

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CED audio embedding extraction and evaluation"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mispeech/ced-base",
        help="CED model name from Hugging Face",
    )

    args = parser.parse_args()

    # Initialize CED model with command line arguments
    ced_instance = CED(model_name=args.model_name)

    evaluate_qvim_system(ced_instance.compute_similarities, data_path="data/DEV/")
