import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse

import librosa
import numpy as np
import torch
import torchopenl3
from torch import nn
from tqdm import tqdm

from src.retrieval.evaluate import evaluate_qvim_system


class OpenL3(nn.Module):
    def __init__(self, input_repr="mel128", embedding_size=512, content_type="env"):
        super(OpenL3, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 48000
        self.input_repr = input_repr
        self.embedding_size = embedding_size
        self.content_type = content_type

        # Load OpenL3 model once to avoid reloading
        self.model = torchopenl3.models.load_audio_embedding_model(
            input_repr=input_repr,
            content_type=content_type,
            embedding_size=embedding_size,
        )

    def forward(self, x):
        """
        Extract OpenL3 embeddings from audio file paths.

        Args:
            x: List of audio file paths

        Returns:
            torch.Tensor: Embeddings for each audio file
        """
        embeddings = []

        for file_path in x:
            # Load audio file
            try:
                audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)

                # Extract embeddings using the new API
                emb, ts = torchopenl3.get_audio_embedding(
                    audio,
                    sr,
                    model=self.model,
                    center=True,  # Keep default centering
                    hop_size=0.1,  # Default 10 Hz frame rate
                )

                # Average over time dimension to get a single embedding per file
                # emb shape is (1, num_frames, embedding_size)
                if emb.ndim == 3:
                    embedding = emb.squeeze(0).mean(
                        dim=0
                    )  # Remove batch dim and average over time
                elif emb.ndim == 2:
                    embedding = emb.mean(dim=0)  # Average over time
                else:
                    embedding = emb.squeeze()

                # Convert to tensor if it's not already
                if not isinstance(embedding, torch.Tensor):
                    embedding = torch.tensor(embedding)

                # Move to device
                embedding = embedding.to(self.device)
                embeddings.append(embedding)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                # Return zero embedding as fallback
                zero_embedding = torch.zeros(self.embedding_size, device=self.device)
                embeddings.append(zero_embedding)

        return torch.stack(embeddings)

    def extract_features(
        self,
        dataset_dir,
        output_dir,
        batch_size=4,
    ):
        # get all .wav files in the dataset_dir and subdirs
        files = glob.glob(f"{dataset_dir}/**/*.wav", recursive=True)

        # Process files in batches to avoid memory issues
        for i in range(0, len(files), batch_size):
            batch_files = files[i : i + batch_size]

            # Get output file paths
            output_files = [
                f"{output_dir}/{file.replace(dataset_dir, '').replace('.wav', '.pt')}"
                for file in batch_files
            ]

            # Filter out files that already have embeddings
            to_process = []
            final_output_files = []
            for file, output_file in zip(batch_files, output_files):
                if os.path.exists(output_file):
                    print(f"Embedding for {file} already exists, skipping...")
                    continue
                to_process.append(file)
                final_output_files.append(output_file)

            if not to_process:
                continue

            # Create directories for output files
            for output_file in final_output_files:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Compute embeddings (from filepaths)
            embeddings = self.forward(to_process)

            # Save embeddings
            for embedding, output_file in zip(embeddings, final_output_files):
                torch.save(embedding, output_file)
                print(f"Saved embedding to {output_file}")

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
            # Adjust path replacement for OpenL3 embeddings
            emb_dir = base_dir.replace("/Items", "/embeddings/OpenL3/Items").replace(
                "/Queries", "/embeddings/OpenL3/Queries"
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
                # Extract features using OpenL3
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
        description="OpenL3 audio embedding extraction and evaluation"
    )
    parser.add_argument(
        "--input_repr",
        type=str,
        default="mel256",
        help="Input representation for OpenL3 model",
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=512,
        help="Embedding size for OpenL3 model",
    )
    parser.add_argument(
        "--content_type", type=str, default="env", help="Content type for OpenL3 model"
    )

    args = parser.parse_args()

    # Initialize OpenL3 model with command line arguments
    openl3_instance = OpenL3(
        input_repr=args.input_repr,
        embedding_size=args.embedding_size,
        content_type=args.content_type,
    )

    evaluate_qvim_system(openl3_instance.compute_similarities, data_path="data/DEV/")
