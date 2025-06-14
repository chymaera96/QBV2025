import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from typing import Optional

import librosa
import numpy as np
import torch
import torch.nn as nn
import wandb
from ced_model.feature_extraction_ced import (
    CedFeatureExtractor as HFCedFeatureExtractor,
)
from ced_model.modeling_ced import CedForAudioClassification
from scipy import interpolate
from torch import nn
from tqdm import tqdm

from src.retrieval.evaluate import evaluate_qvim_system


class MLPProjection(nn.Module):
    """MLP projection layer to match the one used in training"""

    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
        super().__init__()
        layers = []
        current_dim = input_dim

        # Add hidden layers if they exist
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim

        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))
        self.projection = nn.Sequential(*layers)

    def forward(self, x):
        return self.projection(x)


class CED(nn.Module):
    def __init__(
        self,
        model_name="mispeech/ced-base",
        use_acoustic_features=True,
        use_pitch_features=True,
        use_mlp_projection=False,
        wandb_run_path=None,
        pretrained_projection_path=None,
    ):
        super(CED, self).__init__()

        self.use_acoustic_features = use_acoustic_features
        self.use_pitch_features = use_pitch_features
        self.use_mlp_projection = use_mlp_projection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 16000  # CED expects 16kHz audio
        self.model_name = model_name

        try:
            # Load the feature extractor and model
            self.feature_extractor = HFCedFeatureExtractor.from_pretrained(model_name)
            self.model = CedForAudioClassification.from_pretrained(model_name).to(
                self.device
            )
        except (OSError, json.JSONDecodeError) as e:
            print(f"Error loading model {model_name}: {e}")
            print("Trying to clear cache and reload...")

            # Clear the cached files and try again

            import shutil
            from pathlib import Path

            cache_dir = (
                Path.home()
                / ".cache"
                / "huggingface"
                / "hub"
                / f"models--{model_name.replace('/', '--')}"
            )
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                print(f"Cleared cache directory: {cache_dir}")

            # Try loading again
            self.feature_extractor = HFCedFeatureExtractor.from_pretrained(model_name)
            self.model = CedForAudioClassification.from_pretrained(model_name).to(
                self.device
            )

        # Set model to eval mode and freeze parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Load MLP projection if specified
        self.mlp_projection = None
        if use_mlp_projection:
            if pretrained_projection_path:
                self.mlp_projection = self._load_pretrained_projection(
                    pretrained_projection_path
                )
            elif wandb_run_path:
                self.mlp_projection = self._load_mlp_projection(wandb_run_path)

    def _load_pretrained_projection(self, model_path):
        """Load the MLP projection from a local .pt file with hard-coded config"""
        try:
            # Hard-coded configuration based on the provided config
            input_dim = 768  # CED-base dimension
            hidden_dims = []  # Empty hidden dims from config
            output_dim = 256  # projection_output_dim from config
            dropout_rate = 0.2  # dropout_rate from config

            print(
                f"Creating MLP with hard-coded config: input_dim={input_dim}, hidden_dims={hidden_dims}, output_dim={output_dim}"
            )

            # Create MLP with correct architecture
            mlp = MLPProjection(input_dim, hidden_dims, output_dim, dropout_rate)

            # Load the checkpoint
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            print(f"Loaded checkpoint keys: {checkpoint.keys()}")

            # Get the state dict and handle key renaming if needed
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            # Check if keys need renaming from 'mlp.*' to 'projection.*'
            original_keys = list(state_dict.keys())
            if any(key.startswith("mlp.") for key in original_keys):
                print("Renaming keys from 'mlp.*' to 'projection.*'")
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("mlp."):
                        new_key = key.replace("mlp.", "projection.")
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                state_dict = new_state_dict

            print(f"Final state dict keys: {list(state_dict.keys())}")

            # Load the state dict
            mlp.load_state_dict(state_dict)

            mlp.eval()
            mlp.to(self.device)

            for param in mlp.parameters():
                param.requires_grad = False

            print(f"Successfully loaded pretrained MLP projection from {model_path}")
            return mlp

        except Exception as e:
            print(f"Error loading pretrained MLP projection: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _load_mlp_projection(self, run_path):
        """Load the MLP projection from wandb artifact"""
        try:
            api = wandb.Api()
            run = api.run(run_path)

            artifacts = run.logged_artifacts()
            if not artifacts:
                raise ValueError(f"No artifacts found for run {run_path}")

            latest_artifact = max(artifacts, key=lambda x: x.created_at)
            artifact_dir = latest_artifact.download()

            # List all files and find the .pt file
            files = os.listdir(artifact_dir)
            print(f"Available files: {files}")

            # Find any .pt file in the directory
            model_path = None
            for filename in files:
                if filename.endswith(".pt"):
                    model_path = os.path.join(artifact_dir, filename)
                    print(f"Found model file: {filename}")
                    break

            if model_path is None:
                raise FileNotFoundError(
                    f"No .pt model file found. Available files: {files}"
                )

            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            print(f"Loaded checkpoint keys: {checkpoint.keys()}")

            # Get config from the run
            config = run.config
            print(f"Run config: {config}")

            # Extract MLP parameters from config - CED has different dimension than CLAP
            input_dim = 768  # CED-base typically has 768 dimensions

            hidden_dims = config.get("hidden_dims", [])  # This might be empty!
            output_dim = config.get("projection_output_dim", 128)
            dropout_rate = config.get("dropout_rate", 0.2)

            print(
                f"Creating MLP with: input_dim={input_dim}, hidden_dims={hidden_dims}, output_dim={output_dim}"
            )

            # Create MLP with correct architecture
            mlp = MLPProjection(input_dim, hidden_dims, output_dim, dropout_rate)

            # Get the state dict and rename keys
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            # Rename keys from 'mlp.*' to 'projection.*'
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("mlp."):
                    new_key = key.replace("mlp.", "projection.")
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value

            print(f"Original keys: {list(state_dict.keys())}")
            print(f"Renamed keys: {list(new_state_dict.keys())}")

            # Load the renamed state dict
            mlp.load_state_dict(new_state_dict)

            mlp.eval()
            mlp.to(self.device)

            for param in mlp.parameters():
                param.requires_grad = False

            print(f"Successfully loaded MLP projection from {run_path}")
            return mlp

        except Exception as e:
            print(f"Error loading MLP projection: {e}")
            import traceback

            traceback.print_exc()
            return None

    def extract_acoustic_features(self, audio_path, target_length=100):
        """
        Extract RMS and optionally pitch contour features from audio file.
        (Same implementation as CLAP version)

        Args:
            audio_path (str): Path to audio file
            target_length (int): Target length for subsampled features (default: 100)

        Returns:
            torch.Tensor: RMS features (100,) or concatenated [RMS, pitch] features (200,)
        """
        try:
            # Load audio - use CED's sample rate
            y, sr = librosa.load(audio_path, sr=self.sample_rate)

            # Extract RMS (Root Mean Square) energy
            # Using frame_length and hop_length to get reasonable temporal resolution
            frame_length = 2048
            hop_length = 512
            rms = librosa.feature.rms(
                y=y, frame_length=frame_length, hop_length=hop_length
            )[0]

            features_list = []

            # Always extract RMS
            if len(rms) > 1:
                # Create interpolation functions
                x_orig = np.linspace(0, 1, len(rms))
                x_new = np.linspace(0, 1, target_length)

                # Interpolate RMS
                f_rms = interpolate.interp1d(x_orig, rms, kind="linear")
                rms_resampled = f_rms(x_new)
            else:
                # Handle edge case of very short audio
                rms_resampled = np.full(target_length, rms[0] if len(rms) > 0 else 0)

            # Normalize RMS features (log scale often works better for RMS)
            rms_resampled = np.log(
                rms_resampled + 1e-8
            )  # Add small epsilon to avoid log(0)
            rms_resampled = (rms_resampled - np.mean(rms_resampled)) / (
                np.std(rms_resampled) + 1e-8
            )
            features_list.append(rms_resampled)

            # Optionally extract pitch features
            if self.use_pitch_features:
                # Extract pitch using librosa's piptrack (fundamental frequency estimation)
                pitches, magnitudes = librosa.piptrack(
                    y=y, sr=sr, threshold=0.1, fmin=50, fmax=2000
                )

                # Select the pitch with highest magnitude at each time frame
                pitch_contour = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t] if magnitudes[index, t] > 0 else 0
                    pitch_contour.append(pitch)

                pitch_contour = np.array(pitch_contour)

                # Subsample to target_length using interpolation
                if len(pitch_contour) > 1:
                    # Interpolate pitch (handle zeros in pitch)
                    x_orig = np.linspace(0, 1, len(pitch_contour))
                    x_new = np.linspace(0, 1, target_length)

                    non_zero_mask = pitch_contour > 0
                    if np.any(non_zero_mask):
                        f_pitch = interpolate.interp1d(
                            x_orig,
                            pitch_contour,
                            kind="linear",
                            bounds_error=False,
                            fill_value=0,
                        )
                        pitch_resampled = f_pitch(x_new)
                        # Set negative interpolated values to 0
                        pitch_resampled = np.maximum(pitch_resampled, 0)
                    else:
                        pitch_resampled = np.zeros(target_length)
                else:
                    # Handle edge case of very short audio
                    pitch_resampled = np.full(
                        target_length, pitch_contour[0] if len(pitch_contour) > 0 else 0
                    )

                # Pitch normalization (convert to log scale and normalize)
                pitch_resampled_norm = np.copy(pitch_resampled)
                non_zero_pitch = pitch_resampled_norm > 0
                if np.any(non_zero_pitch):
                    pitch_resampled_norm[non_zero_pitch] = np.log(
                        pitch_resampled_norm[non_zero_pitch]
                    )
                    pitch_mean = np.mean(pitch_resampled_norm[non_zero_pitch])
                    pitch_std = np.std(pitch_resampled_norm[non_zero_pitch])
                    if pitch_std > 1e-8:
                        pitch_resampled_norm[non_zero_pitch] = (
                            pitch_resampled_norm[non_zero_pitch] - pitch_mean
                        ) / pitch_std
                    # Keep zero values as zero (representing unvoiced segments)

                features_list.append(pitch_resampled_norm)

            # Concatenate features
            acoustic_features = np.concatenate(features_list)

            return torch.tensor(acoustic_features, dtype=torch.float32)

        except Exception as e:
            print(
                f"Warning: Failed to extract acoustic features from {audio_path}: {e}"
            )
            # Return zero vector as fallback
            expected_size = target_length * (2 if self.use_pitch_features else 1)
            return torch.zeros(expected_size, dtype=torch.float32)

    def forward(self, x):
        """
        Extract CED features and optionally combine with acoustic features.

        Args:
            x: List of audio file paths

        Returns:
            torch.Tensor: Combined features for each audio file
        """
        # Extract CED features
        ced_features_list = []

        for file_path in x:
            try:
                audio, orig_sr = librosa.load(file_path, sr=self.sample_rate)

                # Ensure minimum length for CED
                if len(audio) < self.sample_rate / 4:
                    audio = np.pad(
                        audio,
                        (0, int((self.sample_rate / 4)) - len(audio)),
                        mode="constant",
                    )

                # Use feature extractor with proper parameters
                inputs = self.feature_extractor(
                    audio,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                )

                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get model outputs (we want hidden states, not classification logits)
                with torch.no_grad():
                    encoder_outputs = self.model.encoder(**inputs)
                    last_hidden_state = encoder_outputs["logits"][-1]
                    # Global average pooling over the sequence dimension
                    pooled_features = last_hidden_state.mean(dim=0).squeeze(0)
                    ced_features_list.append(pooled_features)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                # Use zero embedding as fallback
                if ced_features_list:
                    zero_embedding = torch.zeros_like(ced_features_list[0])
                else:
                    zero_embedding = torch.zeros(
                        768, device=self.device
                    )  # CED-base dimension
                ced_features_list.append(zero_embedding)

        # Stack CED features
        ced_features_batch = torch.stack(ced_features_list)

        # Apply MLP projection to CED features FIRST (before acoustic features)
        if self.mlp_projection is not None:
            ced_features_batch = self.mlp_projection(ced_features_batch)

        # If not using acoustic features, return the projected CED features
        if not self.use_acoustic_features:
            return ced_features_batch

        # Extract acoustic features for each audio file
        acoustic_features_list = []
        for audio_path in x:
            acoustic_features = self.extract_acoustic_features(audio_path)
            acoustic_features_list.append(acoustic_features)

        # Stack acoustic features
        acoustic_features_batch = torch.stack(acoustic_features_list)
        acoustic_features_batch = acoustic_features_batch.to(ced_features_batch.device)

        # Normalize each feature component separately
        ced_features_norm = torch.nn.functional.normalize(
            ced_features_batch, p=2, dim=-1
        )

        features_to_concat = [ced_features_norm]

        # Process acoustic features same as CLAP version
        if self.use_pitch_features:
            rms_features = acoustic_features_batch[:, :100]
            pitch_features = acoustic_features_batch[:, 100:]
            rms_features_norm = torch.nn.functional.normalize(rms_features, p=2, dim=-1)
            pitch_features_norm = torch.nn.functional.normalize(
                pitch_features, p=2, dim=-1
            )
            features_to_concat.extend([rms_features_norm, pitch_features_norm])
        else:
            rms_features = acoustic_features_batch
            rms_features_norm = torch.nn.functional.normalize(rms_features, p=2, dim=-1)
            features_to_concat.append(rms_features_norm)

        # Concatenate normalized features
        combined_features = torch.cat(features_to_concat, dim=-1)
        return combined_features

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
                feature = self.forward([file_path])
                # Handle different output shapes
                if feature.ndim == 3 and feature.shape[0] == 1:
                    processed_feature = feature.squeeze(0)
                    if processed_feature.ndim == 2:  # Still has sequence dimension
                        processed_feature = processed_feature.mean(dim=0)
                elif feature.ndim == 2:
                    if feature.shape[0] == 1:  # Single item in batch
                        processed_feature = feature.squeeze(0)
                    else:  # Sequence dimension
                        processed_feature = processed_feature.mean(dim=0)
                else:
                    processed_feature = (
                        feature.squeeze() if feature.ndim > 1 else feature
                    )

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
                # Handle different output shapes
                if feature.ndim == 3 and feature.shape[0] == 1:
                    processed_feature = feature.squeeze(0)
                    if processed_feature.ndim == 2:  # Still has sequence dimension
                        processed_feature = processed_feature.mean(dim=0)
                elif feature.ndim == 2:
                    if feature.shape[0] == 1:  # Single item in batch
                        processed_feature = feature.squeeze(0)
                    else:  # Sequence dimension
                        processed_feature = processed_feature.mean(dim=0)
                else:
                    processed_feature = (
                        feature.squeeze() if feature.ndim > 1 else feature
                    )

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
    ced_instance = CED(
        model_name="mispeech/ced-base",
        use_acoustic_features=False,
        use_pitch_features=False,
        use_mlp_projection=True,
        pretrained_projection_path="src/retrieval/SC_ced-base_aug-both_proj-256.pt",
    )

    evaluate_qvim_system(ced_instance.compute_similarities, data_path="data/DEV/")
