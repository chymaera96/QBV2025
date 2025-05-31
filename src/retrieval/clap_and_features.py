import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from typing import Optional

import laion_clap
import librosa
import numpy as np
import torch
from scipy import interpolate
from torch import nn
from tqdm import tqdm

import AFCLAP.my_laion_clap.CLAP.src.laion_clap as af_laion_clap
from src.retrieval.evaluate import evaluate_qvim_system


class CLAP(nn.Module):
    def __init__(
        self,
        model_id=1,
        target_audio_layer: Optional[str] = None,
        use_acoustic_features=True,
        use_pitch_features=True,
    ):
        super(CLAP, self).__init__()

        self.is_afclap = False
        self.use_acoustic_features = use_acoustic_features
        self.use_pitch_features = use_pitch_features

        if model_id == "AF":
            self.is_afclap = True
            afclap_ckpt_path = "./ckpt/afclap.pt"

            # Initialize AFCLAP model
            # Note: target_audio_layer is not currently passed to AFCLAP.
            # If needed for AFCLAP, its CLAP_Module and underlying model would require similar modifications.
            self.model = af_laion_clap.CLAP_Module(
                enable_fusion=True, amodel="HTSAT-afclap", tmodel="t5"
            ).cuda()

            # Load AFCLAP checkpoint
            self.model.load_afclap_ckpt(ckpt=afclap_ckpt_path, verbose=False)
        else:
            # Original CLAP initialization
            enable_fusion = False
            # Model IDs 2 and 3 are fusion models in standard CLAP
            if str(model_id) == "2" or str(model_id) == "3":
                enable_fusion = True

            # Pass target_audio_layer to laion_clap.CLAP_Module constructor
            self.model = laion_clap.CLAP_Module(
                enable_fusion=enable_fusion, target_audio_layer_name=target_audio_layer
            ).to("cuda")
            self.model.load_ckpt(model_id=model_id)

        self.model.eval()

    def extract_acoustic_features(self, audio_path, target_length=100):
        """
        Extract RMS and optionally pitch contour features from audio file.

        Args:
            audio_path (str): Path to audio file
            target_length (int): Target length for subsampled features (default: 100)

        Returns:
            torch.Tensor: RMS features (100,) or concatenated [RMS, pitch] features (200,)
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)

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
        if self.is_afclap:
            # AFCLAP uses different parameters for embedding extraction
            clap_features = self.model.get_audio_embedding_from_filelist(
                x=x, sr=16000, use_tensor=True
            )
        else:
            # Original CLAP embedding extraction
            # This will now return intermediate features if target_audio_layer was set
            clap_features = self.model.get_audio_embedding_from_filelist(
                x=x, use_tensor=True
            )

        if not self.use_acoustic_features:
            return clap_features

        # Extract acoustic features for each audio file
        acoustic_features_list = []
        for audio_path in x:
            acoustic_features = self.extract_acoustic_features(audio_path)
            acoustic_features_list.append(acoustic_features)

        # Stack acoustic features
        acoustic_features_batch = torch.stack(acoustic_features_list)

        # Move to same device as CLAP features
        acoustic_features_batch = acoustic_features_batch.to(clap_features.device)

        # Handle different dimensionalities of CLAP features
        if clap_features.ndim == 3:  # (batch, seq_len, dim)
            # Average over sequence dimension for CLAP features
            clap_features_pooled = clap_features.mean(dim=1)  # (batch, dim)
        elif clap_features.ndim == 2:  # (batch, dim) or (seq_len, dim)
            if clap_features.shape[0] == len(x):  # (batch, dim)
                clap_features_pooled = clap_features
            else:  # (seq_len, dim) - single item
                clap_features_pooled = clap_features.mean(
                    dim=0, keepdim=True
                )  # (1, dim)
        else:
            clap_features_pooled = clap_features

        # Ensure batch dimension matches
        if clap_features_pooled.shape[0] != len(x):
            # Handle case where CLAP returns single tensor for batch
            clap_features_pooled = clap_features_pooled.unsqueeze(0).repeat(len(x), 1)

        # Normalize each feature component separately to ensure equal contribution
        # CLAP features are typically already normalized, but let's ensure consistency
        clap_features_norm = torch.nn.functional.normalize(
            clap_features_pooled, p=2, dim=-1
        )

        features_to_concat = [clap_features_norm]

        # Split acoustic features based on what was extracted
        if self.use_pitch_features:
            # RMS and pitch features
            rms_features = acoustic_features_batch[:, :100]  # First 100 dims
            pitch_features = acoustic_features_batch[:, 100:]  # Last 100 dims

            # L2 normalize each component separately
            rms_features_norm = torch.nn.functional.normalize(rms_features, p=2, dim=-1)
            pitch_features_norm = torch.nn.functional.normalize(
                pitch_features, p=2, dim=-1
            )

            features_to_concat.extend([rms_features_norm, pitch_features_norm])
        else:
            # Only RMS features
            rms_features = acoustic_features_batch  # All 100 dims are RMS
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
            # Adjust path replacement if intermediate features are stored differently
            emb_dir = base_dir.replace("/Items", "/embeddings/CLAP/Items").replace(
                "/Queries", "/embeddings/CLAP/Queries"
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
                # self.forward now returns potentially intermediate features + acoustic features
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
    clap_instance = CLAP(
        model_id=1,
        use_acoustic_features=True,
        use_pitch_features=False,
    )

    evaluate_qvim_system(clap_instance.compute_similarities, data_path="data/DEV/")
