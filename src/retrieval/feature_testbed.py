import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, List

import librosa
import numpy as np
import torch
import torch.nn as nn
from scipy.fftpack import dct
from scipy.interpolate import interp1d

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ced_model.feature_extraction_ced import (
    CedFeatureExtractor as HFCedFeatureExtractor,
)
from ced_model.modeling_ced import CedForAudioClassification

import wandb
from src.retrieval.evaluate import evaluate_qvim_system


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def forward(self, audio_paths: List[str]) -> torch.Tensor:
        """Extract features from a list of audio file paths

        Args:
            audio_paths: List of paths to audio files

        Returns:
            torch.Tensor: Features of shape (batch_size, feature_dim)
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class AcousticFeatures(FeatureExtractor):
    """Extract RMS, pitch, and spectral centroid contours"""

    def __init__(
        self,
        name: str = "acoustic",
        num_temporal_samples: int = 100,
        sample_rate: int = 16000,
        use_rms: bool = True,
        use_pitch: bool = True,
        use_spectral_centroid: bool = True,
    ):
        super().__init__(name)
        self.num_temporal_samples = num_temporal_samples
        self.sample_rate = sample_rate
        self.use_rms = use_rms
        self.use_pitch = use_pitch
        self.use_spectral_centroid = use_spectral_centroid

    def extract_rms_contour(self, audio_np):
        """Extract RMS energy contour with fixed number of samples"""
        frame_length = 1024
        hop_length = frame_length // 4

        rms = librosa.feature.rms(
            y=audio_np, frame_length=frame_length, hop_length=hop_length, center=True
        )[0]

        if len(rms) != self.num_temporal_samples:
            if len(rms) > 1:
                x_old = np.linspace(0, 1, len(rms))
                x_new = np.linspace(0, 1, self.num_temporal_samples)
                f = interp1d(
                    x_old,
                    rms,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                rms = f(x_new)
            else:
                rms = np.full(
                    self.num_temporal_samples, rms[0] if len(rms) > 0 else 0.0
                )

        rms_normalized = rms / (np.percentile(rms, 95) + 1e-8)
        rms_normalized = np.clip(rms_normalized, 0, 1)
        return rms_normalized.astype(np.float32)

    def extract_pitch_contour(self, audio_np):
        """Extract F0 pitch contour with fixed number of samples"""
        frame_length = 1024
        hop_length = frame_length // 4

        try:
            f0 = librosa.yin(
                audio_np,
                fmin=80,
                fmax=400,
                sr=self.sample_rate,
                frame_length=frame_length,
                hop_length=hop_length,
            )
        except:
            pitches, magnitudes = librosa.piptrack(
                y=audio_np,
                sr=self.sample_rate,
                threshold=0.1,
                fmin=80,
                fmax=400,
                frame_length=frame_length,
                hop_length=hop_length,
            )
            f0 = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t] if magnitudes[index, t] > 0.1 else 0
                f0.append(pitch)
            f0 = np.array(f0)

        f0 = np.where(f0 < 80, 0, f0)

        if len(f0) != self.num_temporal_samples:
            if len(f0) > 1:
                x_old = np.linspace(0, 1, len(f0))
                x_new = np.linspace(0, 1, self.num_temporal_samples)
                f = interp1d(
                    x_old,
                    f0,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                f0 = f(x_new)
            else:
                f0 = np.full(self.num_temporal_samples, f0[0] if len(f0) > 0 else 0.0)

        f0_log = np.where(f0 > 0, np.log(f0 + 1e-8), 0)
        if f0_log.max() > f0_log.min():
            f0_normalized = (f0_log - f0_log.min()) / (
                f0_log.max() - f0_log.min() + 1e-8
            )
        else:
            f0_normalized = f0_log

        return f0_normalized.astype(np.float32)

    def extract_spectral_centroid_contour(self, audio_np):
        """Extract spectral centroid contour with fixed number of samples"""
        frame_length = 1024
        hop_length = frame_length // 4

        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio_np, sr=self.sample_rate, hop_length=hop_length, n_fft=frame_length
        )[0]

        if len(spectral_centroids) != self.num_temporal_samples:
            if len(spectral_centroids) > 1:
                x_old = np.linspace(0, 1, len(spectral_centroids))
                x_new = np.linspace(0, 1, self.num_temporal_samples)
                f = interp1d(
                    x_old,
                    spectral_centroids,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                spectral_centroids = f(x_new)
            else:
                spectral_centroids = np.full(
                    self.num_temporal_samples,
                    spectral_centroids[0] if len(spectral_centroids) > 0 else 0.0,
                )

        sc_log = np.log(spectral_centroids + 1e-8)
        sc_normalized = (sc_log - sc_log.min()) / (sc_log.max() - sc_log.min() + 1e-8)
        return sc_normalized.astype(np.float32)

    def forward(self, audio_paths: List[str]) -> torch.Tensor:
        """Extract acoustic features from audio files"""
        features_list = []

        for audio_path in audio_paths:
            try:
                y, _ = librosa.load(audio_path, sr=self.sample_rate)

                feature_components = []

                if self.use_rms:
                    rms = self.extract_rms_contour(y)
                    feature_components.append(rms)

                if self.use_pitch:
                    pitch = self.extract_pitch_contour(y)
                    feature_components.append(pitch)

                if self.use_spectral_centroid:
                    sc = self.extract_spectral_centroid_contour(y)
                    feature_components.append(sc)

                features = np.concatenate(feature_components)
                features_list.append(features)

            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                # Fallback zero features
                expected_dim = (
                    (self.num_temporal_samples if self.use_rms else 0)
                    + (self.num_temporal_samples if self.use_pitch else 0)
                    + (self.num_temporal_samples if self.use_spectral_centroid else 0)
                )
                features_list.append(np.zeros(expected_dim, dtype=np.float32))

        return torch.tensor(np.stack(features_list), dtype=torch.float32)


class DCTFeatures(FeatureExtractor):
    """Apply DCT to acoustic features"""

    def __init__(
        self,
        base_extractor: AcousticFeatures,
        num_dct_coeffs: int = 16,
        name: str = "dct_acoustic",
    ):
        super().__init__(name)
        self.base_extractor = base_extractor
        self.num_dct_coeffs = num_dct_coeffs

    def forward(self, audio_paths: List[str]) -> torch.Tensor:
        """Extract DCT coefficients of acoustic features"""
        base_features = self.base_extractor.forward(audio_paths)

        dct_features_list = []
        for features in base_features:
            # Split back into components based on base extractor config
            components = []
            start_idx = 0

            if self.base_extractor.use_rms:
                rms = features[
                    start_idx : start_idx + self.base_extractor.num_temporal_samples
                ]
                rms_dct = dct(rms.numpy(), type=2, norm="ortho")[: self.num_dct_coeffs]
                components.append(rms_dct)
                start_idx += self.base_extractor.num_temporal_samples

            if self.base_extractor.use_pitch:
                pitch = features[
                    start_idx : start_idx + self.base_extractor.num_temporal_samples
                ]
                pitch_dct = dct(pitch.numpy(), type=2, norm="ortho")[
                    : self.num_dct_coeffs
                ]
                components.append(pitch_dct)
                start_idx += self.base_extractor.num_temporal_samples

            if self.base_extractor.use_spectral_centroid:
                sc = features[
                    start_idx : start_idx + self.base_extractor.num_temporal_samples
                ]
                sc_dct = dct(sc.numpy(), type=2, norm="ortho")[: self.num_dct_coeffs]
                components.append(sc_dct)

            dct_features = np.concatenate(components)
            dct_features_list.append(dct_features)

        return torch.tensor(np.stack(dct_features_list), dtype=torch.float32)


class MLPProjection(nn.Module):
    """MLP projection layer"""

    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
        super().__init__()
        layers = []
        current_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.projection = nn.Sequential(*layers)

    def forward(self, x):
        return self.projection(x)


class CEDFeatures(FeatureExtractor):
    """Extract CED model features"""

    def __init__(self, name: str = "ced", model_name: str = "mispeech/ced-base"):
        super().__init__(name)
        self.model_name = model_name
        self.sample_rate = 16000
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load CED model
        self.feature_extractor = HFCedFeatureExtractor.from_pretrained(model_name)
        self.model = CedForAudioClassification.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, audio_paths: List[str]) -> torch.Tensor:
        """Extract CED features from audio files"""
        features_list = []

        for audio_path in audio_paths:
            try:
                audio, _ = librosa.load(audio_path, sr=self.sample_rate)

                # Ensure minimum length
                if len(audio) < self.sample_rate / 4:
                    audio = np.pad(
                        audio,
                        (0, int(self.sample_rate / 4) - len(audio)),
                        mode="constant",
                    )

                inputs = self.feature_extractor(
                    audio, sampling_rate=self.sample_rate, return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    encoder_outputs = self.model.encoder(**inputs)
                    last_hidden_state = encoder_outputs["logits"][-1]
                    pooled_features = last_hidden_state.mean(dim=0).squeeze(0)
                    features_list.append(pooled_features.cpu())

            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                # Fallback zero features (CED-base dimension)
                features_list.append(torch.zeros(768))

        return torch.stack(features_list)


class CEDProjectedFeatures(FeatureExtractor):
    """CED features projected through trained MLP"""

    def __init__(
        self,
        wandb_run_path: str,
        name: str = "ced_projected",
        model_name: str = "mispeech/ced-base",
    ):
        super().__init__(name)
        self.ced_extractor = CEDFeatures(name="ced_base", model_name=model_name)
        self.mlp_projection = self._load_mlp_projection(wandb_run_path)

    def _load_mlp_projection(self, run_path):
        """Load MLP projection from wandb artifact"""
        try:
            api = wandb.Api()
            run = api.run(run_path)

            artifacts = run.logged_artifacts()
            if not artifacts:
                raise ValueError(f"No artifacts found for run {run_path}")

            latest_artifact = max(artifacts, key=lambda x: x.created_at)
            artifact_dir = latest_artifact.download()

            # Find model file
            files = os.listdir(artifact_dir)
            model_path = None
            for filename in files:
                if filename.endswith(".pt"):
                    model_path = os.path.join(artifact_dir, filename)
                    break

            if model_path is None:
                raise FileNotFoundError(
                    f"No .pt model file found. Available files: {files}"
                )

            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            config = run.config

            # Create MLP
            input_dim = 768  # CED-base dimension
            hidden_dims = config.get("hidden_dims", [])
            output_dim = config.get("projection_output_dim", 128)
            dropout_rate = config.get("dropout_rate", 0.2)

            mlp = MLPProjection(input_dim, hidden_dims, output_dim, dropout_rate)

            # Load state dict
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("mlp."):
                    new_key = key.replace("mlp.", "projection.")
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value

            mlp.load_state_dict(new_state_dict)
            mlp.eval()
            mlp.to(self.ced_extractor.device)

            for param in mlp.parameters():
                param.requires_grad = False

            print(f"Successfully loaded MLP projection from {run_path}")
            return mlp

        except Exception as e:
            print(f"Error loading MLP projection: {e}")
            return None

    def forward(self, audio_paths: List[str]) -> torch.Tensor:
        """Extract projected CED features"""
        ced_features = self.ced_extractor.forward(audio_paths)

        if self.mlp_projection is not None:
            ced_features = ced_features.to(self.ced_extractor.device)
            with torch.no_grad():
                projected_features = self.mlp_projection(ced_features)
            return projected_features.cpu()
        else:
            return ced_features


class CombinedFeatures(FeatureExtractor):
    """Combine multiple feature extractors by concatenation"""

    def __init__(self, extractors: List[FeatureExtractor], name: str = "combined"):
        super().__init__(name)
        self.extractors = extractors

    def forward(self, audio_paths: List[str]) -> torch.Tensor:
        """Extract and concatenate features from all extractors"""
        all_features = []

        for extractor in self.extractors:
            features = extractor.forward(audio_paths)
            # Normalize each feature type separately
            features_norm = torch.nn.functional.normalize(features, p=2, dim=-1)
            all_features.append(features_norm)

        return torch.cat(all_features, dim=-1)


class CEDOnlyProjectedFeatures(FeatureExtractor):
    """CED features with zeroed handcrafted slots passed through trained MLP"""

    def __init__(
        self,
        wandb_run_path: str,
        name: str = "ced_only_projected",
        model_name: str = "mispeech/ced-base",
    ):
        super().__init__(name)
        self.ced_extractor = CEDFeatures(name="ced_base", model_name=model_name)
        self.mlp_projection, self.input_dim, self.ced_dim, self.handcrafted_dim = (
            self._load_mlp_projection(wandb_run_path)
        )

    def _load_mlp_projection(self, run_path):
        """Load MLP projection from wandb artifact and determine input structure"""
        try:
            api = wandb.Api()
            run = api.run(run_path)

            artifacts = run.logged_artifacts()
            if not artifacts:
                raise ValueError(f"No artifacts found for run {run_path}")

            latest_artifact = max(artifacts, key=lambda x: x.created_at)
            artifact_dir = latest_artifact.download()

            # Find model file
            files = os.listdir(artifact_dir)
            model_path = None
            for filename in files:
                if filename.endswith(".pt"):
                    model_path = os.path.join(artifact_dir, filename)
                    break

            if model_path is None:
                raise FileNotFoundError(
                    f"No .pt model file found. Available files: {files}"
                )

            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            config = run.config

            # Determine input dimensions from the first layer
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            first_layer_key = None
            for key in state_dict.keys():
                if "weight" in key and ("0." in key or "projection.0." in key):
                    first_layer_key = key
                    break

            if first_layer_key is None:
                raise ValueError("Could not find first layer weights in state dict")

            input_dim = state_dict[first_layer_key].shape[1]
            ced_dim = 768  # CED-base dimension
            handcrafted_dim = input_dim - ced_dim

            print(f"Detected input structure:")
            print(f"  Total input dim: {input_dim}")
            print(f"  CED dim: {ced_dim}")
            print(f"  Handcrafted dim: {handcrafted_dim}")

            # Create MLP
            hidden_dims = config.get("hidden_dims", [])
            output_dim = config.get("projection_output_dim", 128)
            dropout_rate = config.get("dropout_rate", 0.2)

            mlp = MLPProjection(input_dim, hidden_dims, output_dim, dropout_rate)

            # Load state dict
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("mlp."):
                    new_key = key.replace("mlp.", "projection.")
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value

            mlp.load_state_dict(new_state_dict)
            mlp.eval()
            mlp.to(self.ced_extractor.device)

            for param in mlp.parameters():
                param.requires_grad = False

            print(f"Successfully loaded MLP projection from {run_path}")
            return mlp, input_dim, ced_dim, handcrafted_dim

        except Exception as e:
            print(f"Error loading MLP projection: {e}")
            return None, None, None, None

    def forward(self, audio_paths: List[str]) -> torch.Tensor:
        """Extract CED features and pad with zeros for handcrafted features"""
        ced_features = self.ced_extractor.forward(audio_paths)

        if self.mlp_projection is not None:
            batch_size = ced_features.shape[0]

            # Create input tensor with CED features + zero-padded handcrafted features
            full_input = torch.zeros(
                batch_size, self.input_dim, device=ced_features.device
            )

            # Assuming CED features come first (adjust if different)
            full_input[:, : self.ced_dim] = ced_features.to(ced_features.device)
            # Handcrafted features remain zero: full_input[:, self.ced_dim:] = 0

            full_input = full_input.to(self.ced_extractor.device)
            with torch.no_grad():
                projected_features = self.mlp_projection(full_input)
            return projected_features.cpu()
        else:
            return ced_features


class RetrievalSystem:
    """System for testing feature combinations on retrieval tasks"""

    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor

    def compute_similarities(
        self, items: Dict[str, str], queries: Dict[str, str]
    ) -> Dict:
        """Compute similarities between queries and items"""

        # Extract features for items
        item_paths = list(items.values())
        item_features = self.feature_extractor.forward(item_paths)

        # Extract features for queries
        query_paths = list(queries.values())
        query_features = self.feature_extractor.forward(query_paths)

        # Normalize for cosine similarity
        item_features_norm = torch.nn.functional.normalize(item_features, p=2, dim=-1)
        query_features_norm = torch.nn.functional.normalize(query_features, p=2, dim=-1)

        # Compute similarities
        similarities = torch.matmul(query_features_norm, item_features_norm.T)

        # Convert to dictionary format expected by evaluation
        results = {}
        item_ids = list(items.keys())
        query_ids = list(queries.keys())

        for i, query_id in enumerate(query_ids):
            results[query_id] = {}
            for j, item_id in enumerate(item_ids):
                results[query_id][item_id] = similarities[i, j].item()

        return results

    def evaluate(self, data_path: str = "data/DEV/"):
        """Evaluate the retrieval system"""
        print(f"Evaluating with feature extractor: {self.feature_extractor}")
        return evaluate_qvim_system(self.compute_similarities, data_path=data_path)


if __name__ == "__main__":
    """Main function for testing different feature combinations"""

    # Define individual feature extractors
    acoustic_full = AcousticFeatures(
        name="acoustic_full", use_rms=True, use_pitch=True, use_spectral_centroid=True
    )

    acoustic_rms_only = AcousticFeatures(
        name="acoustic_rms", use_rms=True, use_pitch=False, use_spectral_centroid=False
    )

    dct_full = DCTFeatures(acoustic_full, num_dct_coeffs=8, name="dct_full")
    dct_rms = DCTFeatures(acoustic_rms_only, num_dct_coeffs=8, name="dct_rms")

    ced_base = CEDFeatures(name="ced_base")

    # CED with MLP projection (replace with your actual wandb run path)
    ced_projected = CEDProjectedFeatures(
        wandb_run_path="cplachouras/qvim/3dpe2sjw", name="ced_projected"
    )

    # NEW: CED-only features through the combined MLP
    ced_only_projected = CEDOnlyProjectedFeatures(
        wandb_run_path="cplachouras/qvim/v1fgvxhf", name="ced_only_projected"
    )

    # Test different combinations
    test_configs = [
        # Individual features
        ("Acoustic (RMS only)", acoustic_rms_only),
        ("Acoustic (Full)", acoustic_full),
        ("DCT (RMS only)", dct_rms),
        ("DCT (Full)", dct_full),
        ("CED Base", ced_base),
        ("CED Projected", ced_projected),
        ("CED Only (through combined MLP)", ced_only_projected),
        # Combinations
        ("Acoustic + CED", CombinedFeatures([acoustic_full, ced_base], "acoustic+ced")),
        ("DCT + CED", CombinedFeatures([dct_full, ced_base], "dct+ced")),
        (
            "Acoustic + CED Projected",
            CombinedFeatures([acoustic_full, ced_projected], "acoustic+ced_proj"),
        ),
        (
            "DCT + CED Projected",
            CombinedFeatures([dct_full, ced_projected], "dct+ced_proj"),
        ),
    ]

    results = {}

    for config_name, feature_extractor in test_configs:
        print(f"\n{'=' * 50}")
        print(f"Testing: {config_name}")
        print(f"{'=' * 50}")

        try:
            system = RetrievalSystem(feature_extractor)
            metrics = system.evaluate()
            results[config_name] = metrics

            print(f"Results for {config_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

        except Exception as e:
            print(f"Error testing {config_name}: {e}")
            import traceback

            traceback.print_exc()
            results[config_name] = None

    # Print summary
    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")

    for config_name, metrics in results.items():
        if metrics is not None:
            # Print key metric (adjust based on what evaluate_qvim_system returns)
            print(f"{config_name}: {metrics}")
        else:
            print(f"{config_name}: FAILED")
