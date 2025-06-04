import json
import os
import sys

import numpy as np
import torch
import torchopenl3
from ced_model.feature_extraction_ced import (
    CedFeatureExtractor as HFCedFeatureExtractor,
)
from ced_model.modeling_ced import CedForAudioClassification

# Add project root to Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import CLAP libraries directly
import laion_clap

import AFCLAP.my_laion_clap.CLAP.src.laion_clap as af_laion_clap


# quantization for CLAP
def int16_to_float32(x):
    if isinstance(x, torch.Tensor):
        return (x / 32767.0).float()
    return (x / 32767.0).astype("float32")


def float32_to_int16(x):
    if isinstance(x, torch.Tensor):
        x = torch.clamp(x, min=-1.0, max=1.0)
        return (x * 32767.0).short()
    x = np.clip(x, a_min=-1.0, a_max=1.0)
    return (x * 32767.0).astype("int16")


class CLAPFeatureExtractor:
    def __init__(self, model_id="AF", device="cuda"):
        self.is_afclap = model_id == "AF"
        self.device = device

        if self.is_afclap:
            if af_laion_clap is None:
                raise ImportError(
                    "AFCLAP not found. Please install it or choose a different model."
                )

            # Initialize AFCLAP model
            self.model = af_laion_clap.CLAP_Module(
                enable_fusion=True, amodel="HTSAT-afclap", tmodel="t5"
            ).to(device)

            afclap_ckpt_path = "./ckpt/afclap.pt"
            self.model.load_afclap_ckpt(ckpt=afclap_ckpt_path, verbose=False)
        else:
            enable_fusion = False
            if model_id == 2 or model_id == 3:
                enable_fusion = True
            self.model = laion_clap.CLAP_Module(enable_fusion=enable_fusion).to(device)
            self.model.load_ckpt(model_id=model_id)

        self.model.eval()

    def __call__(self, audio_tensor):
        """
        Extract features directly from audio tensor

        Args:
            audio_tensor: Audio tensor (1D tensor of audio samples)

        Returns:
            Feature tensor
        """
        with torch.no_grad():
            if isinstance(audio_tensor, np.ndarray):
                audio_tensor = torch.from_numpy(audio_tensor).float().to(self.device)
            else:
                audio_tensor = audio_tensor.float().to(self.device)

            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.reshape(1, -1)

            # Apply quantization
            audio_tensor = int16_to_float32(float32_to_int16(audio_tensor))

            # Extract features
            if self.is_afclap:
                # AFCLAP uses sr=16000
                features = self.model.get_audio_embedding_from_data(
                    x=audio_tensor, sr=16000, use_tensor=True
                )
            else:
                # Original CLAP uses sr=48000
                features = self.model.get_audio_embedding_from_data(
                    x=audio_tensor, use_tensor=True
                )

            return features.squeeze(0).cpu()


class OpenL3FeatureExtractor:
    def __init__(
        self,
        device="cuda",
        input_repr="mel256",
        embedding_size=6144,
        content_type="env",
    ):
        self.device = device
        self.sample_rate = 48000
        self.input_repr = input_repr
        self.embedding_size = embedding_size
        self.content_type = content_type

        # OpenL3 configuration with user-specified parameters
        self.model = torchopenl3.models.load_audio_embedding_model(
            input_repr=input_repr,
            content_type=content_type,
            embedding_size=embedding_size,
        )

        # Add these lines to ensure model is in eval mode and frozen
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def __call__(self, audio_tensor):
        """
        Extract OpenL3 features directly from audio tensor

        Args:
            audio_tensor: Audio tensor (1D tensor of audio samples)

        Returns:
            Feature tensor
        """
        # Wrap everything in no_grad since this is a frozen feature extractor
        with torch.no_grad():
            if isinstance(audio_tensor, torch.Tensor):
                audio_np = audio_tensor.cpu().numpy()
            else:
                audio_np = audio_tensor

            # Ensure mono audio
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=0)

            # Extract embeddings using OpenL3
            emb, ts = torchopenl3.get_audio_embedding(
                audio_np, self.sample_rate, model=self.model, center=True, hop_size=0.1
            )

            # Average over time dimension to get a single embedding
            # emb shape is (1, num_frames, embedding_size)
            if emb.ndim == 3:
                embedding = emb.squeeze(0).mean(
                    dim=0
                )  # Remove batch dim and average over time
            elif emb.ndim == 2:
                embedding = emb.mean(dim=0)  # Average over time
            else:
                embedding = emb.squeeze()

            # Convert to tensor if not already
            if not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding)

            # Ensure we return CPU tensor to avoid GPU memory accumulation
            return embedding.cpu()


class CedFeatureExtractor:
    def __init__(self, model_name="mispeech/ced-base", device="cuda"):
        self.device = device
        self.model_name = model_name
        self.sample_rate = 16000  # CED expects 16kHz audio

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

    def __call__(self, audio_tensor):
        """
        Extract CED features directly from audio tensor

        Args:
            audio_tensor: Audio tensor (1D tensor of audio samples)

        Returns:
            Feature tensor (hidden states from the model)
        """
        with torch.no_grad():
            if isinstance(audio_tensor, torch.Tensor):
                audio_np = audio_tensor.cpu().numpy()
            else:
                audio_np = audio_tensor

            # Ensure mono audio
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=0)

            # ensure longer than 16000
            if audio_np.ndim == 1 and len(audio_np) < self.sample_rate / 4:
                # Pad with zeros if audio is shorter than 0.25 seconds
                audio_np = np.pad(
                    audio_np,
                    (0, int((self.sample_rate / 4)) - len(audio_np)),
                    mode="constant",
                )

            # Prepare inputs using the feature extractor
            inputs = self.feature_extractor(
                audio_np, sampling_rate=self.sample_rate, return_tensors="pt"
            )

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get model outputs - use encoder directly
            encoder_outputs = self.model.encoder(**inputs)

            # Get the last hidden state from encoder outputs
            last_hidden_state = encoder_outputs["logits"][-1]

            # Global average pooling over the sequence dimension
            pooled_features = last_hidden_state.mean(dim=0).squeeze(0)

            return pooled_features.cpu()
