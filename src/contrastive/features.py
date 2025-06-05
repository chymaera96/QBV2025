import json
import os
import sys

import librosa
import numpy as np
import torch
import torchopenl3
from ced_model.feature_extraction_ced import (
    CedFeatureExtractor as HFCedFeatureExtractor,
)
from ced_model.modeling_ced import CedForAudioClassification
from torch import nn

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
                # Pad with zeros if audio is shorter than 1 second
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

            # Get model outputs - use encoder directly like in your ced.py
            encoder_outputs = self.model.encoder(**inputs)

            # Get the last hidden state from encoder outputs
            # Based on your ced.py, the structure is encoder_outputs["logits"][-1]
            last_hidden_state = encoder_outputs["logits"][-1]

            # Global average pooling over the sequence dimension
            pooled_features = last_hidden_state.mean(dim=0).squeeze(0)

            return pooled_features.cpu()


class CedPlusFeaturesExtractor:
    def __init__(
        self, model_name="mispeech/ced-base", device="cuda", num_temporal_samples=50
    ):
        self.device = device
        self.model_name = model_name
        self.sample_rate = 16000  # CED expects 16kHz audio
        self.num_temporal_samples = num_temporal_samples

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

    def extract_rms_contour(self, audio_np):
        """Extract RMS energy contour with fixed number of samples"""
        # Use frame length of 1024 samples (64ms at 16kHz)
        frame_length = 1024
        hop_length = frame_length // 4

        # Calculate RMS for each frame
        rms = librosa.feature.rms(
            y=audio_np, frame_length=frame_length, hop_length=hop_length, center=True
        )[0]  # Shape: (n_frames,)

        # Resample to fixed number of samples
        if len(rms) != self.num_temporal_samples:
            # Use linear interpolation to get exactly num_temporal_samples
            from scipy.interpolate import interp1d

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
                # If only one frame, repeat it
                rms = np.full(
                    self.num_temporal_samples, rms[0] if len(rms) > 0 else 0.0
                )

        # Normalize RMS to [0, 1] range using robust normalization
        rms_normalized = rms / (np.percentile(rms, 95) + 1e-8)
        rms_normalized = np.clip(rms_normalized, 0, 1)

        return rms_normalized.astype(np.float32)

    def extract_pitch_contour(self, audio_np):
        """Extract F0 pitch contour with fixed number of samples"""
        # Use librosa's piptrack for pitch estimation
        frame_length = 1024
        hop_length = frame_length // 4

        try:
            # Extract pitch using librosa's yin algorithm (more robust than piptrack)
            f0 = librosa.yin(
                audio_np,
                fmin=80,  # Minimum frequency (Hz) - good for human voice
                fmax=400,  # Maximum frequency (Hz) - good for human voice
                sr=self.sample_rate,
                frame_length=frame_length,
                hop_length=hop_length,
            )
        except:
            # Fallback to piptrack if yin fails
            pitches, magnitudes = librosa.piptrack(
                y=audio_np,
                sr=self.sample_rate,
                threshold=0.1,
                fmin=80,
                fmax=400,
                frame_length=frame_length,
                hop_length=hop_length,
            )
            # Select the pitch with highest magnitude at each frame
            f0 = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t] if magnitudes[index, t] > 0.1 else 0
                f0.append(pitch)
            f0 = np.array(f0)

        # Handle unvoiced regions (set to 0) and smooth the contour
        f0 = np.where(f0 < 80, 0, f0)  # Remove very low frequencies

        # Resample to fixed number of samples
        if len(f0) != self.num_temporal_samples:
            from scipy.interpolate import interp1d

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

        # Normalize pitch: convert to log scale for voiced regions, normalize to [0, 1]
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

        # Calculate spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio_np, sr=self.sample_rate, hop_length=hop_length, n_fft=frame_length
        )[0]  # Shape: (n_frames,)

        # Resample to fixed number of samples
        if len(spectral_centroids) != self.num_temporal_samples:
            from scipy.interpolate import interp1d

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

        # Normalize spectral centroid to [0, 1] using robust normalization
        # Convert to log scale first to handle the wide range
        sc_log = np.log(spectral_centroids + 1e-8)
        sc_normalized = (sc_log - sc_log.min()) / (sc_log.max() - sc_log.min() + 1e-8)

        return sc_normalized.astype(np.float32)

    def __call__(self, audio_tensor):
        """
        Extract CED features + temporal features

        Args:
            audio_tensor: Audio tensor (1D tensor of audio samples)

        Returns:
            Combined feature tensor (CED features + temporal features)
        """
        with torch.no_grad():
            if isinstance(audio_tensor, torch.Tensor):
                audio_np = audio_tensor.cpu().numpy()
            else:
                audio_np = audio_tensor

            # Ensure mono audio
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=0)

            # Ensure minimum length for CED
            if audio_np.ndim == 1 and len(audio_np) < self.sample_rate / 4:
                audio_np = np.pad(
                    audio_np,
                    (0, int((self.sample_rate / 4)) - len(audio_np)),
                    mode="constant",
                )

            # Extract CED features
            inputs = self.feature_extractor(
                audio_np, sampling_rate=self.sample_rate, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            encoder_outputs = self.model.encoder(**inputs)
            last_hidden_state = encoder_outputs["logits"][-1]
            ced_features = last_hidden_state.mean(dim=0).squeeze(0).cpu()

            # Extract temporal features
            rms_contour = self.extract_rms_contour(audio_np)
            pitch_contour = self.extract_pitch_contour(audio_np)
            spectral_centroid_contour = self.extract_spectral_centroid_contour(audio_np)

            # Convert temporal features to tensors
            rms_tensor = torch.from_numpy(rms_contour).float()
            pitch_tensor = torch.from_numpy(pitch_contour).float()
            sc_tensor = torch.from_numpy(spectral_centroid_contour).float()

            # Scale temporal features to be roughly in the same range as CED features
            # CED features typically have values in range [-2, 2], so we scale temporal features accordingly
            ced_std = ced_features.std().item()
            ced_mean = ced_features.mean().item()

            # Scale temporal features to have similar statistics
            rms_scaled = (rms_tensor - 0.5) * (
                ced_std * 2
            )  # Center around 0, scale appropriately
            pitch_scaled = (pitch_tensor - 0.5) * (ced_std * 2)
            sc_scaled = (sc_tensor - 0.5) * (ced_std * 2)

            # Concatenate all features
            combined_features = torch.cat(
                [ced_features, rms_scaled, pitch_scaled, sc_scaled]
            )

            return combined_features


class PaSSTFeatureExtractor:
    def __init__(self, device="cuda", use_lora=False, lora_r=16, lora_alpha=32):
        self.device = device
        self.sample_rate = 32000  # PaSST expects 32kHz audio
        self.use_lora = use_lora

        try:
            from hear21passt.base import get_basic_model

            # Initialize PaSST model
            self.model = PaSSTModel(
                use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha
            ).to(device)

            # Set to eval mode initially (will be set to train mode during training if using LoRA)
            if not use_lora:
                self.model.eval()
                # Freeze all parameters for fully frozen mode
                for param in self.model.parameters():
                    param.requires_grad = False

        except ImportError as e:
            raise ImportError(
                "hear21passt not found. Please install it: pip install hear21passt"
            ) from e

    def __call__(self, audio_tensor):
        """
        Extract PaSST features directly from audio tensor

        Args:
            audio_tensor: Audio tensor (1D tensor of audio samples)

        Returns:
            Feature tensor (768-dimensional from PaSST backbone)
        """
        # Only use no_grad if fully frozen
        context = torch.no_grad() if not self.use_lora else torch.enable_grad()

        with context:
            if isinstance(audio_tensor, np.ndarray):
                audio_tensor = torch.from_numpy(audio_tensor).float()
            else:
                audio_tensor = audio_tensor.float()

            # Ensure mono audio
            if audio_tensor.ndim > 1:
                audio_tensor = audio_tensor.mean(dim=0)

            # PaSST expects exactly 320000 samples (10 seconds at 32kHz)
            expected_length = 320000

            if len(audio_tensor) < expected_length:
                # Pad with zeros if audio is shorter
                padding = expected_length - len(audio_tensor)
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
            elif len(audio_tensor) > expected_length:
                # Truncate if audio is longer
                audio_tensor = audio_tensor[:expected_length]

            # Add batch dimension and move to device
            audio_tensor = audio_tensor.unsqueeze(0).to(self.device)

            # Extract features
            features = self.model(audio_tensor)

            return (
                features.squeeze(0).cpu() if not self.use_lora else features.squeeze(0)
            )


class PaSSTModel(nn.Module):
    def __init__(self, use_lora=False, lora_r=16, lora_alpha=32):
        super().__init__()
        from hear21passt.base import get_basic_model

        self.backbone = get_basic_model(mode="embed_only")
        self.use_lora = use_lora

        # Remove classifier heads
        self.backbone.net.head = nn.Identity()
        self.backbone.net.head_dist = nn.Identity()
        self.backbone.net.pre_logits = nn.Identity()

        if use_lora:
            self._add_lora_layers(lora_r, lora_alpha)
        else:
            # Freeze all parameters for fully frozen mode
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _add_lora_layers(self, r, alpha):
        """Add LoRA layers to attention modules in the transformer blocks"""
        try:
            import loralib as lora
        except ImportError:
            raise ImportError(
                "loralib not found. Please install it: pip install loralib"
            )

        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Replace attention layers with LoRA versions
        for block in self.backbone.net.blocks:
            # Replace query, key, value projections with LoRA
            attn = block.attn

            # Get original dimensions
            embed_dim = attn.qkv.in_features

            # Replace qkv projection with LoRA version
            original_qkv = attn.qkv
            lora_qkv = lora.Linear(
                embed_dim,
                embed_dim * 3,  # qkv combined
                r=r,
                lora_alpha=alpha,
                bias=original_qkv.bias is not None,
            )

            # Copy original weights
            lora_qkv.weight.data = original_qkv.weight.data.clone()
            if original_qkv.bias is not None:
                lora_qkv.bias.data = original_qkv.bias.data.clone()

            # Freeze the original linear layer weights
            lora_qkv.weight.requires_grad = False
            if lora_qkv.bias is not None:
                lora_qkv.bias.requires_grad = False

            # Replace the layer
            attn.qkv = lora_qkv

            # Replace projection layer with LoRA version if it exists
            if hasattr(attn, "proj") and attn.proj is not None:
                original_proj = attn.proj
                lora_proj = lora.Linear(
                    embed_dim,
                    embed_dim,
                    r=r,
                    lora_alpha=alpha,
                    bias=original_proj.bias is not None,
                )

                # Copy original weights
                lora_proj.weight.data = original_proj.weight.data.clone()
                if original_proj.bias is not None:
                    lora_proj.bias.data = original_proj.bias.data.clone()

                # Freeze the original weights
                lora_proj.weight.requires_grad = False
                if lora_proj.bias is not None:
                    lora_proj.bias.requires_grad = False

                attn.proj = lora_proj

        print(f"Added LoRA layers with r={r}, alpha={alpha}")

        # Print number of trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )

    def forward(self, x):
        assert x.shape[1] == 320000, f"Expected input shape [B, 320000], got {x.shape}"
        features = self.backbone(x)
        return features  # Return raw 768-dim features, no projection
