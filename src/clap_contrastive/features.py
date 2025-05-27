import os
import sys

import numpy as np
import torch

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
