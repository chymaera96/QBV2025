import json
import os
import pickle
import random
import tarfile
import tempfile

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

EXTRA_FEAT_SR = 100  # 100 frames per second
REF_DESCRIPTION_FILENAME = "references_Phi-4_2505240741_temp060.pkl"
IMIT_DESCRIPTION_FILENAME = "vocal_imitations_Phi-4_2505240852_temp060.pkl"


class VimSketchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir,
        sample_rate=32000,
        duration=10.0,
        add_extra_features: bool = False,
        add_descriptions: bool = False,
        n_short_descriptions: int = 10,
    ):
        """
        Args:
            ...
            n_short_descriptions (int): Number of short descriptions to sample with 50% probability.
        """
        self.dataset_dir = dataset_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.add_extra_features = add_extra_features
        self.add_descriptions = add_descriptions
        self.n_short_descriptions = n_short_descriptions

        reference_filenames = pd.read_csv(
            os.path.join(dataset_dir, "reference_file_names.csv"),
            sep="\t",
            header=None,
            names=["filename"],
        )
        reference_filenames["reference_id"] = reference_filenames["filename"].transform(
            lambda x: "_".join(x.split("_")[1:])
        )

        imitation_file_names = pd.read_csv(
            os.path.join(dataset_dir, "vocal_imitation_file_names.csv"),
            sep="\t",
            header=None,
            names=["filename"],
        )
        imitation_file_names["reference_id"] = imitation_file_names[
            "filename"
        ].transform(lambda x: "_".join(x.split("_")[1:]))

        self.all_pairs = imitation_file_names.merge(
            reference_filenames,
            left_on="reference_id",
            right_on="reference_id",
            how="left",
            suffixes=("_imitation", "_reference"),
        )

        self.cached_files = {"wav": {}, "npy": {}}

        # Pre-load descriptions if required
        if add_descriptions:
            ref_pkl = os.path.join(
                self.dataset_dir, "s2s/descriptions", REF_DESCRIPTION_FILENAME
            )
            imit_pkl = os.path.join(
                self.dataset_dir, "s2s/descriptions", IMIT_DESCRIPTION_FILENAME
            )
            with open(ref_pkl, "rb") as f:
                self.ref_descriptions = pickle.load(f)
            with open(imit_pkl, "rb") as f:
                self.imit_descriptions = pickle.load(f)

    def load_audio(self, path):
        if path not in self.cached_files["wav"]:
            audio, sr = librosa.load(
                path, sr=self.sample_rate, mono=True, duration=self.duration
            )
            self.cached_files["wav"][path] = audio
        return self.__pad_or_truncate__(
            self.cached_files["wav"][path], int(self.sample_rate * self.duration)
        )

    def load_npy(self, path):
        if path not in self.cached_files["npy"]:
            arr = np.load(
                path,
            )
            self.cached_files["npy"][path] = arr
        return self.__pad_or_truncate__(
            self.cached_files["npy"][path], int(self.duration * EXTRA_FEAT_SR)
        )

    def sample_from_descriptions(self, descs):
        return (
            descs
            if isinstance(descs, str)
            else random.choice(descs[: self.n_short_descriptions])
            if len(descs) <= self.n_short_descriptions or random.random() < 0.5
            else random.choice(descs[self.n_short_descriptions :])
        )

    def __pad_or_truncate__(self, arr, fixed_length):
        """supports 1D and 2D arrays"""
        arr = np.asarray(arr)
        out = np.zeros((*arr.shape[:-1], fixed_length), dtype=arr.dtype)
        sl = slice(min(arr.shape[-1], fixed_length))
        out[..., : sl.stop] = arr[..., : sl.stop]
        return out

    def __getitem__(self, index):
        row = self.all_pairs.iloc[index]

        out = {
            "reference_filename": row["filename_reference"],
            "imitation_filename": row["filename_imitation"],
            "reference": self.load_audio(
                os.path.join(self.dataset_dir, "references", row["filename_reference"])
            ),
            "imitation": self.load_audio(
                os.path.join(
                    self.dataset_dir, "vocal_imitations", row["filename_imitation"]
                )
            ),
        }

        # Add extra features or prompt if specified
        if self.add_extra_features:
            ref_npy_path = os.path.join(
                self.dataset_dir,
                "s2s/extra_feat",
                row["filename_reference"].replace(".wav", ".npy"),
            )
            imit_npy_path = os.path.join(
                self.dataset_dir,
                "s2s/extra_feat",
                row["filename_imitation"].replace(".wav", ".npy"),
            )
            out["reference_extra_feat"] = self.load_npy(ref_npy_path)
            out["imitation_extra_feat"] = self.load_npy(imit_npy_path)

        if self.add_descriptions:
            ref_desc = self.ref_descriptions.get(row["filename_reference"]) or [
                row["filename_reference"]
            ]
            imit_desc = self.imit_descriptions.get(row["filename_imitation"]) or [
                row["filename_imitation"]
            ]
            out["reference_description"] = self.sample_from_descriptions(ref_desc)
            out["imitation_description"] = self.sample_from_descriptions(imit_desc)

        return out

    def __len__(self):
        return len(self.all_pairs)


class AESAIMLA_DEV_S2S(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir,
        sample_rate=32000,
        duration=10.0,
        add_extra_features: bool = False,
    ):
        self.dataset_dir = dataset_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.add_extra_features = add_extra_features

        pairs = pd.read_csv(os.path.join(dataset_dir, "DEV Dataset.csv"), skiprows=1)[
            ["Label", "Class", "Items", "Query 1", "Query 2", "Query 3"]
        ]

        pairs = pairs.melt(
            id_vars=[col for col in pairs.columns if "Query" not in col],
            value_vars=["Query 1", "Query 2", "Query 3"],
            var_name="Query Type",
            value_name="Query",
        )

        pairs = pairs.dropna()
        print("Total number of imitations: ", len(pairs["Query"].unique()))
        print("Total number of references: ", len(pairs["Items"].unique()))

        self.all_pairs = pairs
        self.check_files()

        print(f"Found {len(self.all_pairs)} pairs.")

        self.cached_files = {"wav": {}, "npy": {}}

    def check_files(self):
        for _, pair in self.all_pairs.iterrows():
            ref_path = os.path.join(
                self.dataset_dir, "Items", pair["Class"], pair["Items"]
            )
            query_path = os.path.join(
                self.dataset_dir, "Queries", pair["Class"], pair["Query"]
            )
            if not os.path.exists(ref_path):
                print("Missing: ", ref_path)
            if not os.path.exists(query_path):
                print("Missing: ", query_path)

    def load_audio(self, path):
        if path not in self.cached_files["wav"]:
            audio, sr = librosa.load(
                path, sr=self.sample_rate, mono=True, duration=self.duration
            )
            self.cached_files["wav"][path] = audio
        return self.__pad_or_truncate__(
            self.cached_files["wav"][path], int(self.sample_rate * self.duration)
        )

    def load_npy(self, path):
        if path not in self.cached_files["npy"]:
            arr = np.load(
                path,
            )
            self.cached_files["npy"][path] = arr
        return self.__pad_or_truncate__(
            self.cached_files["npy"][path], int(self.duration * EXTRA_FEAT_SR)
        )

    def __pad_or_truncate__(self, arr: np.ndarray, fixed_length: int):
        """supports 1D and 2D arrays"""
        arr = np.asarray(arr)
        out = np.zeros((*arr.shape[:-1], fixed_length), dtype=arr.dtype)
        sl = slice(min(arr.shape[-1], fixed_length))
        out[..., : sl.stop] = arr[..., : sl.stop]
        return out

    def __getitem__(self, index):
        row = self.all_pairs.iloc[index]

        ref_path = os.path.join(self.dataset_dir, "Items", row["Class"], row["Items"])
        query_path = os.path.join(
            self.dataset_dir, "Queries", row["Class"], row["Query"]
        )

        ref_audio = torch.tensor(self.load_audio(ref_path)).float()
        query_audio = torch.tensor(self.load_audio(query_path)).float()

        out = {
            "reference_filename": row["Items"],
            "imitation_filename": row["Query"],
            "reference": ref_audio,
            "imitation": query_audio,
            "reference_class": row["Class"],
            "imitation_class": row["Class"],
        }

        # Add extra features or prompt if specified
        if self.add_extra_features:
            ref_npy_path = os.path.join(
                self.dataset_dir, "s2s/extra_feat", row["Items"].replace(".wav", ".npy")
            )
            imit_npy_path = os.path.join(
                self.dataset_dir, "s2s/extra_feat", row["Query"].replace(".wav", ".npy")
            )
            out["reference_extra_feat"] = self.load_npy(ref_npy_path)
            out["imitation_extra_feat"] = self.load_npy(imit_npy_path)
        return out

    def __len__(self):
        return len(self.all_pairs)


if __name__ == "__main__":
    ds_vims = VimSketchDataset(
        dataset_dir="data/Vim_Sketch_Dataset",
        sample_rate=32000,
        duration=10.0,
        add_extra_features=True,
        add_descriptions=True,
        n_short_descriptions=10,
    )
    ds_dev = AESAIMLA_DEV_S2S(
        dataset_dir="data/qvim-dev",
        sample_rate=32000,
        duration=10.0,
        add_extra_features=True,
    )

    from torch.utils.data import DataLoader
    from tqdm import tqdm

    for ds in [ds_vims, ds_dev]:
        dl = DataLoader(ds, batch_size=10, shuffle=False, drop_last=True, num_workers=4)
        for i, batch in enumerate(tqdm(dl)):
            assert batch["reference"].shape == batch["imitation"].shape == (10, 320000)
            assert (
                batch["reference_filename"] is not None
                and batch["imitation_filename"] is not None
            )
            if i < len(dl) - 1:
                assert (
                    batch["reference_extra_feat"].shape
                    == batch["imitation_extra_feat"].shape
                    == (10, 16, 10 * EXTRA_FEAT_SR)
                )
            if "reference_description" in batch.keys():
                assert all(
                    isinstance(b, str) and len(b) > 0
                    for b in batch["reference_description"]
                )
                for i, text in enumerate(batch["reference_description"]):
                    assert ".wav" not in text, (
                        f"Unexpected .wav in reference description: {text}"
                    )
                    assert len(text) > 0, (
                        f"Reference description is empty for{batch['reference_filename'][i]}"
                    )
                for i, text in enumerate(batch["imitation_description"]):
                    assert ".wav" not in text, (
                        f"Unexpected .wav in imitation description: {text}"
                    )
                    assert len(text) > 0, (
                        f"Imitation description is empty for {batch['imitation_filename'][i]}"
                    )
    print("All tests passed.")


def create_vim_webdataset(dataset_dir, output_dir, split="train", sample_rate=48000):
    """
    Convert VimSketchDataset to WebDataset format for CLAP training.
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset = VimSketchDataset(
        dataset_dir=dataset_dir,
        add_descriptions=True,
        duration=None,  # Don't truncate for webdataset creation
    )

    # Create tar file
    tar_path = os.path.join(output_dir, f"vim_sketch_{split}.tar")
    sizes = {}

    with tarfile.open(tar_path, "w") as tar:
        for i, sample in enumerate(tqdm(dataset, desc=f"Creating {split} webdataset")):
            # Create temporary files
            with tempfile.TemporaryDirectory() as tmpdir:
                base_name = f"sample_{i:06d}"

                # Save reference audio
                ref_audio_path = os.path.join(tmpdir, f"{base_name}_ref.flac")
                sf.write(ref_audio_path, sample["reference"], sample_rate)

                # Save imitation audio
                imit_audio_path = os.path.join(tmpdir, f"{base_name}_imit.flac")
                sf.write(imit_audio_path, sample["imitation"], sample_rate)

                # Create metadata for reference
                ref_metadata = {
                    "text": sample["reference_description"],
                    "original_filename": sample["reference_filename"],
                    "type": "reference",
                }
                ref_json_path = os.path.join(tmpdir, f"{base_name}_ref.json")
                with open(ref_json_path, "w") as f:
                    json.dump(ref_metadata, f)

                # Create metadata for imitation
                imit_metadata = {
                    "text": sample["imitation_description"],
                    "original_filename": sample["imitation_filename"],
                    "type": "imitation",
                }
                imit_json_path = os.path.join(tmpdir, f"{base_name}_imit.json")
                with open(imit_json_path, "w") as f:
                    json.dump(imit_metadata, f)

                # Add files to tar
                tar.add(ref_audio_path, arcname=f"{base_name}_ref.flac")
                tar.add(ref_json_path, arcname=f"{base_name}_ref.json")
                tar.add(imit_audio_path, arcname=f"{base_name}_imit.flac")
                tar.add(imit_json_path, arcname=f"{base_name}_imit.json")

    # Create sizes.json
    # Count actual samples in tar file
    sample_count = 0
    with tarfile.open(tar_path, "r") as tar:
        flac_files = [m for m in tar.getmembers() if m.name.endswith(".flac")]
        sample_count = len(flac_files)

    sizes[f"vim_sketch_{split}.tar"] = sample_count

    sizes_path = os.path.join(output_dir, "sizes.json")
    with open(sizes_path, "w") as f:
        json.dump(sizes, f)

    print(f"Created webdataset with {sample_count} samples at {tar_path}")
    return tar_path, sizes_path


if __name__ == "__main__":
    dataset_dir = "/path/to/your/Vim_Sketch_Dataset"
    output_dir = "/path/to/webdataset/output"

    create_vim_webdataset(dataset_dir, output_dir, "train")


import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader
from training.data import get_mel, tokenizer


class VimSketchCLAPDataset:
    """Adapter to make VimSketchDataset compatible with CLAP training"""

    def __init__(self, vim_dataset, audio_cfg, tmodel="roberta", max_len=480000):
        self.vim_dataset = vim_dataset
        self.audio_cfg = audio_cfg
        self.tmodel = tmodel
        self.max_len = max_len

    def __len__(self):
        # Return double length since we have both reference and imitation
        return len(self.vim_dataset) * 2

    def __getitem__(self, idx):
        # Each VimSketch sample contains both reference and imitation
        vim_idx = idx // 2
        is_imitation = idx % 2 == 1

        vim_sample = self.vim_dataset[vim_idx]

        if is_imitation:
            audio_data = vim_sample["imitation"]
            text_data = vim_sample["imitation_description"]
            audio_name = vim_sample["imitation_filename"]
        else:
            audio_data = vim_sample["reference"]
            text_data = vim_sample["reference_description"]
            audio_name = vim_sample["reference_filename"]

        # Convert to torch tensor if numpy
        if isinstance(audio_data, np.ndarray):
            audio_data = torch.from_numpy(audio_data).float()

        # Ensure mono and correct length
        if audio_data.dim() > 1:
            audio_data = audio_data.mean(dim=0)

        # Pad or truncate to max_len
        if len(audio_data) > self.max_len:
            start_idx = torch.randint(
                0, len(audio_data) - self.max_len + 1, (1,)
            ).item()
            audio_data = audio_data[start_idx : start_idx + self.max_len]
        elif len(audio_data) < self.max_len:
            pad_length = self.max_len - len(audio_data)
            audio_data = torch.cat(
                [
                    audio_data,
                    audio_data.repeat(pad_length // len(audio_data) + 1)[:pad_length],
                ]
            )

        # Tokenize text
        tokenized_text = tokenizer(text_data, tmodel=self.tmodel)

        # Get mel spectrogram
        mel_spec = get_mel(audio_data, self.audio_cfg)

        return {
            "waveform": audio_data,
            "mel_fusion": mel_spec.unsqueeze(
                0
            ),  # Add batch dimension for compatibility
            "text": tokenized_text,
            "audio_name": audio_name,
            "text_name": audio_name.replace(".wav", ".txt"),
            "longer": torch.tensor([False]),  # Most audio will be short
            "__key__": f"sample_{idx:06d}",
            "__url__": f"vim_sketch/train/{audio_name}",
        }


def collate_fn(batch):
    """Custom collate function for VimSketch data"""
    # Stack waveforms
    waveforms = torch.stack([item["waveform"] for item in batch])

    # Stack mel spectrograms
    mel_specs = torch.stack([item["mel_fusion"] for item in batch])

    # Handle text tokenization based on model type
    text_sample = batch[0]["text"]
    if isinstance(text_sample, dict):
        # For BERT/RoBERTa tokenizers
        text_batch = {}
        for key in text_sample.keys():
            text_batch[key] = torch.stack([item["text"][key] for item in batch])
    else:
        # For transformer tokenizer
        text_batch = torch.stack([item["text"] for item in batch])

    return {
        "waveform": waveforms,
        "mel_fusion": mel_specs,
        "text": text_batch,
        "audio_name": [item["audio_name"] for item in batch],
        "text_name": [item["text_name"] for item in batch],
        "longer": torch.stack([item["longer"] for item in batch]),
        "__key__": [item["__key__"] for item in batch],
        "__url__": [item["__url__"] for item in batch],
    }


class VimSketchDataInfo:
    """Data info class compatible with CLAP training"""

    def __init__(self, dataset, batch_size=32, num_workers=4, shuffle=True):
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )
        self.sampler = None

    @property
    def num_samples(self):
        return len(self.dataset)

    @property
    def num_batches(self):
        return len(self.dataloader)


def get_vim_data(args, model_cfg):
    """Create data loaders for VimSketch dataset"""
    from .dataset import VimSketchDataset

    # Create VimSketch dataset
    vim_dataset = VimSketchDataset(
        dataset_dir=args.vim_dataset_path,
        sample_rate=args.sample_rate if hasattr(args, "sample_rate") else 48000,
        duration=None,  # Don't truncate here, handle in adapter
        add_descriptions=True,
        n_short_descriptions=10,
    )

    # Create CLAP-compatible dataset
    clap_dataset = VimSketchCLAPDataset(
        vim_dataset=vim_dataset,
        audio_cfg=model_cfg["audio_cfg"],
        tmodel=args.tmodel,
        max_len=args.max_len if hasattr(args, "max_len") else 480000,
    )

    # Create data info objects
    data = {}
    data["train"] = VimSketchDataInfo(
        clap_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True
    )

    return data
