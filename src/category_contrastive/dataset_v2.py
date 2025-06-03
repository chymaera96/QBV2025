import os
import random

import librosa
import numpy as np
import pandas as pd
import torch
import torch.utils.data

from .augmentations import Augment


class UnifiedVimSketch(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir="./data/Vim_Sketch",
        sample_rate=48000,
        feature_extractor=None,
        seed=42,
        augment=None,  # None, "query", "reference", or "both"
        max_transforms=1,
    ):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.feature_extractor = feature_extractor
        self.augment_mode = augment
        self.augmenter = (
            Augment(sample_rate, max_transforms) if self.augment_mode else None
        )
        random.seed(seed)
        np.random.seed(seed)

        # Load all sounds (both queries and references) into a unified list
        self.all_sounds = []

        # Load reference files
        ref_df = pd.read_csv(
            os.path.join(root_dir, "reference_file_names.csv"),
            sep="\t",
            header=None,
            names=["filename"],
        )

        for _, row in ref_df.iterrows():
            category = row["filename"].split("_")[0]
            self.all_sounds.append(
                {
                    "path": os.path.join(self.root_dir, "references", row["filename"]),
                    "id": row["filename"],
                    "category": category,
                    "type": "reference",
                }
            )

        # Load query files
        query_df = pd.read_csv(
            os.path.join(root_dir, "vocal_imitation_file_names.csv"),
            sep="\t",
            header=None,
            names=["filename"],
        )

        for _, row in query_df.iterrows():
            filename_parts = row["filename"].split("_")
            if len(filename_parts) >= 2:
                category = filename_parts[1][:3]  # Extract category from query
                self.all_sounds.append(
                    {
                        "path": os.path.join(
                            self.root_dir, "vocal_imitations", row["filename"]
                        ),
                        "id": row["filename"],
                        "category": category,
                        "type": "query",
                    }
                )

        # Filter to only include sounds with valid categories
        valid_categories = set()
        for sound in self.all_sounds:
            if sound["type"] == "reference":
                valid_categories.add(sound["category"])

        self.all_sounds = [
            s for s in self.all_sounds if s["category"] in valid_categories
        ]

        if not self.all_sounds:
            raise ValueError("No valid sounds found. Check dataset integrity.")

        # Create category mappings
        self.categories = list(valid_categories)
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}

        print(
            f"Found {len(self.categories)} categories with {len(self.all_sounds)} total sounds"
        )
        print(
            f"References: {sum(1 for s in self.all_sounds if s['type'] == 'reference')}"
        )
        print(f"Queries: {sum(1 for s in self.all_sounds if s['type'] == 'query')}")

        self.cached_files = {}

    def load_audio(self, path):
        if path not in self.cached_files:
            audio, sr = librosa.load(path, sr=self.sample_rate, mono=True)
            self.cached_files[path] = audio
        return self.cached_files[path].copy()

    def __len__(self):
        return len(self.all_sounds)

    def __getitem__(self, index):
        sound_info = self.all_sounds[index]

        # Load and process audio
        audio_np = self.load_audio(sound_info["path"])

        # Apply augmentations if needed
        if self.augmenter:
            if (
                (self.augment_mode == "query" and sound_info["type"] == "query")
                or (
                    self.augment_mode == "reference"
                    and sound_info["type"] == "reference"
                )
                or (self.augment_mode == "both")
            ):
                audio_np = self.augmenter.train_transform(audio_np)

        # Feature extraction
        if self.feature_extractor:
            features = self.feature_extractor(audio_np)
        else:
            features = torch.tensor(audio_np).float()

        return {
            "features": features,
            "sound_id": sound_info["id"],
            "category": sound_info["category"],
            "category_idx": self.category_to_idx[sound_info["category"]],
            "type": sound_info["type"],
        }


if __name__ == "__main__":
    # Example usage
    dataset = UnifiedVimSketch(
        root_dir="./data/Vim_Sketch",
        sample_rate=48000,
        feature_extractor=None,
        augment="both",
    )
    print(dataset[0])
