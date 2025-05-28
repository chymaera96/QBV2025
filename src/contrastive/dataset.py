import os
import random

import librosa
import numpy as np
import pandas as pd
import torch
import torch.utils.data

from .augmentations import Augment


class VimSketch(torch.utils.data.Dataset):
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
        np.random.seed(seed)  # Ensure numpy's random choices are also seeded

        # Load reference files
        ref_df = pd.read_csv(
            os.path.join(root_dir, "reference_file_names.csv"),
            sep="\t",
            header=None,
            names=["filename"],
        )
        ref_df["reference_id"] = ref_df["filename"].apply(
            lambda x: "_".join(x.split("_")[1:])
        )
        self.references = {
            row["reference_id"]: os.path.join(
                self.root_dir, "references", row["filename"]
            )
            for _, row in ref_df.iterrows()
        }
        self.all_reference_ids = list(self.references.keys())
        if not self.all_reference_ids:
            raise ValueError("No reference files found or processed.")

        # Load query files
        query_df = pd.read_csv(
            os.path.join(root_dir, "vocal_imitation_file_names.csv"),
            sep="\t",
            header=None,
            names=["filename"],
        )
        query_df["reference_id"] = query_df["filename"].apply(
            lambda x: "_".join(x.split("_")[1:])
        )

        self.queries = []
        for _, row in query_df.iterrows():
            if row["reference_id"] in self.references:
                self.queries.append(
                    {
                        "query_path": os.path.join(
                            self.root_dir, "vocal_imitations", row["filename"]
                        ),
                        "query_id": row["filename"],
                        "positive_ref_id": row["reference_id"],
                    }
                )
            # else:
            # Consider logging a warning if a query's positive reference is missing
            # print(f"Warning: Positive reference {row['reference_id']} for query {row['filename']} not found.")

        if not self.queries:
            raise ValueError(
                "No query-positive pairs found. Check dataset integrity and paths."
            )

        self.cached_files = {}

    def load_audio(self, path):
        if path not in self.cached_files:
            audio, sr = librosa.load(path, sr=self.sample_rate, mono=True)
            self.cached_files[path] = audio
        return self.cached_files[
            path
        ].copy()  # Return a copy to prevent modification of cached audio

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        query_info = self.queries[index]

        anchor_audio_path = query_info["query_path"]
        positive_ref_id = query_info["positive_ref_id"]
        positive_audio_path = self.references[positive_ref_id]

        # Load and process audio
        anchor_audio_np = self.load_audio(anchor_audio_path)
        positive_audio_np = self.load_audio(positive_audio_path)

        # Apply augmentations if needed
        if self.augmenter:
            if self.augment_mode == "query" or self.augment_mode == "both":
                anchor_audio_np = self.augmenter.train_transform(anchor_audio_np)
            if self.augment_mode == "reference" or self.augment_mode == "both":
                positive_audio_np = self.augmenter.train_transform(positive_audio_np)

        # Feature extraction
        if self.feature_extractor:
            anchor_feat = self.feature_extractor(anchor_audio_np)
            positive_feat = self.feature_extractor(positive_audio_np)
        else:
            anchor_feat = torch.tensor(anchor_audio_np).float()
            positive_feat = torch.tensor(positive_audio_np).float()

        return {
            "query_features": anchor_feat,  # Changed key name to match training loop
            "reference_features": positive_feat,  # Changed key name to match training loop
            "query_id": query_info["query_id"],
            "reference_id": positive_ref_id,  # Changed key name to match training loop
        }
