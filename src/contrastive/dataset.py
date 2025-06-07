import os
import random
from typing import Dict, List

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
        augmentations=None,  # List of augmentation names
    ):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.feature_extractor = feature_extractor
        self.augment_mode = augment
        self.augmenter = (
            Augment(sample_rate, max_transforms, augmentations)
            if self.augment_mode
            else None
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


class CrossValidationDataset(torch.utils.data.Dataset):
    """
    Cross-validation dataset that supports different fold strategies:
    - fold 0: vim_train + aimla_dev_val (current setup)
    - fold 1: vim_train + aimla_dev_half_train + aimla_dev_half_val
    - fold 2: vim_90%_train + aimla_dev_train + vim_10%_val
    - fold 3: vim_train + aimla_dev_train + aimla_dev_val
    """

    def __init__(
        self,
        vim_sketch_dir="./data/Vim_Sketch",
        aimla_dev_dir="./data/DEV",
        sample_rate=48000,
        feature_extractor=None,
        fold=0,
        split="train",
        seed=42,
        augment=None,
        max_transforms=1,
        augmentations=None,
    ):
        self.vim_sketch_dir = vim_sketch_dir
        self.aimla_dev_dir = aimla_dev_dir
        self.sample_rate = sample_rate
        self.feature_extractor = feature_extractor
        self.fold = fold
        self.split = split
        self.seed = seed

        # Initialize data list
        self.data = []

        # Initialize augmentation
        self.augment_mode = augment
        self.augmenter = (
            Augment(sample_rate, max_transforms, augmentations)
            if self.augment_mode
            else None
        )

        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)

        # Load data based on fold
        if fold == 0:
            self._load_fold_0()
        elif fold == 1:
            self._load_fold_1()
        elif fold == 2:
            self._load_fold_2()
        elif fold == 3:
            self._load_fold_3()
        else:
            raise ValueError(f"Unsupported fold: {fold}")

        if not self.data:
            raise ValueError(f"No data loaded for fold {fold}, split {split}")

    def _load_vim_sketch_data(self) -> List[Dict]:
        """Load VimSketch data"""
        import pandas as pd

        data = []
        reference_filenames = pd.read_csv(
            os.path.join(self.vim_sketch_dir, "reference_file_names.csv"),
            sep="\t",
            header=None,
            names=["filename"],
        )
        reference_filenames["reference_id"] = reference_filenames["filename"].transform(
            lambda x: "_".join(x.split("_")[1:])
        )

        imitation_file_names = pd.read_csv(
            os.path.join(self.vim_sketch_dir, "vocal_imitation_file_names.csv"),
            sep="\t",
            header=None,
            names=["filename"],
        )
        imitation_file_names["reference_id"] = imitation_file_names[
            "filename"
        ].transform(lambda x: "_".join(x.split("_")[1:]))

        all_pairs = imitation_file_names.merge(
            reference_filenames,
            left_on="reference_id",
            right_on="reference_id",
            how="left",
            suffixes=("_imitation", "_reference"),
        )

        for _, row in all_pairs.iterrows():
            # Try different folder names to match the original VimSketch class
            possible_query_paths = [
                os.path.join(
                    self.vim_sketch_dir, "VocalImitations", row["filename_imitation"]
                ),
                os.path.join(
                    self.vim_sketch_dir, "vocal_imitations", row["filename_imitation"]
                ),
            ]

            possible_reference_paths = [
                os.path.join(
                    self.vim_sketch_dir, "ReferenceAudio", row["filename_reference"]
                ),
                os.path.join(
                    self.vim_sketch_dir, "references", row["filename_reference"]
                ),
            ]

            query_path = None
            reference_path = None

            # Find the correct query path
            for path in possible_query_paths:
                if os.path.exists(path):
                    query_path = path
                    break

            # Find the correct reference path
            for path in possible_reference_paths:
                if os.path.exists(path):
                    reference_path = path
                    break

            if query_path and reference_path:
                data.append(
                    {
                        "query_path": query_path,
                        "reference_path": reference_path,
                        "query_id": row["filename_imitation"],
                        "reference_id": row["filename_reference"],
                        "source": "vim_sketch",
                    }
                )
            else:
                # Debug: print missing files
                if not query_path:
                    print(
                        f"Warning: Query file not found for {row['filename_imitation']}"
                    )
                if not reference_path:
                    print(
                        f"Warning: Reference file not found for {row['filename_reference']}"
                    )

        print(f"Loaded {len(data)} VimSketch pairs from {self.vim_sketch_dir}")
        return data

    def _load_aimla_dev_data(self) -> List[Dict]:
        """Load AIMLA dev data"""
        import pandas as pd

        data = []
        pairs = pd.read_csv(
            os.path.join(self.aimla_dev_dir, "DEV Dataset.csv"), skiprows=1
        )[["Label", "Class", "Items", "Query 1", "Query 2", "Query 3"]]

        pairs = pairs.melt(
            id_vars=[col for col in pairs.columns if "Query" not in col],
            value_vars=["Query 1", "Query 2", "Query 3"],
            var_name="Query Type",
            value_name="Query",
        ).dropna()

        for _, row in pairs.iterrows():
            query_path = os.path.join(
                self.aimla_dev_dir, "Queries", row["Class"], row["Query"]
            )
            reference_path = os.path.join(
                self.aimla_dev_dir, "Items", row["Class"], row["Items"]
            )

            if os.path.exists(query_path) and os.path.exists(reference_path):
                data.append(
                    {
                        "query_path": query_path,
                        "reference_path": reference_path,
                        "query_id": row["Query"],
                        "reference_id": row["Items"],
                        "source": "aimla_dev",
                    }
                )

        return data

    def _load_fold_0(self):
        """
        Fold 0: VimSketch for training, AIMLA for validation
        Train: VimSketch
        Val: AIMLA dev
        """
        if self.split == "train":
            self.data = self._load_vim_sketch_data()
            print(f"Fold 0 train: {len(self.data)} VimSketch pairs")
        else:
            self.data = self._load_aimla_dev_data()
            print(f"Fold 0 val: {len(self.data)} AIMLA pairs")

    def _load_fold_1(self):
        """
        Fold 1: VimSketch + half AIMLA for training, half AIMLA for validation
        Train: VimSketch + AIMLA first half
        Val: AIMLA second half
        """
        vim_data = self._load_vim_sketch_data()
        aimla_data = self._load_aimla_dev_data()

        # Split AIMLA data in half, stratified by class if possible
        random.shuffle(aimla_data)
        split_idx = len(aimla_data) // 2

        if self.split == "train":
            self.data = vim_data + aimla_data[:split_idx]
            print(
                f"Fold 1 train: {len(vim_data)} VimSketch + {len(aimla_data[:split_idx])} AIMLA = {len(self.data)} total"
            )
        else:
            self.data = aimla_data[split_idx:]
            print(f"Fold 1 val: {len(self.data)} AIMLA")

    def _load_fold_2(self):
        """
        Fold 2: 90% VimSketch + AIMLA for training, 10% VimSketch for validation
        Train: 90% VimSketch + AIMLA dev
        Val: 10% VimSketch
        """
        vim_data = self._load_vim_sketch_data()
        aimla_data = self._load_aimla_dev_data()

        # Split VimSketch 90/10
        random.shuffle(vim_data)
        split_idx = int(0.9 * len(vim_data))

        if self.split == "train":
            self.data = vim_data[:split_idx] + aimla_data
            print(
                f"Fold 2 train: {len(vim_data[:split_idx])} VimSketch + {len(aimla_data)} AIMLA = {len(self.data)} total"
            )
        else:
            self.data = vim_data[split_idx:]
            print(f"Fold 2 val: {len(self.data)} VimSketch")

    def _load_fold_3(self):
        """
        Fold 3: Both VimSketch and AIMLA for training, AIMLA for validation
        Train: VimSketch + AIMLA dev
        Val: AIMLA dev (same as training - this tests overfitting/memorization)
        """
        vim_data = self._load_vim_sketch_data()
        aimla_data = self._load_aimla_dev_data()

        if self.split == "train":
            # Use both VimSketch and AIMLA for training
            self.data = vim_data + aimla_data
            print(
                f"Fold 3 train: {len(vim_data)} VimSketch + {len(aimla_data)} AIMLA = {len(self.data)} total"
            )
        else:
            # Use full AIMLA for validation (same as training data)
            self.data = aimla_data
            print(f"Fold 3 val: {len(aimla_data)} AIMLA")

    def load_audio(self, path):
        import librosa

        audio, sr = librosa.load(path, sr=self.sample_rate, mono=True)
        return torch.from_numpy(audio).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        query_audio = self.load_audio(item["query_path"])
        reference_audio = self.load_audio(item["reference_path"])

        # Apply augmentations if enabled
        if self.augmenter is not None:
            if self.augment_mode in ["query", "both"]:
                query_audio = self.augmenter(query_audio.unsqueeze(0)).squeeze(0)
            if self.augment_mode in ["reference", "both"]:
                reference_audio = self.augmenter(reference_audio.unsqueeze(0)).squeeze(
                    0
                )

        # Extract features
        query_features = self.feature_extractor(query_audio)
        reference_features = self.feature_extractor(reference_audio)

        return {
            "query_features": query_features,
            "reference_features": reference_features,
            "query_id": item["query_id"],
            "reference_id": item["reference_id"],
        }


def get_cross_validation_datasets(
    vim_sketch_dir="./data/Vim_Sketch",
    aimla_dev_dir="./data/DEV",
    sample_rate=48000,
    feature_extractor=None,
    fold=0,
    seed=42,
    augment=None,
    max_transforms=1,
    augmentations=None,
):
    """
    Get train and validation datasets for a specific fold

    Returns:
        train_dataset, val_dataset
    """
    train_dataset = CrossValidationDataset(
        vim_sketch_dir=vim_sketch_dir,
        aimla_dev_dir=aimla_dev_dir,
        sample_rate=sample_rate,
        feature_extractor=feature_extractor,
        fold=fold,
        split="train",
        seed=seed,
        augment=augment,
        max_transforms=max_transforms,
        augmentations=augmentations,
    )

    val_dataset = CrossValidationDataset(
        vim_sketch_dir=vim_sketch_dir,
        aimla_dev_dir=aimla_dev_dir,
        sample_rate=sample_rate,
        feature_extractor=feature_extractor,
        fold=fold,
        split="val",
        seed=seed,
        augment=None,  # Never augment validation data
        max_transforms=max_transforms,
        augmentations=augmentations,
    )

    return train_dataset, val_dataset


def get_evaluation_data_path(fold=0, aimla_dev_dir="./data/DEV"):
    """
    Get the appropriate evaluation data path based on fold
    For fold 0, 1, and 3, we evaluate on AIMLA dev
    For fold 2, we evaluate on the held-out VimSketch data (but we'll use AIMLA for consistency)
    """
    return aimla_dev_dir  # For now, always use AIMLA dev for evaluation


if __name__ == "__main__":
    # Test the cross-validation dataset
    for fold in [0, 1, 2, 3]:
        print(f"\n=== Testing Fold {fold} ===")
        train_ds, val_ds = get_cross_validation_datasets(
            fold=fold, feature_extractor=None, seed=42
        )
        print(f"Train size: {len(train_ds)}")
        print(f"Val size: {len(val_ds)}")

        # Check data sources
        train_sources = [train_ds[i]["source"] for i in range(min(10, len(train_ds)))]
        val_sources = [val_ds[i]["source"] for i in range(min(10, len(val_ds)))]
        print(f"Train sources sample: {set(train_sources)}")
        print(f"Val sources sample: {set(val_sources)}")
