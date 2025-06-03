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
        np.random.seed(seed)

        # Load reference files and extract categories
        ref_df = pd.read_csv(
            os.path.join(root_dir, "reference_file_names.csv"),
            sep="\t",
            header=None,
            names=["filename"],
        )

        # Extract category from reference filename (number1 from number1_number2Title)
        ref_df["category"] = ref_df["filename"].apply(lambda x: x.split("_")[0])

        self.references = {}
        self.category_to_reference = {}

        for _, row in ref_df.iterrows():
            ref_path = os.path.join(self.root_dir, "references", row["filename"])
            ref_id = "_".join(row["filename"].split("_")[1:])
            category = row["category"]

            self.references[ref_id] = ref_path
            self.category_to_reference[category] = {
                "ref_id": ref_id,
                "ref_path": ref_path,
            }

        # Load query files and map to categories
        query_df = pd.read_csv(
            os.path.join(root_dir, "vocal_imitation_file_names.csv"),
            sep="\t",
            header=None,
            names=["filename"],
        )

        self.queries = []
        for _, row in query_df.iterrows():
            # Extract category from query filename (number2 from number1_number2Title)
            filename_parts = row["filename"].split("_")
            if len(filename_parts) >= 2:
                query_category = filename_parts[1][:3]  # number2 becomes the category

                # Check if this category exists in our references
                if query_category in self.category_to_reference:
                    self.queries.append(
                        {
                            "query_path": os.path.join(
                                self.root_dir, "vocal_imitations", row["filename"]
                            ),
                            "query_id": row["filename"],
                            "category": query_category,
                            "positive_ref_id": self.category_to_reference[
                                query_category
                            ]["ref_id"],
                            "positive_ref_path": self.category_to_reference[
                                query_category
                            ]["ref_path"],
                        }
                    )

        if not self.queries:
            raise ValueError(
                "No query-category pairs found. Check dataset integrity and paths."
            )

        # Create category mappings for contrastive learning
        self.categories = list(set(query["category"] for query in self.queries))
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}

        print(
            f"Found {len(self.categories)} categories with {len(self.queries)} total queries"
        )
        self.cached_files = {}

    def load_audio(self, path):
        if path not in self.cached_files:
            audio, sr = librosa.load(path, sr=self.sample_rate, mono=True)
            self.cached_files[path] = audio
        return self.cached_files[path].copy()

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        query_info = self.queries[index]

        anchor_audio_path = query_info["query_path"]
        positive_audio_path = query_info["positive_ref_path"]

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
            "query_features": anchor_feat,
            "reference_features": positive_feat,
            "query_id": query_info["query_id"],
            "reference_id": query_info["positive_ref_id"],
            "category": query_info["category"],
            "category_idx": self.category_to_idx[query_info["category"]],
        }


if __name__ == "__main__":
    # Example usage
    dataset = VimSketch(
        root_dir="./data/Vim_Sketch",
        sample_rate=48000,
        feature_extractor=None,
        augment="both",
    )
    print(dataset[0])
