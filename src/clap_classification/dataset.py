import os
import random

import librosa
import numpy as np
import pandas as pd
import torch

from .augmentations import Augment


class VimSketch(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir="./data/Vim_Sketch",
        sample_rate=48000,
        feature_extractor=None,
        negative_ratio=1.0,
        seed=42,
        augment=None,  # None, "query", "reference", or "both"
        max_transforms=1,
    ):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.feature_extractor = feature_extractor
        self.augment = augment
        self.augmenter = Augment(sample_rate, max_transforms) if augment else None
        random.seed(seed)

        reference_filenames = pd.read_csv(
            os.path.join(root_dir, "reference_file_names.csv"),
            sep="\t",
            header=None,
            names=["filename"],
        )
        reference_filenames["reference_id"] = reference_filenames["filename"].transform(
            lambda x: "_".join(x.split("_")[1:])
        )

        query_file_names = pd.read_csv(
            os.path.join(root_dir, "vocal_imitation_file_names.csv"),
            sep="\t",
            header=None,
            names=["filename"],
        )
        query_file_names["reference_id"] = query_file_names["filename"].transform(
            lambda x: "_".join(x.split("_")[1:])
        )

        # Create positive pairs (matching query-reference)
        positive_pairs = query_file_names.merge(
            reference_filenames,
            left_on="reference_id",
            right_on="reference_id",
            how="left",
            suffixes=("_imitation", "_reference"),
        )
        positive_pairs["is_match"] = 1  # Label as positive matches

        # Create negative pairs (non-matching query-reference)
        negative_pairs = []
        all_reference_ids = reference_filenames["reference_id"].unique()

        for _, query_row in query_file_names.iterrows():
            query_ref_id = query_row["reference_id"]
            # Select possible negative references (different from the true match)
            negative_refs = [
                ref_id for ref_id in all_reference_ids if ref_id != query_ref_id
            ]

            # Skip if no negative references available
            if not negative_refs:
                continue

            # Determine number of negative samples to create
            num_negatives = max(1, int(negative_ratio))
            for _ in range(num_negatives):
                # Select a random non-matching reference
                negative_ref_id = random.choice(negative_refs)
                negative_ref_row = reference_filenames[
                    reference_filenames["reference_id"] == negative_ref_id
                ].iloc[0]

                negative_pairs.append(
                    {
                        "filename_imitation": query_row["filename"],
                        "reference_id": negative_ref_id,
                        "filename_reference": negative_ref_row["filename"],
                        "is_match": 0,  # Label as negative match
                    }
                )

        # Convert negative pairs to DataFrame
        negative_pairs_df = pd.DataFrame(negative_pairs)

        # Combine positive and negative pairs
        self.all_pairs = pd.concat(
            [positive_pairs, negative_pairs_df], ignore_index=True
        )

        # Shuffle the dataset
        self.all_pairs = self.all_pairs.sample(frac=1.0, random_state=seed).reset_index(
            drop=True
        )

        self.cached_files = {}

    def load_audio(self, path):
        if path not in self.cached_files:
            audio, sr = librosa.load(path, sr=self.sample_rate, mono=True)
            # check if it's CLAP
            if hasattr(self.feature_extractor, "is_afclap"):
                audio = audio.reshape(1, -1)
                audio = torch.from_numpy(
                    int16_to_float32(float32_to_int16(audio))
                ).float()
            self.cached_files[path] = audio
        return self.cached_files[path]

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, index):
        row = self.all_pairs.iloc[index]

        # Load audio files
        reference_audio = self.load_audio(
            os.path.join(self.root_dir, "references", row["filename_reference"])
        )

        query_audio = self.load_audio(
            os.path.join(self.root_dir, "vocal_imitations", row["filename_imitation"])
        )

        # Convert to tensors properly depending on whether they're already tensors
        if isinstance(reference_audio, torch.Tensor):
            reference_item = reference_audio.clone().detach().float()
        else:
            reference_item = torch.tensor(reference_audio).float()

        if isinstance(query_audio, torch.Tensor):
            query_item = query_audio.clone().detach().float()
        else:
            query_item = torch.tensor(query_audio).float()

        if self.augment and self.augmenter:
            if self.augment == "query" or self.augment == "both":
                query_item = self.augmenter(query_item.numpy())
            if self.augment == "reference" or self.augment == "both":
                reference_item = self.augmenter(reference_item.numpy())

        if self.feature_extractor:
            reference_item = self.feature_extractor(reference_item)
            query_item = self.feature_extractor(query_item)

        return {
            "reference_item": reference_item,
            "query_item": query_item,
            "reference_id": row["reference_id"],
            "query_id": row["filename_imitation"],
            "is_match": row["is_match"],
        }


# quantization for CLAP
def int16_to_float32(x):
    return (x / 32767.0).astype("float32")


def float32_to_int16(x):
    x = np.clip(x, a_min=-1.0, a_max=1.0)
    return (x * 32767.0).astype("int16")
