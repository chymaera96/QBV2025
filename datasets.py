import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np


class DEV(Dataset):
    """Dataset class for the vocal imitation dev set."""

    def __init__(
        self,
        root_dir="./data/DEV/",
        transform=None,
        target_sample_rate=22050,
        mode="triplet",
    ):
        """
        Initialize the DEV dataset.

        Args:
            root_dir (str): Root directory containing the dataset
            transform (callable, optional): Optional transform to be applied on a sample
            target_sample_rate (int): Target sample rate for audio resampling
            mode (str): Dataset mode ('triplet', 'pairs', 'classification')
        """
        self.root_dir = root_dir
        self.csv_path = os.path.join(root_dir, "DEVUpdateComplete.csv")
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        self.mode = mode

        # Read and process the CSV file
        self.data = pd.read_csv(self.csv_path, skiprows=1)  # Skip the header row

        # Clean column names (the CSV has extra commas)
        self.data.columns = ["Label", "Class", "Items", "Query 1", "Query 2", "Query 3"]

        # Create mapping from class names to indices
        self.classes = sorted(self.data["Class"].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        if self.mode == "triplet":
            # Each entry has 1 reference and 3 queries
            return len(self.data) * 3
        elif self.mode == "pairs":
            # Each reference paired with each of its queries
            return len(self.data) * 3
        else:  # classification
            # Count all individual files
            return len(self.data) * 4  # 1 reference + 3 queries

    def __getitem__(self, idx):
        if self.mode == "triplet":
            # For triplet mode, we return (anchor, positive, negative)
            # idx // 3 gives us the row in the dataframe
            # idx % 3 gives us which query to use (0, 1, or 2)
            row_idx = idx // 3
            query_idx = idx % 3

            # Get the current row
            row = self.data.iloc[row_idx]

            # Get reference (item) audio
            reference_path = os.path.join(
                self.root_dir, "Items", row["Class"], row["Items"]
            )
            reference_audio = self._load_and_process_audio(reference_path)

            # Get query audio (positive example)
            query_column = f"Query {query_idx + 1}"
            query_path = os.path.join(
                self.root_dir, "Queries", row["Class"], row[query_column]
            )
            query_audio = self._load_and_process_audio(query_path)

            # Get a negative example (from a different class)
            # Choose a random different class
            negative_class_row_class = row["Class"]
            negative_idx = np.random.randint(0, len(self.data))
            # Ensure the negative example is from a different class
            while self.data.iloc[negative_idx]["Class"] == negative_class_row_class:
                negative_idx = np.random.randint(0, len(self.data))
            
            negative_row = self.data.iloc[negative_idx]
            negative_class = negative_row["Class"]

            # Choose a random query from the negative class
            neg_query_idx = np.random.randint(1, 4)
            negative_path = os.path.join(
                self.root_dir,
                "Queries",
                negative_class,
                negative_row[f"Query {neg_query_idx}"],
            )
            negative_audio = self._load_and_process_audio(negative_path)

            # Get composite label
            label = {
                "item_id": row["Label"],
                "class_idx": self.class_to_idx[row["Class"]]
            }

            return {
                "anchor": reference_audio,
                "positive": query_audio,
                "negative": negative_audio,
                "label": label,
            }

        elif self.mode == "pairs":
            # For pairs mode, we return (reference, query, is_same_class)
            # idx // 3 gives us the row in the dataframe
            # idx % 3 gives us which query to use (0, 1, or 2)
            row_idx = idx // 3
            query_idx = idx % 3

            # Get the current row
            row = self.data.iloc[row_idx]

            # Get reference (item) audio
            reference_path = os.path.join(
                self.root_dir, "Items", row["Class"], row["Items"]
            )
            reference_audio = self._load_and_process_audio(reference_path)

            # Get query audio
            query_column = f"Query {query_idx + 1}"
            query_path = os.path.join(
                self.root_dir, "Queries", row["Class"], row[query_column]
            )
            query_audio = self._load_and_process_audio(query_path)

            # Get composite label
            label = {
                "item_id": row["Label"],
                "class_idx": self.class_to_idx[row["Class"]]
            }

            return {"reference": reference_audio, "query": query_audio, "label": label}

        else:  # classification mode
            # For classification, we return (query, reference, class_label)
            # idx // 3 gives us the row in the dataframe
            # idx % 3 gives us which query to use (0, 1, or 2)
            row_idx = idx // 3
            query_idx = idx % 3

            # Get the current row
            row = self.data.iloc[row_idx]

            # Get reference (item) audio
            reference_path = os.path.join(
                self.root_dir, "Items", row["Class"], row["Items"]
            )
            reference_audio = self._load_and_process_audio(reference_path)

            # Get query audio
            query_column = f"Query {query_idx + 1}"
            query_path = os.path.join(
                self.root_dir, "Queries", row["Class"], row[query_column]
            )
            query_audio = self._load_and_process_audio(query_path)

            # Get composite label
            label = {
                "item_id": row["Label"],
                "class_idx": self.class_to_idx[row["Class"]]
            }

            return {"query": query_audio, "reference": reference_audio, "label": label}

    def _load_and_process_audio(self, audio_path):
        """Load and preprocess audio file."""
        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample if needed
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=self.target_sample_rate
                )
                waveform = resampler(waveform)

            # Apply transform if specified
            if self.transform:
                waveform = self.transform(waveform)

            return waveform

        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return None


if __name__ == "__main__":
    # test all dataset modes

    dataset = DEV(root_dir="./data/DEV/", mode="triplet")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch in dataloader:
        print("Triplet Mode:")
        print("Anchor shape:", batch["anchor"].shape)
        print("Positive shape:", batch["positive"].shape)
        print("Negative shape:", batch["negative"].shape)
        print("Label:", batch["label"])
        break
    dataset = DEV(root_dir="./data/DEV/", mode="pairs")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch in dataloader:
        print("Pairs Mode:")
        print("Reference shape:", batch["reference"].shape)
        print("Query shape:", batch["query"].shape)
        print("Label:", batch["label"])
        break
    dataset = DEV(root_dir="./data/DEV/", mode="classification")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch in dataloader:
        print("Classification Mode:")
        print("Query shape:", batch["query"].shape)
        print("Reference shape:", batch["reference"].shape)
        print("Label:", batch["label"])
        break
