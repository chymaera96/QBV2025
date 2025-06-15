import glob
import os
import librosa
import numpy as np
import pandas as pd
import torch
import copy
import random

from hc_baseline.modules.augmentations import Augment

class VimSketchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, sample_rate=32000, duration=10.0):
        self.dataset_dir = dataset_dir
        self.sample_rate = sample_rate
        self.duration = duration

        # Read references
        reference_filenames = pd.read_csv(
            os.path.join(dataset_dir, 'reference_file_names.csv'),
            sep='\t',
            header=None,
            names=['filename']
        )
        reference_filenames['reference_id'] = reference_filenames['filename'].transform(
            lambda x: "_".join(x.split('_')[1:])
        )

        # Read imitations with named columns
        imitation_file_names = pd.read_csv(
            os.path.join(dataset_dir, 'vim_with_labels.csv')
        )
        imitation_file_names['reference_id'] = imitation_file_names['filename'].transform(
            lambda x: "_".join(x.split('_')[1:])
        )

        # Merge on reference_id to get anchor-positive pairs and keep sound_class
        self.all_pairs = imitation_file_names.merge(
            reference_filenames,
            left_on="reference_id",
            right_on="reference_id",
            how="left",
            suffixes=('_imitation', '_reference')
        )

        self.cached_files = {}


    def load_audio(self, path, sr=None):
        if sr is None:
            sr = self.sample_rate
        if path not in self.cached_files:
            audio, sr = librosa.load(
                path,
                sr=sr,
                mono=True,
                duration=self.duration
            )
            self.cached_files[path] = audio
        return self.__pad_or_truncate__(self.cached_files[path], sr=sr)


    def __pad_or_truncate__(self, audio, sr=None):
        if sr is None:
            sr = self.sample_rate
        fixed_length = int(sr * self.duration)
        if len(audio) < fixed_length:
            array = np.zeros(fixed_length, dtype="float32")
            array[:len(audio)] = audio
        if len(audio) >= fixed_length:
            array = audio[:fixed_length]

        if isinstance(array, np.ndarray):
            return torch.from_numpy(array).float()
        elif isinstance(array, torch.Tensor):
            return array.float() 
        # return torch.tensor(array, dtype=torch.float32)


    def __getitem__(self, index):

        row = self.all_pairs.iloc[index]

        reference_path = os.path.join(self.dataset_dir, 'references', row['filename_reference'])
        # reference = self.augment(self.load_audio(reference_path))
        reference = self.load_audio(reference_path)
        reference = self.__pad_or_truncate__(reference)

        imitation_path = os.path.join(self.dataset_dir, 'vocal_imitations', row['filename_imitation'])
        # imitation = self.augment(self.load_audio(imitation_path))
        imitation = self.load_audio(imitation_path)
        imitation = self.__pad_or_truncate__(imitation)

        # assert reference.shape[-1] == 320000, f"Reference shape mismatch: {reference.shape}"
        # assert imitation.shape[-1] == 320000, f"Imitation shape mismatch: {imitation.shape}"

        return {
            # 'reference_path': os.path.join(self.dataset_dir, 'references', row['filename_reference']),
            'reference_filename': row['filename_reference'],
            'imitation_filename': row['filename_imitation'],
            'reference': reference,
            'imitation': imitation,
            'anchor_class': row.get('sound_class', None),

        }

    def __len__(self):
        return len(self.all_pairs)


class AESAIMLA_DEV(torch.utils.data.Dataset):

    def __init__(
            self,
            dataset_dir,
            sample_rate=32000,
            duration=10.0
    ):
        self.dataset_dir = dataset_dir
        self.sample_rate = sample_rate
        self.duration = duration

        pairs = pd.read_csv(
            os.path.join(dataset_dir, 'DEV Dataset.csv'),
            skiprows=1
        )[['Label', 'Class', 'Items', 'Query 1', 'Query 2', 'Query 3']]

        # pairs.columns = pairs.columns.droplevel()

        pairs = pairs.melt(id_vars=[col for col in pairs.columns if "Query" not in col],
                           value_vars=["Query 1", "Query 2", "Query 3"],
                           var_name="Query Type",
                           value_name="Query")

        pairs = pairs.dropna()
        print("Total number of imitations: ", len(pairs["Query"].unique()))
        print("Total number of references: ", len(pairs["Items"].unique()))

        self.all_pairs = pairs
        self.check_files()

        print(f"Found {len(self.all_pairs)} pairs.")

        self.cached_files = {}


    def check_files(self):
        for i, pair in self.all_pairs.iterrows():
            reference_name = os.path.join(self.dataset_dir, 'Items', pair['Class'], pair['Items'])
            if not os.path.exists(reference_name):
                print("Missing: ", reference_name)
            imitation_name = os.path.join(self.dataset_dir, 'Queries', pair['Class'], pair['Query'])
            if not os.path.exists(imitation_name):
                print("Missing: ", imitation_name)

    def load_audio(self, path, sr=None):
        if sr is None:
            sr = self.sample_rate
        if path not in self.cached_files:
            audio, sr = librosa.load(
                path,
                sr=sr,
                mono=True,
                duration=self.duration
            )
            self.cached_files[path] = audio
        return self.__pad_or_truncate__(self.cached_files[path], sr=sr)



    def __pad_or_truncate__(self, audio, sr=None):
        if sr is None:
            sr = self.sample_rate
        fixed_length = int(sr * self.duration)
        array = np.zeros(fixed_length, dtype="float32")

        if len(audio) < fixed_length:
            array[:len(audio)] = audio
        if len(audio) >= fixed_length:
            array[:fixed_length]  = audio[:fixed_length]

        return torch.tensor(array, dtype=torch.float32)




    def __getitem__(self, index):

        row = self.all_pairs.iloc[index]

        reference_name = os.path.join(self.dataset_dir, 'Items', row['Class'], row['Items'])
        imitation_name = os.path.join(self.dataset_dir, 'Queries', row['Class'], row['Query'])

        reference = self.load_audio(reference_name)
        imitation = self.load_audio(imitation_name)

        return {
            'reference_path': reference_name,
            'reference_filename': row['Items'],
            'imitation_filename': row['Query'],
            'reference': reference,
            'imitation': imitation,
            'reference_class': row['Class'],
            'imitation_class': row['Class']
        }

    def __len__(self):
        return len(self.all_pairs)
    


class VocalSketchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, sample_rate=32000, duration=10.0):
        self.dataset_dir = dataset_dir
        self.sample_rate = sample_rate
        self.duration = duration

        # self.augment = Augment(sample_rate=sample_rate, max_transforms=5)

        imitation_df = pd.read_csv(
            os.path.join(os.path.dirname(dataset_dir), f'{dataset_dir.split("/")[-1]}_class.csv'),
        )

        self.imitation_filenames = imitation_df['filename'].tolist()
        self.sound_labels = imitation_df['sound_class'].tolist()

        self.cached_files = {}

        # Build index by class
        self.class_to_indices = {}
        for idx, label in enumerate(self.sound_labels):
            label = str(label).lower()
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)


    def load_audio(self, path, sr=None):
        if sr is None:
            sr = self.sample_rate

        if path in self.cached_files:
            return self.cached_files[path]

        try:
            audio, _ = librosa.load(
                path,
                sr=sr,
                mono=True,
                duration=self.duration
            )
            self.cached_files[path] = audio
        except Exception as e:
            self.cached_files[path] = None  # mark as invalid

        return self.cached_files[path]

    def __pad_or_truncate__(self, audio, sr=None):
        if sr is None:
            sr = self.sample_rate
        fixed_length = int(sr * self.duration)
        if len(audio) < fixed_length:
            array = np.zeros(fixed_length, dtype="float32")
            array[:len(audio)] = audio
        else:
            array = audio[:fixed_length]
        return torch.from_numpy(array).float() if isinstance(array, np.ndarray) else array.float()

    def __getitem__(self, index):
        filename = self.imitation_filenames[index]
        path = os.path.join(self.dataset_dir, 'excluded', filename)
        audio = self.load_audio(path)
        if audio is None:
            # Return a dummy entry if corrupted
            return self.__getitem__((index + 1) % len(self))

        # imitation = self.augment(audio)
        imitation = audio
        imitation = self.__pad_or_truncate__(imitation)

        return {
            'imitation': imitation,
            'imitation_filename': filename,
            'sound_label': self.sound_labels[index],
        }

    def __len__(self):
        return len(self.imitation_filenames)
    

class TripletBatchDataset(torch.utils.data.Dataset):
    def __init__(self, anchor_positive_dataset, negative_dataset):
        self.anchor_positive_dataset = anchor_positive_dataset
        self.negative_dataset = negative_dataset

        # Pull class mapping from the negative dataset
        self.class_to_neg_indices = getattr(negative_dataset, 'class_to_indices', {})
        self.neg_dataset_len = len(self.negative_dataset)

    def __getitem__(self, index):
        ap_sample = self.anchor_positive_dataset[index]
        anchor = ap_sample['reference']
        positive = ap_sample['imitation']
        
        anchor_class = ap_sample.get('anchor_class', None)
        anchor_class = str(anchor_class).lower() if anchor_class else None

        # Try same-class negative sampling
        if anchor_class in self.class_to_neg_indices and self.class_to_neg_indices[anchor_class]:
            neg_index = random.choice(self.class_to_neg_indices[anchor_class])
        else:
            # Fallback to random negative
            neg_index = random.randint(0, self.neg_dataset_len - 1)

        neg_sample = self.negative_dataset[neg_index]
        negative = neg_sample['imitation']
        negative_class = neg_sample['sound_label']

        return {
            'anchor': anchor,
            'positive': positive,
            'negative': negative,
            'anchor_filename': ap_sample['reference_filename'],
            'positive_filename': ap_sample['imitation_filename'],
            'negative_filename': neg_sample['imitation_filename'],
            'anchor_class': anchor_class,
            'negative_class': negative_class,
        }

    def __len__(self):
        return len(self.anchor_positive_dataset)
