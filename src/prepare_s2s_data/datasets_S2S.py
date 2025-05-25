from typing import Optional, List
import os
import pickle
import random
import pandas as pd
import numpy as np
import torch
import librosa

EXTRA_FEAT_SR = 100  # 100 frames per second
REF_DESCRIPTION_FILENAME = 'references_Phi-4_2505240741_temp060.pkl'
IMIT_DESCRIPTION_FILENAME = 'vocal_imitations_Phi-4_2505240852_temp060.pkl'


class VimSketchDataset(torch.utils.data.Dataset):

    def __init__(self,
                 dataset_dir,
                 sample_rate=32000,
                 duration=10.0,
                 add_extra_features: bool = False,
                 add_descriptions: bool = False,
                 n_short_descriptions: int = 10):
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

        reference_filenames = pd.read_csv(os.path.join(dataset_dir, 'reference_file_names.csv'),
                                          sep='\t',
                                          header=None,
                                          names=['filename'])
        reference_filenames['reference_id'] = reference_filenames['filename'].transform(lambda x: "_".join(x.split('_')[1:]))

        imitation_file_names = pd.read_csv(os.path.join(dataset_dir, 'vocal_imitation_file_names.csv'),
                                           sep='\t',
                                           header=None,
                                           names=['filename'])
        imitation_file_names['reference_id'] = imitation_file_names['filename'].transform(lambda x: "_".join(x.split('_')[1:]))

        self.all_pairs = imitation_file_names.merge(reference_filenames,
                                                    left_on="reference_id",
                                                    right_on="reference_id",
                                                    how="left",
                                                    suffixes=('_imitation', '_reference'))

        self.cached_files = {"wav": {}, "npy": {}}

        # Pre-load descriptions if required
        if add_descriptions:
            ref_pkl = os.path.join(self.dataset_dir, 's2s/descriptions', REF_DESCRIPTION_FILENAME)
            imit_pkl = os.path.join(self.dataset_dir, 's2s/descriptions', IMIT_DESCRIPTION_FILENAME)
            with open(ref_pkl, "rb") as f:
                self.ref_descriptions = pickle.load(f)
            with open(imit_pkl, "rb") as f:
                self.imit_descriptions = pickle.load(f)

    def load_audio(self, path):
        if path not in self.cached_files["wav"]:
            audio, sr = librosa.load(path, sr=self.sample_rate, mono=True, duration=self.duration)
            self.cached_files["wav"][path] = audio
        return self.__pad_or_truncate__(self.cached_files["wav"][path], int(self.sample_rate * self.duration))

    def load_npy(self, path):
        if path not in self.cached_files["npy"]:
            arr = np.load(path,)
            self.cached_files["npy"][path] = arr
        return self.__pad_or_truncate__(self.cached_files["npy"][path], int(self.duration * EXTRA_FEAT_SR))

    def sample_from_descriptions(self, descs):
        return descs if isinstance(descs, str) \
            else random.choice(descs[:self.n_short_descriptions]) if len(descs) <= self.n_short_descriptions or random.random() < 0.5 \
            else random.choice(descs[self.n_short_descriptions:])

    def __pad_or_truncate__(self, arr, fixed_length):
        """supports 1D and 2D arrays"""
        arr = np.asarray(arr)
        out = np.zeros((*arr.shape[:-1], fixed_length), dtype=arr.dtype)
        sl = slice(min(arr.shape[-1], fixed_length))
        out[..., :sl.stop] = arr[..., :sl.stop]
        return out

    def __getitem__(self, index):

        row = self.all_pairs.iloc[index]

        out = {
            'reference_filename': row['filename_reference'],
            'imitation_filename': row['filename_imitation'],
            'reference': self.load_audio(os.path.join(self.dataset_dir, 'references', row['filename_reference'])),
            'imitation': self.load_audio(os.path.join(self.dataset_dir, 'vocal_imitations', row['filename_imitation'])),
        }

        # Add extra features or prompt if specified
        if self.add_extra_features:
            ref_npy_path = os.path.join(self.dataset_dir, 's2s/extra_feat', row['filename_reference'].replace('.wav', '.npy'))
            imit_npy_path = os.path.join(self.dataset_dir, 's2s/extra_feat', row['filename_imitation'].replace('.wav', '.npy'))
            out['reference_extra_feat'] = self.load_npy(ref_npy_path)
            out['imitation_extra_feat'] = self.load_npy(imit_npy_path)

        if self.add_descriptions:
            ref_desc = self.ref_descriptions.get(row['filename_reference']) or [row['filename_reference']]
            imit_desc = self.imit_descriptions.get(row['filename_imitation']) or [row['filename_imitation']]
            out['reference_description'] = self.sample_from_descriptions(ref_desc)
            out['imitation_description'] = self.sample_from_descriptions(imit_desc)

        return out

    def __len__(self):
        return len(self.all_pairs)


class AESAIMLA_DEV_S2S(torch.utils.data.Dataset):

    def __init__(self, dataset_dir, sample_rate=32000, duration=10.0, add_extra_features: bool = False):
        self.dataset_dir = dataset_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.add_extra_features = add_extra_features

        pairs = pd.read_csv(os.path.join(dataset_dir, 'DEV Dataset.csv'),
                            skiprows=1)[['Label', 'Class', 'Items', 'Query 1', 'Query 2', 'Query 3']]

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

        self.cached_files = {"wav": {}, "npy": {}}

    def check_files(self):
        for _, pair in self.all_pairs.iterrows():
            ref_path = os.path.join(self.dataset_dir, 'Items', pair['Class'], pair['Items'])
            query_path = os.path.join(self.dataset_dir, 'Queries', pair['Class'], pair['Query'])
            if not os.path.exists(ref_path):
                print("Missing: ", ref_path)
            if not os.path.exists(query_path):
                print("Missing: ", query_path)

    def load_audio(self, path):
        if path not in self.cached_files["wav"]:
            audio, sr = librosa.load(path, sr=self.sample_rate, mono=True, duration=self.duration)
            self.cached_files["wav"][path] = audio
        return self.__pad_or_truncate__(self.cached_files["wav"][path], int(self.sample_rate * self.duration))

    def load_npy(self, path):
        if path not in self.cached_files["npy"]:
            arr = np.load(path,)
            self.cached_files["npy"][path] = arr
        return self.__pad_or_truncate__(self.cached_files["npy"][path], int(self.duration * EXTRA_FEAT_SR))

    def __pad_or_truncate__(self, arr: np.ndarray, fixed_length: int):
        """supports 1D and 2D arrays"""
        arr = np.asarray(arr)
        out = np.zeros((*arr.shape[:-1], fixed_length), dtype=arr.dtype)
        sl = slice(min(arr.shape[-1], fixed_length))
        out[..., :sl.stop] = arr[..., :sl.stop]
        return out

    def __getitem__(self, index):
        row = self.all_pairs.iloc[index]

        ref_path = os.path.join(self.dataset_dir, 'Items', row['Class'], row['Items'])
        query_path = os.path.join(self.dataset_dir, 'Queries', row['Class'], row['Query'])

        ref_audio = torch.tensor(self.load_audio(ref_path)).float()
        query_audio = torch.tensor(self.load_audio(query_path)).float()

        out = {
            'reference_filename': row['Items'],
            'imitation_filename': row['Query'],
            'reference': ref_audio,
            'imitation': query_audio,
            'reference_class': row['Class'],
            'imitation_class': row['Class']
        }

        # Add extra features or prompt if specified
        if self.add_extra_features:
            ref_npy_path = os.path.join(self.dataset_dir, 's2s/extra_feat', row['Items'].replace('.wav', '.npy'))
            imit_npy_path = os.path.join(self.dataset_dir, 's2s/extra_feat', row['Query'].replace('.wav', '.npy'))
            out['reference_extra_feat'] = self.load_npy(ref_npy_path)
            out['imitation_extra_feat'] = self.load_npy(imit_npy_path)
        return out

    def __len__(self):
        return len(self.all_pairs)


if __name__ == "__main__":
    ds_vims = VimSketchDataset(dataset_dir="data/Vim_Sketch_Dataset",
                               sample_rate=32000,
                               duration=10.0,
                               add_extra_features=True,
                               add_descriptions=True,
                               n_short_descriptions=10)
    ds_dev = AESAIMLA_DEV_S2S(dataset_dir="data/qvim-dev", sample_rate=32000, duration=10.0, add_extra_features=True)

    from torch.utils.data import DataLoader
    from tqdm import tqdm
    for ds in [ds_vims, ds_dev]:
        dl = DataLoader(ds, batch_size=10, shuffle=False, drop_last=True, num_workers=4)
        for i, batch in enumerate(tqdm(dl)):
            assert batch['reference'].shape == batch['imitation'].shape == (10, 320000)
            assert batch['reference_filename'] is not None and batch['imitation_filename'] is not None
            if i < len(dl) - 1:
                assert batch['reference_extra_feat'].shape == batch['imitation_extra_feat'].shape == (10, 16,
                                                                                                      10 * EXTRA_FEAT_SR)
            if "reference_description" in batch.keys():
                assert all(isinstance(b, str) and len(b) > 0 for b in batch['reference_description'])
                for i, text in enumerate(batch['reference_description']):
                    assert '.wav' not in text, f"Unexpected .wav in reference description: {text}"
                    assert len(text) > 0, f"Reference description is empty for{ batch['reference_filename'][i]}"
                for i, text in enumerate(batch['imitation_description']):
                    assert '.wav' not in text, f"Unexpected .wav in imitation description: {text}"
                    assert len(text) > 0, f"Imitation description is empty for {batch['imitation_filename'][i]}"
    print("All tests passed.")
