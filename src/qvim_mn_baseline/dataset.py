import glob
import os
import librosa
import numpy as np
import pandas as pd
import torch

from hc_baseline.modules.augmentations import Augment

class VimSketchDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            dataset_dir,
            sample_rate=32000,
            duration=10.0
    ):
        self.dataset_dir = dataset_dir
        self.sample_rate = sample_rate
        self.duration = duration

        self.augment_ref = Augment(sample_rate=sample_rate, max_transforms=1)
        self.augment_imit = Augment(sample_rate=sample_rate, max_transforms=1)

        reference_filenames = pd.read_csv(
            os.path.join(dataset_dir, 'reference_file_names.csv'),
            sep='\t',
            header=None,
            names=['filename']
        )
        reference_filenames['reference_id'] = reference_filenames['filename'].transform(
            lambda x: "_".join(x.split('_')[1:])
        )

        imitation_file_names = pd.read_csv(
            os.path.join(dataset_dir, 'vocal_imitation_file_names.csv'),
            sep='\t',
            header=None,
            names=['filename']
        )
        imitation_file_names['reference_id'] = imitation_file_names['filename'].transform(
            lambda x: "_".join(x.split('_')[1:])
        )

        self.all_pairs = imitation_file_names.merge(
            reference_filenames,
            left_on="reference_id",
            right_on="reference_id", how="left",
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
        reference = self.augment_ref(self.load_audio(reference_path))
        # reference = self.load_audio(reference_path, sr=48000)
        reference = self.__pad_or_truncate__(reference)

        imitation_path = os.path.join(self.dataset_dir, 'vocal_imitations', row['filename_imitation'])
        imitation = self.augment_imit(self.load_audio(imitation_path))
        # imitation = self.load_audio(imitation_path)
        imitation = self.__pad_or_truncate__(imitation)

        assert reference.shape[-1] == 320000, f"Reference shape mismatch: {reference.shape}"
        assert imitation.shape[-1] == 320000, f"Imitation shape mismatch: {imitation.shape}"

        return {
            # 'reference_path': os.path.join(self.dataset_dir, 'references', row['filename_reference']),
            'reference_filename': row['filename_reference'],
            'imitation_filename': row['filename_imitation'],
            'reference': reference,
            'imitation': imitation,

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

        reference = self.load_audio(reference_name, sr=48000)
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


# if __name__ == "__main__":
#     dataset = VimSketchDataset(dataset_dir='data/Vim_Sketch_Dataset', sample_rate=32000, duration=10.0)
#     for idx in range(len(dataset)):
#         sample = dataset[idx]
#         # print(f"Sample {idx}:")
#         if sample['reference'].shape != torch.Size([480000]):
#             print(f"Sample reference {idx}: {sample['reference_filename']}")
#             print(f"  reference shape: {sample['reference'].shape} | type: {type(sample['reference'])}")
#         if sample['imitation'].shape != torch.Size([320000]):
#             print(f"Sample imitation {idx}: {sample['reference_filename']}")
#             print(f"  imitation shape: {sample['imitation'].shape} | type: {type(sample['imitation'])}")
