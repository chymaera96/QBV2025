import os
import pandas as pd
import numpy as np
import torch
import librosa
import scipy.signal
import torchaudio
import torchcrepe
import torchaudio.transforms as T


def extract_control_features(waveform, sample_rate, frame_size, hop_length, feature_len):
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    mel_spec = T.MelSpectrogram(sample_rate=sample_rate, n_fft=frame_size,
                                hop_length=hop_length, n_mels=64)(waveform)
    loudness = torch.log1p(mel_spec.mean(dim=1)).squeeze(0)

    centroid = T.SpectralCentroid(sample_rate=sample_rate, n_fft=frame_size,
                                  hop_length=hop_length)(waveform).squeeze(0)

    with torch.no_grad():
        pitch, pd = torchcrepe.predict(
            waveform,
            sample_rate,
            hop_length,
            fmin=50.0,
            fmax=1100.0,
            model="tiny",
            batch_size=64,
            return_periodicity=True,
            return_harmonicity=False,
        )
    pitch_probs = pd.squeeze(0)
    pitch_probs[pitch_probs < 0.1] = 0.0

    # Median filtering
    loudness = torch.tensor(scipy.signal.medfilt(loudness.numpy(), kernel_size=9))
    centroid = torch.tensor(scipy.signal.medfilt(centroid.numpy(), kernel_size=9))
    pitch_probs = torch.tensor(scipy.signal.medfilt(pitch_probs.numpy(), kernel_size=9))

    # Stack and interpolate
    features = torch.stack([loudness, centroid, pitch_probs])  # [3, T]
    features = torch.nn.functional.interpolate(features.unsqueeze(0), size=feature_len, mode="linear", align_corners=True)
    return features.squeeze(0).float()  # [3, feature_len]


class VimSketchDataset_S2S(torch.utils.data.Dataset):

    def __init__(
            self,
            dataset_dir,
            sample_rate=32000,
            duration=10.0,
            feature_len=256              # <-- added
    ):
        self.dataset_dir = dataset_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.feature_len = feature_len

        self.frame_size = 1024
        self.hop_length = int(sample_rate * duration / feature_len)

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

    def load_audio(self, path):
        if path not in self.cached_files:
            audio, sr = librosa.load(
                path,
                sr=self.sample_rate,
                mono=True,
                duration=self.duration
            )
            self.cached_files[path] = audio
        return self.__pad_or_truncate__(self.cached_files[path])

    def __pad_or_truncate__(self, audio):
        fixed_length = int(self.sample_rate * self.duration)
        if len(audio) < fixed_length:
            array = np.zeros(fixed_length, dtype="float32")
            array[:len(audio)] = audio
        if len(audio) >= fixed_length:
            array = audio[:fixed_length]
        return array

    def __getitem__(self, index):

        row = self.all_pairs.iloc[index]

        reference_audio = torch.tensor(
            self.load_audio(os.path.join(self.dataset_dir, 'references', row['filename_reference']))
        ).float()

        imitation_audio = torch.tensor(
            self.load_audio(os.path.join(self.dataset_dir, 'vocal_imitations', row['filename_imitation']))
        ).float()

        reference_features = extract_control_features(
            reference_audio, self.sample_rate, self.frame_size, self.hop_length, self.feature_len
        )

        imitation_features = extract_control_features(
            imitation_audio, self.sample_rate, self.frame_size, self.hop_length, self.feature_len
        )

        return {
            'reference_filename': row['filename_reference'],
            'imitation_filename': row['filename_imitation'],
            'reference': reference_features,
            'imitation': imitation_features,
        }

    def __len__(self):
        return len(self.all_pairs)


class AESAIMLA_DEV_S2S(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_dir,
            sample_rate=32000,
            duration=10.0,
            feature_len=256
    ):
        self.dataset_dir = dataset_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.feature_len = feature_len

        self.frame_size = 1024
        self.hop_length = int(sample_rate * duration / feature_len)

        pairs = pd.read_csv(
            os.path.join(dataset_dir, 'DEV Dataset.csv'),
            skiprows=1
        )[['Label', 'Class', 'Items', 'Query 1', 'Query 2', 'Query 3']]

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
        for _, pair in self.all_pairs.iterrows():
            ref_path = os.path.join(self.dataset_dir, 'Items', pair['Class'], pair['Items'])
            query_path = os.path.join(self.dataset_dir, 'Queries', pair['Class'], pair['Query'])
            if not os.path.exists(ref_path):
                print("Missing: ", ref_path)
            if not os.path.exists(query_path):
                print("Missing: ", query_path)

    def load_audio(self, path):
        if path not in self.cached_files:
            audio, _ = librosa.load(path, sr=self.sample_rate, mono=True, duration=self.duration)
            self.cached_files[path] = self.__pad_or_truncate__(audio)
        return self.cached_files[path]

    def __pad_or_truncate__(self, audio):
        fixed_length = int(self.sample_rate * self.duration)
        array = np.zeros(fixed_length, dtype="float32")
        array[:min(len(audio), fixed_length)] = audio[:min(len(audio), fixed_length)]
        return array

    def __getitem__(self, index):
        row = self.all_pairs.iloc[index]

        ref_path = os.path.join(self.dataset_dir, 'Items', row['Class'], row['Items'])
        query_path = os.path.join(self.dataset_dir, 'Queries', row['Class'], row['Query'])

        ref_audio = torch.tensor(self.load_audio(ref_path)).float()
        query_audio = torch.tensor(self.load_audio(query_path)).float()

        ref_feat = extract_control_features(ref_audio, self.sample_rate, self.frame_size, self.hop_length, self.feature_len)
        query_feat = extract_control_features(query_audio, self.sample_rate, self.frame_size, self.hop_length, self.feature_len)

        return {
            'reference_filename': row['Items'],
            'imitation_filename': row['Query'],
            'reference': ref_feat,             # [3, T]
            'imitation': query_feat,           # [3, T]
            'reference_class': row['Class'],
            'imitation_class': row['Class']
        }

    def __len__(self):
        return len(self.all_pairs)
