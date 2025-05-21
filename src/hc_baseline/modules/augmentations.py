import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from audiomentations import PitchShift, Gain, TimeStretch, Shift

from hc_baseline.modules.fx_util import FrameLevelCorruption

class Augment(nn.Module):
    def __init__(self, sample_rate, max_transforms=1):
        super(Augment, self).__init__()

        self.max_transforms = max_transforms
        self.sample_rate = sample_rate
        self.train_transform_options = [
            Shift(min_shift=-0.2, max_shift=0.2, p=1.0),
            Gain(min_gain_db=-10, max_gain_db=10, p=1.0),
            PitchShift(min_semitones=-3, max_semitones=3, p=1.0),
            TimeStretch(min_rate=0.8, max_rate=1.2, p=1.0),
            FrameLevelCorruption(remove_prob=0.0, silence_prob=0.0),
            FrameLevelCorruption(duplicate_prob=0.0, silence_prob=0.0),
            FrameLevelCorruption(duplicate_prob=0.0, remove_prob=0.0),
        ]


    def apply_random_transforms(self, audio, transform_options, max_transforms):
        if len(transform_options) == 0 or max_transforms <= 0:
            return audio

        selected_transforms = np.random.choice(transform_options, size=min(max_transforms, len(transform_options)), replace=False)
        for transform in selected_transforms:
            audio = transform(audio, sample_rate=self.sample_rate)
        return audio

    def train_transform(self, audio):
        return self.apply_random_transforms(audio, self.train_transform_options, self.max_transforms)


    def forward(self, x):
        x = self.train_transform(x)

        return torch.from_numpy(x).float()

