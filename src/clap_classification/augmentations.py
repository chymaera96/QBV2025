import random

import numpy as np
import torch
import torch.nn as nn
from audiomentations import Gain, PitchShift, Shift, TimeStretch
from audiomentations.core.transforms_interface import BaseWaveformTransform


class Augment(nn.Module):
    def __init__(self, sample_rate, max_transforms=1):
        super(Augment, self).__init__()

        self.max_transforms = max_transforms
        self.sample_rate = sample_rate
        self.train_transform_options = [
            Shift(min_shift=-0.2, max_shift=0.2, p=1.0),
            Gain(min_gain_db=-10, max_gain_db=10, p=1.0),
            PitchShift(min_semitones=-3, max_semitones=3, p=1.0),
            TimeStretch(min_rate=0.67, max_rate=1.33, p=1.0),
            FrameLevelCorruption(remove_prob=0.0, silence_prob=0.0),
            FrameLevelCorruption(duplicate_prob=0.0, silence_prob=0.0),
            FrameLevelCorruption(duplicate_prob=0.0, remove_prob=0.0),
        ]

    def apply_random_transforms(self, audio, transform_options, max_transforms):
        if len(transform_options) == 0 or max_transforms <= 0:
            return audio

        selected_transforms = np.random.choice(
            transform_options,
            size=min(max_transforms, len(transform_options)),
            replace=False,
        )
        for transform in selected_transforms:
            audio = transform(audio, sample_rate=self.sample_rate)
        return audio

    def train_transform(self, audio):
        return self.apply_random_transforms(
            audio, self.train_transform_options, self.max_transforms
        )

    def forward(self, x):
        x = self.train_transform(x)

        return torch.from_numpy(x).float()


class FrameLevelCorruption(BaseWaveformTransform):
    """
    Applies frame-level corruption by dividing the audio into
    contiguous segments based on a randomly chosen FPS and then
    applying duplication, removal, and silence insertion
    with their corresponding probabilities.
    """

    def __init__(
        self,
        min_fps=0.5,
        max_fps=5.0,
        duplicate_prob=0.1,
        remove_prob=0.1,
        silence_prob=0.1,
        p=1.0,
    ):
        super().__init__(p)
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.duplicate_prob = duplicate_prob
        self.remove_prob = remove_prob
        self.silence_prob = silence_prob

    def apply(self, samples, sample_rate):
        num_samples = len(samples)
        fps = random.uniform(self.min_fps, self.max_fps)
        frame_size = int(sample_rate / fps)

        corrupted_samples = []
        i = 0

        while i < num_samples:
            frame = samples[i : i + frame_size]
            if len(frame) == 0:
                break

            # Apply duplication
            if random.random() < self.duplicate_prob:
                frame = np.concatenate((frame, frame))

            # Apply removal
            if random.random() < self.remove_prob:
                frame = np.array([])  # Remove this frame

            # Apply silence
            if random.random() < self.silence_prob:
                frame = np.zeros_like(frame)

            corrupted_samples.append(frame)
            i += frame_size

        return (
            np.concatenate(corrupted_samples).astype(np.float32)
            if corrupted_samples
            else samples.astype(np.float32)
        )
