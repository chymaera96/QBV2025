import random
import numpy as np
from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.transforms_interface import BaseTransform


class FrameLevelCorruption(BaseWaveformTransform):
    """
    Applies frame-level corruption by dividing the audio into contiguous segments
    based on a randomly chosen FPS and then applying duplication, removal, and silence insertion
    with their corresponding probabilities.
    """
    def __init__(self, 
                 min_fps=0.5, max_fps=5.0,
                 duplicate_prob=0.1, 
                 remove_prob=0.1, 
                 silence_prob=0.1,
                 p=1.0 
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
            frame = samples[i:i+frame_size]
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

        return np.concatenate(corrupted_samples).astype(np.float32) if corrupted_samples else samples.astype(np.float32)
    


class Identity(BaseTransform):
    """
    A no-op augmentation that simply returns the input audio unchanged.
    """
    def __init__(self, p=1.0):
        super().__init__(p)

    def apply(self, samples, sample_rate):
        return samples

    def __call__(self, samples, sample_rate):
        """
        Makes the Identity class callable and compatible with apply_random_transforms.
        """
        return self.apply(samples, sample_rate)