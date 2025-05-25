import argparse
import os
import glob

import torch
import numpy as np
from tqdm import tqdm
import torchaudio
import torchcrepe
import torchaudio.transforms as T
from librosa import yin
from librosa.feature import chroma_cqt
from scipy.signal import medfilt

from multiprocessing import Pool

from online_normalization import OnlineStats
# debug
# from types import SimpleNamespace
# config = SimpleNamespace(dataset_path="../../data", num_workers=2, nfft=1024, hop_ms=10)

# mu: [2.366447687149048, 1688.7119140625, 0.3718608617782593]
# std: [0.046666089445352554, 314.2520751953125, 0.006295355036854744]


def extract_s2s_features(waveform, sample_rate=32000, nfft=1024, hop_ms=10):
    """
    Extracts features for S2S training. By default, 100 feature frames per second.

    NOTE: 
    - Due to the torchcrepe library, input audio with a batch dimension is not supported. 
    - centroid valuess are set to 0 for sielnce or NaN values.
    
    Args:
        waveform (Tensor): The input audio (mono) waveform. Shape: [1, T] or [T] with total number of samples T.
        sample_rate (int): The sample rate of the audio.
        nfft (int): The size of the FFT window.
        hop_ms (int): The number of samples between frames.
    
    Returns:
        Tensor (16, feature_len): A tensor containing the extracted features 
                                  where feature_len = T // hop_length.
                                  loudness (d=0),
                                  centroid (d=1), 
                                  pitch_crepe (d=2), 
                                  pitch_yin (d=3),
                                  chroma (d=4:)
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    hop_length = int(sample_rate * hop_ms / 1000)
    feature_len = waveform.shape[-1] // hop_length

    mel_spec = T.MelSpectrogram(sample_rate=sample_rate, n_fft=nfft, hop_length=hop_length, n_mels=64)(waveform)
    loudness = torch.log1p(mel_spec.mean(dim=1)).squeeze(0)
    loudness[torch.isnan(loudness)] = 0.  # avoid NaN values

    centroid = T.SpectralCentroid(sample_rate=sample_rate, n_fft=nfft, hop_length=hop_length)(waveform).squeeze(0)
    centroid = centroid * (loudness > 0.05)  # set 0 to almost silent frames
    centroid[torch.isnan(centroid)] = 0.  # avoid NaN values

    waveform = waveform.to(device="cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        pitch_crepe, periodicity = torchcrepe.predict(waveform,
                                                      sample_rate,
                                                      hop_length,
                                                      fmin=50.0,
                                                      fmax=1100.0,
                                                      model="tiny",
                                                      batch_size=64,
                                                      return_periodicity=True,
                                                      return_harmonicity=False,
                                                      device="cuda" if torch.cuda.is_available() else "cpu")
    periodicity = periodicity.squeeze(0)[:feature_len].detach().cpu()
    periodicity = torch.nan_to_num(periodicity, nan=0.0)  # avoid NaN values
    pitch_crepe = pitch_crepe.squeeze(0)[:feature_len].detach().cpu()
    pitch_crepe = torch.nan_to_num(pitch_crepe, nan=0.0)  # avoid NaN values
    pitch_crepe[periodicity < 0.1] = 0.0

    # YIN pitch
    waveform_np = waveform.detach().cpu().numpy()
    pitch_yin = yin(waveform_np, fmin=50.0, fmax=1100.0, sr=sample_rate, win_length=nfft, hop_length=hop_length)
    pitch_yin = torch.tensor(medfilt(pitch_yin.squeeze(0), kernel_size=5)).cpu()
    pitch_yin = torch.nan_to_num(pitch_yin, nan=0.0)

    # Chroma CQT
    target_len = sample_rate * 2
    pad_width = max(0, target_len - waveform_np.shape[1])
    waveform_np = np.pad(waveform_np, ((0, 0), (0, pad_width)))  # at least 2 seconds for CQT
    chroma = chroma_cqt(y=waveform_np, fmin=60.0, sr=sample_rate, hop_length=hop_length, n_chroma=12)
    chroma = medfilt(chroma.squeeze(0), kernel_size=(1, 5))  # [12, t], smooth over time
    chroma = torch.tensor(chroma).cpu()
    chroma = torch.nan_to_num(chroma, nan=0.0)  # avoid NaN values

    # Median filtering is omitted. We can apply it online.
    min_len = min(len(loudness), len(centroid), len(pitch_crepe), len(pitch_yin), chroma.shape[1])
    assert min_len > 0, "Feature length is zero. Check the input audio."

    # Chroma * loudness?
    chroma = chroma[:, :min_len] * loudness[:min_len]
    ret = torch.stack([loudness[:min_len], centroid[:min_len], pitch_crepe[:min_len], pitch_yin[:min_len]],
                      dim=0)  # [4, feature_len]
    return torch.cat([ret, chroma[:, :min_len]], dim=0)  # [16, feature_len]


def process_split(config, audio_files):
    """ Single worker function to extract features from audio files. """
    stats = OnlineStats(num_features=16)
    for i, audio_file in enumerate(tqdm(audio_files)):
        x, sample_rate = torchaudio.load(audio_file)
        if x.shape[0] > 1:
            x = x.mean(dim=0, keepdim=True)
        feat = extract_s2s_features(x, sample_rate=sample_rate, nfft=config.nfft, hop_ms=config.hop_ms)
        stats.update(feat)  # update online stats

        # Save features as numpy files
        _dir, _fname = os.path.split(audio_file)
        ds_main = os.path.relpath(_dir, config.dataset_path)  # e.g. Vim_Sketch_Dataset/.../
        npy_file = os.path.join(config.dataset_path, ds_main.split('/')[0], 's2s/extra_feat', _fname.replace('.wav', '.npy'))
        os.makedirs(os.path.dirname(npy_file), exist_ok=True)
        np.save(npy_file, feat.numpy(), allow_pickle=True)

    return {'mean': stats.get_mean(), 'std': stats.get_std(), 'count': i + 1}


def main(config):
    # Collect all file paths: VimSketch and qvim-dev
    root_dirs = [os.path.join(config.dataset_path, 'Vim_Sketch_Dataset'), os.path.join(config.dataset_path, 'qvim-dev')]
    audio_files = [
        os.path.abspath(f)
        for d in root_dirs
        for f in glob.glob(os.path.join(d, '**', '*'), recursive=True)
        if os.path.splitext(f)[1].lower() in {'.wav', '.mp3', '.flac'}
    ]
    print(f"Found {len(audio_files)} audio files in {root_dirs}.")

    # Multi-process feature extraction
    print(f"Extracting features with {config.num_workers} workers...")
    chunks = np.array_split(audio_files, config.num_workers)
    mp_args = [(config, chunk.tolist()) for chunk in chunks]
    with Pool(config.num_workers) as pool:
        results = pool.starmap(process_split, mp_args)
    print("Feature extraction completed.")

    # Combine stats
    n = sum(r['count'] for r in results)
    mu = sum(r['mean'] * r['count'] for r in results) / n
    var = sum((r['std'] + (r['std'] - mu)**2) * r['std'] for r in results) / n
    std = np.sqrt(var)
    mu = torch.Tensor(mu).numpy().tolist()
    std = torch.Tensor(std).numpy().tolist()

    stat_file = os.path.join(config.dataset_path, 'extra_feat_stats.txt')
    with open(stat_file, 'w') as f:
        f.write(f'mu: {mu}\n')
        f.write(f'std: {std} \n')
    print(f"Saved the stats (Mean={mu}, Std={std}) to {stat_file}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Argument parser for S2S feature extraction on the VimSketch and QVIM devset.")

    # General
    parser.add_argument('--num_workers', type=int, default=8, help="Number of data loader workers.")
    parser.add_argument('--num_gpus', type=int, default=1, help="Number of GPUs to use for training.")
    parser.add_argument('--dataset_path', type=str, default='data', help="Path to the data sets.")
    parser.add_argument('--nfft', type=int, default=1024, help="FFT size for STFT.")
    parser.add_argument('--hop_ms', type=int, default=10, help="Hop size in milliseconds for STFT.")
    args = parser.parse_args()

    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main(args)
