import torch
import torchcrepe
import librosa
import glob
import argparse
from tqdm import tqdm
import os

def extract_pitch_probs(audio_path, sample_rate=16000, hop_length=160):
    # Load and resample
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    audio = torch.tensor(audio).unsqueeze(0)  # shape: [1, T]

    # Run torchcrepe (batched frame prediction)
    with torch.no_grad():
        pitch, periodicity = torchcrepe.predict(
            audio,
            sample_rate=sample_rate,
            hop_length=hop_length,
            fmin=50.0,
            fmax=1100.0,
            model='tiny',
            batch_size=64,
            return_periodicity=True,
            return_harmonicity=False,
        )

    pitch_probs = periodicity.squeeze(0)  # shape: [T]

    return pitch_probs


def main():
    parser = argparse.ArgumentParser(description="Extract pitch probabilities from audio files using torchcrepe.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with input .wav files")
    parser.add_argument("--output_dir", type=str, default="pitch_probs", help="Directory to save pitch .pt files")
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--hop_length", type=int, default=320)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    audio_files = glob.glob(os.path.join(args.input_dir, "**/*.wav"), recursive=True)
    for audio_path in tqdm(audio_files):
        audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        pitch_curve = extract_pitch_probs(audio_path, sample_rate=args.sample_rate, hop_length=args.hop_length)
        torch.save(pitch_curve, os.path.join(args.output_dir, f"{audio_name}.pt"))

    print("Pitch probability curves extracted and saved.")


if __name__ == "__main__":
    main()
