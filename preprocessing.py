from laion_clap import CLAP_Module
import glob
import torch
import librosa
import argparse
from tqdm import tqdm

argparse = argparse.ArgumentParser(description="Extract CLAP embeddings from audio files.")
argparse.add_argument(
    "--input_dir",
    type=str,
    help="Path to the directory containing audio files.",
)

def extract_clap_embedding(audio_path, clap_model):

    audio_data, sr = librosa.load(audio_path, sr=48000)
    audio_data = audio_data.reshape(1, -1)
    # audio_data = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float()
    emb = clap_model.get_audio_embedding_from_data(x=audio_data, use_tensor=True)
    return emb.detach().cpu()

def main():
    args = argparse.parse_args()
    clap_model = CLAP_Module(enable_fusion=False)
    clap_model.load_ckpt()

    for audio_path in tqdm(glob.glob(args.input_dir + "/**/*.wav", recursive=True)):
        audio_name = audio_path.split("/")[-1].split(".")[0]
        embedding = extract_clap_embedding(audio_path, clap_model)
        torch.save(embedding, f"clap_embeddings/{audio_name}.pt")
    print("CLAP embeddings extracted and saved successfully.")

if __name__ == "__main__":
    main()