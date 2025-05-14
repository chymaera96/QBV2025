import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import AFCLAP.my_laion_clap.CLAP.src.laion_clap as af_laion_clap
import laion_clap
import torch
from evaluate import evaluate_qvim_system
from torch import nn
from tqdm import tqdm


class CLAP(nn.Module):
    def __init__(self, model_id=1):
        super(CLAP, self).__init__()

        self.is_afclap = False

        if model_id == "AF":
            self.is_afclap = True
            afclap_ckpt_path = "./ckpt/afclap.pt"

            # Initialize AFCLAP model
            self.model = af_laion_clap.CLAP_Module(
                enable_fusion=True, amodel="HTSAT-afclap", tmodel="t5"
            ).cuda()

            # Load AFCLAP checkpoint
            self.model.load_afclap_ckpt(ckpt=afclap_ckpt_path, verbose=False)
        else:
            # Original CLAP initialization
            enable_fusion = False
            if model_id == 2 or model_id == 3:
                enable_fusion = True
            self.model = laion_clap.CLAP_Module(enable_fusion=enable_fusion).to("cuda")
            self.model.load_ckpt(model_id=model_id)

        self.model.eval()

    def forward(self, x):
        if self.is_afclap:
            # AFCLAP uses different parameters for embedding extraction
            features = self.model.get_audio_embedding_from_filelist(
                x=x, sr=16000, use_tensor=True
            )
        else:
            # Original CLAP embedding extraction
            features = self.model.get_audio_embedding_from_filelist(
                x=x, use_tensor=True
            )
        return features

    def extract_features(
        self,
        dataset_dir,
        output_dir,
        batch_size=4,
    ):
        # get all .wav files in the dataset_dir and subdirs
        files = glob.glob(f"{dataset_dir}/**/*.wav", recursive=True)

        # Process files in batches to avoid memory issues
        for i in range(0, len(files), batch_size):
            batch_files = files[i : i + batch_size]

            # Get output file paths
            output_files = [
                f"{output_dir}/{file.replace(dataset_dir, '').replace('.wav', '.pt')}"
                for file in batch_files
            ]

            # Filter out files that already have embeddings
            to_process = []
            final_output_files = []
            for file, output_file in zip(batch_files, output_files):
                if os.path.exists(output_file):
                    print(f"Embedding for {file} already exists, skipping...")
                    continue
                to_process.append(file)
                final_output_files.append(output_file)

            if not to_process:
                continue

            # Create directories for output files
            for output_file in final_output_files:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Compute embeddings (from filepaths)
            embeddings = self.forward(to_process)

            # Save embeddings
            for embedding, output_file in zip(embeddings, final_output_files):
                torch.save(embedding, output_file)
                print(f"Saved embedding to {output_file}")

    def compute_similarities(self, items, queries):
        """
        Compute similarities between queries and items.
        Extract features on GPU, but store and compute similarities on CPU.

        Args:
            items (dict): Dictionary of item file paths.
            queries (dict): Dictionary of query file paths.

        Returns:
            dict: Dictionary containing similarities for each query.
        """

        # Infer embedding paths based on common directory structure
        def get_embedding_path(file_path):
            # Check if it's already an embedding file
            if file_path.endswith(".pt"):
                return file_path

            # Try to find corresponding embedding
            base_dir = os.path.dirname(os.path.dirname(file_path))
            emb_dir = base_dir.replace("/Items", "/embeddings/CLAP/Items").replace(
                "/Queries", "/embeddings/CLAP/Queries"
            )

            relative_path = os.path.relpath(file_path, base_dir)
            embedding_path = os.path.join(
                emb_dir, os.path.splitext(relative_path)[0] + ".pt"
            )
            if os.path.exists(embedding_path):
                return embedding_path

            # If no embedding found, return None to indicate need for extraction
            return None

        # Extract or load features for items
        item_features = {}
        for item_id, file_path in tqdm(
            items.items(), total=len(items), desc="Processing items"
        ):
            embedding_path = get_embedding_path(file_path)

            if embedding_path and os.path.exists(embedding_path):
                item_features[item_id] = torch.load(embedding_path, map_location="cpu")
            else:
                feature = self.forward([file_path])
                item_features[item_id] = feature.mean(dim=0).detach().cpu()

            torch.cuda.empty_cache()

        # Extract or load features for queries
        query_features = {}
        for query_id, file_path in tqdm(
            queries.items(), total=len(queries), desc="Processing queries"
        ):
            embedding_path = get_embedding_path(file_path)

            if embedding_path and os.path.exists(embedding_path):
                # Load pre-computed embedding to CPU
                query_features[query_id] = torch.load(
                    embedding_path, map_location="cpu"
                )
            else:
                feature = self.forward([file_path])
                query_features[query_id] = feature.mean(dim=0).detach().cpu()

            torch.cuda.empty_cache()

        results = {}
        for query_id, query_feature in tqdm(
            query_features.items(), desc="Computing similarities"
        ):
            results[query_id] = {}
            # Normalize query feature for cosine similarity
            query_feature_normalized = query_feature / query_feature.norm(
                dim=-1, keepdim=True
            )

            for item_id, item_feature in item_features.items():
                # Normalize item feature for cosine similarity
                item_feature_normalized = item_feature / item_feature.norm(
                    dim=-1, keepdim=True
                )

                similarity = torch.matmul(
                    query_feature_normalized, item_feature_normalized.T
                ).item()
                results[query_id][item_id] = similarity

        return results


if __name__ == "__main__":
    clap = CLAP("AF")
    evaluate_qvim_system(clap.compute_similarities, data_path="data/DEV/")
