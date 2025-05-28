"""Simplified evaluation as a function call."""

import json
import os
from glob import glob
from typing import Callable, Dict

import numpy as np
import pandas as pd


def evaluate_qvim_system(
    compute_similarities_fn: Callable[
        [Dict[str, str], Dict[str, str]], Dict[str, Dict[str, float]]
    ],
    data_path: str = "data/qvim-dev",
    output_path: str = None,
) -> Dict[str, float]:
    """
    Evaluate a Query by Vocal Imitation system.

    Args:
        compute_similarities_fn: A function that computes similarity scores between items and queries
        data_path: Path to the dataset containing Items/ and Queries/ directories
        output_path: Path to save similarities.json (if None, not saved)

    Returns:
        Dictionary containing evaluation metrics (MRR and NDCG)
    """
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)

    # Load item and query files
    items_path = os.path.join(data_path, "Items")
    item_files = pd.DataFrame(
        {"path": list(glob(os.path.join(items_path, "**", "*.wav"), recursive=True))}
    )
    item_files["Class"] = item_files["path"].transform(
        lambda x: x.split(os.path.sep)[-2]
    )
    item_files["Items"] = item_files["path"].transform(
        lambda x: x.split(os.path.sep)[-1]
    )

    queries_path = os.path.join(data_path, "Queries")
    query_files = pd.DataFrame(
        {"path": list(glob(os.path.join(queries_path, "**", "*.wav"), recursive=True))}
    )
    query_files["Class"] = query_files["path"].transform(
        lambda x: x.split(os.path.sep)[-2]
    )
    query_files["Query"] = query_files["path"].transform(
        lambda x: x.split(os.path.sep)[-1]
    )

    print("Total item files:", len(item_files))
    print("Total query files:", len(query_files))

    if len(query_files) == 0 or len(item_files) == 0:
        raise ValueError("No query files found! Check your dataset path.")

    # Compute similarities
    scores = compute_similarities_fn(
        items={row["Items"]: row["path"] for i, row in item_files.iterrows()},
        queries={row["Query"]: row["path"] for i, row in query_files.iterrows()},
    )

    # Save similarities
    if output_path is not None:
        with open(os.path.join(output_path, "similarities.json"), "w") as f:
            json.dump(scores, f)

    # Load ground truth and evaluate
    rankings = pd.DataFrame(
        dict(
            **{"id": [i for i in list(scores.keys())]},
            **{
                k: [v[k] for v in scores.values()]
                for k in scores[list(scores.keys())[0]].keys()
            },
        )
    ).set_index("id")

    df = pd.read_csv(os.path.join(data_path, "DEV Dataset.csv"), skiprows=1)[
        ["Label", "Class", "Items", "Query 1", "Query 2", "Query 3"]
    ]

    df = df.melt(
        id_vars=[col for col in df.columns if "Query" not in col],
        value_vars=["Query 1", "Query 2", "Query 3"],
        var_name="Query Type",
        value_name="Query",
    ).dropna()

    # Remove missing files
    rankings = rankings.loc[df["Query"].unique(), df["Items"].unique()]

    # Individual item evaluation
    ground_truth = {row["Query"]: [row["Items"]] for i, row in df.iterrows()}

    position_of_correct = {}
    missing_query_files = []
    for query, correct_item_list in ground_truth.items():
        # Skip if query is not in the DataFrame
        if query not in rankings.index:
            missing_query_files.append(query)
            continue
        # Get row and sort items by similarity in descending order
        sorted_items = rankings.loc[query].sort_values(ascending=False)
        # Find rank of correct items
        position_of_correct[query] = {
            item: sorted_items.index.get_loc(item)
            for item in correct_item_list
            if item in sorted_items.index
        }
        assert len(position_of_correct[query]) == len(correct_item_list), (
            f"Missing item! Got: {list(position_of_correct[query].keys())}. Expected: {correct_item_list}"
        )

    # Compute MRR
    normalized_rrs = []
    for query, items_ranks in position_of_correct.items():
        rr, irr = [], []  # summed RR and ideal RR
        for i, (item, rank) in enumerate(items_ranks.items()):
            rr.append(1 / (rank + 1))
            irr.append(1 / (i + 1))
        normalized_rrs.append(sum(rr) / sum(irr))  # normalize MRR with ideal one
    mrr = np.mean(normalized_rrs)

    print("MRR random:", round((1 / np.arange(1, len(df["Items"].unique()))).mean(), 4))
    print("MRR       :", round(mrr, 4))

    # Class-wise evaluation
    ground_truth = {
        row["Query"]: [
            row_["Items"]
            for j, row_ in df.drop_duplicates("Items").iterrows()
            if row_["Class"] == row["Class"]
        ]
        for i, row in df.drop_duplicates("Query").iterrows()
    }

    position_of_correct = {}
    missing_query_files = []
    for query, correct_item_list in ground_truth.items():
        # Skip if query is not in the DataFrame
        if query not in rankings.index:
            missing_query_files.append(query)
            continue
        # Get row and sort items by similarity in descending order
        sorted_items = rankings.loc[query].sort_values(ascending=False)
        # Find rank of correct items
        position_of_correct[query] = {
            item: sorted_items.index.get_loc(item)
            for item in correct_item_list
            if item in sorted_items.index
        }
        assert len(position_of_correct[query]) == len(correct_item_list), (
            f"Missing item!"
        )

    # Compute class-wise MRR
    normalized_rrs = []
    for query, items_ranks in position_of_correct.items():
        rr, irr = [], []  # summed RR and ideal RR
        for i, (item, rank) in enumerate(items_ranks.items()):
            rr.append(1 / (rank + 1))
            irr.append(1 / (i + 1))
        normalized_rrs.append(sum(rr) / sum(irr))  # normalize MRR with ideal one
    class_wise_mrr = np.mean(normalized_rrs)

    # Compute NDCG
    normalized_dcg = []
    ndcgs = {}
    for query, items_ranks in position_of_correct.items():
        dcg, idcg = [], []  # summed RR and ideal RR
        for i, (item, rank) in enumerate(items_ranks.items()):
            dcg.append(1 / np.log2(rank + 2))
            idcg.append(1 / np.log2(i + 2))
        normalized_dcg.append(sum(dcg) / sum(idcg))  # normalize MRR with ideal one
        ndcgs[query] = sum(dcg) / sum(idcg)
    ndcg = np.mean(normalized_dcg)

    print("Class-wise MRR :", round(class_wise_mrr, 4))
    print("Class-wise NDCG:", round(ndcg, 4))

    # Return metrics
    return {
        "mrr": float(mrr),
        "class_wise_mrr": float(class_wise_mrr),
        "ndcg": float(ndcg),
    }


if __name__ == "__main__":
    # Example usage with a dummy model
    def dummy_compute_similarities(items, queries):
        """Dummy similarity computation that returns random scores"""
        import random

        return {q_id: {i_id: random.random() for i_id in items} for q_id in queries}

    # Evaluate
    metrics = evaluate_qvim_system(dummy_compute_similarities)
    print("Evaluation complete. Metrics:", metrics)
