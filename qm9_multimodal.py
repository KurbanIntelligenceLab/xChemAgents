# %%
import os
import ast
import torch
import pandas as pd
from torch_geometric.datasets import QM9
from torch_geometric.data import Dataset

# %%
def attach_clip_embeddings(dataset_path: str, csv_path: str, output_path: str):
    """
    Attaches CLIP embeddings to QM9 molecules and saves the updated dataset.

    Steps:
    1) Load the existing PyG QM9 dataset from `dataset_path`.
    2) Load a CSV file containing ['index', 'clip_embedding'].
    3) Match each molecule in the dataset with its corresponding embedding using the index parsed from `data.name`.
    4) Attach the CLIP embedding as a tensor to each matching molecule.
    5) Save the updated dataset to `output_path`.

    Args:
        dataset_path (str): Path to the original PyG dataset (unused in this function, kept for compatibility).
        csv_path (str): Path to the CSV file containing clip embeddings.
        output_path (str): Path to save the updated dataset with embeddings.
    """
    # 1. Load the original QM9 dataset
    dataset = QM9(root='data/QM9')

    # 2. Load the CLIP embeddings CSV
    emb_df = pd.read_csv(csv_path)
    print(f"Loaded embeddings CSV with {len(emb_df)} rows from '{csv_path}'")

    # Ensure 'index' is integer for matching
    if emb_df['index'].dtype != int:
        emb_df['index'] = emb_df['index'].astype(int)

    # 3. Build a lookup table: index -> clip_embedding tensor
    id_to_embedding = {}
    for _, row in emb_df.iterrows():
        idx = row['index']
        clip_str = row['clip_embedding']
        if isinstance(clip_str, str) and clip_str.strip():
            try:
                clip_list = ast.literal_eval(clip_str)
                id_to_embedding[idx] = torch.tensor(clip_list, dtype=torch.float)
            except (SyntaxError, ValueError):
                # Skip malformed embedding strings
                pass

    # 4. Attach embeddings to the matching molecules
    updated_list = []
    count_with_embed = 0

    for data in dataset:
        # Parse index from data.name (e.g., "gdb_000123" -> 123)
        try:
            id_num = int(data.name.split('_')[-1])
        except Exception:
            id_num = None

        clip_emb = id_to_embedding.get(id_num, None)
        if clip_emb is not None:
            data.clip_embedding = clip_emb
            count_with_embed += 1
            updated_list.append(data)

    print(f"Attached embeddings for {count_with_embed} / {len(updated_list)} molecules.")

    # 5. Save the updated dataset with embeddings
    torch.save(updated_list, output_path)
    print(f"Saved updated dataset (with clip_embedding) to '{output_path}'\n")

# %%
class QM9AnalysisPipeline:
    # Target properties from the QM9 dataset
    QM9_PROPERTIES = [
        "Dipole Moment",
        "Isotropic polarizability",
        "HOMO",
        "LUMO",
        "HOMO-LUMO gap",
        "Electronic spatial extent",
        "Zero point vibrational energy",
        "Internal energy at 0K",
        "Internal energy at 298.15K",
    ]

# %%
if __name__ == "__main__":
    dataset_path = "./data/QM9/processed/data_v3.pt"  # Base QM9 dataset (not used directly here)

    for prop in QM9AnalysisPipeline.QM9_PROPERTIES:
        safe_name = prop.replace(" ", "_").replace('-', '_')  # Sanitize for filenames

        csv_path = f"./agent_results/results_{safe_name}_CLIP.csv"
        output_path = f"./agent_results/results_{safe_name}_CLIP.pt"

        print(f"Attaching CLIP embeddings for: {prop}")
        try:
            attach_clip_embeddings(dataset_path, csv_path, output_path)
        except Exception as e:
            print(f"Failed to process {prop}: {e}\n")
