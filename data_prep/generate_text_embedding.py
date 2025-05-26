# %%
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from transformers import CLIPModel, CLIPTokenizer

# %%
def generate_clip_embeddings(input_csv, output_csv,
                             model_name="openai/clip-vit-large-patch14",
                             device="cpu"):
    """
    Generate CLIP embeddings for molecule descriptions based on selected features.

    This function reads a CSV file containing selected features for molecules,
    constructs a text prompt by retrieving actual feature values from the QM9 dataset,
    computes CLIP text embeddings, and saves the enriched data with embeddings to a new CSV.

    Args:
        input_csv (str): Path to the input CSV containing molecule indices and selected features.
        output_csv (str): Path to save the output CSV with added CLIP embeddings.
        model_name (str): HuggingFace model identifier for CLIP.
        device (str): Compute device ("cpu" or "cuda").
    """

    # 1. Load the feature selection results and filter out rows with invalid iterations
    df = pd.read_csv(input_csv)
    df = df[df["iterations_needed"] != 4]  # Exclude unreliable entries
    # Convert empty strings to NaN
    df = df.replace("", np.nan)
    # Drop any row that contains at least one NaN value
    df = df.dropna(axis=0, how='any')

    # 2. Load QM9 dataset with molecular descriptors and set index for fast lookup
    df_qm9 = pd.read_csv("./data/qm9_parsed_merged_data.csv")
    df_qm9 = df_qm9.set_index("index")

    # 3. Load pretrained CLIP tokenizer and model
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    # 4. Prepare containers for constructed texts and embeddings
    text_all = []
    embeddings = []

    # 5. Iterate through each molecule row to construct input text and compute embeddings
    #for i, row in df.iterrows():
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Generating CLIP embeddings"):

        # Get selected features and molecule index
        lst = row["selected_features"].split(",")
        index = row["index"]

        # Construct descriptive text: e.g., "Formula: C2H6 Molecular weight: 30.07 ..."
        text = ""
        for feature in lst:
            text += f"{feature}: {df_qm9.loc[index][feature]} "

        text_all.append(text)

        # Tokenize and encode the text with CLIP
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            text_embeds = model.get_text_features(**inputs)
            emb_list = text_embeds[0].cpu().numpy().tolist()
            embeddings.append(emb_list)

    # 6. Add generated text and embeddings back to the DataFrame
    df["selected_feature_values"] = text_all        
    df["clip_embedding"] = embeddings

    # 7. Save the final DataFrame to a CSV
    df.to_csv(output_csv, index=False)
    print(f"Done! Saved the molecules with embeddings to '{output_csv}'.\n")

# %%
class QM9AnalysisPipeline:
    # List of target molecular properties from the QM9 dataset
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
    # Batch process each property in the QM9 dataset
    for prop in QM9AnalysisPipeline.QM9_PROPERTIES:
        # Sanitize property name for use in filenames
        safe_name = prop.replace(" ", "_").replace('-', '_')

        # Construct file paths for input and output
        input_csv_path = f"./agent_results/results_{safe_name}.csv"
        output_csv_path = f"./agent_results/results_{safe_name}_CLIP.csv"

        print(f"Processing: {prop}")
        try:
            generate_clip_embeddings(
                input_csv=input_csv_path,
                output_csv=output_csv_path,
                device="cuda"  # Change to "cpu" if GPU is not available
            )
        except Exception as e:
            print(f"Failed to process {prop}: {e}")
