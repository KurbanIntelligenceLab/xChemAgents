"""
Description:
This script processes and merges two CSV files related to the QM9 dataset:
1. `qm9_representation_data.csv` - Contains structural/representation features for molecules.
2. `qm9_pubchems.csv` - Contains textual descriptions of molecules, including chemical and physical properties.

Main Steps:
- Extract structured molecular properties (e.g., IUPAC name, molecular weight, XLogP, etc.) from unstructured text using regular expressions.
- Normalize the parsed data into a structured DataFrame.
- Merge this structured property data with the molecule representation data via the 'index' column.
- Drop rows with any missing values in the parsed features to ensure clean data.
- Save the final cleaned dataset as `qm9_parsed_merged_data.csv`.

Requirements:
- The 'description' field in `qm9_pubchems.csv` must contain consistent labeled information.
- Both CSVs must share the 'index' column (used as a join key).
"""
# %%
import pandas as pd
import re

# %%
def parse_description(text):
    if pd.isna(text) or not isinstance(text, str):
        return {}

    try:
        return {
            "IUPAC name": re.search(r"IUPAC name:\s*(.+?)\s+Formula:", text).group(1),
            "Formula": re.search(r"Formula:\s*(\S+)", text).group(1),
            "Molecular weight": float(re.search(r"Molecular weight:\s*~?([\d.]+)", text).group(1)),
            "XLogP": float(re.search(r"XLogP:\s*([\d.\-]+)", text).group(1)),
            "H-bond donors": int(re.search(r"H-bond donors:\s*(\d+)", text).group(1)),
            "H-bond acceptors": int(re.search(r"H-bond acceptors:\s*(\d+)", text).group(1)),
            "Rotatable bonds": int(re.search(r"Rotatable bonds:\s*(\d+)", text).group(1)),
            "Polar Surface Area": float(re.search(r"Polar Surface Area:\s*([\d.]+)", text).group(1)),
            "Synonyms": re.search(r"Synonyms \(partial list\):\s*(.+)", text).group(1)
        }
    except Exception:
        return {}

# %% Load your data
df_rep = pd.read_csv("./data/qm9_representation_data.csv")        # Must contain 'gdb_id'
df_desc = pd.read_csv("./data/qm9_pubchems.csv", low_memory=False)     # Must contain 'gdb_id' and 'description'

# %% Parse 'description'
parsed = df_desc["description"].apply(parse_description)
parsed_df = pd.json_normalize(parsed)

# %% Add gdb_id to parsed data
df_desc_cleaned = pd.concat([df_desc["index"], parsed_df], axis=1)

# %% Merge with df_rep
df_merged = df_rep.merge(df_desc_cleaned, on="index", how="left")

# %% Drop rows with any missing parsed values
parsed_columns = parsed_df.columns.tolist()
df_merged_cleaned = df_merged.dropna(subset=parsed_columns)

# %% Save the final output
df_merged_cleaned.to_csv("./data/qm9_parsed_merged_data.csv", index=False)
print("Final processed data saved to 'qm9_parsed_merged_data.csv'")

# %%
