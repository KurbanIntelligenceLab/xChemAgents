# %%
"""
QM9 XYZ → CSV Converter

This script walks through a directory of QM9-format .xyz files,
extracts key metadata (atom count, GDB index, SMILES and InChI
for both original and relaxed geometries), and writes them
out in a single, sorted CSV for downstream analysis.
"""
# Dataset: Quantum chemistry structures and properties of 134 kilo molecules
#          Link: https://springernature.figshare.com/collections/_/978904

# %%
import os
import pandas as pd

# %%
# Configuration: adjust these paths as needed
data_folder = './data/dsgdb9nsd.xyz'
output_file_name = './data/qm9_representation_data.csv'

# %%
def extract_info_from_xyz(file_path):
    """
    Read a single .xyz file and pull out:
      - atom_count: number of atoms (first line)
      - gdb_id: GDB index from the header
      - SMILES (GDB-9) & SMILES (relaxed geometry): 3rd-from-last line
      - InChI  (GDB-9) & InChI  (relaxed geometry): last line
    Returns a dict suitable for a DataFrame row.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 5:
        raise ValueError("File too short to contain required fields")

    # Atom count
    atom_count = int(lines[0].strip())

    # Header line (2nd line): find GDB index
    header_line = lines[1].strip().split()
    gdb_id = None
    for i, val in enumerate(header_line):
        if val == "gdb" and i + 1 < len(header_line):
            gdb_id = header_line[i + 1]
            break

    # SMILES line: 3rd from the end
    smiles_line = lines[-2].strip().split()
    smiles1 = smiles_line[0] if len(smiles_line) > 0 else ""
    smiles2 = smiles_line[1] if len(smiles_line) > 1 else ""

    # InChI line: last line
    inchi_line = lines[-1].strip().split()
    inchi1 = inchi_line[0] if len(inchi_line) > 0 else ""
    inchi2 = inchi_line[1] if len(inchi_line) > 1 else ""

    return {
        "filename": os.path.basename(file_path),
        "atom_count": atom_count,
        "index": gdb_id,
        "SMILES (GDB-9)": smiles1,
        "SMILES (relaxed geometry)": smiles2,
        "InChI  (GDB-9)": inchi1,
        "InChI  (relaxed geometry)": inchi2
    }

# %%
def process_all_xyz(folder_path, output_csv):
    """
    Walks through folder_path, finds all .xyz files,
    extracts metadata via extract_info_from_xyz, and
    aggregates into a single CSV sorted by the numeric GDB index.
    """
    rows = []
    # Recursively traverse the directory
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.xyz'):
                file_path = os.path.join(root, file)
                try:
                    row = extract_info_from_xyz(file_path)
                    rows.append(row)
                except Exception as e:
                    print(f"Skipping {file}: {e}")

    df = pd.DataFrame(rows)

    # Convert index (gdb_id) to int for proper sorting (if possible)
    df["index"] = pd.to_numeric(df["index"], errors='coerce')
    df = df.sort_values(by="index").reset_index(drop=True)

    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} entries to {output_csv} (sorted by index)")

# %% Run the full processing pipeline
process_all_xyz(data_folder, output_file_name)

# %%
