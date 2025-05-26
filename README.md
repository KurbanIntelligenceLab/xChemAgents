# Official Implementation of 'xChemAgents: Agentic AI for Explainable Quantum Chemistry'

Please follow the steps below to set up the necessary libraries, frameworks, and data, and to train and evaluate the models.

Parsed Dataset link: [Data](https://figshare.com/s/7e858bb1c98bbba706b4) with results.

## Ollama with LLM Model Installation

**Install Ollama**

```$ curl -fsSL https://ollama.com/install.sh | sh```

**Pull LLama 3**

```$ ollama pull llama3```

**Check the version**
```
$ ollama list
NAME               ID              SIZE      MODIFIED   
llama3:latest      365c0bd3c000    4.7 GB    10 days ago    
```

## Conda Environment Setup

Create and activate the Conda environment:

```bash
$ conda create -n icml-agent-env python=3.11 -y
$ conda activate icml-agent-env
```

Install the required packages:

```
$ conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda==11.8 -c pytorch -c nvidia
$ conda install pytorch-cluster -c pyg
$ conda install pytorch-sparse -c pyg
$ pip install -r requirements.txt
```

## Download Data for 133885 GDB-9 molecules

After downloading and unzipping you should have a folder of `.xyz` files (e.g. `dsgdb9nsd.xyz/`) under `data/`.

```
$ cd data
$ curl -L -J -O "https://springernature.figshare.com/ndownloader/files/3195389"
$ mkdir -p dsgdb9nsd.xyz
$ tar -xjf dsgdb9nsd.xyz.tar.bz2 -C dsgdb9nsd.xyz/
```


## Running the Pipeline

**1. Retrieve PubChem Data**

Find the `qm9_pubchem.csv` file, containing data retrieved from PubChem, into the `data/` directory.


**2. Parse QM9 `.xyz` data**

Parse the QM9 `.xyz` dataset files and extract molecular metadata (atom count, GDB index, SMILES & InChI strings) into a single CSV for downstream tasks.
```
$ python process_xyz_files.py
```

**3. Parse QM9 PubChem Data**

Extract structured molecular properties (e.g., IUPAC name, molecular weight, XLogP, etc.) from unstructured text using regular expressions.

Merge this structured property data with the molecule representation data 

```
$ python process_qm9_descriptions.py
```

**4. Run Multi-Agentic System**
```
$ python agentic_qm9
```

**5. CLIP-Based Embedding Generation**

Generates CLIP text embeddings for QM9 molecules using selected features determined by the multi-agent system.

```
$ python generate_text_embedding.py
```

**6. Attaching CLIP Embeddings to QM9 Dataset**

This module enriches the PyTorch Geometric QM9 dataset by attaching CLIP text embeddings to each molecule.

```
python qm9_multimodal.py
```

Here's a clear and concise improved version:

---

**7. Train the Models**

Locate the folders containing state-of-the-art (SOTA) models:

* **SchNet**
* **DimeNet++**
* **Equiformer**
* **FAENet**

Within each folder:

* **Base Model:** Run `train_<model_name>.py`
* **Multi-Model:** Run `train_modified_<model_name>.py`

All target properties will be trained iteratively. Results will be saved in the `logs/` and `runs/` directories.

**8. Evaluate the Results**

Evaluate the trained models' performance using the provided evaluation script:

```bash
python calc_Avg.py
```

**Note:** Before running, ensure you update the script to specify the correct directory containing your results.


**Test Bed**
* OS: Ubuntu 22.04

* CPU: AMD Ryzen Threadripper PRO 3975WX 32-Cores

* GPU: NVIDIA RTX A6000

This repository utilizes PyG for SchNet, DimeNet++, and QM9 dataset. The official [Equiformer](https://github.com/atomicarchitects/equiformer) and [FAENet](https://github.com/vict0rsch/faenet/tree/main) repositories utilized for the implementations.