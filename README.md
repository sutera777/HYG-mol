## 🤖 Assistant

Of course! Here is a very simple and clear English version of the `README.md` file, focusing on the essential steps to get the project running.

---

# HYG-mol: Explainable Molecular Property Prediction

This project uses a deep learning model called an **Attention Hypergraph Network** to predict properties of molecules (like toxicity or solubility).


## Key Features

* Predicts molecular properties for both classification and regression tasks.
* **Explains its predictions** by highlighting important chemical substructures.
* Built with PyTorch, PyTorch Geometric, and RDKit.

## Setup

It is highly recommended to use **Conda** to install the dependencies.

**1. Clone the repository**
```bash
git clone https://github.com/sutera777/HYG-mol.git
cd HYG-mol
```

**2. Create and activate a Conda environment**
```bash
# Create a new environment named "hyg-mol"
conda create -n hyg-mol python=3.9 -y

# Activate it
conda activate hyg-mol
```

**3. Install all dependencies**
```bash
# Install PyTorch (CPU version is simplest)
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install PyG (PyTorch Geometric)
conda install pyg -c pyg

# Install RDKit
conda install -c conda-forge rdkit


```
> **Note**: For GPU support, please find the correct PyTorch installation command on their [official website](https://pytorch.org/get-started/locally/).

**4. Add your data**

Place your dataset `.csv` files (e.g., `bace.csv`, `esol.csv`) into the `data/` folder.

---

## How to Run

All commands should be run from the project's root directory (`HYG-mol/`).


This is the main command to train, validate, and test a model.

```bash
python src/main.py \
    --dataset bace \
    --model_type attention \
    --epochs 50 \
    --batch_size 64
```

* This command trains the `attention` model on the `bace` dataset for 50 epochs.
* Results, including the best model (`.pt` file) and performance plots, will be saved to a new folder inside the `results/` directory.

