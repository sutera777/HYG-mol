


---

# HYG-mol

This project uses a deep learning model called an **Attention Hypergraph Network** to predict properties of molecules (like toxicity or solubility).


## Key Features

* Predicts molecular properties for both classification and regression tasks.
* **Explains its predictions** by highlighting important chemical substructures.
* Built with PyTorch, PyTorch Geometric, and RDKit.

##  Installation

### 1. Environment Setup
We recommend using Python 3.9 and Conda to manage your environment:

```bash
conda create -n HYG-mol python=3.9
conda activate HYG-mol
```

### 2. Dependency Installation
Since some PyTorch Geometric extensions depend on specific CUDA versions, follow this order:

```bash
# Install PyTorch Core (CUDA 11.6)
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install Graph Extensions
pip install torch-scatter torch-sparse torch-cluster torch-geometric==2.3.1 -f https://data.pyg.org/whl/torch-1.13.1+cu116.html

# Install remaining dependencies
pip install -r requirements.txt
```



##  Usage

### Training and Evaluation
To run the model on the BBBP dataset:

```bash
python -m src.main --dataset BBBP --task_type classification --model_type attention 
```

### Supported Datasets
The framework currently supports:
- **Classification**: BBBP, BACE, ClinTox, Tox21, SIDER, ToxCast.
- **Regression**: ESOL, FreeSolv, Lipophilicity, QM8, QM9.

