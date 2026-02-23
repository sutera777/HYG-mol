# src/config.py

import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()

DATA_DIR: Path = ROOT_DIR / 'data'
PROCESSED_DATA_DIR: Path = ROOT_DIR / 'processed_data'
RESULTS_DIR: Path = ROOT_DIR / 'results'

DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

CHEMBERTA_MODEL_NAME: str = "seyonec/ChemBERTa-zinc-base-v1" 

HYPEREDGE_INPUT_DIM: int = 5

SUPPORTED_DATASETS: list[str] = [
    'bace', 'bbbp', 'sider', 'clintox', 'tox21', 'toxcast', 
    'esol', 'freesolv', 'lipophilicity', 'qm8', 'qm9'
]

EXPLAINABILITY_SETTINGS = {
    'attention_temperature': 2.0,     
    'smoothing_strength': 0.3,         
    'visualization': {
        'image_width': 600,
        'image_height': 450,
        'highlight_bond_color': (0.7, 0.7, 0.7), 
    }
}


DEFAULT_TRAINING_ARGS = {
    'seed': 42,
    'batch_size': 32,
    'epochs': 100,
    'lr': 0.001,
    'hidden_channels': 128,
    'patience': 10,
    'heads': 4, # for attention model
}


