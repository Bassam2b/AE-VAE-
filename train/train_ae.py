"""
train/train_ae.py
Training script for the Autoencoder (AE).
"""

import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from utils.data_loader import load_dataset
from model.ae import build_autoencoder

DATASET_PATH = BASE_DIR / "data/raw/medical_mnist"
RESULTS_DIR  = BASE_DIR / "results/models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def train_ae(epochs=20):
    train_ds, val_ds, class_names, _ = load_dataset(DATASET_PATH)
    print("Classes:", class_names)

    autoencoder, encoder, decoder, _ = build_autoencoder()
    autoencoder.summary()

    start = time.time()
    history = autoencoder.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
    )
    print(f"AE Training Time: {time.time() - start:.1f}s")

    autoencoder.save(RESULTS_DIR / "autoencoder.keras")
    print("AE saved.")
    return autoencoder, encoder, decoder, history


if __name__ == "__main__":
    train_ae()
