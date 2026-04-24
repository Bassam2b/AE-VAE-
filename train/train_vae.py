"""
train/train_vae.py
Training script for the Variational Autoencoder (VAE).
"""

import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from utils.data_loader import load_dataset
from model.vae import build_vae

DATASET_PATH = BASE_DIR / "data/raw/medical_mnist"
RESULTS_DIR  = BASE_DIR / "results/models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def train_vae(epochs=30):
    train_ds, val_ds, class_names, _ = load_dataset(DATASET_PATH)
    print("Classes:", class_names)

    vae, encoder_vae, decoder_vae = build_vae()

    start = time.time()
    history = vae.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
    )
    print(f"VAE Training Time: {time.time() - start:.1f}s")

    encoder_vae.save(RESULTS_DIR / "vae_encoder.keras")
    decoder_vae.save(RESULTS_DIR / "vae_decoder.keras")
    print("VAE saved.")
    return vae, encoder_vae, decoder_vae, history


if __name__ == "__main__":
    train_vae()
