"""
train/train_all_regions.py
Train AE and VAE on each Medical MNIST class separately.
"""

import os
import sys
import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from model.ae  import build_autoencoder
from model.vae import build_vae

DATASET_PATH = BASE_DIR / "data/raw/medical_mnist"
RESULTS_DIR  = BASE_DIR / "results/models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE   = (64, 64)
BATCH_SIZE = 32


def preprocess_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, image


def load_class_dataset(class_dir):
    file_list = [
        str(class_dir / f)
        for f in os.listdir(class_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    ds = tf.data.Dataset.from_tensor_slices(file_list)
    ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(1000).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
    return ds


def train_all(ae_epochs=5, vae_epochs=5):
    classes = sorted(os.listdir(DATASET_PATH))
    results = {}

    for cls in classes:
        class_dir = DATASET_PATH / cls
        if not class_dir.is_dir():
            continue

        print(f"\n{'='*40}")
        print(f"Training on class: {cls}")
        print(f"{'='*40}")

        ds = load_class_dataset(class_dir)

        ae, _, _, _ = build_autoencoder()
        ae.fit(ds, epochs=ae_epochs, verbose=1)
        ae.save(RESULTS_DIR / f"ae_{cls}.keras")

        vae, _, _ = build_vae()
        vae.fit(ds, epochs=vae_epochs, verbose=1)

        results[cls] = {"ae": ae, "vae": vae}
        print(f"Done: {cls}")

    return results


if __name__ == "__main__":
    train_all()
