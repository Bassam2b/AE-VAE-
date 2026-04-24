"""
data/loader.py
Dataset loading and preprocessing utilities for Medical MNIST.
"""

import tensorflow as tf
from pathlib import Path

IMG_SIZE   = (64, 64)
BATCH_SIZE = 128
SEED       = 42

def preprocess(image, label):
    "Normalize image to [0, 1] and return (image, image) for reconstruction."
    image = tf.cast(image, tf.float32) / 255.0
    return image, image


def load_dataset(extract_path: Path, img_size=IMG_SIZE,
                 batch_size=BATCH_SIZE, seed=SEED):

    raw_train_ds, raw_val_ds = tf.keras.utils.image_dataset_from_directory(
        extract_path,
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        color_mode="grayscale",
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=0.2,
        subset="both",
    )

    class_names = raw_train_ds.class_names

    train_ds = (
    raw_train_ds
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .shuffle(1000, seed=SEED)  
    .prefetch(tf.data.AUTOTUNE)
)

    val_ds = (
        raw_val_ds
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) 
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, class_names, (raw_train_ds, raw_val_ds)