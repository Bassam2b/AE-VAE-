"""
utils/visualization.py
All plotting and visualization utilities for AE and VAE experiments.
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from sklearn.decomposition import PCA

def _ensure_dir(save_path):
    """Create parent directory if it doesn't exist."""
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

def show_samples(raw_train_ds, class_names, n=12):
    """Display a grid of sample images from the training set."""
    for images, labels in raw_train_ds.take(1):
        plt.figure(figsize=(12, 8))
        for i in range(n):
            plt.subplot(3, 4, i + 1)
            plt.imshow(images[i].numpy().squeeze(), cmap="gray")
            plt.title(class_names[labels[i]])
            plt.axis("off")
        plt.tight_layout()
        plt.show()

def plot_loss(history, title="Loss", save_path=None):
    """Plot training (and optional validation) loss curves."""
    _ensure_dir(save_path)
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_vae_loss(history, save_path=None):
    """Plot VAE total, reconstruction, and KL divergence loss curves."""
    _ensure_dir(save_path)
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Total Loss")
    if "reconstruction_loss" in history.history:
        plt.plot(history.history["reconstruction_loss"], label="Reconstruction")
    if "kl_loss" in history.history:
        plt.plot(history.history["kl_loss"], label="KL Divergence")
    plt.title("VAE Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_reconstructions(model, val_ds, n=8, title="Reconstructions", save_path=None):
    """Plot original vs reconstructed images."""
    _ensure_dir(save_path)
    for batch_images, _ in val_ds.take(1):
        preds = model.predict(batch_images[:n], verbose=0)

        plt.figure(figsize=(2 * n, 4))
        for i in range(n):
            plt.subplot(2, n, i + 1)
            plt.imshow(batch_images[i].numpy().squeeze(), cmap="gray")
            plt.axis("off")
            if i == 0:
                plt.ylabel("Original", fontsize=10)

            plt.subplot(2, n, i + n + 1)
            plt.imshow(preds[i].squeeze(), cmap="gray")
            plt.axis("off")
            if i == 0:
                plt.ylabel("Reconstructed", fontsize=10)

        plt.suptitle(title)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

def plot_ae_vs_vae(autoencoder, encoder_vae, decoder_vae, val_ds, n=6, save_path=None):
    """Comparison: Original / AE reconstruction / VAE reconstruction."""
    _ensure_dir(save_path)
    for batch_images, _ in val_ds.take(1):
        ae_preds         = autoencoder.predict(batch_images[:n], verbose=0)
        z_mean, _, z     = encoder_vae.predict(batch_images[:n], verbose=0)
        vae_preds        = decoder_vae.predict(z, verbose=0)

        plt.figure(figsize=(3 * n, 6))
        for i in range(n):
            plt.subplot(3, n, i + 1)
            plt.imshow(batch_images[i].numpy().squeeze(), cmap="gray")
            plt.axis("off")
            if i == 0: plt.ylabel("Original", fontsize=10)

            plt.subplot(3, n, i + n + 1)
            plt.imshow(ae_preds[i].squeeze(), cmap="gray")
            plt.axis("off")
            if i == 0: plt.ylabel("AE", fontsize=10)

            plt.subplot(3, n, i + 2 * n + 1)
            plt.imshow(vae_preds[i].squeeze(), cmap="gray")
            plt.axis("off")
            if i == 0: plt.ylabel("VAE", fontsize=10)

        plt.suptitle("Original  /  AE  /  VAE")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

def plot_latent_space_with_labels(encoder_vae, raw_val_ds, class_names, save_path=None):
    """
    Scatter plot of latent space with real class labels.
    """
    _ensure_dir(save_path)
    latent_vectors, labels_all = [], []

    for batch_images, batch_labels in raw_val_ds:
        images = tf.cast(batch_images, tf.float32) / 255.0
        z_mean, _, _ = encoder_vae.predict(images, verbose=0)
        latent_vectors.append(z_mean)
        labels_all.append(batch_labels.numpy())

    latent_vectors = np.concatenate(latent_vectors)
    labels_all     = np.concatenate(labels_all)

    # Reduce to 2D if latent_dim > 2
    if latent_vectors.shape[1] > 2:
        print(f"Reducing latent dim {latent_vectors.shape[1]} → 2 using PCA...")
        latent_vectors = PCA(n_components=2).fit_transform(latent_vectors)

    plt.figure(figsize=(10, 8))
    for cls_idx, cls_name in enumerate(class_names):
        mask = labels_all == cls_idx
        plt.scatter(
            latent_vectors[mask, 0],
            latent_vectors[mask, 1],
            label=cls_name, s=8, alpha=0.6
        )

    plt.legend(markerscale=3)
    plt.title("VAE Latent Space (PCA 2D)" if latent_vectors.shape[1] > 2 else "VAE 2D Latent Space")
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_generated_grid(decoder_vae, latent_dim=2, grid_size=5, z_range=2.0, save_path=None):
    """
    Generate images by sampling from the latent space.
    - If latent_dim == 2: uses a 2D grid scan
    - If latent_dim > 2:  samples randomly from N(0,1)
    """
    _ensure_dir(save_path)
    plt.figure(figsize=(10, 10))

    if latent_dim == 2:
        # 2D grid scan
        grid = np.linspace(-z_range, z_range, grid_size)
        count = 1
        for yi in grid:
            for xi in grid:
                z_sample  = np.array([[xi, yi]])
                x_decoded = decoder_vae.predict(z_sample, verbose=0)
                plt.subplot(grid_size, grid_size, count)
                plt.imshow(x_decoded[0].squeeze(), cmap="gray")
                plt.axis("off")
                count += 1
        plt.suptitle("VAE Generated Samples (2D Latent Grid)")

    else:
        # Random sampling for higher dims
        n = grid_size * grid_size
        z_samples = np.random.normal(size=(n, latent_dim))
        decoded   = decoder_vae.predict(z_samples, verbose=0)
        for i in range(n):
            plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(decoded[i].squeeze(), cmap="gray")
            plt.axis("off")
        plt.suptitle(f"VAE Generated Samples (Random, latent_dim={latent_dim})")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_denoising(autoencoder, val_ds, add_noise_fn, n=6, save_path=None):
    """Plots: Original / Noisy / Denoised."""
    _ensure_dir(save_path)
    for batch_images, _ in val_ds.take(1):
        noisy = add_noise_fn(batch_images[:n])
        preds = autoencoder.predict(noisy, verbose=0)

        plt.figure(figsize=(3 * n, 6))
        for i in range(n):
            plt.subplot(3, n, i + 1)
            plt.imshow(batch_images[i].numpy().squeeze(), cmap="gray")
            plt.axis("off")
            if i == 0: plt.ylabel("Original", fontsize=10)

            plt.subplot(3, n, i + n + 1)
            plt.imshow(noisy[i].numpy().squeeze(), cmap="gray")
            plt.axis("off")
            if i == 0: plt.ylabel("Noisy", fontsize=10)

            plt.subplot(3, n, i + 2 * n + 1)
            plt.imshow(preds[i].squeeze(), cmap="gray")
            plt.axis("off")
            if i == 0: plt.ylabel("Denoised", fontsize=10)

        plt.suptitle("Original  /  Noisy  /  Denoised")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()