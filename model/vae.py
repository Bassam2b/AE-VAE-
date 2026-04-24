"""
models/vae.py
Variational Autoencoder (VAE) encoder, decoder, and custom training class.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K

LATENT_DIM = 16

def build_vae_encoder(input_shape=(64, 64, 1), latent_dim=LATENT_DIM):

    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32,  3, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64,  3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)

    shape_before_flatten = K.int_shape(x)[1:]

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)

    z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    # Reparameterization
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])

    encoder_vae = Model(inputs, [z_mean, z_log_var, z], name="vae_encoder")
    return encoder_vae, shape_before_flatten

def build_vae_decoder(shape_before_flatten, latent_dim=LATENT_DIM):
   
    latent_inputs = layers.Input(shape=(latent_dim,))

    x = layers.Dense(int(np.prod(shape_before_flatten)))(latent_inputs)
    x = layers.Reshape(shape_before_flatten)(x)

    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64,  3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32,  3, strides=2, padding="same", activation="relu")(x)

    outputs = layers.Conv2DTranspose(1, 3, padding="same", activation="sigmoid")(x)

    decoder_vae = Model(latent_inputs, outputs, name="vae_decoder")
    return decoder_vae

class VAE(Model):
    def __init__(self, encoder, decoder, beta=0.5, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta  # controls KL vs reconstruction balance

    def train_step(self, data):
        x, _ = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)

            recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(x, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1,
                )
            )
            total_loss = recon_loss + self.beta * kl_loss  # ← weighted

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    def test_step(self, data):
        x, _ = data
        z_mean, z_log_var, z = self.encoder(x, training=False)
        reconstruction = self.decoder(z, training=False)

        recon_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(x, reconstruction),
                axis=(1, 2),
            )
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1,
            )
        )

        return {
            "loss": recon_loss + kl_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
        }


def build_vae(input_shape=(64, 64, 1), latent_dim=LATENT_DIM, beta=0.5):
    encoder_vae, shape_before_flatten = build_vae_encoder(input_shape, latent_dim)
    decoder_vae = build_vae_decoder(shape_before_flatten, latent_dim)

    vae = VAE(encoder_vae, decoder_vae, beta=beta)
    vae.compile(optimizer="adam")

    return vae, encoder_vae, decoder_vae