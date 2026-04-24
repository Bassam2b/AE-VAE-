"""
models/autoencoder.py
Convolutional Autoencoder (AE) encoder, decoder, and full model.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K


LATENT_DIM = 64

def build_encoder(input_shape=(64, 64, 1), latent_dim=LATENT_DIM):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32,  3, strides=2, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64,  3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    shape_before_flatten = K.int_shape(x)[1:]
    x      = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, name="latent_vector")(x)

    encoder = Model(inputs, latent, name="encoder")
    return encoder, shape_before_flatten

def build_decoder(shape_before_flatten, latent_dim=LATENT_DIM):
   
    latent_inputs = layers.Input(shape=(latent_dim,))

    x = layers.Dense(int(np.prod(shape_before_flatten)))(latent_inputs)
    x = layers.Reshape(shape_before_flatten)(x)

    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64,  3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32,  3, strides=2, padding="same", activation="relu")(x)

    outputs = layers.Conv2DTranspose(1, 3, padding="same", activation="sigmoid")(x)

    decoder = Model(latent_inputs, outputs, name="decoder")
    return decoder

def build_autoencoder(input_shape=(64, 64, 1), latent_dim=LATENT_DIM):
 
    encoder, shape_before_flatten = build_encoder(input_shape, latent_dim)
    decoder                       = build_decoder(shape_before_flatten, latent_dim)

    ae_inputs = layers.Input(shape=input_shape)
    encoded   = encoder(ae_inputs)
    decoded   = decoder(encoded)

    autoencoder = Model(ae_inputs, decoded, name="autoencoder")
    autoencoder.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["mse"],
    )

    return autoencoder, encoder, decoder, shape_before_flatten