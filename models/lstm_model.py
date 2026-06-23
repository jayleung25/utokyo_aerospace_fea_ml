"""Track C — Stacked LSTM exploiting temporal sequence structure.

Input format:  (N, 20, 16) — pipeline output used directly, no reshaping.

Architecture:
    Input(20, 16)
    → LSTM(128, return_sequences=True) → Dropout(0.2)
    → LSTM(64,  return_sequences=False) → Dropout(0.2)
    → Dense(32, relu)
    → Dense(1, sigmoid)

Why LSTM is the strongest candidate to outperform the flat ANN:
  CZM damage is irreversible and path-dependent — once the interface
  softens, it cannot recover. An LSTM hidden state carries this loading
  history explicitly through each of the 20 timesteps, applying an
  inductive bias that matches the physics. The flat ANN, by contrast,
  must learn the same sequential relationships from a single 320-dimensional
  vector with no built-in notion of order.

  The transition zone (D = 0.3–0.7) is where the LSTM advantage is
  largest: damage initiation depends on the rate of loading change
  (ΔseparN, ΔfailureIndex), which the recurrent hidden state naturally
  encodes without needing explicit delta features.

Same sample weights, optimizer, and LR schedule as Track B.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

_L2 = 1e-4


def build_lstm_model() -> keras.Model:
    """Build and compile the stacked LSTM model."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(20, 16), name="input"),
            keras.layers.LSTM(
                128,
                return_sequences=True,
                kernel_regularizer=keras.regularizers.l2(_L2),
                recurrent_regularizer=keras.regularizers.l2(_L2),
                name="lstm_128",
            ),
            keras.layers.Dropout(0.2, name="drop_128"),
            keras.layers.LSTM(
                64,
                return_sequences=False,
                kernel_regularizer=keras.regularizers.l2(_L2),
                recurrent_regularizer=keras.regularizers.l2(_L2),
                name="lstm_64",
            ),
            keras.layers.Dropout(0.2, name="drop_64"),
            keras.layers.Dense(
                32,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(_L2),
                name="dense_32",
            ),
            keras.layers.Dense(1, activation="sigmoid", name="output"),
        ],
        name="lstm_model",
    )

    optimizer = keras.optimizers.Adam(
        learning_rate=1e-3,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        clipnorm=1.0,
    )
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def prepare_inputs(X: np.ndarray) -> np.ndarray:
    """Pipeline output (N, 20, 16) is already the correct shape; just cast."""
    return X.astype("float32")
