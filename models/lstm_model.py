"""Track C — Stacked LSTM exploiting temporal sequence structure.

Input format:  (N, 20, 16) — pipeline output used directly, no reshaping.

Architecture (Round 2):
    Input(20, 16)
    → LSTM(128, return_sequences=True, recurrent_dropout=0.1)
    → LayerNormalization → Dropout(0.2)
    → LSTM(64,  return_sequences=False, recurrent_dropout=0.1)
    → LayerNormalization → Dropout(0.2)
    → Dense(64, relu) → Dense(32, relu)
    → Dense(1, sigmoid)

Round 2 changes vs Round 1:
  - LR 1e-3 → 1e-4: LSTMs need a lower LR than dense ANNs; 1e-3 caused
    val_loss oscillation that stalled convergence at epoch 300.
  - clipnorm 1.0 → 0.5: tighter bound prevents gradient spikes across
    20 unrolled BPTT steps.
  - recurrent_dropout=0.1: regularises the recurrent kernel independently
    of the output dropout; reduces co-adaptation to the D≈1.0 majority.
  - LayerNormalization: preferred over BatchNorm for RNNs (normalises
    across the feature axis per timestep, not across the batch axis).
  - Dense(32) → Dense(64)→Dense(32): extra capacity in the prediction
    head lets the LSTM features combine more expressively before output.

Same sample weights and LR schedule as Track B (patience raised in
train_model.py to give each LR level more time to converge).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

_L2 = 1e-4


def build_lstm_model() -> keras.Model:
    """Build and compile the stacked LSTM model (Round 2 config)."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(20, 16), name="input"),
            keras.layers.LSTM(
                128,
                return_sequences=True,
                recurrent_dropout=0.1,
                kernel_regularizer=keras.regularizers.l2(_L2),
                recurrent_regularizer=keras.regularizers.l2(_L2),
                name="lstm_128",
            ),
            keras.layers.LayerNormalization(name="ln_128"),
            keras.layers.Dropout(0.2, name="drop_128"),
            keras.layers.LSTM(
                64,
                return_sequences=False,
                recurrent_dropout=0.1,
                kernel_regularizer=keras.regularizers.l2(_L2),
                recurrent_regularizer=keras.regularizers.l2(_L2),
                name="lstm_64",
            ),
            keras.layers.LayerNormalization(name="ln_64"),
            keras.layers.Dropout(0.2, name="drop_64"),
            keras.layers.Dense(
                64,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(_L2),
                name="dense_64",
            ),
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
        learning_rate=1e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        clipnorm=0.5,
    )
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def prepare_inputs(X: np.ndarray) -> np.ndarray:
    """Pipeline output (N, 20, 16) is already the correct shape; just cast."""
    return X.astype("float32")
