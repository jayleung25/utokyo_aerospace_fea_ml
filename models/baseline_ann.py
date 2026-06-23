"""Track A — Professor's exact history-formulation ANN (70k version).

Input format:  (N, 160) — 20 timesteps × 8 absolute features, flattened.
               Extract from pipeline output: X[:, :, :8].reshape(N, -1)
               Drops the delta features so the feature set exactly matches
               the professor's training data (raw 160-column CSV).

Architecture (professor's 70k improved model, ML_CZM_ToJAY.pdf):
    Input(160) → Dense(256, relu) → Dense(128, relu) → Dense(128, relu)
               → Dense(64, relu) → Dense(32, relu) → Dense(1, linear)

Training config (exact match):
    Optimizer : Adam, lr=5e-4, β1=0.9, β2=0.999, ε=1e-8
    Loss      : MSE, no sample weights
    Batch     : 128
    Max epochs: 200
    Target    : Val MSE ≈ 3.83e-05
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

N_FEATURES = 160  # 20 timesteps × 8 absolute features


def build_baseline_ann() -> keras.Model:
    """Build and compile the professor's ANN architecture."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(N_FEATURES,), name="input"),
            keras.layers.Dense(256, activation="relu", name="dense_256"),
            keras.layers.Dense(128, activation="relu", name="dense_128a"),
            keras.layers.Dense(128, activation="relu", name="dense_128b"),
            keras.layers.Dense(64,  activation="relu", name="dense_64"),
            keras.layers.Dense(32,  activation="relu", name="dense_32"),
            keras.layers.Dense(1,   activation="linear", name="output"),
        ],
        name="baseline_ann",
    )

    optimizer = keras.optimizers.Adam(
        learning_rate=5e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
    )
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def prepare_inputs(X: np.ndarray) -> np.ndarray:
    """Convert pipeline output (N, 20, 16) → (N, 160).

    Keeps only the 8 absolute features per timestep (drops the 8 delta
    columns) so the feature set exactly matches the professor's raw CSV.
    """
    N = X.shape[0]
    return X[:, :, :8].reshape(N, -1).astype("float32")
