"""Track B — Enhanced ANN with delta features, sample weights, and regularization.

Input format:  (N, 320) — 20 timesteps × 16 (8 absolute + 8 delta), flattened.
               Extract from pipeline output: X.reshape(N, -1)

Architecture:
    Input(320)
    → Dense(512, relu) → BatchNorm → Dropout(0.2)
    → Dense(256, relu) → BatchNorm → Dropout(0.2)
    → Dense(128, relu) → BatchNorm
    → Dense(64, relu)  → Dense(32, relu)
    → Dense(1, sigmoid)     ← enforces D̂ ∈ (0, 1)

Key improvements over professor's baseline:
  - Delta features (ΔseparN, Δtractness, …) expose rate-of-change signals
    that distinguish active loading from plateau, helping the model localise
    the transition zone (D = 0.3–0.7) where the load curve peaks.
  - BatchNorm stabilises gradients across the wider layers.
  - Dropout(0.2) prevents over-fitting to the 68.8% majority at D ≈ 1.
  - sigmoid output constrains predictions to [0, 1], eliminating the small
    fraction of D > 1 extrapolations the linear output can produce.
  - Inverse-frequency sample weights (14.8× for transition zone) make the
    loss treat rare critical samples as heavily as the D ≈ 1 majority.
  - AdamW + L2 regularisation on Dense layers (weight decay equivalent).
  - ReduceLROnPlateau decays LR when val_loss plateaus, extracting more
    signal without a fixed schedule.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

N_FEATURES = 320  # 20 timesteps × 16 (8 absolute + 8 delta)
_L2 = 1e-4        # L2 weight decay on all Dense layers


def _dense(units: int) -> keras.layers.Dense:
    return keras.layers.Dense(
        units,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(_L2),
    )


def build_enhanced_ann() -> keras.Model:
    """Build and compile the enhanced ANN."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(N_FEATURES,), name="input"),
            _dense(512),
            keras.layers.BatchNormalization(name="bn_512"),
            keras.layers.Dropout(0.2, name="drop_512"),
            _dense(256),
            keras.layers.BatchNormalization(name="bn_256"),
            keras.layers.Dropout(0.2, name="drop_256"),
            _dense(128),
            keras.layers.BatchNormalization(name="bn_128"),
            _dense(64),
            _dense(32),
            keras.layers.Dense(1, activation="sigmoid", name="output"),
        ],
        name="enhanced_ann",
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
    """Convert pipeline output (N, 20, 16) → (N, 320)."""
    N = X.shape[0]
    return X.reshape(N, -1).astype("float32")
