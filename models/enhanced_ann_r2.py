"""Track B Round 2 — Enhanced ANN with transition-zone weighted loss.

Architecture (unchanged from enhanced_ann):
    Input(320)
    → Dense(512, relu) → BatchNorm → Dropout(0.2)
    → Dense(256, relu) → BatchNorm → Dropout(0.2)
    → Dense(128, relu) → BatchNorm
    → Dense(64, relu)  → Dense(32, relu)
    → Dense(1, sigmoid)

Key change vs enhanced_ann:
  - Custom loss weights the transition zone (D=0.3–0.7) 3× vs elsewhere.
    Combined with the 14.8× inverse-frequency sample weights, transition
    samples receive ~44× more gradient signal than the D≈1 majority —
    directly targeting the physics bottleneck (FEA peak-load overestimation).
  - Explicit mse metric exposed so EarlyStopping can monitor val_mse
    (pure MSE) rather than val_loss (weighted loss + L2 noise), preventing
    premature LR decay and early stopping on regularization artifacts.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

N_FEATURES = 320  # 20 timesteps × 16 (8 absolute + 8 delta)
_L2 = 1e-4        # L2 weight decay on all Dense layers

_TRANSITION_LOW  = 0.3
_TRANSITION_HIGH = 0.7
_TRANSITION_MULT = 3.0  # 3× loss weight inside transition zone


def transition_weighted_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """MSE with 3× multiplier for samples in the transition zone (D ∈ [0.3, 0.7)).

    Stacks on top of per-sample inverse-frequency weights supplied via
    model.fit(sample_weight=...), giving the physically critical transition
    zone ~44× more gradient influence than the dominant D≈1 failed zone.
    """
    sq_err = tf.square(y_true - y_pred)
    in_transition = tf.cast(
        (y_true >= _TRANSITION_LOW) & (y_true < _TRANSITION_HIGH), tf.float32
    )
    zone_weight = 1.0 + (_TRANSITION_MULT - 1.0) * in_transition
    return tf.reduce_mean(zone_weight * sq_err)


def _dense(units: int) -> keras.layers.Dense:
    return keras.layers.Dense(
        units,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(_L2),
    )


def build_enhanced_ann_r2() -> keras.Model:
    """Build and compile the enhanced ANN R2 with transition-zone loss."""
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
        name="enhanced_ann_r2",
    )

    optimizer = keras.optimizers.Adam(
        learning_rate=1e-3,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        clipnorm=1.0,
    )
    model.compile(
        optimizer=optimizer,
        loss=transition_weighted_mse,
        metrics=["mse", "mae"],  # val_mse exposed for EarlyStopping
    )
    return model


def prepare_inputs(X: np.ndarray) -> np.ndarray:
    """Convert pipeline output (N, 20, 16) → (N, 320)."""
    N = X.shape[0]
    return X.reshape(N, -1).astype("float32")
