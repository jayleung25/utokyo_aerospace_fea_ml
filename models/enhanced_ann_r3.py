"""Track B Round 3 — Initiation-zone loss weight + monotonicity penalty.

Architecture (unchanged from enhanced_ann_r2):
    Input(320)
    → Dense(512, relu) → BatchNorm → Dropout(0.2)
    → Dense(256, relu) → BatchNorm → Dropout(0.2)
    → Dense(128, relu) → BatchNorm
    → Dense(64, relu)  → Dense(32, relu)
    → Dense(1, sigmoid)

Motivation: Enhanced ANN R2 is the best model on every metric except
Initiation MAE (D < 0.3), which regressed 49% vs the plain baseline ANN
(0.00433 vs 0.00290) — the same regression also shows up in LSTM R4 (+79%).
Three rounds have flagged this without a fix. Separately, the README's
"Critical Additions" list has called for a D-monotonicity physics
constraint since round 1 and it has never been implemented in any model.
Both are addressed here, isolated to the loss function only (architecture,
LR, L2, dropout all held fixed vs R2) so any change in results is
attributable to the loss change alone.

Key changes vs enhanced_ann_r2:
  1. `physics_weighted_mse` adds a 2× multiplier for the initiation zone
     (D < 0.3) on top of R2's existing 3× transition-zone multiplier.
     Stacked on the 14.8× inverse-frequency sample weights, initiation
     samples now get ~2× more gradient signal than they did under R2's
     loss, without touching the transition-zone weighting that already
     hit its target.
  2. Monotonicity penalty via `model.add_loss`: damage is physically
     irreversible, but the dataset has no trajectory/element ID column,
     so consecutive timesteps of the same simulation can't be paired
     directly (row order after ingest/split does not preserve adjacency).
     Instead this uses `failureIndex` — the accumulated loading criterion
     already identified as the single strongest live feature — as a
     monotonicity proxy: within each training batch, for any pair of
     samples where failureIndex_i > failureIndex_j, the model should not
     predict D_i < D_j. This is a standard soft/pairwise monotonicity
     regularizer (isotonic-style), computed per-batch since no ground-truth
     ordering across the whole dataset exists. Implemented as a small Keras
     layer taking (failureIndex, D_pred) so it participates in the
     autodiff graph via `add_loss`, independent of the main weighted-MSE
     term and unaffected by per-sample weights or the explicit "mse" metric
     used for EarlyStopping.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

N_FEATURES = 320  # 20 timesteps × 16 (8 absolute + 8 delta)
_L2 = 1e-4        # L2 weight decay on all Dense layers

_INITIATION_HIGH = 0.3
_INITIATION_MULT = 2.0  # 2× loss weight inside initiation zone (D < 0.3)

_TRANSITION_LOW  = 0.3
_TRANSITION_HIGH = 0.7
_TRANSITION_MULT = 3.0  # 3× loss weight inside transition zone (unchanged from R2)

# Flat index of the current (most recent) timestep's failureIndex in the
# (N, 320) input: timestep 19 (newest, chronological order) × 16 features
# + feature 0 (failureIndex is the first of the 8 absolute features).
_CURRENT_FAILURE_INDEX_FLAT = 19 * 16 + 0  # = 304

_MONO_WEIGHT = 1e-3  # keeps the penalty a gentle regularizer vs ~1e-5-scale MSE


def physics_weighted_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """MSE with extra multipliers for the initiation and transition zones.

    Zones do not overlap (initiation = D<0.3, transition = D in [0.3,0.7)),
    so the multipliers add independently with no double-counting.
    """
    sq_err = tf.square(y_true - y_pred)
    in_initiation = tf.cast(y_true < _INITIATION_HIGH, tf.float32)
    in_transition = tf.cast(
        (y_true >= _TRANSITION_LOW) & (y_true < _TRANSITION_HIGH), tf.float32
    )
    zone_weight = (
        1.0
        + (_INITIATION_MULT - 1.0) * in_initiation
        + (_TRANSITION_MULT - 1.0) * in_transition
    )
    return tf.reduce_mean(zone_weight * sq_err)


class MonotonicityPenalty(keras.layers.Layer):
    """Pairwise soft-monotonicity penalty over a proxy ordering variable.

    For each pair (i, j) in the batch where proxy_i > proxy_j, penalizes
    d_pred_i < d_pred_j via a hinge on the predicted-damage difference.
    Computed per-batch (no cross-batch or dataset-wide ordering) since the
    raw data carries no trajectory ID to pair true consecutive timesteps.
    """

    def __init__(self, weight: float = _MONO_WEIGHT, **kwargs) -> None:
        super().__init__(**kwargs)
        self.weight = weight

    def call(self, inputs):
        proxy, d_pred = inputs  # each (batch, 1)
        proxy_diff = proxy - tf.transpose(proxy)     # proxy_i - proxy_j
        d_diff = d_pred - tf.transpose(d_pred)       # d_pred_i - d_pred_j
        violation = tf.nn.relu(-d_diff) * tf.cast(proxy_diff > 0, tf.float32)
        penalty = self.weight * tf.reduce_mean(violation)
        self.add_loss(penalty)
        return d_pred  # pass-through so the layer can sit in the main graph


def _dense(units: int) -> keras.layers.Dense:
    return keras.layers.Dense(
        units,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(_L2),
    )


def build_enhanced_ann_r3() -> keras.Model:
    """Build and compile the enhanced ANN R3 (initiation weight + monotonicity)."""
    inputs = keras.Input(shape=(N_FEATURES,), name="input")

    x = _dense(512)(inputs)
    x = keras.layers.BatchNormalization(name="bn_512")(x)
    x = keras.layers.Dropout(0.2, name="drop_512")(x)
    x = _dense(256)(x)
    x = keras.layers.BatchNormalization(name="bn_256")(x)
    x = keras.layers.Dropout(0.2, name="drop_256")(x)
    x = _dense(128)(x)
    x = keras.layers.BatchNormalization(name="bn_128")(x)
    x = _dense(64)(x)
    x = _dense(32)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    current_fi = keras.layers.Lambda(
        lambda t: t[:, _CURRENT_FAILURE_INDEX_FLAT : _CURRENT_FAILURE_INDEX_FLAT + 1],
        name="current_failure_index",
    )(inputs)
    outputs = MonotonicityPenalty(name="monotonicity_penalty")([current_fi, outputs])

    model = keras.Model(inputs=inputs, outputs=outputs, name="enhanced_ann_r3")

    optimizer = keras.optimizers.Adam(
        learning_rate=1e-3,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        clipnorm=1.0,
    )
    model.compile(
        optimizer=optimizer,
        loss=physics_weighted_mse,
        metrics=["mse", "mae"],  # val_mse exposed for EarlyStopping; unaffected by add_loss
    )
    return model


def prepare_inputs(X: np.ndarray) -> np.ndarray:
    """Convert pipeline output (N, 20, 16) → (N, 320)."""
    N = X.shape[0]
    return X.reshape(N, -1).astype("float32")
