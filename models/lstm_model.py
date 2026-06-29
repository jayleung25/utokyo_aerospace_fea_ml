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

Round 3 changes vs Round 2:
  - Feature selection Lambda layer: drops 10 structural-zero DCB-only
    columns (modeMixity, separT1/T2, tractT1/T2 and their deltas); keeps
    6 live features [failureIndex, separN, tractN, and their deltas].
    Input shrinks (N,20,16)→(N,20,6) inside the model; pipeline unchanged.
  - LSTM units scaled up 128→192, 64→96 to compensate for denser input.
  - L2 1e-4 → 5e-5: val_loss was ~9× test MSE, indicating over-regularisation.
  - Dropout 0.2 → 0.15: paired reduction to avoid double-regularising.
  - Switched from Sequential to Functional API (Lambda requires named tensors).

Round 4 changes vs Round 3:
  - Bidirectional(LSTM(128)) on first layer: all 20 steps are available at
    inference, so a backward pass encodes remaining loading capacity — a
    physically meaningful signal for damage near peak load.
  - AttentionPooling: soft attention over all 20 timesteps replaces the
    final-hidden-state-only readout; lets the model weight the step where
    failureIndex first crosses 1.0 (damage initiation signal).
  - Second LSTM keeps return_sequences=True to feed all steps to attention.

Round 5 changes vs Round 4:
  - Huber loss (delta=0.05) replaces MSE: quadratic for |error| < 0.05
    (precision-sensitive in the transition zone), linear beyond (stops
    D≈1 majority from saturating gradients).
  - "mse" added as explicit metric so val_mse is comparable across rounds.

Same sample weights and LR schedule as Track B (patience raised in
train_model.py to give each LR level more time to converge).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

_L2 = 1e-4

# Indices of the 6 live features in the (N, 20, 16) pipeline tensor.
# The remaining 10 are structural zeros in the DCB-only dataset.
_ACTIVE_FEATURES_R3 = [0, 2, 5, 8, 10, 13]  # failureIndex, separN, tractN + deltas
_L2_R3 = 5e-5


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


def build_lstm_model_r3() -> keras.Model:
    """Round 3: feature selection + scaled units + reduced L2."""
    inputs = keras.Input(shape=(20, 16), name="input")

    x = keras.layers.Lambda(
        lambda t: tf.gather(t, _ACTIVE_FEATURES_R3, axis=2),
        name="feature_select",
    )(inputs)  # (batch, 20, 6)

    x = keras.layers.LSTM(
        192,
        return_sequences=True,
        recurrent_dropout=0.1,
        kernel_regularizer=keras.regularizers.l2(_L2_R3),
        recurrent_regularizer=keras.regularizers.l2(_L2_R3),
        name="lstm_192",
    )(x)
    x = keras.layers.LayerNormalization(name="ln_192")(x)
    x = keras.layers.Dropout(0.15, name="drop_192")(x)

    x = keras.layers.LSTM(
        96,
        return_sequences=False,
        recurrent_dropout=0.1,
        kernel_regularizer=keras.regularizers.l2(_L2_R3),
        recurrent_regularizer=keras.regularizers.l2(_L2_R3),
        name="lstm_96",
    )(x)
    x = keras.layers.LayerNormalization(name="ln_96")(x)
    x = keras.layers.Dropout(0.15, name="drop_96")(x)

    x = keras.layers.Dense(
        96,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(_L2_R3),
        name="dense_96",
    )(x)
    x = keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(_L2_R3),
        name="dense_32",
    )(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="lstm_model_r3")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=0.5),
        loss="mse",
        metrics=["mae"],
    )
    return model


class AttentionPooling(keras.layers.Layer):
    """Dot-product attention over T timesteps → weighted sum of hidden states."""

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attn_W",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, x):
        scores = tf.squeeze(tf.matmul(x, self.W), axis=-1)  # (batch, T)
        weights = tf.nn.softmax(scores, axis=-1)             # (batch, T)
        return tf.reduce_sum(x * tf.expand_dims(weights, -1), axis=1)  # (batch, H)


def build_lstm_model_r4() -> keras.Model:
    """Round 4: Bidirectional LSTM + attention pooling over all timesteps."""
    inputs = keras.Input(shape=(20, 16), name="input")

    x = keras.layers.Lambda(
        lambda t: tf.gather(t, _ACTIVE_FEATURES_R3, axis=2),
        name="feature_select",
    )(inputs)  # (batch, 20, 6)

    x = keras.layers.Bidirectional(
        keras.layers.LSTM(
            128,
            return_sequences=True,
            recurrent_dropout=0.1,
            kernel_regularizer=keras.regularizers.l2(_L2_R3),
            recurrent_regularizer=keras.regularizers.l2(_L2_R3),
        ),
        name="bilstm_128",
    )(x)  # (batch, 20, 256)
    x = keras.layers.LayerNormalization(name="ln_bi")(x)
    x = keras.layers.Dropout(0.15, name="drop_bi")(x)

    x = keras.layers.LSTM(
        96,
        return_sequences=True,
        recurrent_dropout=0.1,
        kernel_regularizer=keras.regularizers.l2(_L2_R3),
        recurrent_regularizer=keras.regularizers.l2(_L2_R3),
        name="lstm_96",
    )(x)  # (batch, 20, 96)
    x = keras.layers.LayerNormalization(name="ln_96")(x)
    x = keras.layers.Dropout(0.15, name="drop_96")(x)

    x = AttentionPooling(name="attn_pool")(x)  # (batch, 96)

    x = keras.layers.Dense(
        96,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(_L2_R3),
        name="dense_96",
    )(x)
    x = keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(_L2_R3),
        name="dense_32",
    )(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="lstm_model_r4")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=0.5),
        loss="mse",
        metrics=["mse", "mae"],
    )
    return model


def _huber_loss(y_true, y_pred):
    return tf.keras.losses.huber(y_true, y_pred, delta=0.05)


def build_lstm_model_r5() -> keras.Model:
    """Round 5: R4 architecture with Huber loss (delta=0.05) for transition-zone precision."""
    inputs = keras.Input(shape=(20, 16), name="input")

    x = keras.layers.Lambda(
        lambda t: tf.gather(t, _ACTIVE_FEATURES_R3, axis=2),
        name="feature_select",
    )(inputs)

    x = keras.layers.Bidirectional(
        keras.layers.LSTM(
            128,
            return_sequences=True,
            recurrent_dropout=0.1,
            kernel_regularizer=keras.regularizers.l2(_L2_R3),
            recurrent_regularizer=keras.regularizers.l2(_L2_R3),
        ),
        name="bilstm_128",
    )(x)
    x = keras.layers.LayerNormalization(name="ln_bi")(x)
    x = keras.layers.Dropout(0.15, name="drop_bi")(x)

    x = keras.layers.LSTM(
        96,
        return_sequences=True,
        recurrent_dropout=0.1,
        kernel_regularizer=keras.regularizers.l2(_L2_R3),
        recurrent_regularizer=keras.regularizers.l2(_L2_R3),
        name="lstm_96",
    )(x)
    x = keras.layers.LayerNormalization(name="ln_96")(x)
    x = keras.layers.Dropout(0.15, name="drop_96")(x)

    x = AttentionPooling(name="attn_pool")(x)

    x = keras.layers.Dense(
        96,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(_L2_R3),
        name="dense_96",
    )(x)
    x = keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(_L2_R3),
        name="dense_32",
    )(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="lstm_model_r5")
    # Use explicit mse metric so val_mse is comparable to MSE-trained rounds
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=0.5),
        loss=_huber_loss,
        metrics=["mse", "mae"],
    )
    return model


def prepare_inputs(X: np.ndarray) -> np.ndarray:
    """Pipeline output (N, 20, 16) is already the correct shape; just cast."""
    return X.astype("float32")
