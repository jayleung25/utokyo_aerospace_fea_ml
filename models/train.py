"""Shared training loop, checkpointing, and diagnostic callbacks.

All three model tracks call run_training() with their own model instance
and input arrays. The StratifiedMAECallback surfaces per-bin MAE every
10 epochs so transition-zone degradation is visible during training rather
than hidden inside aggregate val_loss.
"""

from __future__ import annotations

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from pipeline import config


class StratifiedMAECallback(keras.callbacks.Callback):
    """Logs per-damage-bin MAE on the validation set every N epochs."""

    def __init__(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        log_every: int = 10,
    ) -> None:
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.log_every = log_every
        self._interior = np.array(config.D_BIN_EDGES[1:-1], dtype=np.float32)

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if (epoch + 1) % self.log_every != 0:
            return
        y_pred = self.model.predict(self.X_val, verbose=0).flatten()
        bin_idx = np.searchsorted(self._interior, self.y_val, side="right")
        bin_idx = np.clip(bin_idx, 0, len(config.D_BIN_LABELS) - 1)
        parts: list[str] = []
        for i, label in enumerate(config.D_BIN_LABELS):
            mask = bin_idx == i
            if mask.sum() > 0:
                mae = float(np.mean(np.abs(y_pred[mask] - self.y_val[mask])))
                parts.append(f"{label}={mae:.5f}")
        print(f"\n  [epoch {epoch + 1}] Stratified MAE — " + " | ".join(parts))


def run_training(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    weights_train: np.ndarray | None = None,
    save_dir: str = "models/saved",
    model_name: str = "model",
    batch_size: int = 128,
    epochs: int = 200,
    patience: int = 20,
    use_lr_schedule: bool = False,
    lr_schedule_patience: int = 10,
    early_stopping_monitor: str = "val_loss",
) -> keras.callbacks.History:
    """Fit a model with EarlyStopping, checkpointing, and stratified MAE logging.

    Args:
        model:                  Compiled Keras model.
        X_train:                Training features in the shape expected by this model.
        y_train:                Training labels (damage D).
        X_val:                  Validation features (same shape as X_train).
        y_val:                  Validation labels.
        weights_train:          Per-sample loss weights from pipeline/weights.py.
                                Pass None for the professor-replication baseline.
        save_dir:               Root directory for saved weights and logs.
        model_name:             Subdirectory name under save_dir.
        batch_size:             Mini-batch size.
        epochs:                 Maximum number of training epochs.
        patience:               EarlyStopping patience (epochs without improvement).
        use_lr_schedule:        If True, adds ReduceLROnPlateau. Disabled for baseline
                                replication so LR exactly matches the professor's setup.
        lr_schedule_patience:   ReduceLROnPlateau patience (epochs before halving LR).
                                LSTMs need a higher value (20) than dense ANNs (10) to
                                avoid killing the LR before the model can converge.
        early_stopping_monitor: Metric for EarlyStopping and ReduceLROnPlateau.
                                Use "val_mse" for models with a custom loss + explicit
                                mse metric, so callbacks track pure MSE rather than
                                val_loss (custom loss + L2 noise). Default "val_loss"
                                preserves behaviour for baseline and LSTM rounds.

    Returns:
        Keras History object with full training log.
    """
    out_dir = os.path.join(save_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)

    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor=early_stopping_monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(out_dir, "best_model.weights.h5"),
            monitor=early_stopping_monitor,
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
        keras.callbacks.CSVLogger(
            os.path.join(out_dir, "train.log"),
            append=False,
        ),
        StratifiedMAECallback(X_val, y_val),
    ]

    if use_lr_schedule:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=early_stopping_monitor,
                factor=0.5,
                patience=lr_schedule_patience,
                min_lr=1e-6,
                verbose=1,
            )
        )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        sample_weight=weights_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Use val_mse when available (pure MSE, comparable to professor baseline).
    # Fall back to val_loss for models without explicit mse metric.
    mse_key = "val_mse" if "val_mse" in history.history else "val_loss"
    best_val_mse = min(history.history[mse_key])
    label = "val_mse" if mse_key == "val_mse" else "val_loss≈mse"
    print(f"\n[train] {model_name} — best {label}: {best_val_mse:.6e}")
    print(f"[train] Professor baseline: 3.83e-05")
    ratio = best_val_mse / 3.83e-5
    if ratio < 1.0:
        print(f"[train] Improvement over professor: {(1 - ratio) * 100:.1f}% lower MSE")
    else:
        print(f"[train] vs professor: {ratio:.2f}x (>{ratio:.2f}x means worse)")
    print(f"[train] Artifacts saved to: {out_dir}/")
    return history
