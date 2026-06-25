"""CLI entry point for training CZM surrogate models.

Usage:
    python scripts/train_model.py --model baseline
    python scripts/train_model.py --model enhanced
    python scripts/train_model.py --model lstm

Each invocation:
  1. Loads preprocessed data from data/processed/
  2. Prepares inputs in the format expected by the chosen model
  3. Trains via the shared run_training() loop (EarlyStopping, checkpointing)
  4. Prints final val MSE and per-bin MAE summary for comparison

After training all three models, open notebooks/02_model_comparison.ipynb
to generate the visualisation suite (Plots 1–5).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Allow running from the project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.train import run_training
from models import baseline_ann, enhanced_ann, lstm_model
from pipeline import config

# ---------------------------------------------------------------------------
# Default paths (relative to project root, not to scripts/)
# ---------------------------------------------------------------------------

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROCESSED_DIR = os.path.join(_ROOT, "data", "processed")
_SAVE_DIR = os.path.join(_ROOT, "models", "saved")

# ---------------------------------------------------------------------------
# Per-model configuration
# ---------------------------------------------------------------------------

MODEL_CONFIGS: dict[str, dict] = {
    "baseline": {
        "build_fn":       baseline_ann.build_baseline_ann,
        "prepare_fn":     baseline_ann.prepare_inputs,
        "use_weights":    False,   # professor used no sample weighting
        "batch_size":     128,
        "epochs":         200,
        "patience":       20,
        "use_lr_schedule": False,  # professor used fixed Adam lr=5e-4
    },
    "enhanced": {
        "build_fn":       enhanced_ann.build_enhanced_ann,
        "prepare_fn":     enhanced_ann.prepare_inputs,
        "use_weights":    True,
        "batch_size":     256,
        "epochs":         300,
        "patience":       30,
        "use_lr_schedule": True,
    },
    "lstm": {
        "build_fn":           lstm_model.build_lstm_model,
        "prepare_fn":         lstm_model.prepare_inputs,
        "use_weights":        True,
        "batch_size":         128,
        "epochs":             600,
        "patience":           50,
        "use_lr_schedule":    True,
        "lr_schedule_patience": 20,
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_data(processed_dir: str) -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray,
]:
    """Load all processed arrays from data/processed/."""
    train = np.load(os.path.join(processed_dir, "train.npz"))
    val   = np.load(os.path.join(processed_dir, "val.npz"))
    test  = np.load(os.path.join(processed_dir, "test.npz"))
    w     = np.load(os.path.join(processed_dir, "weights_train.npy"))
    return (
        train["X"].astype("float32"), train["y"].astype("float32"),
        val["X"].astype("float32"),   val["y"].astype("float32"),
        test["X"].astype("float32"),  test["y"].astype("float32"),
        w.astype("float32"),
    )


def print_stratified_mae(
    model: object,
    X_val: np.ndarray,
    y_val: np.ndarray,
    prepare_fn,
    label: str = "val",
) -> None:
    """Print per-bin MAE for the trained model on the given split."""
    X_in = prepare_fn(X_val)
    y_pred = model.predict(X_in, verbose=0).flatten()
    interior = np.array(config.D_BIN_EDGES[1:-1], dtype=np.float32)
    bin_idx = np.searchsorted(interior, y_val, side="right")
    bin_idx = np.clip(bin_idx, 0, len(config.D_BIN_LABELS) - 1)

    print(f"\nFinal stratified MAE on {label} set:")
    print(f"  {'Bin':<15} {'N':>8}  {'MAE':>10}")
    print(f"  {'-'*38}")
    for i, bin_label in enumerate(config.D_BIN_LABELS):
        mask = bin_idx == i
        n = int(mask.sum())
        if n > 0:
            mae = float(np.mean(np.abs(y_pred[mask] - y_val[mask])))
            marker = " ← transition zone" if bin_label == "transition" else ""
            print(f"  {bin_label:<15} {n:>8,}  {mae:>10.6f}{marker}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CZM Surrogate — Model Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        required=True,
        help="Which model to train",
    )
    parser.add_argument(
        "--data-dir",
        default=_PROCESSED_DIR,
        help="Path to data/processed/ directory",
    )
    parser.add_argument(
        "--save-dir",
        default=_SAVE_DIR,
        help="Root directory for saved model weights and logs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MODEL_CONFIGS[args.model]

    print("=" * 65)
    print(f"  UTokyo Aerospace FEA/ML — Training: {args.model}")
    print(f"  Professor baseline val MSE target: 3.83e-05")
    print("=" * 65)

    # Load data
    print("\n[data] Loading preprocessed data from:", args.data_dir)
    train_X, train_y, val_X, val_y, test_X, test_y, weights = load_data(
        args.data_dir
    )
    print(
        f"  Train: {train_X.shape}  Val: {val_X.shape}  Test: {test_X.shape}"
    )

    # Prepare model-specific input shapes
    prepare_fn = cfg["prepare_fn"]
    train_X_in = prepare_fn(train_X)
    val_X_in   = prepare_fn(val_X)
    print(f"  Model input shape: {train_X_in.shape}")

    # Build model
    print(f"\n[model] Building {args.model}...")
    model = cfg["build_fn"]()
    model.summary()

    train_weights = weights if cfg["use_weights"] else None

    # Train
    t0 = time.perf_counter()
    run_training(
        model=model,
        X_train=train_X_in,
        y_train=train_y,
        X_val=val_X_in,
        y_val=val_y,
        weights_train=train_weights,
        save_dir=args.save_dir,
        model_name=args.model,
        batch_size=cfg["batch_size"],
        epochs=cfg["epochs"],
        patience=cfg["patience"],
        use_lr_schedule=cfg["use_lr_schedule"],
        lr_schedule_patience=cfg.get("lr_schedule_patience", 10),
    )
    elapsed = time.perf_counter() - t0
    print(f"\n[train] Total wall time: {elapsed:.1f}s")

    # Final diagnostics
    print_stratified_mae(model, val_X, val_y, prepare_fn, label="val")

    print("=" * 65)
    print("  Done. Next steps:")
    print("    1. Train remaining models (baseline / enhanced / lstm)")
    print("    2. Open notebooks/02_model_comparison.ipynb to visualise")
    print("=" * 65)


if __name__ == "__main__":
    main()
