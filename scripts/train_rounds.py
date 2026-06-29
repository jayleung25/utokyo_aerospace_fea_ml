"""Run LSTM Rounds 3, 4, and 5 sequentially in one shot.

Usage:
    python scripts/train_rounds.py

Loads processed data once, trains R3 → R4 → R5 back-to-back, then prints
a final comparison table of all rounds vs the professor baseline.
Results and weights are saved under models/saved/lstm_r{3,4,5}/.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.train import run_training
from models import lstm_model
from models.lstm_model import (
    build_lstm_model_r3,
    build_lstm_model_r4,
    build_lstm_model_r5,
)
from pipeline import config

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROCESSED_DIR = os.path.join(_ROOT, "data", "processed")
_SAVE_DIR = os.path.join(_ROOT, "models", "saved")

PROFESSOR_BASELINE_MSE = 3.83e-5
BASELINE_ANN_MSE = 3.455e-5

ROUNDS = [
    {
        "name":                 "lstm_r3",
        "build_fn":             build_lstm_model_r3,
        "batch_size":           256,
        "epochs":               1000,
        "patience":             80,
        "use_lr_schedule":      True,
        "lr_schedule_patience": 25,
    },
    {
        "name":                 "lstm_r4",
        "build_fn":             build_lstm_model_r4,
        "batch_size":           256,
        "epochs":               1200,
        "patience":             100,
        "use_lr_schedule":      True,
        "lr_schedule_patience": 30,
    },
    {
        "name":                 "lstm_r5",
        "build_fn":             build_lstm_model_r5,
        "batch_size":           256,
        "epochs":               1200,
        "patience":             100,
        "use_lr_schedule":      True,
        "lr_schedule_patience": 30,
    },
]


def load_data():
    train = np.load(os.path.join(_PROCESSED_DIR, "train.npz"))
    val   = np.load(os.path.join(_PROCESSED_DIR, "val.npz"))
    test  = np.load(os.path.join(_PROCESSED_DIR, "test.npz"))
    w     = np.load(os.path.join(_PROCESSED_DIR, "weights_train.npy"))
    return (
        train["X"].astype("float32"), train["y"].astype("float32"),
        val["X"].astype("float32"),   val["y"].astype("float32"),
        test["X"].astype("float32"),  test["y"].astype("float32"),
        w.astype("float32"),
    )


def evaluate_model(model, X, y, label):
    """Return test MSE and per-bin MAE dict."""
    y_pred = model.predict(X, verbose=0).flatten()
    mse = float(np.mean((y_pred - y) ** 2))

    interior = np.array(config.D_BIN_EDGES[1:-1], dtype=np.float32)
    bin_idx = np.searchsorted(interior, y, side="right")
    bin_idx = np.clip(bin_idx, 0, len(config.D_BIN_LABELS) - 1)

    bin_mae = {}
    for i, bin_label in enumerate(config.D_BIN_LABELS):
        mask = bin_idx == i
        if mask.sum() > 0:
            bin_mae[bin_label] = float(np.mean(np.abs(y_pred[mask] - y[mask])))

    print(f"\n  [{label}] Test MSE: {mse:.4e}  "
          f"(prof: {PROFESSOR_BASELINE_MSE:.2e}, ANN: {BASELINE_ANN_MSE:.2e})")
    print(f"  Stratified MAE:")
    for bin_label, mae in bin_mae.items():
        marker = " ← transition zone" if bin_label == "transition" else ""
        print(f"    {bin_label:<15} {mae:.6f}{marker}")

    return mse, bin_mae


def print_summary(results):
    print("\n" + "=" * 72)
    print("  FINAL COMPARISON — LSTM Optimization Rounds")
    print("=" * 72)
    header = f"  {'Model':<14} {'Test MSE':>12} {'vs Prof':>9} {'vs ANN':>9}  {'Trans MAE':>10}"
    print(header)
    print("  " + "-" * 68)

    rows = [
        ("professor",     PROFESSOR_BASELINE_MSE, None),
        ("baseline_ann",  BASELINE_ANN_MSE,       None),
    ] + [(r["label"], r["test_mse"], r["bin_mae"]) for r in results]

    for name, mse, bin_mae in rows:
        vs_prof = mse / PROFESSOR_BASELINE_MSE
        vs_ann  = mse / BASELINE_ANN_MSE
        trans   = bin_mae.get("transition", float("nan")) if bin_mae else float("nan")
        trans_s = f"{trans:.6f}" if not (trans != trans) else "  —"
        print(f"  {name:<14} {mse:>12.4e} {vs_prof:>8.2f}x {vs_ann:>8.2f}x  {trans_s:>10}")

    print("=" * 72)
    best = min(results, key=lambda r: r["test_mse"])
    improvement = (1 - best["test_mse"] / PROFESSOR_BASELINE_MSE) * 100
    if improvement > 0:
        print(f"  Best: {best['label']}  —  {improvement:.1f}% lower MSE than professor")
    else:
        print(f"  Best: {best['label']}  —  still above professor baseline")
    print("=" * 72)


def main():
    print("=" * 72)
    print("  UTokyo Aerospace FEA/ML — Sequential LSTM Optimization")
    print(f"  Rounds: {', '.join(r['name'] for r in ROUNDS)}")
    print(f"  Professor baseline: {PROFESSOR_BASELINE_MSE:.2e}")
    print("=" * 72)

    print("\n[data] Loading preprocessed data...")
    train_X, train_y, val_X, val_y, test_X, test_y, weights = load_data()
    print(f"  Train {train_X.shape}  Val {val_X.shape}  Test {test_X.shape}")

    # prepare_inputs is a no-op cast — same function for all LSTM rounds
    prepare_fn = lstm_model.prepare_inputs
    train_X_in = prepare_fn(train_X)
    val_X_in   = prepare_fn(val_X)
    test_X_in  = prepare_fn(test_X)

    results = []
    wall_start = time.perf_counter()

    for cfg in ROUNDS:
        name = cfg["name"]
        print(f"\n{'='*72}")
        print(f"  Training {name}")
        print(f"{'='*72}")

        model = cfg["build_fn"]()
        model.summary(line_length=80)

        t0 = time.perf_counter()
        run_training(
            model=model,
            X_train=train_X_in,
            y_train=train_y,
            X_val=val_X_in,
            y_val=val_y,
            weights_train=weights,
            save_dir=_SAVE_DIR,
            model_name=name,
            batch_size=cfg["batch_size"],
            epochs=cfg["epochs"],
            patience=cfg["patience"],
            use_lr_schedule=cfg["use_lr_schedule"],
            lr_schedule_patience=cfg["lr_schedule_patience"],
        )
        elapsed = time.perf_counter() - t0
        print(f"\n  [{name}] Wall time: {elapsed/60:.1f} min")

        test_mse, bin_mae = evaluate_model(model, test_X_in, test_y, name)
        results.append({"label": name, "test_mse": test_mse, "bin_mae": bin_mae})

    total = time.perf_counter() - wall_start
    print(f"\n[done] Total wall time: {total/60:.1f} min ({total/3600:.2f} hr)")

    print_summary(results)


if __name__ == "__main__":
    main()
