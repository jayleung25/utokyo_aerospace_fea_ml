"""
Data pipeline entry point.

Usage:
    python scripts/run_pipeline.py --data data/raw_70000.csv --output data/processed/

Runs all pipeline stages in order:
  1. Ingest & validate
  2. Stratified split
  3. Preprocess (scale → temporal reshape → delta features)
  4. Compute sample weights
  5. Build tf.data datasets  [skipped if TensorFlow not installed]
  6. Export all artifacts
"""

import argparse
import sys
import os
import time

# Allow running from the project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import ingest, split, preprocess, weights, export
from pipeline import dataset as dataset_module

# TF availability is determined inside dataset.py at import time
_TF_AVAILABLE = dataset_module._TF_AVAILABLE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CZM Surrogate — Data Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to raw CSV (e.g. data/raw_70000.csv)",
    )
    parser.add_argument(
        "--output",
        default="data/processed",
        help="Directory for processed artifacts",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size used when constructing tf.data datasets",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.perf_counter()

    print("=" * 60)
    print("  UTokyo Aerospace FEA/ML — Data Pipeline")
    print("=" * 60)

    # Stage 1: Ingest
    print("\n--- Stage 1: Ingest & Validate ---")
    df = ingest.load_raw(args.data)

    # Stage 2: Split (before preprocessing so scaler is fit on train only)
    print("--- Stage 2: Stratified Split ---")
    train_df, val_df, test_df = split.stratified_split(df)

    # Stage 3: Preprocess
    print("--- Stage 3: Preprocess ---")
    train_X_raw, train_y = preprocess.extract_xy(train_df)
    val_X_raw,   val_y   = preprocess.extract_xy(val_df)
    test_X_raw,  test_y  = preprocess.extract_xy(test_df)

    # Fit scaler on train only, apply to val/test (no data leakage)
    train_X, scaler = preprocess.build_features(train_X_raw, scaler=None)
    val_X,   _      = preprocess.build_features(val_X_raw,   scaler=scaler)
    test_X,  _      = preprocess.build_features(test_X_raw,  scaler=scaler)

    # Stage 4: Sample weights (training only)
    print("--- Stage 4: Sample Weights ---")
    train_weights = weights.compute_sample_weights(train_y)

    # Stage 5: tf.data datasets (optional — requires TensorFlow)
    train_ds = val_ds = test_ds = None
    if _TF_AVAILABLE:
        print("--- Stage 5: tf.data Datasets ---")
        train_ds = dataset_module.make_train_dataset(train_X, train_y, train_weights, args.batch_size)
        val_ds   = dataset_module.make_eval_dataset(val_X,   val_y,   args.batch_size, name="Val")
        test_ds  = dataset_module.make_eval_dataset(test_X,  test_y,  args.batch_size, name="Test")
    else:
        print("--- Stage 5: tf.data Datasets --- [SKIPPED — install TensorFlow to enable]")

    # Stage 6: Export
    print("--- Stage 6: Export ---")
    export.export_all(
        out_dir=args.output,
        train_X=train_X, train_y=train_y,
        val_X=val_X,     val_y=val_y,
        test_X=test_X,   test_y=test_y,
        weights=train_weights,
        scaler=scaler,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    elapsed = time.perf_counter() - t0
    print("=" * 60)
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Artifacts written to: {os.path.abspath(args.output)}/")
    print(f"  Feature tensor shape: (N, 20, 16)  — 8 absolute + 8 delta per step")
    print(f"  Temporal order: index 0 = h19 (oldest) -> index 19 = h0 (newest)")
    if not _TF_AVAILABLE:
        print("  NOTE: install tensorflow to also build tf.data datasets")
    print("=" * 60)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    main()
