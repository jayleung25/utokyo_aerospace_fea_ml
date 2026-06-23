"""
Unit tests for the data pipeline.

Run with:
    python tests/test_pipeline.py

Uses a 200-row synthetic fixture (no real data required) so tests
run offline and fast. Each stage is tested in isolation.
"""

import sys
import os
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import config, ingest, preprocess, split, weights, export


# ---------------------------------------------------------------------------
# Synthetic data fixture
# ---------------------------------------------------------------------------

def make_synthetic_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Generate a small DataFrame that matches the real CSV schema."""
    rng = np.random.default_rng(seed)
    data = {}
    for col in config.FEATURE_COLS:
        data[col] = rng.standard_normal(n).astype(np.float32)
    # Ensure modeMixity = 0 to pass DCB validation
    for step in range(config.N_TIMESTEPS):
        data[f"h{step}_modeMixity"] = np.zeros(n, dtype=np.float32)
    # Labels spread across all bins: 50 in each of 4 bins
    labels = np.concatenate([
        rng.uniform(0.0, 0.3, 50),
        rng.uniform(0.3, 0.7, 50),
        rng.uniform(0.7, 0.9, 50),
        rng.uniform(0.9, 1.0, 50),
    ])
    rng.shuffle(labels)
    data[config.LABEL_COL] = labels.astype(np.float32)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConfig(unittest.TestCase):
    def test_feature_col_count(self):
        self.assertEqual(len(config.FEATURE_COLS), 160)

    def test_feature_col_naming(self):
        self.assertEqual(config.FEATURE_COLS[0], "h0_failureIndex")
        self.assertEqual(config.FEATURE_COLS[-1], "h19_tractT2")

    def test_split_ratios_sum(self):
        total = sum(config.SPLIT_RATIOS.values())
        self.assertAlmostEqual(total, 1.0, places=10)

    def test_bin_edges_and_labels(self):
        self.assertEqual(len(config.D_BIN_EDGES) - 1, len(config.D_BIN_LABELS))


class TestIngest(unittest.TestCase):
    def setUp(self):
        self.df = make_synthetic_df(200)

    def test_load_validates_columns(self):
        bad = self.df.drop(columns=["h0_failureIndex"])
        with self.assertRaises(ValueError):
            ingest._validate_columns(bad)

    def test_load_catches_nulls(self):
        bad = self.df.copy()
        bad.loc[0, "h0_separN"] = np.nan
        with self.assertRaises(ValueError):
            ingest._validate_no_nulls(bad)

    def test_load_catches_out_of_range_labels(self):
        bad = self.df.copy()
        bad.loc[0, config.LABEL_COL] = 1.5
        with self.assertRaises(ValueError):
            ingest._validate_label_range(bad)

    def test_valid_df_passes_all_checks(self):
        ingest._validate_columns(self.df)
        ingest._validate_no_nulls(self.df)
        ingest._validate_label_range(self.df)
        ingest._validate_mode_mixity(self.df)


class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.df = make_synthetic_df(200)
        self.X_raw, self.y = preprocess.extract_xy(self.df)

    def test_extract_xy_shapes(self):
        self.assertEqual(self.X_raw.shape, (200, 160))
        self.assertEqual(self.y.shape, (200,))

    def test_scale_features_fit(self):
        X_scaled, scaler = preprocess.scale_features(self.X_raw)
        self.assertEqual(X_scaled.shape, (200, 160))
        # After scaling, mean should be near 0 and std near 1 per feature
        self.assertAlmostEqual(float(X_scaled.mean()), 0.0, places=4)

    def test_scale_features_apply_no_fit(self):
        _, scaler = preprocess.scale_features(self.X_raw)
        X_scaled2, scaler2 = preprocess.scale_features(self.X_raw, scaler=scaler)
        self.assertEqual(X_scaled2.shape, (200, 160))
        self.assertIs(scaler, scaler2)

    def test_reshape_temporal(self):
        X_scaled, _ = preprocess.scale_features(self.X_raw)
        X_temporal = preprocess.reshape_temporal(X_scaled)
        self.assertEqual(X_temporal.shape, (200, 20, 8))
        # After reversal: index 0 must hold h19 values (flat positions 152..159)
        # and index 19 must hold h0 values (flat positions 0..7).
        np.testing.assert_allclose(X_temporal[0, 0, :], X_scaled[0, 152:160],
                                   err_msg="index 0 should be h19 (oldest)")
        np.testing.assert_allclose(X_temporal[0, 19, :], X_scaled[0, 0:8],
                                   err_msg="index 19 should be h0 (most recent)")

    def test_compute_deltas_shape_and_zero_pad(self):
        X_scaled, _ = preprocess.scale_features(self.X_raw)
        X_temporal = preprocess.reshape_temporal(X_scaled)
        deltas = preprocess.compute_deltas(X_temporal)
        self.assertEqual(deltas.shape, (200, 20, 8))
        # First timestep delta should be all zeros (pad)
        np.testing.assert_array_equal(deltas[:, 0, :], 0.0)

    def test_build_features_output_shape(self):
        X_out, scaler = preprocess.build_features(self.X_raw)
        self.assertEqual(X_out.shape, (200, 20, 16))

    def test_no_data_leakage_between_splits(self):
        """Scaler fit on half the data must not be re-fit on the other half."""
        X_train = self.X_raw[:100]
        X_val = self.X_raw[100:]
        X_t, scaler = preprocess.build_features(X_train)
        X_v, scaler2 = preprocess.build_features(X_val, scaler=scaler)
        self.assertIs(scaler, scaler2)


class TestSplit(unittest.TestCase):
    def setUp(self):
        self.df = make_synthetic_df(200)

    def test_split_sizes(self):
        train_df, val_df, test_df = split.stratified_split(self.df)
        total = len(train_df) + len(val_df) + len(test_df)
        self.assertEqual(total, 200)
        # Train should be ~80%
        self.assertGreater(len(train_df), 140)
        self.assertLess(len(train_df), 175)

    def test_all_bins_in_every_split(self):
        train_df, val_df, test_df = split.stratified_split(self.df)
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            bins = pd.cut(
                df[config.LABEL_COL],
                bins=config.D_BIN_EDGES,
                labels=config.D_BIN_LABELS,
                right=False,
            )
            for label in config.D_BIN_LABELS:
                count = (bins == label).sum()
                self.assertGreater(count, 0, f"Bin '{label}' missing from {name} split")

    def test_no_overlap_between_splits(self):
        train_df, val_df, test_df = split.stratified_split(self.df)
        # Verify no row appears in more than one split by checking that the
        # concatenation of all three has the same row count as the original df
        # (reset_index means integer indices repeat, so we compare by values)
        combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
        self.assertEqual(len(combined), len(self.df))
        # Also verify the combined set reconstructs the full label distribution
        self.assertAlmostEqual(
            combined[config.LABEL_COL].sum(),
            self.df[config.LABEL_COL].sum(),
            places=3,
        )


class TestWeights(unittest.TestCase):
    def test_weights_mean_is_one(self):
        y = np.concatenate([
            np.random.uniform(0.0, 0.3, 50),
            np.random.uniform(0.3, 0.7, 50),
            np.random.uniform(0.7, 0.9, 50),
            np.random.uniform(0.9, 1.0, 200),
        ])
        w = weights.compute_sample_weights(y)
        self.assertAlmostEqual(float(w.mean()), 1.0, places=5)

    def test_rare_bin_gets_higher_weight(self):
        y = np.concatenate([
            np.random.uniform(0.0, 0.3, 10),   # rare
            np.random.uniform(0.9, 1.0, 190),  # abundant
        ])
        w = weights.compute_sample_weights(y)
        rare_mean = w[y < 0.3].mean()
        abundant_mean = w[y > 0.9].mean()
        self.assertGreater(rare_mean, abundant_mean)

    def test_weights_positive(self):
        y = np.random.uniform(0, 1, 100)
        w = weights.compute_sample_weights(y)
        self.assertTrue((w > 0).all())


class TestExport(unittest.TestCase):
    def setUp(self):
        self.df = make_synthetic_df(200)
        train_df, val_df, test_df = split.stratified_split(self.df)
        X_raw, y = preprocess.extract_xy(train_df)
        X_val_raw, y_val = preprocess.extract_xy(val_df)
        X_test_raw, y_test = preprocess.extract_xy(test_df)
        self.train_X, self.scaler = preprocess.build_features(X_raw)
        self.val_X, _ = preprocess.build_features(X_val_raw, self.scaler)
        self.test_X, _ = preprocess.build_features(X_test_raw, self.scaler)
        self.train_y = y
        self.val_y = y_val
        self.test_y = y_test
        self.train_w = weights.compute_sample_weights(y)
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

    def test_export_creates_all_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            export.export_all(
                tmpdir,
                self.train_X, self.train_y,
                self.val_X, self.val_y,
                self.test_X, self.test_y,
                self.train_w, self.scaler,
                self.train_df, self.val_df, self.test_df,
            )
            expected = ["train.npz", "val.npz", "test.npz",
                        "weights_train.npy", "scaler.json", "pipeline_config.json"]
            for fname in expected:
                self.assertTrue(os.path.exists(os.path.join(tmpdir, fname)), f"Missing {fname}")

    def test_npz_shapes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            export.export_all(
                tmpdir,
                self.train_X, self.train_y,
                self.val_X, self.val_y,
                self.test_X, self.test_y,
                self.train_w, self.scaler,
                self.train_df, self.val_df, self.test_df,
            )
            # Load and immediately extract arrays, then close the NpzFile handle.
            # On Windows, keeping the handle open prevents TemporaryDirectory cleanup.
            with np.load(os.path.join(tmpdir, "train.npz")) as train_data:
                X_shape = train_data["X"].shape
            self.assertEqual(len(X_shape), 3)
            self.assertEqual(X_shape[1], 20)
            self.assertEqual(X_shape[2], 16)

    def test_scaler_json_readable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            export.export_all(
                tmpdir,
                self.train_X, self.train_y,
                self.val_X, self.val_y,
                self.test_X, self.test_y,
                self.train_w, self.scaler,
                self.train_df, self.val_df, self.test_df,
            )
            with open(os.path.join(tmpdir, "scaler.json")) as f:
                scaler_data = json.load(f)
            self.assertIn("mean", scaler_data)
            self.assertIn("scale", scaler_data)
            self.assertEqual(len(scaler_data["mean"]), 160)


if __name__ == "__main__":
    unittest.main(verbosity=2)
