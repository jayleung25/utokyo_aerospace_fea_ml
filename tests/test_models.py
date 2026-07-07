"""
Unit tests for model-building and custom-loss code (models/*.py).

Run with:
    python tests/test_models.py

Uses synthetic tensors (no real data required) so tests run offline and
fast. Focused on the enhanced_ann_r3 round: the initiation/transition
zone-weighted loss and the pairwise monotonicity penalty, since neither
had any test coverage before this round.
"""

import sys
import os
import unittest

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.enhanced_ann_r3 import (
    physics_weighted_mse,
    MonotonicityPenalty,
    build_enhanced_ann_r3,
    prepare_inputs,
    N_FEATURES,
    _CURRENT_FAILURE_INDEX_FLAT,
    _INITIATION_HIGH,
    _INITIATION_MULT,
    _TRANSITION_LOW,
    _TRANSITION_HIGH,
    _TRANSITION_MULT,
)


class TestPhysicsWeightedMSE(unittest.TestCase):
    def test_zero_error_gives_zero_loss(self):
        y = np.array([0.1, 0.5, 0.95], dtype=np.float32)
        loss = physics_weighted_mse(y, y)
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_initiation_zone_weighted_more_than_failed_zone(self):
        # Same absolute error (0.1) in each zone; initiation should incur
        # a larger loss contribution than the failed zone (weight 1x).
        y_true_init = np.array([0.1], dtype=np.float32)
        y_true_failed = np.array([0.99], dtype=np.float32)
        y_pred_init = np.array([0.2], dtype=np.float32)
        y_pred_failed = np.array([1.09], dtype=np.float32)

        loss_init = float(physics_weighted_mse(y_true_init, y_pred_init))
        loss_failed = float(physics_weighted_mse(y_true_failed, y_pred_failed))
        self.assertAlmostEqual(loss_init / loss_failed, _INITIATION_MULT, places=4)

    def test_transition_zone_weight_unchanged_from_r2(self):
        y_true = np.array([0.5], dtype=np.float32)
        y_pred = np.array([0.6], dtype=np.float32)
        y_true_failed = np.array([0.99], dtype=np.float32)
        y_pred_failed = np.array([1.09], dtype=np.float32)

        loss_trans = float(physics_weighted_mse(y_true, y_pred))
        loss_failed = float(physics_weighted_mse(y_true_failed, y_pred_failed))
        self.assertAlmostEqual(loss_trans / loss_failed, _TRANSITION_MULT, places=4)

    def test_zone_edges_are_half_open_and_non_overlapping(self):
        # D exactly at 0.3 must be transition (2nd bin), not initiation.
        y_at_edge = np.array([_INITIATION_HIGH], dtype=np.float32)
        y_pred = np.array([_INITIATION_HIGH + 0.1], dtype=np.float32)
        y_failed = np.array([0.99], dtype=np.float32)
        y_pred_failed = np.array([1.09], dtype=np.float32)

        loss_edge = float(physics_weighted_mse(y_at_edge, y_pred))
        loss_failed = float(physics_weighted_mse(y_failed, y_pred_failed))
        self.assertAlmostEqual(loss_edge / loss_failed, _TRANSITION_MULT, places=4)


class TestMonotonicityPenalty(unittest.TestCase):
    def test_no_penalty_when_already_monotonic(self):
        proxy = tf.constant([[0.0], [1.0], [2.0]], dtype=tf.float32)
        d_pred = tf.constant([[0.1], [0.2], [0.3]], dtype=tf.float32)  # non-decreasing with proxy
        layer = MonotonicityPenalty(weight=1.0)
        out = layer([proxy, d_pred])
        np.testing.assert_allclose(out.numpy(), d_pred.numpy())
        self.assertAlmostEqual(float(layer.losses[0]), 0.0, places=6)

    def test_penalty_positive_when_inverted(self):
        proxy = tf.constant([[0.0], [1.0], [2.0]], dtype=tf.float32)
        d_pred = tf.constant([[0.9], [0.5], [0.1]], dtype=tf.float32)  # fully inverted
        layer = MonotonicityPenalty(weight=1.0)
        layer([proxy, d_pred])
        self.assertGreater(float(layer.losses[0]), 0.0)

    def test_penalty_scales_with_weight(self):
        proxy = tf.constant([[0.0], [1.0]], dtype=tf.float32)
        d_pred = tf.constant([[0.9], [0.1]], dtype=tf.float32)
        layer_a = MonotonicityPenalty(weight=1e-3)
        layer_b = MonotonicityPenalty(weight=1e-2)
        layer_a([proxy, d_pred])
        layer_b([proxy, d_pred])
        self.assertAlmostEqual(
            float(layer_b.losses[0]) / float(layer_a.losses[0]), 10.0, places=4
        )

    def test_output_is_pass_through(self):
        proxy = tf.constant([[0.0], [1.0]], dtype=tf.float32)
        d_pred = tf.constant([[0.3], [0.7]], dtype=tf.float32)
        out = MonotonicityPenalty()([proxy, d_pred])
        np.testing.assert_allclose(out.numpy(), d_pred.numpy())


class TestBuildEnhancedAnnR3(unittest.TestCase):
    def test_builds_and_compiles(self):
        model = build_enhanced_ann_r3()
        self.assertEqual(model.input_shape, (None, N_FEATURES))
        self.assertEqual(model.output_shape, (None, 1))

    def test_forward_pass_shape_and_range(self):
        model = build_enhanced_ann_r3()
        X = np.random.default_rng(0).standard_normal((8, N_FEATURES)).astype(np.float32)
        y_pred = model.predict(X, verbose=0)
        self.assertEqual(y_pred.shape, (8, 1))
        # sigmoid output must stay in [0, 1]
        self.assertTrue(np.all(y_pred >= 0.0) and np.all(y_pred <= 1.0))

    def test_current_failure_index_flat_offset(self):
        # timestep 19 (newest, chronological) * 16 features + feature 0 (failureIndex)
        self.assertEqual(_CURRENT_FAILURE_INDEX_FLAT, 19 * 16 + 0)

    def test_fit_one_step_no_errors(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((32, N_FEATURES)).astype(np.float32)
        y = rng.uniform(0.0, 1.0, 32).astype(np.float32)
        w = np.ones(32, dtype=np.float32)
        model = build_enhanced_ann_r3()
        history = model.fit(X, y, sample_weight=w, batch_size=16, epochs=1, verbose=0)
        self.assertIn("mse", history.history)


class TestPrepareInputs(unittest.TestCase):
    def test_flattens_to_320(self):
        X = np.random.default_rng(0).standard_normal((5, 20, 16)).astype(np.float32)
        X_flat = prepare_inputs(X)
        self.assertEqual(X_flat.shape, (5, 320))
        np.testing.assert_allclose(X_flat.reshape(5, 20, 16), X)


if __name__ == "__main__":
    unittest.main(verbosity=2)
