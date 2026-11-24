"""B's non-noise evaluation, order-unit, shared-input and adapter contracts."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score

from algorithms import dbscan, hierarchical
from analysis.b_eda import aggregate_orders, monthly_summary
from experiments.b_runner import (compare_on_common_customers, evaluate_partition,
                                  load_shared_matrix, original_scale_profiles, select_dbscan)
from preprocessing.feature_engineering import FEATURE_COLUMNS


class BWorkTests(unittest.TestCase):
    def test_adapters_recover_known_groups_without_changing_coordinates(self):
        X, truth = make_blobs(n_samples=80, n_features=6, centers=3, cluster_std=.08, random_state=17)
        original = X.copy()
        for method in ["ward", "complete", "average"]:
            result = hierarchical.fit_predict(X, {"n_clusters": 3, "linkage": method})
            self.assertGreater(adjusted_rand_score(truth, result.labels), .99)
            np.testing.assert_array_equal(X, original)
            self.assertEqual(result.diagnostics["linkage_matrix"].shape, (79, 4))
        X = np.vstack([X, np.full(6, 1000.)])
        result = dbscan.fit_predict(X, {"eps": .6, "min_samples": 5})
        self.assertEqual(result.labels[-1], -1)
        self.assertGreater(adjusted_rand_score(truth, result.labels[:-1]), .99)

    def test_degenerate_tree_still_returns_exact_requested_cluster_count(self):
        # A distance-threshold cut can silently collapse tied merge heights.
        for k in [2, 4, 8]:
            result = hierarchical.fit_predict(np.ones((8, 6)), {"n_clusters": k})
            self.assertEqual(len(np.unique(result.labels)), k)

    def test_noise_is_excluded_and_single_effective_cluster_is_unavailable(self):
        X = np.array([[0., 0.], [.1, .2], [5., 5.], [5.1, 5.2], [100., 100.]])
        labels = np.array([0, 0, 1, 1, -1])
        result = evaluate_partition(X, labels)
        self.assertEqual(result["effective_clusters"], 2)
        self.assertEqual(result["evaluated_customers"], 4)
        self.assertAlmostEqual(result["coverage"], .8)
        self.assertAlmostEqual(result["silhouette"], silhouette_score(X[:4], labels[:4]))
        for invalid in [np.array([0, 0, 0, 0, -1]), np.full(5, -1), np.arange(5)]:
            result = evaluate_partition(X, invalid)
            self.assertIsNone(result["silhouette"])
            self.assertFalse(result["metrics_available"])
            self.assertTrue(result["metric_note"])

    def test_common_comparison_recalculates_both_on_identical_customers(self):
        X = np.array([[0., 0.], [.1, .2], [5., 5.], [5.1, 5.2], [100., 100.]])
        ward, density = np.array([0, 0, 1, 1, 2]), np.array([0, 0, 1, 1, -1])
        table, mask = compare_on_common_customers(X, ward, density)
        self.assertEqual(mask.tolist(), [True, True, True, True, False])
        np.testing.assert_allclose(table.silhouette, silhouette_score(X[mask], ward[mask]))
        self.assertEqual(table.common_customers.tolist(), [4, 4])
        self.assertEqual(table.original_coverage.tolist(), [1., .8])

    def test_composite_order_key_and_incomplete_observation_month(self):
        transactions = pd.DataFrame({"CustomerID": ["a", "a", "b", "a"],
            "InvoiceNo": ["same", "same", "same", "other"],
            "InvoiceDate": pd.to_datetime(["2010-12-01 08:00", "2010-12-01 08:00", "2010-12-01 08:00", "2011-01-09 12:00"]),
            "LineAmount": [2., 3., 7., 11.]})
        orders = aggregate_orders(transactions)
        self.assertEqual(len(orders), 3)
        self.assertEqual(orders.OrderAmount.sum(), 23.)
        monthly = monthly_summary(transactions, orders)
        self.assertEqual(monthly.Orders.tolist(), [2, 1])
        self.assertEqual(monthly.DetailRows.tolist(), [3, 1])
        self.assertEqual(monthly.IsIncomplete.tolist(), [False, True])
        self.assertEqual(monthly.CalendarDaysObserved.tolist(), [31, 9])

    def test_shared_preprocessing_is_called_once_and_input_order_validated(self):
        raw = pd.DataFrame({"CustomerID": pd.Series(["z", "a"], dtype="string")})
        X = np.arange(12, dtype=float).reshape(2, 6)
        with patch("experiments.b_runner.feature_scaling.load_customer_features", return_value=(raw.CustomerID, X)) as loader:
            ids, actual = load_shared_matrix(Path("fixture"), raw)
        loader.assert_called_once()
        self.assertIs(actual, X)
        self.assertFalse(actual.flags.writeable)
        self.assertEqual(ids.tolist(), ["z", "a"])
        with patch("experiments.b_runner.feature_scaling.load_customer_features", return_value=(raw.CustomerID.iloc[::-1], X)):
            with self.assertRaises(ValueError):
                load_shared_matrix(Path("fixture"), raw)

    def test_original_scale_profile_contains_noise_and_all_six_features(self):
        raw = pd.DataFrame({"CustomerID": pd.Series(["b", "a", "c"], dtype="string"),
                            **{name: [10., 30., 500.] for name in FEATURE_COLUMNS}})
        diagnostics = pd.DataFrame({"CustomerID": raw.CustomerID.iloc[::-1], "Monetary": [500., 30., 10.],
                                    "CancelledRows": [1, 0, 0], "CancellationAmountAbs": [500., 0., 0.]})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            summary = original_scale_profiles(raw, raw.CustomerID, np.array([0, 0, -1]), diagnostics, output)
            self.assertEqual(summary.Customers.sum(), 3)
            self.assertEqual(summary.loc[summary.Cluster.eq(0), "Monetary_median"].item(), 20.)
            noise = pd.read_csv(output / "noise_feature_statistics.csv")
            self.assertEqual(set(noise.Feature), set(FEATURE_COLUMNS))
            self.assertTrue(noise["mean"].eq(500).all())

    def test_selection_does_not_prefer_high_score_with_tiny_coverage(self):
        table = pd.DataFrame({"run_id": ["tiny", "usable", "small_cluster"], "metrics_available": [True]*3,
                              "coverage": [.1, .9, .95], "min_cluster_size": [50, 50, 2],
                              "silhouette": [.99, .3, .9], "eps": [.2, .6, .8], "min_samples": [5, 10, 5]})
        selected, stage = select_dbscan(table, 1000)
        self.assertEqual(selected.run_id, "usable")
        self.assertEqual(stage, "coverage_and_cluster_size")

    def test_adapters_reject_identifier_and_invalid_settings(self):
        for method in [hierarchical.fit_predict, dbscan.fit_predict]:
            with self.assertRaises(ValueError):
                method(pd.DataFrame({"CustomerID": [1, 2], "value": [3, 4]}))
            with self.assertRaises(ValueError):
                method(np.array([[0., np.nan], [1., 1.]]))
        for params in [{"eps": 0}, {"eps": float("inf")}, {"min_samples": 0}]:
            with self.assertRaises(ValueError):
                dbscan.fit_predict(np.ones((10, 6)), params)
        with self.assertRaises(ValueError):
            hierarchical.fit_predict(np.ones((10, 6)), {"linkage": "single"})


if __name__ == "__main__":
    unittest.main()
