"""Shared-function input, exact saved coordinates and full-data metric contracts."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from analyze_run import analyze_run
from experiments.inputs import ROOT, load_input, matrix_hash
from experiments.group_metrics import evaluate_full
from experiments.runner import run_suite
from preprocessing import feature_scaling
from preprocessing.feature_engineering import FEATURE_COLUMNS


class SharedPreprocessingTests(unittest.TestCase):
    def test_shared_loader_called_once_and_coordinates_not_scaled_again(self):
        ids, X = feature_scaling.load_customer_features(ROOT / "data/customer_features.csv")
        with patch.object(feature_scaling, "load_customer_features", return_value=(ids, X)) as loader:
            bundle = load_input("formal")
        loader.assert_called_once()
        np.testing.assert_array_equal(bundle.matrix(FEATURE_COLUMNS), X)
        self.assertEqual(bundle.frame.CustomerID.tolist(), ids.tolist())
        self.assertEqual(bundle.provenance["used_rows"], 4338)
        self.assertEqual(bundle.provenance["matrix_sha256"], matrix_hash(bundle.frame))
        self.assertTrue(bundle.provenance["preprocessing"]["code_sha256"])

    def test_shared_function_matches_independent_log_and_population_scaling(self):
        raw = pd.read_csv(ROOT / "data/customer_features.csv")
        values = raw[FEATURE_COLUMNS].to_numpy(dtype=float)
        for index, column in enumerate(FEATURE_COLUMNS):
            if column in feature_scaling.LOG_COLUMNS:
                values[:, index] = np.log1p(values[:, index])
        expected = (values - values.mean(axis=0)) / values.std(axis=0, ddof=0)
        actual = load_input("formal").matrix(FEATURE_COLUMNS)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_invalid_shared_output_or_customer_order_rejected(self):
        ids, X = feature_scaling.load_customer_features(ROOT / "data/customer_features.csv")
        for returned_ids, values in [(ids.iloc[::-1], X), (ids, X[:, :2]), (ids, X * np.nan)]:
            with patch.object(feature_scaling, "load_customer_features", return_value=(returned_ids, values)):
                with self.assertRaises(ValueError):
                    load_input("formal")

    def test_saved_coordinates_round_trip_and_analysis_needs_no_external_csv(self):
        bundle = load_input("formal")
        bundle.frame = bundle.frame.sample(90, random_state=17).reset_index(drop=True)
        bundle.provenance["used_rows"] = 90
        bundle.provenance["matrix_sha256"] = matrix_hash(bundle.frame)
        config = json.loads((ROOT / "config/smoke_experiments.json").read_text())
        config["baseline_k"] = 2
        config["gmm"].update(max_iter=300, n_init=1)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "experiment"
            runs = run_suite(bundle, config, output, make_plots=False)
            self.assertTrue(runs.status.eq("ok").all())
            saved = pd.read_csv(output / "model_input.csv", dtype={"CustomerID": "string"}, float_precision="round_trip")
            np.testing.assert_array_equal(saved[FEATURE_COLUMNS].to_numpy(), bundle.matrix(FEATURE_COLUMNS))
            for run_id in runs.run_id:
                assignments, summary = analyze_run(output, run_id, Path(temporary) / run_id)
                self.assertEqual(len(assignments), 90)
                self.assertEqual(summary.Customers.sum(), 90)
            with (output / "model_input.csv").open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "differs from the matrix"):
                analyze_run(output, runs.run_id.iloc[0], Path(temporary) / "tampered")

    def test_full_metrics_match_c_entrypoint_definitions(self):
        X = np.array([[0., 0.], [.1, .2], [.2, .1], [5., 5.], [5.2, 5.1], [4.9, 5.1]])
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = evaluate_full(X, labels)
        self.assertEqual(scores["silhouette_samples"], len(X))
        self.assertAlmostEqual(scores["silhouette"], silhouette_score(X, labels))
        self.assertAlmostEqual(scores["davies_bouldin"], davies_bouldin_score(X, labels))
        self.assertAlmostEqual(scores["calinski_harabasz"], calinski_harabasz_score(X, labels))


if __name__ == "__main__":
    unittest.main()
