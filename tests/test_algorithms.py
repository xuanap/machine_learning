"""Behavior checks with known synthetic groups, not retail-quality claims."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from algorithms import gmm, kmeans
from experiments.inputs import ExperimentInput, ROOT, file_hash, load_input
from experiments.reference_metrics import evaluate
from experiments.runner import make_tasks, run_suite
from preprocessing.feature_engineering import FEATURE_COLUMNS


def synthetic_input(n=90):
    values, labels = make_blobs(n_samples=n, n_features=6, centers=3, cluster_std=.15, random_state=17)
    frame = pd.DataFrame(values, columns=FEATURE_COLUMNS)
    frame.insert(0, "CustomerID", [f"test_{index:03d}" for index in reversed(range(n))])
    return ExperimentInput(frame, {"notice": "synthetic unit-test fixture"}, "smoke"), labels


class AlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.bundle, self.truth = synthetic_input()
        self.X = self.bundle.matrix(FEATURE_COLUMNS)

    def test_kmeans_recovers_groups_and_keeps_input_scale(self):
        original = self.X.copy()
        result = kmeans.fit_predict(self.X, {"n_clusters": 3}, 42)
        self.assertGreater(adjusted_rand_score(self.truth, result.labels), .99)
        self.assertTrue(np.array_equal(original, self.X))
        for cluster in range(3):
            np.testing.assert_allclose(result.centers[cluster], self.X[result.labels == cluster].mean(axis=0), atol=1e-10)
        self.assertIsNone(result.converged)

    def test_gmm_all_covariances_and_probabilities(self):
        for covariance in ["full", "tied", "diag", "spherical"]:
            with self.subTest(covariance=covariance):
                result = gmm.fit_predict(self.X, {"n_components": 3, "covariance_type": covariance, "n_init": 1})
                self.assertGreater(adjusted_rand_score(self.truth, result.labels), .99)
                self.assertEqual(result.probabilities.shape, (len(self.X), 3))
                np.testing.assert_allclose(result.probabilities.sum(axis=1), 1, atol=1e-10)
                self.assertTrue((result.labels == result.probabilities.argmax(axis=1)).all())
                self.assertTrue(result.converged)
                self.assertTrue(np.isfinite([result.diagnostics["aic"], result.diagnostics["bic"]]).all())

    def test_fixed_seed_reproduces_each_algorithm(self):
        for method, params in [(kmeans.fit_predict, {"n_clusters": 3}), (gmm.fit_predict, {"n_components": 3})]:
            first = method(self.X, params, 17)
            second = method(self.X, params, 17)
            np.testing.assert_array_equal(first.labels, second.labels)
            np.testing.assert_allclose(first.centers, second.centers)

    def test_invalid_inputs_and_identifier_are_rejected(self):
        for values in [np.array([1, 2]), np.array([[1.0], [np.nan]]), np.empty((0, 6))]:
            with self.assertRaises(ValueError):
                kmeans.fit_predict(values)
        with self.assertRaises(ValueError):
            kmeans.fit_predict(self.bundle.frame)
        with self.assertRaises(ValueError):
            kmeans.fit_predict(self.X, {"n_clusters": len(self.X) + 1})
        with self.assertRaises(ValueError):
            kmeans.fit_predict(self.X, {"n_init": "auto"})

    def test_nonconvergence_and_iteration_limits_are_not_hidden(self):
        values = np.random.default_rng(1).normal(size=(150, 3))
        result = gmm.fit_predict(values, {"n_components": 4, "init_params": "random", "n_init": 1, "max_iter": 1, "tol": 1e-12})
        self.assertFalse(result.converged)
        self.assertEqual(result.status, "not_converged")
        self.assertTrue(result.warnings)
        limited = kmeans.fit_predict(values, {"n_clusters": 4, "max_iter": 1})
        self.assertEqual(limited.status, "iteration_limit_reached")
        self.assertIsNone(limited.converged)

    def test_collapsed_clusters_have_explicit_status(self):
        result = kmeans.fit_predict(np.ones((10, 2)), {"n_clusters": 3})
        self.assertEqual(result.status, "collapsed_clusters")
        self.assertTrue(result.warnings)

    def test_single_cluster_metric_is_unavailable(self):
        scores = evaluate(self.X, np.zeros(len(self.X), dtype=int))
        self.assertIsNone(scores["silhouette"])
        self.assertTrue(scores["metric_note"])

    def test_one_factor_configs_preserve_other_settings(self):
        config = json.loads((ROOT / "config/smoke_experiments.json").read_text())
        tasks = make_tasks(config, "parameters")
        initializations = [task for task in tasks if task["study"] == "kmeans_initialization"]
        self.assertEqual({task["params"]["n_init"] for task in initializations}, {2})
        self.assertEqual({task["params"]["n_clusters"] for task in initializations}, {3})
        self.assertEqual({task["params"]["init"] for task in initializations}, {"random", "k-means++"})

    def test_runner_saves_ids_labels_probabilities_and_provenance(self):
        config = json.loads((ROOT / "config/smoke_experiments.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            table = run_suite(self.bundle, config, output, make_plots=False)
            self.assertTrue(table.status.eq("ok").all())
            for run_id in table.run_id:
                labels = pd.read_csv(output / run_id / "labels.csv", dtype={"CustomerID": "string"})
                self.assertEqual(labels.CustomerID.tolist(), self.bundle.frame.CustomerID.tolist())
            probabilities = pd.read_csv(output / table.loc[table.algorithm.eq("gmm"), "run_id"].item() / "probabilities.csv")
            np.testing.assert_allclose(probabilities.filter(like="Probability_").sum(axis=1), 1)
            self.assertIn("SMOKE TEST ONLY", (output / "NOTICE.txt").read_text())
            with self.assertRaises(FileExistsError):
                run_suite(self.bundle, config, output, make_plots=False)

    def test_runner_preserves_failed_configuration(self):
        config = json.loads((ROOT / "config/smoke_experiments.json").read_text())
        config["baseline_k"] = 1000
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "failed"
            table = run_suite(self.bundle, config, output, make_plots=False)
            self.assertTrue(table.status.eq("failed").all())
            self.assertTrue(table.error.str.contains("ValueError").all())
            self.assertTrue((output / table.run_id.iloc[0] / "run.json").exists())


class PreparedInputTests(unittest.TestCase):
    def test_formal_override_requires_both_files(self):
        with self.assertRaisesRegex(ValueError, "Supply both"):
            load_input("formal", ROOT / "data/customer_features.csv")

    def test_smoke_sample_reproducible_and_identified(self):
        first = load_input("smoke", sample_size=20)
        second = load_input("smoke", sample_size=20)
        self.assertEqual(first.frame.CustomerID.tolist(), second.frame.CustomerID.tolist())
        self.assertEqual(len(first.frame), 20)
        self.assertIn("SMOKE TEST ONLY", first.provenance["notice"])

    def test_formal_input_keeps_cs_order_and_validates_hashes(self):
        source = ROOT / "data/customer_features.csv"
        raw = pd.read_csv(source, dtype={"CustomerID": "string"})
        # Test fixture only: never exported to the real shared data directory.
        prepared = raw.sample(frac=1, random_state=4).reset_index(drop=True)
        for column in FEATURE_COLUMNS:
            prepared[column] = (prepared[column] - prepared[column].mean()) / prepared[column].std(ddof=0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prepared.csv"
            metadata_path = Path(temporary) / "prepared.json"
            prepared.to_csv(path, index=False)
            metadata = {"data_version": "online-retail-six-features-v1",
                        "source_features_sha256": file_hash(source), "prepared_csv_sha256": file_hash(path),
                        "feature_columns": FEATURE_COLUMNS,
                        "preprocessing": {"scaler": "StandardScaler", "representation": "scaled_features",
                                          "version": "unit-test-only", "log1p_columns": []}}
            metadata_path.write_text(json.dumps(metadata))
            loaded = load_input("formal", path, metadata_path)
            self.assertEqual(loaded.frame.CustomerID.tolist(), prepared.CustomerID.tolist())
            metadata["prepared_csv_sha256"] = "wrong"
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "prepared_csv_sha256"):
                load_input("formal", path, metadata_path)
            metadata["prepared_csv_sha256"] = file_hash(path)
            metadata["preprocessing"]["representation"] = "pca_features"
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "not PCA"):
                load_input("formal", path, metadata_path)


if __name__ == "__main__":
    unittest.main()
