"""Known label permutations/splits and missing-run uncertainty examples."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.compare_clusters import compare_partitions
from compare_runs import compare_runs
from experiments.inputs import ROOT
from experiments.plots import plot_summary, summarize_runs


class PartitionComparisonTests(unittest.TestCase):
    def test_permuted_labels_and_customer_order_are_identical_partition(self):
        left = pd.DataFrame({"CustomerID": ["A", "B", "C", "D"], "Cluster": [0, 0, 1, 1]})
        right = pd.DataFrame({"CustomerID": ["D", "B", "A", "C"], "Cluster": [10, 20, 20, 10]})
        customers, counts, fractions, mapping, summary = compare_partitions(left, right)
        self.assertEqual(customers.CustomerID.tolist(), ["A", "B", "C", "D"])
        self.assertTrue(customers.SameAfterMatching.all())
        self.assertEqual(summary["ari"], 1)
        self.assertEqual(summary["nmi"], 1)
        self.assertEqual(summary["matched_overlap_fraction"], 1)
        self.assertEqual(counts.to_numpy().sum(), 4)
        np.testing.assert_allclose(fractions.sum(axis=1), 1)
        self.assertEqual(dict(zip(mapping.RightCluster, mapping.MatchedLeftCluster)), {10: 1, 20: 0})

    def test_rectangular_matching_leaves_split_group_unmatched(self):
        left = pd.DataFrame({"CustomerID": list("ABCD"), "Cluster": [0, 0, 1, 1]})
        right = pd.DataFrame({"CustomerID": list("ABCD"), "Cluster": [0, 1, 2, 2]})
        _, _, _, mapping, summary = compare_partitions(left, right)
        self.assertEqual(summary["matched_customers"], 3)
        self.assertEqual(summary["matched_overlap_fraction"], .75)
        self.assertEqual(mapping.MatchedLeftCluster.isna().sum(), 1)
        self.assertEqual(len(summary["unmatched_right_clusters"]), 1)

    def test_zero_overlap_assignment_is_not_a_real_mapping(self):
        pairs = [(0, 0)] * 10 + [(0, 1), (1, 0), (2, 2)]
        left = pd.DataFrame({"CustomerID": [str(i) for i in range(13)], "Cluster": [a for a, _ in pairs]})
        right = pd.DataFrame({"CustomerID": [str(i) for i in range(13)], "Cluster": [b for _, b in pairs]})
        _, _, _, _, summary = compare_partitions(left, right)
        self.assertEqual(summary["matched_customers"], 11)
        self.assertEqual(summary["unmatched_right_clusters"], [1])

    def test_mismatched_customers_or_duplicate_ids_are_rejected(self):
        left = pd.DataFrame({"CustomerID": ["A", "B"], "Cluster": [0, 1]})
        for right in [pd.DataFrame({"CustomerID": ["A", "C"], "Cluster": [0, 1]}),
                      pd.DataFrame({"CustomerID": ["A", "A"], "Cluster": [0, 1]})]:
            with self.assertRaises(ValueError):
                compare_partitions(left, right)

    def test_saved_smoke_comparison_retains_provenance(self):
        experiment = ROOT / "results/smoke/a_framework_check_v2"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "compare"
            summary = compare_runs(experiment, "000_kmeans_seed42", experiment, "001_gmm_seed42", output)
            self.assertEqual(summary["customers"], 300)
            self.assertIn("SMOKE ONLY", (output / "NOTICE.txt").read_text())
            metadata = json.loads((output / "comparison.json").read_text())
            self.assertEqual(metadata["mode"], "smoke")
            self.assertTrue(metadata["left"]["labels_sha256"])
            with self.assertRaisesRegex(ValueError, "same ordered features"):
                compare_runs(experiment, "000_kmeans_seed42", experiment, "012_kmeans_seed42", Path(temporary) / "bad")


class PlotUncertaintyTests(unittest.TestCase):
    def table(self):
        return pd.DataFrame([
            {"study": "test_k", "algorithm": "kmeans", "factor": "K", "value": 2, "seed": 1, "status": "ok", "silhouette": .4, "fit_seconds": 1.},
            {"study": "test_k", "algorithm": "kmeans", "factor": "K", "value": 2, "seed": 2, "status": "failed", "silhouette": np.nan, "fit_seconds": np.nan},
            {"study": "test_k", "algorithm": "kmeans", "factor": "K", "value": 2, "seed": 3, "status": "not_converged", "silhouette": .9, "fit_seconds": 2.},
            {"study": "test_k", "algorithm": "kmeans", "factor": "K", "value": 3, "seed": 1, "status": "failed", "silhouette": np.nan, "fit_seconds": np.nan},
        ])

    def test_missing_and_single_runs_keep_counts_and_undefined_std(self):
        support = summarize_runs(self.table())
        single = support.loc[support.value.eq(2) & support.metric.eq("silhouette")].iloc[0]
        self.assertEqual((single.valid_runs, single.planned_runs, single.non_ok_runs), (1, 3, 2))
        self.assertEqual(single["mean"], .4)
        self.assertTrue(pd.isna(single["std"]))
        missing = support.loc[support.value.eq(3) & support.metric.eq("silhouette")].iloc[0]
        self.assertEqual(missing.valid_runs, 0)
        self.assertTrue(pd.isna(missing["mean"]))

    def test_sample_standard_deviation_from_two_valid_runs(self):
        table = self.table().iloc[:2].copy()
        table.loc[1, ["status", "silhouette"]] = ["ok", .6]
        support = summarize_runs(table)
        value = support.loc[support.metric.eq("silhouette")].iloc[0]
        self.assertEqual(value.valid_runs, 2)
        self.assertAlmostEqual(value["mean"], .5)
        self.assertAlmostEqual(value["std"], np.sqrt(.02))

    def test_plots_keep_fully_failed_parameter_and_support_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            plot_summary(self.table(), output, "smoke")
            self.assertTrue((output / "test_k_silhouette.png").exists())
            table = pd.read_csv(output / "plot_support.csv")
            self.assertEqual(set(table.value), {2, 3})
            self.assertTrue(table["std"].isna().all())


if __name__ == "__main__":
    unittest.main()
