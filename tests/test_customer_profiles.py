"""Hand-calculated profiles, ID alignment, and post-fit analysis boundaries."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.customer_profiles import build_profiles, gmm_boundaries, kmeans_boundaries
from analyze_run import analyze_run
from experiments.inputs import ROOT


def features():
    return pd.DataFrame({"CustomerID": ["A", "B", "C"], "Recency": [1, 3, 9],
                         "Frequency": [2, 4, 1], "Monetary": [100., 200., 80.],
                         "AvgOrderValue": [50., 50., 80.], "ProductDiversity": [3, 5, 2],
                         "ActiveDays": [20, 40, 0]})


class CustomerProfileTests(unittest.TestCase):
    def test_profiles_reconcile_and_keep_label_order(self):
        labels = pd.DataFrame({"CustomerID": ["C", "B", "A"], "Cluster": [1, 0, 0]})
        assignments, summary, statistics = build_profiles(features(), labels)
        self.assertEqual(assignments.CustomerID.tolist(), ["C", "B", "A"])
        first = summary.set_index("Cluster").loc[0]
        self.assertEqual(first.Customers, 2)
        self.assertEqual(first.TotalMonetary, 300)
        self.assertAlmostEqual(first.CustomerShare, 2 / 3)
        self.assertAlmostEqual(first.MonetaryShare, 300 / 380)
        self.assertEqual(first.Recency_Mean, 2)
        self.assertEqual(first.Recency_Median, 2)
        singleton = statistics.loc[statistics.Cluster.eq(1)]
        self.assertTrue(singleton["std"].isna().all())
        self.assertAlmostEqual(summary.CustomerShare.sum(), 1)
        self.assertAlmostEqual(summary.MonetaryShare.sum(), 1)

    def test_unknown_duplicate_and_invalid_cluster_rejected(self):
        for labels in [pd.DataFrame({"CustomerID": ["missing"], "Cluster": [0]}),
                       pd.DataFrame({"CustomerID": ["A", "A"], "Cluster": [0, 1]}),
                       pd.DataFrame({"CustomerID": ["A"], "Cluster": [.5]})]:
            with self.assertRaises(ValueError):
                build_profiles(features(), labels)

    def test_gmm_probability_rows_and_columns_align_by_id(self):
        assignments = pd.DataFrame({"CustomerID": ["B", "A"], "Cluster": [1, 0]})
        probabilities = pd.DataFrame({"CustomerID": ["A", "B"],
                                      "Probability_1": [.49, .99], "Probability_0": [.51, .01]})
        result = gmm_boundaries(assignments, probabilities)
        np.testing.assert_allclose(result.ProbabilityGap, [.98, .02])
        self.assertGreater(result.NormalizedEntropy.iloc[1], result.NormalizedEntropy.iloc[0])

    def test_gmm_rejects_wrong_probability_and_label(self):
        assignments = pd.DataFrame({"CustomerID": ["A"], "Cluster": [0]})
        for values in [(.6, .3), (-.1, 1.1), (.2, .8)]:
            with self.assertRaises(ValueError):
                gmm_boundaries(assignments, pd.DataFrame({"CustomerID": ["A"],
                                                          "Probability_0": [values[0]], "Probability_1": [values[1]]}))

    def test_gmm_single_component_does_not_invent_margin(self):
        assignments = pd.DataFrame({"CustomerID": ["A"], "Cluster": [0]})
        result = gmm_boundaries(assignments, pd.DataFrame({"CustomerID": ["A"], "Probability_0": [1.]}))
        self.assertTrue(result.ProbabilityGap.isna().all())
        self.assertEqual(result.NormalizedEntropy.item(), 0)

    def test_kmeans_uses_fitted_coordinates_and_center_ids(self):
        assignments = pd.DataFrame({"CustomerID": ["B", "C", "A"], "Cluster": [0, 1, 0]})
        matrix = pd.DataFrame({"CustomerID": ["C", "A", "B"], "Recency": [10., 0., 4.9], "Frequency": [0., 0., 0.]})
        centers = pd.DataFrame({"Cluster": [1, 0], "Recency": [10., 0.], "Frequency": [0., 0.]})
        result = kmeans_boundaries(assignments, matrix, centers, ["Recency", "Frequency"])
        np.testing.assert_allclose(result.AssignedCenterDistance, [4.9, 0, 0])
        np.testing.assert_allclose(result.DistanceGap, [.2, 10, 10])
        self.assertTrue(result.AssignedIsNearest.all())

    def test_end_to_end_analysis_preserves_smoke_notice(self):
        experiment = ROOT / "results/smoke/a_framework_check_v2"
        if not experiment.exists():
            self.skipTest("Recorded smoke fixture not included in this checkout")
        with tempfile.TemporaryDirectory() as temporary:
            for run_id in ["000_kmeans_seed42", "001_gmm_seed42"]:
                output = Path(temporary) / run_id
                assignments, summary = analyze_run(experiment, run_id, output)
                self.assertEqual(len(assignments), 300)
                self.assertEqual(summary.Customers.sum(), 300)
                self.assertIn("SMOKE ONLY", (output / "NOTICE.txt").read_text())
                self.assertEqual(json.loads((output / "analysis_manifest.json").read_text())["mode"], "smoke")
                self.assertTrue((output / "boundary_customers.csv").exists())
                with self.assertRaises(FileExistsError):
                    analyze_run(experiment, run_id, output)


if __name__ == "__main__":
    unittest.main()
