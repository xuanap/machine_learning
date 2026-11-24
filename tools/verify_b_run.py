"""Verify B's saved artifacts, IDs, profiles and selected/common-subset metrics.

Usage: python tools/verify_b_run.py results/b/<run>
Does not fit any clustering model or modify source data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocessing.feature_engineering import FEATURE_COLUMNS
from preprocessing.feature_scaling import load_customer_features


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return pd.read_csv(path, dtype={"CustomerID": "string", "InvoiceNo": "string"}, float_precision="round_trip")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    output = args.run.resolve()
    if not output.is_relative_to(ROOT / "results/b"):
        parser.error("run must be inside results/b")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    checks = []

    def check(name, passed):
        if not passed:
            raise AssertionError(name)
        checks.append(name)

    check("run_complete", manifest["status"] == "complete")
    for source, expected in manifest["sources_sha256"].items():
        check(f"source_hash:{source}", digest(ROOT / source) == expected)
    for filename, expected in manifest["outputs_sha256"].items():
        check(f"output_hash:{filename}", digest(output / filename) == expected)
    raw = read(ROOT / "data/customer_features.csv")
    if (output / "eda").exists():
        orders = read(output / "eda/orders.csv")
        monthly = read(output / "eda/monthly.csv")
        summary = json.loads((output / "eda/summary.json").read_text(encoding="utf-8"))
        check("order_composite_keys_unique", not orders.duplicated(["CustomerID", "InvoiceNo"]).any())
        check("detail_rows_sum", orders.DetailRows.sum() == summary["detail_rows"] == monthly.DetailRows.sum())
        check("order_count_sum", len(orders) == summary["orders"] == monthly.Orders.sum())
        check("positive_sales_reconcile", np.isclose(orders.OrderAmount.sum(), raw.Monetary.sum(), rtol=1e-12))
        check("single_purchase_reconcile", summary["single_order_customers"] == raw.Frequency.eq(1).sum())
        check("incomplete_month_matches_manifest", monthly.loc[monthly.IsIncomplete, "Month"].tolist() == summary["incomplete_months"])
    folder = output / "clustering"
    if folder.exists():
        ids, expected_X = load_customer_features(ROOT / "data/customer_features.csv")
        X = np.load(folder / "model_input.npy")
        snapshot = read(folder / "model_input.csv")
        check("matrix_exactly_matches_C", np.array_equal(X, expected_X))
        check("matrix_csv_round_trip", np.array_equal(snapshot[FEATURE_COLUMNS].to_numpy(), X))
        check("matrix_customer_order", snapshot.CustomerID.tolist() == ids.tolist())
        check("matrix_hash", hashlib.sha256(X.tobytes()).hexdigest() == manifest["matrix"]["sha256_float64_c_order"])

        def labels_at(path):
            labels = read(path)
            check(f"id_alignment:{path.relative_to(folder)}", labels.CustomerID.tolist() == ids.tolist())
            check(f"integer_labels:{path.relative_to(folder)}", np.issubdtype(labels.Cluster.dtype, np.integer))
            return labels.Cluster.to_numpy()

        hierarchy = read(folder / "hierarchical/parameter_summary.csv")
        check("all_21_hierarchy_configs", set(zip(hierarchy.linkage, hierarchy.k)) == {(method, k) for method in ["ward", "complete", "average"] for k in range(2, 9)})
        for row in hierarchy.itertuples():
            labels = labels_at(folder / "hierarchical" / f"{row.linkage}_k{row.k}_labels.csv")
            check(f"hierarchy_K:{row.linkage}:{row.k}", len(set(labels)) == row.k and (labels >= 0).all())
        scan = read(folder / "dbscan/parameter_summary.csv")
        check("all_30_dbscan_configs", len(scan) == 30 and not scan.duplicated(["eps", "min_samples"]).any())
        for row in scan.itertuples():
            labels = labels_at(folder / "dbscan" / f"{row.run_id}_labels.csv")
            groups, sizes = np.unique(labels, return_counts=True)
            counts = {str(int(k)): int(v) for k, v in zip(groups, sizes)}
            check(f"dbscan_counts:{row.run_id}", counts == json.loads(row.cluster_sizes))
            check(f"dbscan_coverage:{row.run_id}", np.isclose((labels != -1).mean(), row.coverage))
            valid = 2 <= len(set(labels) - {-1}) < (labels != -1).sum()
            check(f"dbscan_metric_availability:{row.run_id}", bool(row.metrics_available) == valid)
            if not valid:
                check(f"dbscan_null_metrics:{row.run_id}", pd.isna([row.silhouette, row.davies_bouldin, row.calinski_harabasz]).all())
        ward = labels_at(folder / "ward_k4/labels.csv")
        density = labels_at(folder / "dbscan_selected/labels.csv")
        selected = json.loads((folder / "dbscan/selection.json").read_text())
        check("selected_labels_match_scan", np.array_equal(density, labels_at(folder / "dbscan" / f"{selected['run_id']}_labels.csv")))
        for name, labels in [("ward_k4", ward), ("dbscan_selected", density)]:
            profiles = read(folder / name / "feature_statistics.csv")
            for group in np.unique(labels):
                stats = profiles[profiles.Cluster.eq(group)].set_index("Feature").reindex(FEATURE_COLUMNS)
                actual = raw.loc[labels == group, FEATURE_COLUMNS]
                check(f"profile:{name}:{group}", len(stats) == 6 and np.allclose(stats["count"], len(actual))
                      and np.allclose(stats["mean"], actual.mean(), rtol=1e-12)
                      and np.allclose(stats["50%"], actual.median(), rtol=1e-12))
        common = read(folder / "comparison_common_customers.csv")
        assignments = read(folder / "comparison_customer_assignments.csv")
        mask = density != -1
        check("comparison_exact_same_customers", assignments.CustomerID.tolist() == ids[mask].tolist())
        with threadpool_limits(limits=1):
            for row, labels in zip(common.itertuples(), [ward[mask], density[mask]]):
                if row.metrics_available:
                    check(f"common_metrics:{row.algorithm}", np.allclose(
                        [row.silhouette, row.davies_bouldin, row.calinski_harabasz],
                        [silhouette_score(X[mask], labels), davies_bouldin_score(X[mask], labels), calinski_harabasz_score(X[mask], labels)], rtol=1e-12))
    result = {"passed": True, "checks": len(checks), "check_names": checks}
    (output / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS: {len(checks)} artifact checks. {output / 'verification.json'}")


if __name__ == "__main__":
    main()
