"""Summarize A's recorded full-customer experiments without fitting models."""
import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from analysis.compare_clusters import compare_partitions
from experiments.inputs import file_hash
from experiments.plots import plt
from experiments.runner import write_json

KEYS = ["study", "algorithm", "factor", "value"]
METRICS = ["silhouette", "davies_bouldin", "calinski_harabasz", "fit_seconds", "aic", "bic"]


def summarize(baseline, parameters, ablation, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tables, manifests, runs = [], [], {}
    for folder in [baseline, parameters, ablation]:
        folder = Path(folder)
        manifest = json.loads((folder / "experiment_manifest.json").read_text(encoding="utf-8"))
        if manifest["mode"] != "formal":
            raise ValueError("This study summary requires full-customer prepared-input runs")
        manifests.append((folder, manifest))
        table = pd.read_csv(folder / "summary.csv")
        table["experiment"] = folder.name
        for row in table.itertuples():
            runs[(folder.name, row.run_id)] = folder / row.run_id
        tables.append(table)
    reference = manifests[0][1]
    for _, manifest in manifests[1:]:
        for field in ["source_features_sha256", "matrix_sha256", "preprocessing"]:
            if manifest["input"][field] != reference["input"][field]:
                raise ValueError(f"Study inputs differ: {field}")
        if manifest["metrics_provider"] != reference["metrics_provider"]:
            raise ValueError("Metric protocols differ across suites")
    combined = pd.concat(tables, ignore_index=True)
    combined["value"] = combined.value.astype(str)
    for index, row in combined.iterrows():
        sizes = pd.read_csv(runs[(row.experiment, row.run_id)] / "cluster_sizes.csv")
        combined.loc[index, "minimum_cluster_customers"] = sizes.Customers.min()
        combined.loc[index, "maximum_cluster_customers"] = sizes.Customers.max()
        combined.loc[index, "clusters_below_one_percent"] = (sizes.Customers < row.samples * .01).sum()
    combined.to_csv(output / "all_runs.csv", index=False, encoding="utf-8-sig")
    records, agreements = [], []
    for group, rows in combined.groupby(KEYS, sort=False):
        ok = rows.loc[rows.status.eq("ok")]
        record = {**dict(zip(KEYS, group)), "planned_runs": len(rows), "valid_runs": len(ok)}
        for metric in [*METRICS, "minimum_cluster_customers", "clusters_below_one_percent"]:
            values = pd.to_numeric(ok[metric], errors="coerce").dropna() if metric in ok else pd.Series(dtype=float)
            record[metric + "_mean"] = float(values.mean()) if len(values) else None
            record[metric + "_std"] = float(values.std(ddof=1)) if len(values) >= 2 else None
        records.append(record)
        for left, right in itertools.combinations(ok.itertuples(), 2):
            a = pd.read_csv(runs[(left.experiment, left.run_id)] / "labels.csv", dtype={"CustomerID": "string"})
            b = pd.read_csv(runs[(right.experiment, right.run_id)] / "labels.csv", dtype={"CustomerID": "string"})
            if set(a.CustomerID) != set(b.CustomerID):
                raise ValueError("Seed runs cover different customer sets")
            joined = a.merge(b, on="CustomerID", validate="one_to_one", suffixes=("_a", "_b"))
            agreements.append({**dict(zip(KEYS, group)), "left_seed": left.seed, "right_seed": right.seed,
                               "ari": adjusted_rand_score(joined.Cluster_a, joined.Cluster_b),
                               "nmi": normalized_mutual_info_score(joined.Cluster_a, joined.Cluster_b)})
    summary = pd.DataFrame(records)
    summary.to_csv(output / "configuration_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(agreements).to_csv(output / "seed_pair_agreement.csv", index=False, encoding="utf-8-sig")
    feature_agreements = []
    subsets = combined.loc[combined.study.eq("feature_ablation") & combined.status.eq("ok")]
    for (algorithm, seed), rows in subsets.groupby(["algorithm", "seed"]):
        rows = rows.set_index("value")
        left, right = rows.loc["RFM"], rows.loc["six_features"]
        a = pd.read_csv(runs[(left.experiment, left.run_id)] / "labels.csv", dtype={"CustomerID": "string"})
        b = pd.read_csv(runs[(right.experiment, right.run_id)] / "labels.csv", dtype={"CustomerID": "string"})
        customers, counts, fractions, mapping, agreement = compare_partitions(a, b)
        # Across feature spaces compare partitions, not the magnitudes of internal metrics.
        feature_agreements.append({"algorithm": algorithm, "seed": int(seed), **agreement})
        if seed == 42:
            folder = output / f"feature_agreement_{algorithm}_seed42"
            folder.mkdir()
            customers.to_csv(folder / "customers.csv", index=False, encoding="utf-8-sig")
            counts.to_csv(folder / "overlap.csv", encoding="utf-8-sig")
            mapping.to_csv(folder / "mapping.csv", index=False, encoding="utf-8-sig")
    write_json(output / "feature_partition_agreement.json", feature_agreements)

    for study in ["kmeans_k", "gmm_k"]:
        data = summary.loc[summary.study.eq(study)].copy()
        data["K"] = data.value.astype(int)
        data = data.sort_values("K")
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
        for axis, metric, label in zip(axes, METRICS[:3], ["Silhouette (higher)", "Davies-Bouldin (lower)", "Calinski-Harabasz (higher)"]):
            axis.errorbar(data.K, data[metric + "_mean"], yerr=data[metric + "_std"], marker="o", capsize=3)
            axis.set(xlabel="K", ylabel=label, xticks=data.K)
            axis.grid(alpha=.2)
        fig.suptitle(f"{study}: 4,338 customers, 3 seeds; mean and sample std")
        fig.savefig(output / f"{study}_three_metrics.png", dpi=160)
        plt.close(fig)
    covariance = summary.loc[summary.study.eq("gmm_covariance")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for axis, metric, label in zip(axes, ["silhouette", "bic"], ["Silhouette (higher)", "BIC (lower; same six features)"]):
        axis.errorbar(covariance.value, covariance[metric + "_mean"], yerr=covariance[metric + "_std"], marker="o", capsize=3)
        axis.set(ylabel=label, xlabel="Covariance type, K=4")
        axis.grid(alpha=.2)
    fig.suptitle("GMM objective trade-off; mean and sample std over 3 seeds")
    fig.savefig(output / "gmm_covariance_objectives.png", dpi=160)
    plt.close(fig)
    write_json(output / "study_summary.json", {
        "customers": reference["input"]["used_rows"], "runs": len(combined),
        "status_counts": combined.status.value_counts().to_dict(), "seeds": reference["config"]["seeds"],
        "input": reference["input"], "metrics_provider": reference["metrics_provider"],
        "source_manifests": {folder.name: file_hash(folder / "experiment_manifest.json") for folder, _ in manifests},
        "summary_code_sha256": file_hash(Path(__file__)),
        "scope": "A-member study using C's shared preprocessing; not the final five-algorithm comparison; ablation scope proposed by A",
    })
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "results/formal/a_shared_baseline_v1")
    parser.add_argument("--parameters", type=Path, default=ROOT / "results/formal/a_shared_parameters_v1")
    parser.add_argument("--ablation", type=Path, default=ROOT / "results/formal/a_shared_ablation_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/study/a_shared_v1")
    args = parser.parse_args()
    table = summarize(args.baseline, args.parameters, args.ablation, args.output)
    print(f"Saved {len(table)} configuration summaries to {args.output.resolve()}")
