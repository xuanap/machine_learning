"""Rescore archived assignments on the selected DBSCAN non-noise cohort."""
import hashlib
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from algorithms.registry import cluster
from experiments.group_metrics import evaluate_full

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "results/formal/a_shared_baseline_v1"
B = ROOT / "results/b/full_20261124_v2"
THREE = ROOT / "results/comparison/shared_three_algorithms"
OUT = ROOT / "results/comparison/shared_five_algorithms/common_customers"
METRICS = ["silhouette", "davies_bouldin", "calinski_harabasz"]


def read(path):
    return pd.read_csv(path, dtype={"CustomerID": "string"}, float_precision="round_trip")


def load_labels(path, ids):
    labels = read(path).set_index("CustomerID", verify_integrity=True)["Cluster"]
    if len(labels) != len(ids) or not ids.isin(labels.index).all() or not pd.api.types.is_integer_dtype(labels):
        raise ValueError(f"Invalid customer labels: {path}")
    return labels.reindex(ids)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((THREE / "manifest.json").read_text())
    b_manifest = json.loads((B / "manifest.json").read_text())
    frame = read(A / "model_input.csv").set_index("CustomerID", verify_integrity=True)
    X = np.ascontiguousarray(frame.to_numpy(dtype=np.float64))
    digest = hashlib.sha256(X.tobytes()).hexdigest()
    if digest != manifest["model_input"]["matrix_sha256"] or digest != b_manifest["matrix"]["sha256_float64_c_order"]:
        raise ValueError("Model input does not match both archives")

    runs, sources = {}, {}
    for row in read(THREE / "a_archive_audit.csv").itertuples():
        if row.ari_archive_vs_rerun != 1:
            raise ValueError(f"Archive partition differs: {row.archive_run}")
        key = (row.algorithm, row.seed)
        path = A / row.archive_run / "labels.csv"
        runs[key] = load_labels(path, frame.index)
        sources[key] = str(path.relative_to(ROOT))
    for name, folder in [("ward", "ward_k4"), ("dbscan", "dbscan_selected")]:
        path = B / "clustering" / folder / "labels.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != b_manifest["outputs_sha256"][str(path.relative_to(B))]:
            raise ValueError(f"Label checksum mismatch: {path}")
        runs[name, 42] = load_labels(path, frame.index)
        sources[name, 42] = str(path.relative_to(ROOT))

    mask = runs["dbscan", 42].ne(-1)
    if len(mask) != 4338 or mask.sum() != 4306:
        raise ValueError("Unexpected DBSCAN cohort size")
    archive_scores = read(THREE / "per_seed.csv").set_index(["algorithm", "seed"])
    audit = []
    for seed in manifest["seeds"]:
        key = ("spectral", seed)
        labels = cluster(X, "spectral", manifest["params"]["spectral"], random_state=seed,
                         threads=manifest["threads"])
        runs[key] = pd.Series(labels, index=frame.index, name="Cluster")
        path = OUT / f"spectral_seed{seed}_full_labels.csv"
        runs[key].to_csv(path)
        sources[key] = str(path.relative_to(ROOT))
        full = evaluate_full(X, labels)
        audit.append({"seed": seed, **{m: full[m] for m in METRICS},
                      **{f"{m}_delta_vs_archive": full[m] - archive_scores.loc[key, m] for m in METRICS}})
    pd.DataFrame(audit).to_csv(OUT / "spectral_reproduction.csv", index=False)

    records, cohort = [], {}
    for (name, seed), labels in runs.items():
        labels = labels.loc[mask]
        cohort[name, seed] = labels
        sizes = labels.value_counts().sort_index()
        records.append(dict(algorithm=name, seed=seed, customers=len(labels), clusters=len(sizes),
                            min_cluster_size=int(sizes.min()), cluster_sizes=json.dumps(sizes.to_dict()),
                            source=sources[name, seed], **evaluate_full(X[mask], labels)))
    scores = pd.DataFrame(records)
    reference = read(B / "clustering/comparison_common_customers.csv").set_index("algorithm")
    for name, archived in [("ward", "ward_k4"), ("dbscan", "dbscan_selected")]:
        np.testing.assert_allclose(scores.set_index(["algorithm", "seed"]).loc[(name, 42), METRICS]
                                   .to_numpy(dtype=float), reference.loc[archived, METRICS].to_numpy(dtype=float))
    scores.to_csv(OUT / "scores_per_run.csv", index=False)
    summary = scores.groupby("algorithm", sort=False)[METRICS].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary.insert(0, "runs", scores.groupby("algorithm").size())
    summary.to_csv(OUT / "score_summary.csv")

    names = ["kmeans", "gmm", "spectral", "ward", "dbscan"]
    assignments = pd.DataFrame({name: cohort[name, 42] for name in names})
    assignments.to_csv(OUT / "assignments_seed42.csv")
    pd.DataFrame({f"{name}_seed{seed}": labels for (name, seed), labels in cohort.items()}).to_csv(
        OUT / "assignments_all_runs.csv")
    ari = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    differences, overlaps = [], []
    for left, right in combinations(names, 2):
        a, b = assignments[left], assignments[right]
        ari.loc[left, right] = ari.loc[right, left] = adjusted_rand_score(a, b)
        table = pd.crosstab(a, b)
        i, j = linear_sum_assignment(-table.to_numpy())
        mapping = {table.columns[col]: table.index[row] for row, col in zip(i, j)
                   if table.iat[row, col] > 0}
        matched = b.map(mapping)
        for left_cluster in table.index:
            for right_cluster in table.columns:
                overlaps.append(dict(left=left, right=right, left_cluster=left_cluster,
                                     right_cluster=right_cluster, customers=table.loc[left_cluster, right_cluster],
                                     matched_pair=mapping.get(right_cluster) == left_cluster))
        for cid in a.index[matched != a]:
            differences.append(dict(CustomerID=cid, left=left, right=right, left_cluster=a[cid],
                                    right_cluster=b[cid], matched_left_cluster=matched[cid]))
    pd.DataFrame(overlaps).to_csv(OUT / "pairwise_overlaps_seed42.csv", index=False)
    pd.DataFrame(differences).to_csv(OUT / "different_assignments_seed42.csv", index=False)
    ari.to_csv(OUT / "ari_seed42.csv")
    stability = [{"algorithm": name, "seed_a": a, "seed_b": b,
                  "ari": adjusted_rand_score(cohort[name, a], cohort[name, b])}
                 for name in names[:3] for a, b in combinations(manifest["seeds"], 2)]
    pd.DataFrame(stability).to_csv(OUT / "ari_between_seeds.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ["K-means", "GMM", "Spectral", "Ward", "DBSCAN"]
    image = ax.imshow(ari, cmap="Blues", vmin=min(0, ari.to_numpy().min()), vmax=1)
    ax.set_xticks(range(len(names)), labels)
    ax.set_yticks(range(len(names)), labels)
    for i, j in np.ndindex(ari.shape):
        value = ari.iat[i, j]
        ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                color="white" if value > .55 else "#123047")
    ax.set_title("Agreement on 4,306 non-noise customers\nAdjusted Rand index | seed 42")
    fig.colorbar(image, ax=ax, shrink=.8)
    fig.tight_layout()
    fig.savefig(OUT / "ari_seed42.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
