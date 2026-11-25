"""PCA comparison, archived spectral profiles, and five-algorithm performance heatmap."""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from algorithms.registry import cluster
from analysis.customer_profiles import build_profiles
from experiments.group_metrics import evaluate_comparison, evaluate_full

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/comparison/shared_five_algorithms"
COMMON = OUT / "common_customers"
ARCHIVE = ROOT / "results/formal/a_shared_baseline_v1"
B = ROOT / "results/b/full_20261124_v2/clustering"
METRICS = ["silhouette", "davies_bouldin", "calinski_harabasz"]
NAMES = {"kmeans": "K-means", "gmm": "GMM", "spectral": "Spectral",
         "hierarchical": "Ward", "dbscan": "DBSCAN"}


def read(path):
    return pd.read_csv(path, dtype={"CustomerID": str}, float_precision="round_trip")


def pca_comparison():
    output = OUT / "pca"
    output.mkdir(exist_ok=True)
    manifest = json.loads((OUT / "timing/manifest.json").read_text())
    frame = read(ARCHIVE / "model_input.csv").set_index("CustomerID")
    X = np.ascontiguousarray(frame.to_numpy(dtype=np.float64))
    digest = hashlib.sha256(X.tobytes()).hexdigest()
    if digest != manifest["matrix_sha256"] or not frame.index.is_unique:
        raise ValueError("Unexpected baseline matrix")
    with threadpool_limits(limits=1):
        pca = PCA(n_components=2, svd_solver="full")
        Z = pca.fit_transform(X)
    pd.DataFrame(Z, index=frame.index, columns=["PC1", "PC2"]).to_csv(output / "coordinates.csv")
    pd.DataFrame(pca.components_, index=["PC1", "PC2"], columns=frame.columns).to_csv(output / "loadings.csv")
    (output / "manifest.json").write_text(json.dumps({
        "original_matrix_sha256": digest, "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "n_components": 2, "whiten": False, "solver": "full", "threads": 1,
        "parameters": manifest["params"], "label_references": manifest["label_references"],
        "seeds": manifest["seeds_by_round"], "common_scope": "intersection of assigned customers across all runs",
    }, indent=2) + "\n")
    runs = []
    assignments = pd.DataFrame(index=frame.index)
    for name, params in manifest["params"].items():
        seeds = [42] if name in ("hierarchical", "dbscan") else manifest["seeds_by_round"]
        for seed in seeds:
            key = f"{name}_{seed}"
            path = ROOT / manifest["label_references"][key]
            saved = read(path).set_index("CustomerID")["Cluster"]
            if not saved.index.is_unique or set(saved.index) != set(frame.index):
                raise ValueError(f"Customer IDs differ in {path}")
            before = saved.reindex(frame.index).to_numpy()
            after = cluster(Z, name, params, random_state=seed, threads=1)
            runs.append((name, seed, before, after))
            assignments[f"{key}_before"] = before
            assignments[f"{key}_after"] = after
    common = assignments.ne(-1).all(axis=1).to_numpy()
    assignments["common_assigned"] = common
    assignments.to_csv(output / "assignments.csv")
    records = []
    for name, seed, before, after in runs:
        ari = adjusted_rand_score(before[common], after[common])
        for space, labels, fit_X in [("6D", before, X), ("PCA2", after, Z)]:
            native = evaluate_comparison(fit_X, labels)
            fixed = evaluate_full(X[common], labels[common])
            values, counts = np.unique(labels[labels != -1], return_counts=True)
            records.append(dict(algorithm=name, seed=seed, fit_space=space,
                                common_customers=int(common.sum()), ari_before_after_common=ari,
                                **native, **{f"common_6d_{m}": fixed[m] for m in METRICS},
                                cluster_sizes=json.dumps(dict(zip(values.tolist(), counts.tolist())))))
    detail = pd.DataFrame(records)
    detail.to_csv(output / "scores_per_run.csv", index=False)
    measures = ["observed_clusters", "coverage", "ari_before_after_common", *METRICS,
                *[f"common_6d_{m}" for m in METRICS]]
    summary = detail.groupby(["algorithm", "fit_space"], sort=False)[measures].mean()
    summary.to_csv(output / "summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    y = np.arange(len(NAMES))
    for ax, metric, title in zip(axes, METRICS, ["Silhouette ↑", "Davies–Bouldin ↓", "Calinski–Harabasz ↑"]):
        for space, offset, color in [("6D", -.16, "#176581"), ("PCA2", .16, "#d58a45")]:
            vals = [summary.loc[(name, space), f"common_6d_{metric}"] for name in NAMES]
            ax.barh(y + offset, vals, height=.3, color=color, label=space)
            for position, value in zip(y + offset, vals):
                if pd.isna(value):
                    ax.annotate("N/A (one cluster)", (0, position), xytext=(4, 0),
                                textcoords="offset points", va="center", fontsize=8)
        ax.set_yticks(y, list(NAMES.values()))
        ax.invert_yaxis()
        ax.set_title(title)
    axes[-1].legend()
    fig.suptitle(f"PCA comparison | Same 6D scoring space, {common.sum():,} customers")
    fig.tight_layout()
    fig.savefig(output / "pca_comparison.png", dpi=240)
    plt.close(fig)


def spectral_profiles():
    output = OUT / "spectral_profiles"
    output.mkdir(exist_ok=True)
    raw = read(ROOT / "data/customer_features.csv")
    labels = read(COMMON / "spectral_seed42_full_labels.csv")
    if len(raw) != 4338 or set(raw.CustomerID) != set(labels.CustomerID):
        raise ValueError("Raw features and spectral labels cover different customers")
    _, summary, statistics = build_profiles(raw, labels)
    summary.to_csv(output / "cluster_summary.csv", index=False)
    statistics.to_csv(output / "feature_statistics.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, feature in zip(axes.flat, statistics["Feature"].unique()):
        ax.bar(summary.Cluster.astype(str), summary[f"{feature}_Median"], color="#176581")
        ax.set_title(feature)
        ax.set_xlabel("Cluster")
    fig.suptitle("Spectral clustering | Original-scale customer medians (seed 42)")
    fig.tight_layout()
    fig.savefig(output / "profile_medians.png", dpi=240)
    plt.close(fig)


def performance_heatmap():
    scores = read(COMMON / "score_summary.csv").rename(columns={"algorithm": "key"}).replace({"key": {"ward": "hierarchical"}}).set_index("key")
    runtime = read(OUT / "timing/summary.csv").set_index("algorithm")
    table = scores[[f"{m}_mean" for m in METRICS]].join(runtime[["mean"]].rename(columns={"mean": "wall_seconds"}))
    table = table.reindex(NAMES).rename(index=NAMES)
    table.to_csv(OUT / "performance_heatmap_data.csv")
    ranks = table.rank(ascending=True)
    for col in ["davies_bouldin_mean", "wall_seconds"]:
        ranks[col] = table[col].rank(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.imshow(ranks, cmap="Blues", vmin=1, vmax=len(table), aspect="auto")
    ax.set_xticks(range(4), ["Silhouette ↑", "Davies–Bouldin ↓", "Calinski–Harabasz ↑", "Wall seconds ↓"])
    ax.set_yticks(range(len(table)), table.index)
    for i, j in np.ndindex(table.shape):
        precision = 1 if j == 2 else 4
        ax.text(j, i, f"{table.iat[i, j]:.{precision}f}", ha="center", va="center",
                color="white" if ranks.iat[i, j] >= 4 else "#142f48")
    ax.set_title("Five-algorithm performance | Column-wise ranks", pad=16)
    fig.text(.5, .065, "Scores: same 4,306 customers. Wall time: 4,338 customers, same machine, 3 runs.",
             ha="center")
    fig.text(.5, .025, "Darker = better rank within a column. DBSCAN has two clusters (4,302 + 4).",
             ha="center")
    fig.tight_layout(rect=(0, .11, 1, 1))
    fig.savefig(OUT / "performance_heatmap.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    pca_comparison()
    spectral_profiles()
    performance_heatmap()
