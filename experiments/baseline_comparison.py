"""Run the shared three-algorithm baseline and reconcile A's archived labels."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from algorithms.registry import cluster
from experiments.group_metrics import evaluate_comparison
from experiments.inputs import ROOT, file_hash, load_input, matrix_hash
from experiments.runner import write_json

METRICS = ("silhouette", "davies_bouldin", "calinski_harabasz", "runtime_seconds")
DISPLAY = {"kmeans": "K-means", "gmm": "GMM", "spectral": "Spectral"}


def heatmap(summary, path, k, seed_count):
    values = summary[[metric + "_mean" for metric in METRICS]].to_numpy()
    scores = np.column_stack([
        pd.Series(values[:, i]).rank(method="average", pct=True, ascending=i in (0, 2))
        for i in range(len(METRICS))
    ])
    fig, ax = plt.subplots(figsize=(8.4, 3.2), layout="constrained")
    image = ax.imshow(scores, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), ["Silhouette ↑", "Davies–Bouldin ↓", "Calinski–Harabasz ↑", "Runtime (s) ↓"])
    ax.set_yticks(range(len(summary)), summary.algorithm.map(DISPLAY))
    for i, j in np.ndindex(values.shape):
        ax.text(j, i, f"{values[i, j]:.2f}" if j == 2 else f"{values[i, j]:.4f}",
                ha="center", va="center", color="white" if scores[i, j] > 0.8 else "#14243a")
    ax.set_title(f"Customer clustering baseline · K={k} · {seed_count} seeds")
    fig.colorbar(image, ax=ax, label="Rank percentile within metric")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/comparison/shared_three_algorithms")
    parser.add_argument("--archive", type=Path, default=ROOT / "results/formal/a_shared_baseline_v1")
    args = parser.parse_args()
    bundle = load_input("formal")
    X = bundle.matrix(bundle.provenance["feature_columns"])
    archive = json.loads((args.archive / "experiment_manifest.json").read_text())
    source = archive["input"]
    if source["source_features_sha256"] != bundle.provenance["source_features_sha256"]:
        raise ValueError("A's archived source data does not match the current source")
    model = archive["model_input"]
    model_path = args.archive / model["filename"]
    if file_hash(model_path) != model["sha256"]:
        raise ValueError("A's archived model-input file hash does not match")
    saved_input = pd.read_csv(model_path, dtype={"CustomerID": "string"}, float_precision="round_trip")
    if matrix_hash(saved_input) != source["matrix_sha256"]:
        raise ValueError("A's archived model matrix hash does not match")
    if saved_input.CustomerID.tolist() != bundle.frame.CustomerID.tolist():
        raise ValueError("A's archived customer order does not match")
    archived_X = saved_input[bundle.provenance["feature_columns"]].to_numpy(dtype=np.float64)
    if not np.allclose(archived_X, X, rtol=0, atol=1e-12):
        raise ValueError("A's archived coordinates differ beyond floating-point rounding")
    max_input_difference = float(np.max(np.abs(archived_X - X)))
    X = archived_X
    config = archive["config"]
    k = config["baseline_k"]
    params = {
        "kmeans": dict(config["kmeans"], n_clusters=k),
        "gmm": dict(config["gmm"], n_components=k),
        "spectral": {"n_clusters": k, "affinity": "rbf", "gamma": 0.1},
    }
    old = pd.read_csv(args.archive / "summary.csv")
    rows, audit = [], []
    for algorithm, settings in params.items():
        for seed in config["seeds"]:
            start = perf_counter()
            labels = cluster(X, algorithm, settings, random_state=seed, threads=config["threads"])
            elapsed = perf_counter() - start
            metrics = evaluate_comparison(X, labels)
            rows.append(dict(algorithm=algorithm, seed=seed, k=k, customers=len(X),
                             runtime_seconds=elapsed, **metrics))
            if algorithm == "spectral":
                continue
            source, = old[(old.algorithm == algorithm) & (old.seed == seed)].itertuples(index=False)
            if json.loads(source.requested_params) != settings:
                raise ValueError(f"Archive parameters differ for {source.run_id}")
            saved = pd.read_csv(args.archive / source.run_id / "labels.csv", dtype={"CustomerID": "string"})
            aligned = saved.set_index("CustomerID").loc[bundle.frame.CustomerID, "Cluster"].to_numpy()
            rescored = evaluate_comparison(X, aligned)
            audit.append({
                "algorithm": algorithm, "seed": seed, "archive_run": source.run_id,
                "ari_archive_vs_rerun": adjusted_rand_score(aligned, labels),
                **{f"archive_{key}": getattr(source, key) for key in METRICS[:3]},
                **{f"rescored_{key}": rescored[key] for key in METRICS[:3]},
            })

    table = pd.DataFrame(rows)
    summary = table.groupby("algorithm", sort=False).agg(
        runs=("seed", "size"), coverage_mean=("coverage", "mean"),
        clusters_min=("observed_clusters", "min"), clusters_max=("observed_clusters", "max"),
        **{f"{metric}_{stat}": (metric, stat) for metric in METRICS for stat in ("mean", "std")},
    ).reset_index()
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "per_seed.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    pd.DataFrame(audit).to_csv(args.output / "a_archive_audit.csv", index=False)
    heatmap(summary, args.output / "heatmap.png", k, len(config["seeds"]))
    write_json(args.output / "manifest.json", {
        "input": source, "current_input": bundle.provenance, "model_input": model,
        "max_coordinate_difference": max_input_difference, "k": k, "seeds": config["seeds"], "threads": config["threads"],
        "params": params, "archive_summary_sha256": file_hash(args.archive / "summary.csv"),
        "timing": "wall time around cluster(), including validation, fit and label generation",
        "noise_policy": "exclude -1 from internal scores; report coverage and observed cluster count",
    })
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
