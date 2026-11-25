"""Measure the five fixed baselines on one machine through registry.cluster."""
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_info

from algorithms.registry import cluster

ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / "results/comparison"
A = ROOT / "results/formal/a_shared_baseline_v1"
B = ROOT / "results/b/full_20261124_v2/clustering"
OUT = COMPARISON / "shared_five_algorithms/timing"


def main():
    three = COMPARISON / "shared_three_algorithms"
    config = json.loads((three / "manifest.json").read_text())
    frame = pd.read_csv(A / "model_input.csv", dtype={"CustomerID": str}, float_precision="round_trip")
    ids = frame.pop("CustomerID")
    X = np.ascontiguousarray(frame.to_numpy(dtype=np.float64))
    digest = hashlib.sha256(X.tobytes()).hexdigest()
    if digest != config["model_input"]["matrix_sha256"] or not ids.is_unique:
        raise ValueError("Unexpected benchmark input")
    selection = json.loads((B / "dbscan/selection.json").read_text())
    params = dict(config["params"], hierarchical={"n_clusters": 4, "linkage": "ward"},
                  dbscan={key: selection[key] for key in ("eps", "min_samples")})

    paths = {(r.algorithm, r.seed): A / r.archive_run / "labels.csv"
             for r in pd.read_csv(three / "a_archive_audit.csv").itertuples()}
    for seed in config["seeds"]:
        paths["spectral", seed] = OUT.parent / "common_customers" / f"spectral_seed{seed}_full_labels.csv"
    reference = {}
    for (name, seed), path in paths.items():
        reference[name, seed] = path
    reference.update({("hierarchical", 42): B / "ward_k4/labels.csv",
                      ("dbscan", 42): B / "dbscan_selected/labels.csv"})
    expected = {}
    for name, path in reference.items():
        labels = pd.read_csv(path, dtype={"CustomerID": str}).set_index("CustomerID")["Cluster"]
        if not labels.index.is_unique or set(labels.index) != set(ids):
            raise ValueError(f"Customer IDs differ in {path}")
        expected[name] = labels.reindex(ids)

    # Warm each entry once. Rotate execution order during the three measured rounds.
    names = list(params)
    for name in names:
        cluster(X, name, params[name], random_state=42, threads=1)
    rows = []
    for round_index, seed in enumerate(config["seeds"]):
        for order, name in enumerate(names[round_index:] + names[:round_index], start=1):
            start = perf_counter()
            labels = cluster(X, name, params[name], random_state=seed, threads=1)
            seconds = perf_counter() - start
            reference_seed = 42 if name in ("hierarchical", "dbscan") else seed
            ari = adjusted_rand_score(expected[name, reference_seed], labels)
            rows.append(dict(algorithm=name, round=round_index + 1, order=order, seed=seed,
                             wall_seconds=seconds, ari_reference=ari, noise=int((labels == -1).sum()),
                             clusters=len(set(labels) - {-1})))
    OUT.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "per_run.csv", index=False)
    summary = runs.groupby("algorithm")["wall_seconds"].agg(["count", "mean", "std", "min", "max"]).reindex(names)
    summary.to_csv(OUT / "summary.csv")
    (OUT / "manifest.json").write_text(json.dumps({
        "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version, "libraries": {pkg: version(pkg) for pkg in
            ("numpy", "scipy", "scikit-learn", "pandas", "threadpoolctl")},
        "threadpool_libraries": threadpool_info(), "threads": 1,
        "matrix_sha256": digest, "params": params, "seeds_by_round": config["seeds"],
        "warmups_per_algorithm": 1, "timing": "perf_counter around algorithms.registry.cluster only",
        "label_references": {f"{name}_{seed}": str(path.relative_to(ROOT))
                             for (name, seed), path in reference.items()},
    }, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    y = np.arange(len(summary))
    ax.barh(y, summary["mean"], xerr=summary["std"], capsize=4, color="#176581", height=.55)
    ax.set_yticks(y, ["K-means", "GMM", "Spectral", "Ward", "DBSCAN"])
    ax.invert_yaxis()
    for i, (avg, sd) in enumerate(zip(summary["mean"], summary["std"])):
        ax.annotate(f"{avg:.3f} ± {sd:.3f}", (avg + sd, i), xytext=(7, 0),
                    textcoords="offset points", va="center")
    ax.set_xlim(0, (summary["mean"] + summary["std"]).max() * 1.32)
    ax.set_title("Customer clustering | Same-machine runtime")
    ax.set_xlabel("Wall-clock seconds · 3 runs per algorithm (mean ± sample SD)")
    fig.text(.5, .02, "4,338 customers · one shared registry entry point · single-threaded",
             ha="center")
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(OUT / "runtime.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
