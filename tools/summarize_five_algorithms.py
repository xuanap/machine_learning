"""Assemble the five-algorithm report from saved runs, without fitting models."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
THREE = ROOT / "results/comparison/shared_three_algorithms"
B = ROOT / "results/b/full_20261124_v2"
OUTPUT = THREE.with_name("shared_five_algorithms")
METRICS = ("silhouette", "davies_bouldin", "calinski_harabasz")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def one(rows):
    row, = rows
    return row


def main():
    three_manifest = read_json(THREE / "manifest.json")
    b_manifest = read_json(B / "manifest.json")
    if (three_manifest["model_input"]["matrix_sha256"] != b_manifest["matrix"]["sha256_float64_c_order"]
            or three_manifest["current_input"]["source_features_sha256"]
            != b_manifest["sources_sha256"]["data/customer_features.csv"]):
        raise ValueError("Archive feature inputs differ")
    per_seed = read_csv(THREE / "per_seed.csv")
    rows = []
    for key, name in [("kmeans", "K-means"), ("gmm", "GMM"), ("spectral", "Spectral")]:
        group = [row for row in per_seed if row["algorithm"] == key]
        if sorted(int(row["seed"]) for row in group) != sorted(three_manifest["seeds"]):
            raise ValueError(f"Unexpected seed set for {key}")
        times = [float(row["runtime_seconds"]) for row in group]
        rows.append(dict(algorithm=name, params=json.dumps(three_manifest["params"][key]),
                         seeds="0/42/2026", runs=len(group), input_customers=4338,
                         evaluated_customers=4338, clusters=4, noise=0, coverage=1,
                         **{m: mean(float(row[m]) for row in group) for m in METRICS},
                         registry_wall_seconds=mean(times), registry_wall_sd=stdev(times),
                         b_native_seconds=None, b_timing_scope=None,
                         source=str((THREE / "per_seed.csv").relative_to(ROOT))))
    selection = read_json(B / "clustering/dbscan/selection.json")
    for name, filename, filters, scope in [
        ("Ward", "hierarchical/parameter_summary.csv", {"linkage": "ward", "k": "4"},
         "linkage + cut_tree"),
        ("DBSCAN", "dbscan/parameter_summary.csv", {"run_id": selection["run_id"]}, "estimator.fit"),
    ]:
        path = B / "clustering" / filename
        expected = b_manifest["outputs_sha256"][str(path.relative_to(B))]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Archive checksum mismatch for {path}")
        row = one(r for r in read_csv(path) if all(r[k] == v for k, v in filters.items()))
        params = {k: row[k] for k in (("linkage", "k") if name == "Ward" else ("eps", "min_samples"))}
        rows.append(dict(algorithm=name, params=json.dumps(params), seeds="42 (unused)",
                         runs=1, input_customers=4338, evaluated_customers=int(row["evaluated_customers"]),
                         clusters=int(row["effective_clusters"]), noise=int(row["noise_customers"]),
                         coverage=float(row["coverage"]), **{m: float(row[m]) for m in METRICS},
                         registry_wall_seconds=None, registry_wall_sd=None,
                         b_native_seconds=float(row["fit_seconds"]), b_timing_scope=scope,
                         source=str(path.relative_to(ROOT))))

    timing_path = OUTPUT / "timing/summary.csv"
    if timing_path.exists():
        timing_manifest = read_json(timing_path.with_name("manifest.json"))
        if timing_manifest["matrix_sha256"] != three_manifest["model_input"]["matrix_sha256"]:
            raise ValueError("Timing run used a different matrix")
        timing = {row["algorithm"]: row for row in read_csv(timing_path)}
        keys = ["kmeans", "gmm", "spectral", "hierarchical", "dbscan"]
        for row, key in zip(rows, keys):
            row.update(same_machine_wall_seconds=timing[key]["mean"],
                       same_machine_wall_sd=timing[key]["std"],
                       same_machine_runs=timing[key]["count"],
                       same_machine_timing_source=str(timing_path.relative_to(ROOT)))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "benchmark.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.75))
    names = [r["algorithm"] for r in rows[:-1]] + ["DBSCAN*"]
    columns = [
        ("silhouette", "Silhouette ↑", ".4f"),
        ("davies_bouldin", "Davies–Bouldin ↓", ".4f"),
        ("calinski_harabasz", "Calinski–Harabasz ↑", ".1f"),
        ("coverage", "Customer coverage (%)", ".2f"),
    ]
    for ax, (key, title, precision) in zip(axes.flat, columns):
        values = [r[key] * (100 if key == "coverage" else 1) for r in rows]
        bars = ax.barh(names, values, color=["#176581"] * 4 + ["#ac5c25"], height=.58)
        ax.bar_label(bars, labels=[format(v, precision) for v in values], padding=5, fontsize=10)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlim(0, max(values) * 1.23)
        ax.invert_yaxis()
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Customer clustering | Five archived baselines", fontsize=21, fontweight="bold")
    fig.tight_layout(rect=(.02, .115, .98, .93))
    fig.text(.04, .067, "* DBSCAN scores use 4,306 assigned customers (32 noise). Other scores use all 4,338.",
             fontsize=11)
    fig.text(.04, .033, "K-means, GMM and Spectral: three-seed means. Ward and DBSCAN: one archived run.",
             fontsize=11)
    fig.savefig(OUTPUT / "benchmark.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
