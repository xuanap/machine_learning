"""Plot spectral clustering scores against K using the experiment CSV."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    ("silhouette", "Silhouette ↑"),
    ("davies_bouldin", "Davies–Bouldin ↓"),
    ("calinski_harabasz", "Calinski–Harabasz ↑"),
)


def plot_scores(input_path, output_path):
    with input_path.open(encoding="utf-8-sig", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["status"] == "ok"]

    series = []
    for affinity, label, parameter, color in (
        ("nearest_neighbors", "kNN", "n_neighbors", "#2462a0"),
        ("rbf", "RBF", "gamma", "#b44e26"),
    ):
        group = sorted(
            (row for row in rows if row["affinity"] == affinity),
            key=lambda row: int(row["n_clusters"]),
        )
        if not group:
            continue
        settings = {float(row[parameter]) for row in group}
        if len(settings) != 1:
            raise ValueError(f"Expected one fixed {parameter} value for {label}.")
        name = "neighbors" if affinity == "nearest_neighbors" else "gamma"
        series.append((group, f"{label} · {name}={settings.pop():g}", color))

    if not series:
        raise ValueError("The input CSV contains no successful kNN or RBF configurations.")

    ticks = sorted({int(row["n_clusters"]) for group, _, _ in series for row in group})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.45), layout="constrained")
    for ax, (key, title) in zip(axes, METRICS):
        for group, label, color in series:
            ax.plot(
                [int(row["n_clusters"]) for row in group],
                [float(row[key]) for row in group],
                marker="o", markersize=4, color=color, label=label,
            )
        ax.set(title=title, xlabel="Number of clusters (K)", xticks=ticks)
        ax.grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Spectral clustering · K sensitivity")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/tables/spectral_parameters/k.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/figures/spectral_k_metrics.png")
    args = parser.parse_args()
    plot_scores(args.input, args.output)
    print(f"Saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
