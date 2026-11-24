"""Plot spectral affinity, neighbor-count, and gamma experiments from CSV."""

import argparse
import csv
from pathlib import Path

from matplotlib.ticker import PercentFormatter

from experiments.plot_spectral_k import METRICS, ROOT, plt

TITLES = {
    "affinity": "Affinity comparison",
    "n_neighbors": "kNN neighbor count",
    "gamma": "RBF gamma",
}
OUTPUT_NAMES = {
    "affinity": "spectral_affinity_metrics.png",
    "n_neighbors": "spectral_n_neighbors_metrics.png",
    "gamma": "spectral_gamma_metrics.png",
}


def affinity_label(row):
    if row["affinity"] == "nearest_neighbors":
        return f"kNN\n{float(row['n_neighbors']):g} neighbors"
    if row["affinity"] == "rbf":
        return f"RBF\ngamma={float(row['gamma']):g}"
    raise ValueError(f"Unknown affinity: {row['affinity']}")


def plot_study(study, input_path, output_path):
    with input_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No configurations in {input_path}")

    categorical = study == "affinity"
    if categorical:
        x = list(range(len(rows)))
        labels = [affinity_label(row) for row in rows]
    else:
        rows.sort(key=lambda row: float(row[study]))
        x = [float(row[study]) for row in rows]
        labels = [f"{value:g}" for value in x]
    ks = {int(row["n_clusters"]) for row in rows}
    if len(ks) != 1:
        raise ValueError(f"Expected one fixed K in {input_path}")

    fig, axes = plt.subplots(2, 2, figsize=(9.7, 6.2), layout="constrained")
    panels = (*METRICS, ("smallest_cluster_share", "Smallest cluster share"))
    for ax, (metric, title) in zip(axes.flat, panels):
        values = [
            float(row[metric]) if row["status"] == "ok" and row.get(metric) else float("nan")
            for row in rows
        ]
        candidates = [(i, value) for i, value in enumerate(values) if value == value]
        best = set()
        if candidates and metric != "smallest_cluster_share":
            choose = min if metric == "davies_bouldin" else max
            target = choose(value for _, value in candidates)
            best = {i for i, value in candidates if value == target}

        if categorical:
            ax.bar(x, values, width=0.58,
                   color=["#b44e26" if i in best else "#2462a0" for i in range(len(x))])
        else:
            ax.plot(x, values, color="#2462a0", marker="o", markersize=5)
            ax.scatter([x[i] for i in best], [values[i] for i in best],
                       color="#b44e26", s=45, zorder=3)
        for i in sorted(best):
            precision = 2 if metric == "calinski_harabasz" else 4
            ax.annotate(f"{values[i]:.{precision}f}", (x[i], values[i]),
                        xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
        tick_indices = range(len(x))
        if study == "n_neighbors" and len(x) > 12:
            tick_indices = [i for i, value in enumerate(x) if value in (5, 50, 100, 200, 300, 400, 600)]
        ax.set(title=title, xticks=[x[i] for i in tick_indices],
               xticklabels=[labels[i] for i in tick_indices])
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        ax.margins(y=0.20)
        if not categorical:
            ax.set_xlabel(study)
        if metric == "smallest_cluster_share":
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    unavailable = sum(row["status"] != "ok" for row in rows)
    subtitle = f"K={ks.pop()} · Orange marks the best tested metric value"
    if unavailable:
        subtitle += f" · {unavailable} failed configuration(s)"
    fig.suptitle(f"{TITLES[study]}\n{subtitle}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=["all", *TITLES], default="all")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results/tables/spectral_parameters")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/figures")
    args = parser.parse_args()
    steps = TITLES if args.step == "all" else (args.step,)
    for study in steps:
        output = args.output_dir / OUTPUT_NAMES[study]
        plot_study(study, args.input_dir / f"{study}.csv", output)
        print(f"Saved to {output.resolve()}")


if __name__ == "__main__":
    main()
