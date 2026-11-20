"""Plot parameter summaries, keeping test runs visibly labelled."""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summarize_runs(table, metrics=("silhouette", "fit_seconds")):
    """Count scheduled runs, including failures; keep undefined sample std missing."""
    records = []
    keys = ["study", "algorithm", "factor", "value"]
    for group, rows in table.groupby(keys, sort=False, dropna=False):
        for metric in metrics:
            values = pd.to_numeric(rows[metric], errors="coerce") if metric in rows else pd.Series(np.nan, index=rows.index)
            usable = rows.status.eq("ok") & np.isfinite(values)
            valid = values.loc[usable]
            records.append(dict(zip(keys, group), metric=metric, planned_runs=len(rows),
                                valid_runs=len(valid), non_ok_runs=int(rows.status.ne("ok").sum()),
                                unavailable_metric_runs=int((rows.status.eq("ok") & ~np.isfinite(values)).sum()),
                                mean=float(valid.mean()) if len(valid) else np.nan,
                                std=float(valid.std(ddof=1)) if len(valid) >= 2 else np.nan))
    return pd.DataFrame(records)


def plot_summary(table, destination, mode):
    destination.mkdir(parents=True, exist_ok=True)
    support = summarize_runs(table)
    support.to_csv(destination / "plot_support.csv", index=False, encoding="utf-8-sig")
    for (study, metric), subset in support.groupby(["study", "metric"], sort=False):
            fig, axis = plt.subplots(figsize=(8, 4.6), layout="constrained")
            categories = list(dict.fromkeys(subset.value.tolist()))
            for series, (algorithm, rows) in enumerate(subset.groupby("algorithm", sort=False)):
                summary = rows.reset_index(drop=True)
                if rows.factor.iloc[0] == "K":
                    positions = summary.value.astype(float)
                else:
                    positions = [categories.index(value) for value in summary.value]
                    axis.set_xticks(np.arange(len(categories)), [str(value) for value in categories])
                line, = axis.plot(positions, summary["mean"], marker="o", label=algorithm)
                repeated = summary.valid_runs.ge(2)
                if repeated.any():
                    axis.errorbar(np.asarray(positions)[repeated], summary.loc[repeated, "mean"],
                                  yerr=summary.loc[repeated, "std"], fmt="none", capsize=4, color=line.get_color())
                for position, row in zip(positions, summary.itertuples()):
                    label = f"n={row.valid_runs}/{row.planned_runs}"
                    if row.valid_runs == 1:
                        label += "\nstd n/a"
                    if row.valid_runs:
                        axis.annotate(label, (position, row.mean), xytext=(0, 10 + series * 15),
                                      textcoords="offset points", ha="center", fontsize=8, color=line.get_color())
                    else:
                        axis.text(position, .03 + series * .08, label, transform=axis.get_xaxis_transform(),
                                  ha="center", fontsize=8, color=line.get_color())
            axis.set_xlabel(subset.factor.iloc[0])
            if subset.factor.iloc[0] == "K":
                coordinates = [float(value) for value in categories]
                padding = max(max(coordinates) - min(coordinates), 1) * .2
                axis.set_xlim(min(coordinates) - padding, max(coordinates) + padding)
            else:
                axis.set_xlim(-.4, max(len(categories) - 1, 0) + .4)
            axis.set_ylabel("Fit time (seconds)" if metric == "fit_seconds" else "Silhouette (local reference)")
            prefix = "SMOKE ONLY - raw diagnostic data" if mode == "smoke" else "Prepared input - local metric protocol"
            axis.set_title(f"{prefix}\n{study}")
            axis.grid(alpha=.2)
            axis.margins(y=.25)
            axis.text(.01, -.22, "n = valid / planned runs; sample std is available only with at least 2 valid runs",
                      transform=axis.transAxes, fontsize=8)
            axis.legend()
            fig.savefig(destination / f"{study}_{metric}.png", dpi=150)
            plt.close(fig)
    return support
