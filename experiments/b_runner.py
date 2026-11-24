"""B's full-data clustering, noise-aware metrics and original-scale profiles."""
import json
import math

from analysis.b_eda import save_csv, save_figure

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from algorithms import dbscan, hierarchical
from experiments.group_metrics import evaluate_full
from preprocessing import feature_scaling
from preprocessing.feature_engineering import FEATURE_COLUMNS

EPS_GRID = [.25, .35, .45, .55, .65, .8, 1., 1.25, 1.5, 2.]
MIN_SAMPLES_GRID = [5, 10, 20]
SELECTION_RULE = ("Prefer >=2 clusters, coverage>=0.80 and smallest non-noise cluster "
                  ">=max(20,ceil(0.005*n)); rank by silhouette, then coverage, then "
                  "eps/min_samples ascending. If empty, relax cluster size, then coverage "
                  "(ranking coverage first); if no valid metrics, show maximum coverage descriptively.")


def load_shared_matrix(root, raw):
    ids, X = feature_scaling.load_customer_features(root / "data/customer_features.csv")
    if ids.astype("string").tolist() != raw.CustomerID.tolist():
        raise ValueError("Shared loader changed CustomerID order")
    if X.shape != (len(raw), 6) or not np.isfinite(X).all():
        raise ValueError("Shared preprocessing must return a finite six-dimensional matrix")
    X.setflags(write=False)
    return ids, X


def evaluate_partition(X, labels):
    labels = np.asarray(labels)
    if labels.shape != (len(X),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Labels must be one integer per input customer")
    mask = labels != -1
    groups, sizes = np.unique(labels[mask], return_counts=True)
    all_groups, all_sizes = np.unique(labels, return_counts=True)
    metrics = evaluate_full(X[mask], labels[mask])
    return {"effective_clusters": len(groups),
            "cluster_sizes": json.dumps({str(int(k)): int(v) for k, v in zip(all_groups, all_sizes)}),
            "min_cluster_size": int(sizes.min()) if len(sizes) else 0,
            "max_cluster_size": int(sizes.max()) if len(sizes) else 0,
            "coverage": float(mask.mean()) if len(labels) else 0.,
            "noise_fraction": float((~mask).mean()) if len(labels) else 0.,
            "noise_customers": int((~mask).sum()), "evaluated_customers": int(mask.sum()),
            "metrics_available": metrics["silhouette"] is not None, **metrics}


def compare_on_common_customers(X, ward_labels, dbscan_labels):
    """Re-evaluate both partitions on precisely the same non-noise customers."""
    mask = np.asarray(dbscan_labels) != -1
    rows = []
    for name, labels, coverage in [("ward_k4", ward_labels, 1.),
                                   ("dbscan_selected", dbscan_labels, float(mask.mean()))]:
        rows.append({"algorithm": name, "original_coverage": coverage,
                     "common_fraction_of_all_customers": float(mask.mean()),
                     "common_customers": int(mask.sum()),
                     **evaluate_partition(X[mask], np.asarray(labels)[mask])})
    return pd.DataFrame(rows), mask


def original_scale_profiles(raw, ids, labels, diagnostics, output):
    assigned = pd.DataFrame({"CustomerID": ids.astype("string"), "Cluster": labels})
    members = assigned.merge(raw, on="CustomerID", validate="one_to_one")
    members = members.merge(diagnostics.drop(columns="Monetary"), on="CustomerID", validate="one_to_one")
    if len(members) != len(raw):
        raise ValueError("Profiles must cover all customers, including noise")
    rows, summaries = [], []
    for cluster, group in members.groupby("Cluster", sort=True):
        stats = group[FEATURE_COLUMNS].describe().T
        stats.insert(0, "Cluster", int(cluster))
        rows.append(stats.rename_axis("Feature").reset_index())
        row = {"Cluster": int(cluster), "Customers": len(group), "CustomerShare": len(group) / len(raw),
               "PositiveSalesGBP": float(group.Monetary.sum()),
               "PositiveSalesShare": float(group.Monetary.sum() / raw.Monetary.sum()),
               "CustomersWithCancellation": int(group.CancelledRows.gt(0).sum()),
               "CancellationAmountAbs": float(group.CancellationAmountAbs.sum())}
        for column in FEATURE_COLUMNS:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_median"] = float(group[column].median())
        summaries.append(row)
    profiles, summary = pd.concat(rows, ignore_index=True), pd.DataFrame(summaries)
    save_csv(assigned, output / "labels.csv")
    save_csv(profiles, output / "feature_statistics.csv")
    save_csv(summary, output / "cluster_summary.csv")
    save_csv(profiles[profiles.Cluster.eq(-1)], output / "noise_feature_statistics.csv")
    save_csv(summary[summary.Cluster.eq(-1)], output / "noise_summary.csv")
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    for ax, column in zip(axes.flat, FEATURE_COLUMNS):
        ax.bar(summary.Cluster.astype(str), summary[f"{column}_median"],
               color=["#999999" if label == -1 else "#377eb8" for label in summary.Cluster])
        ax.set(xlabel="Cluster (-1 = noise)", ylabel=f"Median {column}", title="Original scale")
    save_figure(fig, output / "profile_medians.png")
    return summary


def select_dbscan(table, customers):
    available = table[table.metrics_available].copy()
    stable = available[(available.coverage >= .8) & (available.min_cluster_size >= max(20, math.ceil(.005 * customers)))]
    stage = "coverage_and_cluster_size"
    if stable.empty:
        stable = available[available.coverage >= .8]
        stage = "relaxed_cluster_size"
    if stable.empty:
        stable = available
        stage = "relaxed_coverage"
    if stable.empty:
        stable = table
        stage = "no_valid_multicluster_solution"
    columns = (["coverage", "silhouette", "eps", "min_samples"] if stage in {"relaxed_coverage", "no_valid_multicluster_solution"}
               else ["silhouette", "coverage", "eps", "min_samples"])
    return stable.sort_values(columns, ascending=[False, False, True, True], na_position="last").iloc[0], stage


def run_clustering(root, output, raw, progress=print):
    output.mkdir(parents=True)
    ids, X = load_shared_matrix(root, raw)
    snapshot = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    snapshot.insert(0, "CustomerID", ids.astype("string"))
    save_csv(snapshot, output / "model_input.csv")
    np.save(output / "model_input.npy", X)
    progress(f"Shared matrix: {len(X)} customers x 6 features; Ward K=4 baseline first.")
    ward = hierarchical.fit_predict(X, {"n_clusters": 4, "linkage": "ward"})
    rows = []
    for method in ["ward", "complete", "average"]:
        progress(f"Hierarchical: {method}, K=2..8 (full data and full metrics).")
        for k in range(2, 9):
            result = ward if method == "ward" and k == 4 else hierarchical.fit_predict(X, {"n_clusters": k, "linkage": method})
            rows.append({"linkage": method, "k": k, "fit_seconds": result.fit_seconds,
                         **evaluate_partition(X, result.labels)})
            save_csv(pd.DataFrame({"CustomerID": ids, "Cluster": result.labels}), output / "hierarchical" / f"{method}_k{k}_labels.csv")
            if k == 4:
                tree = result.diagnostics["linkage_matrix"]
                save_csv(pd.DataFrame(tree, columns=["Left", "Right", "Distance", "Customers"]), output / "hierarchical" / f"{method}_linkage.csv")
                fig, ax = plt.subplots(figsize=(13, 5))
                cut_height = (tree[-4, 2] + tree[-3, 2]) / 2
                dendrogram(tree, truncate_mode="lastp", p=35, show_leaf_counts=True,
                           color_threshold=cut_height, ax=ax)
                ax.axhline(cut_height, linestyle="--", color="black", label="K=4 cut height")
                ax.set(title=f"{method.title()} | full tree, last 35 merged groups", xlabel="Leaf customer / (merged customer count)", ylabel="Ward distance = sqrt(2 x SSE increment)" if method == "ward" else "Linkage distance (Euclidean)")
                ax.legend()
                save_figure(fig, output / "hierarchical" / f"{method}_dendrogram.png")
    hierarchy_table = pd.DataFrame(rows)
    save_csv(hierarchy_table, output / "hierarchical" / "parameter_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric in zip(axes.flat, ["silhouette", "davies_bouldin", "calinski_harabasz", "min_cluster_size"]):
        for method, group in hierarchy_table.groupby("linkage"):
            ax.plot(group.k, group[metric], "o-", label=method)
        ax.set(xlabel="K", ylabel=metric)
        ax.legend()
    save_figure(fig, output / "hierarchical" / "parameter_metrics.png")
    diagnostics = pd.read_csv(root / "results/tables/customer_cancellation_diagnostics.csv", dtype={"CustomerID": "string"})
    ward_profile = original_scale_profiles(raw, ids, ward.labels, diagnostics, output / "ward_k4")
    progress("DBSCAN: nearest-neighbor distance curves, including self in min_samples.")
    fig, ax = plt.subplots(figsize=(11, 5))
    distance_rows, quantiles = [], []
    for minimum in MIN_SAMPLES_GRID:
        with threadpool_limits(limits=1):
            distances = NearestNeighbors(n_neighbors=minimum, n_jobs=1).fit(X).kneighbors(X, return_distance=True)[0][:, -1]
        ordered = np.sort(distances)
        ax.plot(np.arange(1, len(X) + 1), ordered, label=f"min_samples={minimum} (self included)")
        distance_rows.append(pd.DataFrame({"Rank": np.arange(1, len(X) + 1), "min_samples": minimum, "Distance": ordered}))
        quantiles.append({"min_samples": minimum, **{f"q{int(q * 100)}": float(np.quantile(distances, q)) for q in [.5, .75, .9, .95, .99]}})
    ax.set(xlabel="Customers sorted by neighbor distance", ylabel="Euclidean distance in C's six-dimensional matrix", title="Neighbor-distance diagnostic; parameter scan follows")
    ax.legend()
    save_figure(fig, output / "dbscan" / "neighbor_distances.png")
    save_csv(pd.concat(distance_rows), output / "dbscan" / "neighbor_distances.csv")
    save_csv(pd.DataFrame(quantiles), output / "dbscan" / "neighbor_distance_quantiles.csv")
    db_rows, fitted = [], {}
    for minimum in MIN_SAMPLES_GRID:
        progress(f"DBSCAN scan: min_samples={minimum}, {len(EPS_GRID)} eps values.")
        for eps in EPS_GRID:
            result = dbscan.fit_predict(X, {"eps": eps, "min_samples": minimum})
            key = f"eps{eps:g}_min{minimum}"
            fitted[key] = result
            db_rows.append({"run_id": key, "eps": eps, "min_samples": minimum,
                            "fit_seconds": result.fit_seconds, **evaluate_partition(X, result.labels)})
            save_csv(pd.DataFrame({"CustomerID": ids, "Cluster": result.labels}), output / "dbscan" / f"{key}_labels.csv")
    db_table = pd.DataFrame(db_rows)
    save_csv(db_table, output / "dbscan" / "parameter_summary.csv")
    selected, stage = select_dbscan(db_table, len(X))
    chosen = fitted[selected.run_id]
    selection = {"run_id": selected.run_id, "eps": float(selected.eps),
                 "min_samples": int(selected.min_samples), "stage": stage, "rule": SELECTION_RULE}
    (output / "dbscan" / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    db_profile = original_scale_profiles(raw, ids, chosen.labels, diagnostics, output / "dbscan_selected")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, metric in zip(axes, ["silhouette", "coverage", "effective_clusters"]):
        table = db_table.pivot(index="min_samples", columns="eps", values=metric).reindex(index=MIN_SAMPLES_GRID, columns=EPS_GRID)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#ededed")
        im = ax.imshow(table.to_numpy(), aspect="auto", cmap=cmap)
        ax.set(xticks=range(len(EPS_GRID)), yticks=range(len(MIN_SAMPLES_GRID)),
               xticklabels=[f"{v:g}" for v in EPS_GRID], yticklabels=MIN_SAMPLES_GRID,
               xlabel="eps", ylabel="min_samples", title=metric)
        for i in range(len(MIN_SAMPLES_GRID)):
            for j in range(len(EPS_GRID)):
                value = table.iloc[i, j]
                ax.text(j, i, "NA" if pd.isna(value) else f"{value:.2f}" if metric != "effective_clusters" else f"{value:.0f}", ha="center", va="center", fontsize=7, color="black" if pd.isna(value) else "white")
        fig.colorbar(im, ax=ax)
    save_figure(fig, output / "dbscan" / "scan_metrics.png")
    common, mask = compare_on_common_customers(X, ward.labels, chosen.labels)
    save_csv(common, output / "comparison_common_customers.csv")
    save_csv(pd.DataFrame({"CustomerID": ids[mask], "WardCluster": ward.labels[mask], "DBSCANCluster": chosen.labels[mask]}), output / "comparison_customer_assignments.csv")
    scopes = pd.DataFrame([{"algorithm": "ward_k4", "scope": "all_customers", **evaluate_partition(X, ward.labels)},
                          {"algorithm": "dbscan_selected", "scope": "non_noise_customers", **evaluate_partition(X, chosen.labels)}])
    save_csv(scopes, output / "metrics_with_separate_scopes.csv")
    return {"matrix": X, "hierarchy": hierarchy_table, "dbscan": db_table,
            "selected": selected, "selection": selection, "ward_profile": ward_profile,
            "dbscan_profile": db_profile, "common": common}
