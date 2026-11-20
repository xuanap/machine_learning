"""Customer profiles and boundary diagnostics with explicit ID joins."""
import numpy as np
import pandas as pd

from preprocessing.feature_engineering import FEATURE_COLUMNS


def validate_ids(frame, name):
    if "CustomerID" not in frame:
        raise ValueError(f"{name} is missing CustomerID")
    ids = frame["CustomerID"]
    if ids.isna().any() or ids.astype("string").str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError(f"{name} must have unique nonempty CustomerID")


def validate_clusters(labels):
    values = pd.to_numeric(labels["Cluster"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or not (values == np.floor(values)).all():
        raise ValueError("Cluster must contain nonnegative integer labels for K-means/GMM")
    return values.astype(int)


def build_profiles(raw_features, labels):
    validate_ids(raw_features, "raw_features")
    validate_ids(labels, "labels")
    if not set(FEATURE_COLUMNS) <= set(raw_features.columns) or "Cluster" not in labels:
        raise ValueError("Six original features and Cluster are required")
    if labels.empty:
        raise ValueError("Cannot profile an empty label table")
    labels = labels[["CustomerID", "Cluster"]].copy()
    labels["Cluster"] = validate_clusters(labels)
    joined = labels.merge(raw_features[["CustomerID", *FEATURE_COLUMNS]], on="CustomerID", how="left", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Some labelled customers are missing from the original feature table")
    joined = joined.drop(columns="_merge")
    if not np.isfinite(joined[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Original feature values must be finite")
    grouped = joined.groupby("Cluster", sort=True)
    summary = grouped.agg(Customers=("CustomerID", "size"), TotalMonetary=("Monetary", "sum"))
    summary["CustomerShare"] = summary["Customers"] / len(joined)
    total = joined["Monetary"].sum()
    summary["MonetaryShare"] = summary["TotalMonetary"] / total if total != 0 else np.nan
    for feature in FEATURE_COLUMNS:
        summary[f"{feature}_Mean"] = grouped[feature].mean()
        summary[f"{feature}_Median"] = grouped[feature].median()
    statistics = []
    for feature in FEATURE_COLUMNS:
        table = grouped[feature].agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
        table.insert(1, "Feature", feature)
        statistics.append(table)
    return joined, summary.reset_index(), pd.concat(statistics, ignore_index=True)


def gmm_boundaries(assignments, probabilities):
    validate_ids(assignments, "assignments")
    validate_ids(probabilities, "probabilities")
    if set(assignments["CustomerID"]) != set(probabilities["CustomerID"]):
        raise ValueError("GMM probability customers do not match label customers")
    columns = [column for column in probabilities if column.startswith("Probability_")]
    try:
        columns.sort(key=lambda name: int(name.split("_")[-1]))
    except ValueError as error:
        raise ValueError("Invalid GMM probability column name") from error
    if not columns or columns != [f"Probability_{index}" for index in range(len(columns))]:
        raise ValueError("GMM probabilities must cover consecutive component labels from zero")
    aligned = probabilities.set_index("CustomerID").loc[assignments["CustomerID"], columns]
    values = aligned.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any() or not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-8):
        raise ValueError("GMM probabilities must be finite, in [0,1], and sum to one")
    labels = validate_clusters(assignments)
    if not np.array_equal(labels, values.argmax(axis=1)):
        raise ValueError("GMM labels disagree with maximum posterior probabilities")
    ranked = np.argsort(-values, axis=1, kind="stable")
    rows = np.arange(len(values))
    result = assignments.copy()
    result["HighestProbability"] = values[rows, ranked[:, 0]]
    if len(columns) >= 2:
        result["SecondCluster"] = ranked[:, 1]
        result["SecondProbability"] = values[rows, ranked[:, 1]]
        result["ProbabilityGap"] = result["HighestProbability"] - result["SecondProbability"]
        result["NormalizedEntropy"] = -(values * np.log(np.clip(values, 1e-300, 1))).sum(axis=1) / np.log(len(columns))
    else:
        result["SecondCluster"] = pd.NA
        result["SecondProbability"] = np.nan
        result["ProbabilityGap"] = np.nan
        result["NormalizedEntropy"] = 0.0
    return result


def kmeans_boundaries(assignments, matrix, centers, columns):
    validate_ids(assignments, "assignments")
    validate_ids(matrix, "matrix")
    if "Cluster" not in centers or centers["Cluster"].duplicated().any():
        raise ValueError("Centers must have unique Cluster labels")
    center_labels = validate_clusters(centers)
    if set(center_labels) != set(range(len(center_labels))):
        raise ValueError("Centers must cover consecutive labels from zero")
    if not set(assignments["CustomerID"]) <= set(matrix["CustomerID"]):
        raise ValueError("Some customers have no matching model-input row")
    ordered_centers = centers.assign(Cluster=center_labels).sort_values("Cluster")
    values = matrix.set_index("CustomerID").loc[assignments["CustomerID"], columns].to_numpy(dtype=float)
    means = ordered_centers[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(means).all():
        raise ValueError("Model inputs and centers must be finite")
    distances = np.linalg.norm(values[:, None, :] - means[None, :, :], axis=2)
    ranked = np.argsort(distances, axis=1, kind="stable")
    labels = validate_clusters(assignments)
    if (labels >= len(means)).any():
        raise ValueError("Saved labels refer to missing centers")
    rows = np.arange(len(values))
    result = assignments.copy()
    result["AssignedCenterDistance"] = distances[rows, labels]
    result["NearestCluster"] = ranked[:, 0]
    result["NearestCenterDistance"] = distances[rows, ranked[:, 0]]
    result["AssignedIsNearest"] = labels == ranked[:, 0]
    if len(means) >= 2:
        result["SecondCluster"] = ranked[:, 1]
        result["SecondCenterDistance"] = distances[rows, ranked[:, 1]]
        result["DistanceGap"] = result["SecondCenterDistance"] - result["NearestCenterDistance"]
    else:
        result["SecondCluster"] = pd.NA
        result["SecondCenterDistance"] = np.nan
        result["DistanceGap"] = np.nan
    return result
