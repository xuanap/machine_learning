"""Partition agreement after ID alignment; matched overlap is not accuracy."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .customer_profiles import validate_clusters, validate_ids


def compare_partitions(left, right):
    prepared = []
    for name, frame in [("left", left), ("right", right)]:
        validate_ids(frame, name)
        if "Cluster" not in frame or frame.empty:
            raise ValueError(f"{name} must contain nonempty Cluster labels")
        current = frame[["CustomerID", "Cluster"]].copy()
        current["CustomerID"] = current.CustomerID.astype("string")
        validate_ids(current, name)
        current["Cluster"] = validate_clusters(current)
        prepared.append(current.rename(columns={"Cluster": name.title() + "Cluster"}))
    left, right = prepared
    if set(left.CustomerID) != set(right.CustomerID):
        raise ValueError("Both partitions must cover exactly the same customer set")
    customers = left.merge(right, on="CustomerID", how="left", validate="one_to_one", sort=False)
    overlap = pd.crosstab(customers.LeftCluster, customers.RightCluster)
    rows, columns = linear_sum_assignment(-overlap.to_numpy())
    matched = {int(overlap.columns[column]): int(overlap.index[row]) for row, column in zip(rows, columns)
               if overlap.iloc[row, column] > 0}
    mappings = []
    for cluster in overlap.columns:
        target = matched.get(int(cluster))
        count = int(overlap.loc[target, cluster]) if target is not None else 0
        total = int(overlap[cluster].sum())
        mappings.append({"RightCluster": int(cluster), "MatchedLeftCluster": target,
                         "OverlappedCustomers": count, "RightClusterCustomers": total,
                         "RightClusterMatchedFraction": count / total})
    mapping = pd.DataFrame(mappings)
    mapping["MatchedLeftCluster"] = mapping.MatchedLeftCluster.astype("Int64")
    customers["MatchedRightCluster"] = customers.RightCluster.map(matched).astype("Int64")
    customers["SameAfterMatching"] = customers.LeftCluster.eq(customers.MatchedRightCluster).fillna(False).astype(bool)
    summary = {
        "customers": len(customers), "left_clusters": len(overlap), "right_clusters": len(overlap.columns),
        "ari": float(adjusted_rand_score(customers.LeftCluster, customers.RightCluster)),
        "nmi": float(normalized_mutual_info_score(customers.LeftCluster, customers.RightCluster, average_method="arithmetic")),
        "matched_customers": int(customers.SameAfterMatching.sum()),
        "matched_overlap_fraction": float(customers.SameAfterMatching.mean()),
        "unmatched_right_clusters": [int(label) for label in overlap.columns if int(label) not in matched],
        "interpretation": "Partition agreement, not accuracy or real customer behavior change. One-to-one matching may leave split clusters unmatched.",
    }
    fractions = overlap.div(overlap.sum(axis=1), axis=0)
    return customers, overlap, fractions, mapping, summary
