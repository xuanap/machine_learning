"""Shared customer feature preprocessing for clustering experiments."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from preprocessing.feature_engineering import FEATURE_COLUMNS

LOG_COLUMNS = ["Frequency", "Monetary", "AvgOrderValue", "ProductDiversity"]


def load_customer_features(path):
    """Load customer IDs and prepare the six clustering features."""
    customers = pd.read_csv(path, dtype={"CustomerID": str})
    ids = customers["CustomerID"]
    if ids.isna().any() or not ids.is_unique:
        raise ValueError("CustomerID must be present and unique for each row.")

    features = customers[FEATURE_COLUMNS].astype(float)
    if not np.isfinite(features.to_numpy()).all() or (features < 0).any().any():
        raise ValueError("Customer features must be finite and nonnegative.")
    features[LOG_COLUMNS] = np.log1p(features[LOG_COLUMNS])
    X = StandardScaler().fit_transform(features)
    return ids, X
