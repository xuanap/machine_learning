"""Read raw data for smoke checks or C's versioned prepared data for formal runs."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.feature_engineering import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class ExperimentInput:
    frame: pd.DataFrame
    provenance: dict
    mode: str

    def matrix(self, columns):
        if not columns or len(set(columns)) != len(columns) or not set(columns) <= set(FEATURE_COLUMNS):
            raise ValueError("Select distinct columns from the agreed six features")
        return self.frame[columns].to_numpy(dtype=np.float64)


def load_input(mode, input_path=None, metadata_path=None, sample_size=300, sample_seed=17):
    if mode not in {"smoke", "formal"}:
        raise ValueError("mode must be smoke or formal")
    manifest = json.loads((ROOT / "data/dataset_manifest.json").read_text(encoding="utf-8"))
    raw_path = ROOT / "data/customer_features.csv"
    source_hash = file_hash(raw_path)
    if source_hash != manifest["files"]["data/customer_features.csv"]["sha256"]:
        raise ValueError("Shared source features differ from the data manifest")
    if mode == "formal" and (input_path is None or metadata_path is None):
        raise ValueError("Formal runs require C's prepared CSV and preprocessing metadata; use --mode smoke to test code")
    path = Path(input_path) if input_path is not None else raw_path
    frame = pd.read_csv(path, dtype={"CustomerID": "string"}, encoding="utf-8-sig")
    if set(frame.columns) != {"CustomerID", *FEATURE_COLUMNS} or len(frame.columns) != 7:
        raise ValueError("Input must contain exactly CustomerID and the six agreed feature columns")
    ids = frame["CustomerID"]
    if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError("CustomerID must be nonempty and unique")
    for column in FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Prepared features contain missing or infinite values")
    provenance = {"input_filename": path.name, "input_sha256": file_hash(path),
                  "source_features_sha256": source_hash, "data_version": manifest["data_version"],
                  "feature_columns": FEATURE_COLUMNS, "original_input_rows": len(frame)}
    if mode == "formal":
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("Preprocessing metadata must be a JSON object")
        for field, expected in [("data_version", manifest["data_version"]),
                                ("source_features_sha256", source_hash),
                                ("prepared_csv_sha256", provenance["input_sha256"]),
                                ("feature_columns", FEATURE_COLUMNS)]:
            if metadata.get(field) != expected:
                raise ValueError(f"C's metadata does not match {field}")
        preprocessing = metadata.get("preprocessing", {})
        if not isinstance(preprocessing, dict):
            raise ValueError("preprocessing must be a JSON object")
        if preprocessing.get("scaler") not in {"StandardScaler", "RobustScaler", "MinMaxScaler"}:
            raise ValueError("Specify C's feature-wise scaler (StandardScaler, RobustScaler or MinMaxScaler)")
        if preprocessing.get("representation") != "scaled_features" or not preprocessing.get("version"):
            raise ValueError("Baseline inputs must be versioned scaled features, not PCA plotting coordinates")
        log_columns = preprocessing.get("log1p_columns")
        if not isinstance(log_columns, list) or not set(log_columns) <= set(FEATURE_COLUMNS):
            raise ValueError("Declare the log1p column list, including [] when not used")
        raw_ids = pd.read_csv(raw_path, usecols=["CustomerID"], dtype="string")["CustomerID"]
        if set(ids) != set(raw_ids) or len(ids) != len(raw_ids):
            raise ValueError("C's prepared data must cover the same customer set as the shared source")
        if provenance["input_sha256"] == source_hash:
            raise ValueError("The formal input is the unchanged raw feature CSV")
        provenance["preprocessing"] = preprocessing
        provenance["preprocessing_metadata_sha256"] = file_hash(metadata_path)
    else:
        if not isinstance(sample_size, int) or sample_size < 2:
            raise ValueError("smoke sample_size must be at least 2")
        frame = frame.sample(n=min(sample_size, len(frame)), random_state=sample_seed).sort_index()
        provenance.update({"preprocessing": "none applied by A; diagnostic input only",
                           "sample_seed": sample_seed,
                           "notice": "SMOKE TEST ONLY: no model-quality conclusions or best-parameter selection"})
    frame = frame.reset_index(drop=True)
    provenance["used_rows"] = len(frame)
    provenance["ordered_customer_ids_sha256"] = hashlib.sha256("\n".join(frame.CustomerID).encode()).hexdigest()
    return ExperimentInput(frame, provenance, mode)
