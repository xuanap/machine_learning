"""Audit recorded artifacts, exact input snapshots and selected full-data scores."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    source = ROOT / "data/customer_features.csv"
    source_ids = pd.read_csv(source, dtype={"CustomerID": "string"}).CustomerID.tolist()
    counts = {"runs": 0, "label_rows": 0, "gmm_probability_rows": 0, "full_metric_recomputations": 0}
    study_path = ROOT / "results/study/a_shared_v1/study_summary.json"
    study = json.loads(study_path.read_text(encoding="utf-8"))
    for name, expected_hash in study["source_manifests"].items():
        folder = ROOT / "results/formal" / name
        manifest_path = folder / "experiment_manifest.json"
        assert digest(manifest_path) == expected_hash, name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert digest(source) == manifest["input"]["source_features_sha256"]
        for relative, expected in manifest["code_sha256"].items():
            assert digest(ROOT / relative) == expected, relative
        model_path = folder / manifest["model_input"]["filename"]
        assert digest(model_path) == manifest["model_input"]["sha256"]
        model = pd.read_csv(model_path, dtype={"CustomerID": "string"}, float_precision="round_trip")
        assert model.CustomerID.tolist() == source_ids
        features = manifest["input"]["feature_columns"]
        values = model[features].to_numpy(dtype="<f8")
        matrix_digest = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
        assert matrix_digest == manifest["input"]["matrix_sha256"]
        table = pd.read_csv(folder / "summary.csv")
        for row in table.itertuples():
            run_folder = folder / row.run_id
            details = json.loads((run_folder / "run.json").read_text(encoding="utf-8"))
            assert row.status == details["summary"]["status"] == "ok"
            labels = pd.read_csv(run_folder / "labels.csv", dtype={"CustomerID": "string"})
            assert labels.CustomerID.tolist() == source_ids
            assert len(labels) == row.samples == 4338
            sizes = pd.read_csv(run_folder / "cluster_sizes.csv").set_index("Cluster").Customers
            assert sizes.to_dict() == labels.groupby("Cluster").size().to_dict()
            centers = pd.read_csv(run_folder / "centers.csv")
            assert len(centers) == labels.Cluster.nunique()
            if row.algorithm == "gmm":
                probability = pd.read_csv(run_folder / "probabilities.csv", dtype={"CustomerID": "string"})
                assert probability.CustomerID.tolist() == source_ids
                columns = [f"Probability_{i}" for i in range(len(centers))]
                p = probability[columns].to_numpy()
                assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
                np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-8, rtol=0)
                np.testing.assert_array_equal(labels.Cluster, p.argmax(axis=1))
                counts["gmm_probability_rows"] += len(p)
            if row.study == "baseline":
                with threadpool_limits(limits=1):
                    X = model[details["task"]["features"]].to_numpy()
                    expected_scores = [silhouette_score(X, labels.Cluster), davies_bouldin_score(X, labels.Cluster),
                                       calinski_harabasz_score(X, labels.Cluster)]
                np.testing.assert_allclose([row.silhouette, row.davies_bouldin, row.calinski_harabasz],
                                           expected_scores, rtol=1e-10, atol=1e-10)
                assert row.silhouette_samples == 4338
                counts["full_metric_recomputations"] += 1
            counts["runs"] += 1
            counts["label_rows"] += len(labels)
    for method in ["kmeans_seed42", "gmm_seed42"]:
        folder = ROOT / "results/analysis/formal/a_shared_baseline_v1" / method
        metadata = json.loads((folder / "analysis_manifest.json").read_text(encoding="utf-8"))
        source_folder = ROOT / "results/formal/a_shared_baseline_v1"
        paths = {"experiment_manifest.json": source_folder / "experiment_manifest.json",
                 "model_input.csv": source_folder / "model_input.csv"}
        for name, expected in metadata["artifacts_sha256"].items():
            actual = paths.get(name, source_folder / metadata["run_id"] / name)
            assert digest(actual) == expected, actual
        profiles = pd.read_csv(folder / "cluster_summary.csv")
        assert profiles.Customers.sum() == 4338
        np.testing.assert_allclose(profiles.CustomerShare.sum(), 1)
    assert counts["runs"] == study["runs"] == 78
    return {"passed": True, **counts, "source_sha256": digest(source),
            "scope": "Artifact and metric audit; not an independent model/code review"}


if __name__ == "__main__":
    result = verify()
    output = ROOT / "results/study/a_shared_v1/verification.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
