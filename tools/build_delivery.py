"""Package the B/C handoff with an explicit file allowlist and verified zip entries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((ROOT / "results/tables/independent_validation.json").read_text(encoding="utf-8"))
    if report["status"] != "passed":
        raise ValueError("Run independent verification before packaging")
    manifest = json.loads((ROOT / "data/dataset_manifest.json").read_text(encoding="utf-8"))
    for name, metadata in manifest["files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError(f"Data changed after manifest generation: {name}")
    names = ["README.md", "requirements.txt", "prepare_data.py",
             "config/data_config.json", "data/source_manifest.json", "data/dataset_manifest.json",
             "data/cleaned_data.csv", "data/customer_features.csv", "data/customer_dates.csv", "data/excluded_rows.csv.gz"]
    for folder, suffix in [("preprocessing", ".py"), ("tools", ".py"), ("tests", ".py"),
                           ("docs", ".md"), ("results/tables", None)]:
        names.extend(str(path.relative_to(ROOT)).replace("\\", "/") for path in sorted((ROOT / folder).iterdir())
                     if path.is_file() and (suffix is None or path.suffix == suffix))
    names = sorted(set(names))
    destination = ROOT / "delivery/Customer_Clustering_BC_v1.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name in names:
            archive.write(ROOT / name, "Customer_Clustering/" + name)
    with ZipFile(destination) as archive:
        bad_entry = archive.testzip()
        if bad_entry:
            raise ValueError(f"Corrupt zip entry: {bad_entry}")
        for name in names:
            if hashlib.sha256(archive.read("Customer_Clustering/" + name)).digest() != hashlib.sha256((ROOT / name).read_bytes()).digest():
                raise ValueError(f"Packaged file differs: {name}")
    summary = {"filename": destination.name, "bytes": destination.stat().st_size,
               "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "entries": len(names)}
    destination.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
