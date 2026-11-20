"""Download the unmodified UCI Online Retail workbook from its official archive."""
from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
PAGE = "https://archive.ics.uci.edu/dataset/352/online+retail"


def main() -> None:
    destination = ROOT / "data" / "Online Retail.xlsx"
    manifest_path = ROOT / "data" / "source_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Using existing workbook: {destination}", flush=True)
        return
    print(f"Downloading official archive: {URL}", flush=True)
    request = Request(URL, headers={"User-Agent": "Coursework-OnlineRetail/1.0"})
    with urlopen(request, timeout=90) as response:
        archive_bytes = response.read()
    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        matches = [name for name in archive.namelist() if Path(name).name == "Online Retail.xlsx"]
        if len(matches) != 1:
            raise ValueError(f"Expected one workbook; found {matches}")
        workbook_bytes = archive.read(matches[0])
    temporary = destination.with_suffix(".xlsx.part")
    temporary.write_bytes(workbook_bytes)
    temporary.replace(destination)
    manifest = {
        "dataset": "UCI Online Retail (id=352)",
        "page_url": PAGE,
        "download_url": URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "workbook_sha256": hashlib.sha256(workbook_bytes).hexdigest(),
        "workbook_bytes": len(workbook_bytes),
        "filename": destination.name,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
