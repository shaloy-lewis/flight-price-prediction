"""Download the original public Kaggle archive and record source metadata.

No credentials are read. If anonymous access is unavailable, use Kaggle's official CLI
after configuring your own account and accepting applicable source terms.
"""

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://www.kaggle.com/api/v1/datasets/"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "kaggle.zip"
    urllib.request.urlretrieve(
        BASE + "download/shubhambathwal/flight-price-prediction?datasetVersionNumber=2", archive
    )
    expected = {"economy.csv", "business.csv", "Clean_Dataset.csv"}
    with zipfile.ZipFile(archive) as zipped:
        if not expected.issubset(zipped.namelist()):
            raise ValueError("Archive is missing the expected original CSV files")
        for name in sorted(expected):
            (args.output / name).write_bytes(zipped.read(name))
    with urllib.request.urlopen(BASE + "view/shubhambathwal/flight-price-prediction") as response:
        metadata = json.load(response)
    (args.output / "source_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("Downloaded Kaggle version 2; run python -m src.cli validate to record content hashes.")


if __name__ == "__main__":
    main()
