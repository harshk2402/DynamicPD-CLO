import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.wrds import WRDSClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data/raw/wrds/sp_issuer_ratings.parquet"
MANIFEST_PATH = PROJECT_ROOT / "data/raw/wrds/sp_issuer_ratings_manifest.json"

# S&P withdrew its ratings from Compustat in Feb 2017, so this monthly issuer-level table cannot serve
# as a label source for 2019-2024. It is kept as the external benchmark that validates the ratings
# reconstructed from FISD bond ratings.
SQL = """
    SELECT gvkey, datadate, splticrm
    FROM comp.adsprate
    WHERE splticrm IS NOT NULL
"""


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def extract(force: bool = False) -> dict:
    if OUTPUT_PATH.exists() and not force:
        log(f"{OUTPUT_PATH} exists; pass --force to re-extract")
        return json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}

    df = WRDSClient().query(SQL)
    df["gvkey"] = df["gvkey"].astype("string")
    df["datadate"] = pd.to_datetime(df["datadate"])
    df["splticrm"] = df["splticrm"].astype("string").str.strip()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_PATH.with_name(f".{OUTPUT_PATH.name}.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, OUTPUT_PATH)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "comp.adsprate",
        "rows": int(len(df)),
        "firms": int(df["gvkey"].nunique()),
        "date_range": [str(df["datadate"].min().date()), str(df["datadate"].max().date())],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    log(f"saved {len(df):,} rows to {OUTPUT_PATH}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract S&P long-term issuer ratings from Compustat.")
    parser.add_argument("--force", action="store_true")
    extract(force=parser.parse_args().force)


if __name__ == "__main__":
    main()
