import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.utils.connections import get_fred_client

load_dotenv()

START_DATE = "2000-01-01"
END_DATE = "2024-12-31"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data/raw/macro"
MANIFEST_PATH = OUTPUT_DIR / "extraction_manifest.json"

DATASETS = (
    "treasury_rates",
    "gdp_growth",
    "fed_funds",
    "unemployment",
    "cpi",
    "credit_spreads",
    "vix",
)

DATASET_CONFIG = {
    "treasury_rates": {
        "fetcher": "fetch_treasury_rates",
        "filename": "treasury_rates.parquet",
        "fred_series_ids": ["DGS3MO", "DGS10"],
    },
    "gdp_growth": {
        "fetcher": "fetch_gdp_growth",
        "filename": "gdp_growth.parquet",
        "fred_series_ids": ["A191RL1Q225SBEA"],
    },
    "fed_funds": {
        "fetcher": "fetch_fed_funds",
        "filename": "fed_funds.parquet",
        "fred_series_ids": ["FEDFUNDS"],
    },
    "unemployment": {
        "fetcher": "fetch_unemployment",
        "filename": "unemployment.parquet",
        "fred_series_ids": ["UNRATE"],
    },
    "cpi": {
        "fetcher": "fetch_cpi",
        "filename": "cpi.parquet",
        "fred_series_ids": ["CPIAUCSL"],
    },
    "credit_spreads": {
        "fetcher": "fetch_credit_spreads",
        "filename": "credit_spreads.parquet",
        "fred_series_ids": ["AAA10Y", "BAA10Y"],
    },
    "vix": {
        "fetcher": "fetch_vix",
        "filename": "vix.parquet",
        "fred_series_ids": ["VIXCLS"],
    },
}


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def write_manifest(manifest: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = MANIFEST_PATH.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp_path, MANIFEST_PATH)


def safe_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    df.to_parquet(tmp_path)
    os.replace(tmp_path, path)


def validate_macro_frame(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Fetched macro dataset is empty")
    if df.index.name != "date":
        raise ValueError("Expected datetime index named date")
    index = pd.to_datetime(df.index, errors="coerce")
    if index.isna().any():
        raise ValueError("Date index contains unparseable values")
    if index.has_duplicates:
        raise ValueError("Date index contains duplicate values")


def validate_saved_file(path: Path, expected_rows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected output file was not created: {path}")
    saved = pd.read_parquet(path)
    validate_macro_frame(saved)
    if expected_rows is not None and len(saved) != expected_rows:
        raise ValueError(
            f"Saved row count mismatch for {path}: expected {expected_rows}, got {len(saved)}"
        )
    return saved


def is_valid_nonempty_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return len(validate_saved_file(path)) > 0
    except Exception:
        return False


def manifest_record(
    *,
    dataset: str,
    fred_series_ids: list[str],
    path: Path,
    status: str,
    df: pd.DataFrame | None = None,
    error: str | None = None,
) -> dict:
    record = {
        "dataset": dataset,
        "fred_series_ids": fred_series_ids,
        "requested_start_date": START_DATE,
        "requested_end_date": END_DATE,
        "actual_min_date": None,
        "actual_max_date": None,
        "row_count": None,
        "column_count": None,
        "missing_value_counts": None,
        "output_filepath": str(path),
        "file_size": path.stat().st_size if path.exists() else None,
        "status": status,
        "error_message": error,
        "extraction_timestamp_utc": utc_now(),
    }
    if df is not None:
        index = pd.to_datetime(df.index, errors="coerce").dropna()
        record["row_count"] = int(len(df))
        record["column_count"] = int(len(df.columns))
        record["missing_value_counts"] = {
            col: int(df[col].isna().sum()) for col in df.columns
        }
        if len(index):
            record["actual_min_date"] = index.min().date().isoformat()
            record["actual_max_date"] = index.max().date().isoformat()
    return record


def normalize_only(value: str | None) -> set[str]:
    if value is None or value == "all":
        return set(DATASETS)
    selected = {part.strip() for part in value.split(",") if part.strip()}
    unknown = selected.difference(DATASETS).difference({"all"})
    if unknown:
        raise ValueError(f"Unknown --only value(s): {', '.join(sorted(unknown))}")
    if "all" in selected:
        return set(DATASETS)
    return selected


def fetch_treasury_rates() -> pd.DataFrame:
    fred = get_fred_client()
    t3m = fred.get_series(
        "DGS3MO", observation_start=START_DATE, observation_end=END_DATE
    )
    t10y = fred.get_series(
        "DGS10", observation_start=START_DATE, observation_end=END_DATE
    )
    df = pd.DataFrame({"treasury_3m": t3m, "treasury_10y": t10y})
    df["term_spread"] = df["treasury_10y"] - df["treasury_3m"]
    df.index.name = "date"
    return df.sort_index()


def fetch_gdp_growth() -> pd.DataFrame:
    fred = get_fred_client()
    s = fred.get_series(
        "A191RL1Q225SBEA", observation_start=START_DATE, observation_end=END_DATE
    )
    df = s.rename("gdp_growth_qoq").to_frame()
    df.index.name = "date"
    return df.sort_index()


def fetch_fed_funds() -> pd.DataFrame:
    fred = get_fred_client()
    s = fred.get_series(
        "FEDFUNDS", observation_start=START_DATE, observation_end=END_DATE
    )
    df = s.rename("fed_funds_rate").to_frame()
    df.index.name = "date"
    return df.sort_index()


def fetch_unemployment() -> pd.DataFrame:
    fred = get_fred_client()
    s = fred.get_series(
        "UNRATE", observation_start=START_DATE, observation_end=END_DATE
    )
    df = s.rename("unemployment_rate").to_frame()
    df["unemployment_change"] = df["unemployment_rate"].diff()
    df.index.name = "date"
    return df.sort_index()


def fetch_cpi() -> pd.DataFrame:
    fred = get_fred_client()
    s = fred.get_series(
        "CPIAUCSL", observation_start=START_DATE, observation_end=END_DATE
    )
    df = s.rename("cpi").to_frame()
    df["cpi_yoy"] = df["cpi"].pct_change(12) * 100
    df.index.name = "date"
    return df.sort_index()


def fetch_credit_spreads() -> pd.DataFrame:
    fred = get_fred_client()
    aaa10y = fred.get_series(
        "AAA10Y",
        observation_start=START_DATE,
        observation_end=END_DATE,
    )
    baa10y = fred.get_series(
        "BAA10Y",
        observation_start=START_DATE,
        observation_end=END_DATE,
    )
    df = pd.DataFrame(
        {
            "aaa10y": aaa10y,
            "baa10y": baa10y,
        }
    )
    df.index.name = "date"
    return df.sort_index()


def fetch_vix() -> pd.DataFrame:
    fred = get_fred_client()
    s = fred.get_series(
        "VIXCLS",
        observation_start=START_DATE,
        observation_end=END_DATE,
    )
    df = s.rename("vix").to_frame()
    df.index.name = "date"
    return df.sort_index()


def extract_dataset(dataset: str, force: bool, manifest: dict) -> None:
    config = DATASET_CONFIG[dataset]
    path = OUTPUT_DIR / config["filename"]
    try:
        if not force and is_valid_nonempty_file(path):
            saved = validate_saved_file(path)
            log(f"Skipping valid cache: {path}")
            manifest[dataset] = manifest_record(
                dataset=dataset,
                fred_series_ids=config["fred_series_ids"],
                path=path,
                status="skipped",
                df=saved,
            )
            write_manifest(manifest)
            return

        log(f"Extracting FRED macro dataset: {dataset}")
        fetcher = globals()[config["fetcher"]]
        df = fetcher()
        validate_macro_frame(df)
        safe_write_parquet(df, path)
        saved = validate_saved_file(path, expected_rows=len(df))
        manifest[dataset] = manifest_record(
            dataset=dataset,
            fred_series_ids=config["fred_series_ids"],
            path=path,
            status="success",
            df=saved,
        )
        write_manifest(manifest)
    except Exception as exc:
        log(f"FAILED {dataset}: {exc}")
        manifest[dataset] = manifest_record(
            dataset=dataset,
            fred_series_ids=config["fred_series_ids"],
            path=path,
            status="failed",
            error=str(exc),
        )
        write_manifest(manifest)


def run_extraction(only: set[str], force: bool) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    for dataset in DATASETS:
        if dataset in only:
            extract_dataset(dataset, force, manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DynamicPD-CLO raw FRED macro extractor")
    parser.add_argument("--extract-macro", action="store_true")
    parser.add_argument(
        "--only",
        default="all",
        help=(
            "treasury_rates, gdp_growth, fed_funds, unemployment, cpi, "
            "credit_spreads, vix, or all"
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.extract_macro:
        return
    only = normalize_only(args.only)
    run_extraction(only, force=args.force)


if __name__ == "__main__":
    main()
