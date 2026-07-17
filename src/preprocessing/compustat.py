import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUARTERLY_DIR = PROJECT_ROOT / "data/raw/wrds/compustat_quarterly"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/compustat"
OUTPUT_PATH = OUTPUT_DIR / "compustat_features.parquet"
MANIFEST_PATH = OUTPUT_DIR / "compustat_features_manifest.json"

DEFAULT_START_YEAR = 2000
DEFAULT_END_YEAR = 2024
RAW_LOOKBACK_YEAR = 1999

STANDARD_FILTERS = {
    "indfmt": "INDL",
    "datafmt": "STD",
    "popsrc": "D",
    "consol": "C",
}

RAW_COLUMNS = (
    "gvkey",
    "datadate",
    "fyearq",
    "fqtr",
    "fyr",
    "datafqtr",
    "datacqtr",
    "rdq",
    "consol",
    "indfmt",
    "datafmt",
    "popsrc",
    "costat",
    "sic",
    "sich",
    "atq",
    "ltq",
    "dlttq",
    "dlcq",
    "xintq",
    "oibdpq",
    "niq",
    "actq",
    "lctq",
    "cheq",
)

REQUIRED_COLUMNS = (
    "gvkey",
    "datadate",
    "rdq",
    "indfmt",
    "datafmt",
    "popsrc",
    "consol",
    "atq",
    "dlttq",
    "dlcq",
    "xintq",
    "oibdpq",
    "niq",
    "actq",
    "lctq",
    "cheq",
)

FEATURE_COLUMNS = (
    "leverage",
    "coverage",
    "profitability",
    "liquidity",
    "log_assets",
    "cash_ratio",
)

OUTPUT_COLUMNS = (
    "gvkey",
    "quarter_end",
    "datadate",
    "rdq",
    "fyearq",
    "fqtr",
    "datafqtr",
    "datacqtr",
    "atq",
    "ltq",
    "total_debt",
    "dlttq",
    "dlcq",
    "xintq",
    "oibdpq",
    "niq",
    "actq",
    "lctq",
    "cheq",
    "accounting_age_days",
    "rdq_missing",
    "debt_component_missing",
    "coverage_missing",
    *FEATURE_COLUMNS,
)


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def parquet_columns(path: Path) -> list[str]:
    return pq.read_schema(path).names


def yearly_files(raw_start_year: int, end_year: int) -> list[Path]:
    files = []
    for year in range(raw_start_year, end_year + 1):
        path = QUARTERLY_DIR / f"compustat_quarterly_{year}.parquet"
        if path.exists():
            files.append(path)
        else:
            log(f"Missing raw Compustat quarterly partition for {year}: {path}")
    return files


def read_raw_partition(path: Path) -> pd.DataFrame:
    available = parquet_columns(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in available]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    selected = [col for col in RAW_COLUMNS if col in available]
    try:
        return pd.read_parquet(path, columns=selected)
    except OSError as exc:
        if "Repetition level histogram size mismatch" not in str(exc):
            raise
        log(f"PyArrow could not read {path.name}; retrying with fastparquet")
        return pd.read_parquet(path, columns=selected, engine="fastparquet")


def fiscal_quarter_end(year: int, quarter: int) -> pd.Timestamp:
    return pd.Period(f"{year}Q{quarter}", freq="Q").to_timestamp(
        freq="D",
        how="end",
    ).normalize()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gvkey"] = df["gvkey"].astype("string")
    df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
    df["rdq"] = pd.to_datetime(df["rdq"], errors="coerce")
    df["rdq_missing"] = df["rdq"].isna()

    for col in (
        "atq",
        "ltq",
        "dlttq",
        "dlcq",
        "xintq",
        "oibdpq",
        "niq",
        "actq",
        "lctq",
        "cheq",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    debt_parts = df[["dlttq", "dlcq"]]
    df["debt_component_missing"] = debt_parts.isna().any(axis=1)
    df["total_debt"] = debt_parts.fillna(0).sum(axis=1)
    df.loc[debt_parts.isna().all(axis=1), "total_debt"] = np.nan

    atq_positive = df["atq"] > 0
    xint_positive = df["xintq"] > 0
    lct_positive = df["lctq"] > 0

    df["leverage"] = np.where(atq_positive, df["total_debt"] / df["atq"], np.nan)
    df["coverage"] = np.where(xint_positive, df["oibdpq"] / df["xintq"], np.nan)
    df["profitability"] = np.where(atq_positive, df["niq"] / df["atq"], np.nan)
    df["liquidity"] = np.where(lct_positive, df["actq"] / df["lctq"], np.nan)
    df["log_assets"] = np.nan
    df.loc[atq_positive, "log_assets"] = np.log(df.loc[atq_positive, "atq"])
    df["cash_ratio"] = np.where(atq_positive, df["cheq"] / df["atq"], np.nan)
    df["coverage_missing"] = df["coverage"].isna()

    return df


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    metrics = {
        "raw_rows": int(len(df)),
        "after_standard_filters": None,
        "after_sic_exclusions": None,
        "after_required_dates": None,
        "missing_rdq_rows": None,
        "missing_rdq_percent": None,
        "exact_duplicate_rows": None,
        "duplicate_gvkey_datadate_keys": None,
        "duplicate_observations_dropped": None,
        "sic_source": None,
        "sic_rows_from_sic": 0,
        "sic_rows_from_sich": 0,
        "sic_exclusion_applied": False,
    }

    out = df.copy()
    for col, expected in STANDARD_FILTERS.items():
        out = out[out[col] == expected]
    metrics["after_standard_filters"] = int(len(out))

    sic = pd.Series(pd.NA, index=out.index, dtype="Float64")
    if "sic" in out.columns:
        sic_from_sic = pd.to_numeric(out["sic"], errors="coerce")
        sic = sic_from_sic.astype("Float64")
        metrics["sic_rows_from_sic"] = int(sic_from_sic.notna().sum())
    if "sich" in out.columns:
        sic_from_sich = pd.to_numeric(out["sich"], errors="coerce")
        fallback_mask = sic.isna() & sic_from_sich.notna()
        sic.loc[fallback_mask] = sic_from_sich.loc[fallback_mask]
        metrics["sic_rows_from_sich"] = int(fallback_mask.sum())

    if sic.notna().any():
        if metrics["sic_rows_from_sic"] and metrics["sic_rows_from_sich"]:
            metrics["sic_source"] = "sic_with_sich_fallback"
        elif metrics["sic_rows_from_sic"]:
            metrics["sic_source"] = "sic"
        else:
            metrics["sic_source"] = "sich"
        excluded = sic.between(6000, 6999, inclusive="both") | sic.between(
            4900,
            4999,
            inclusive="both",
        )
        out = out[~excluded.fillna(False)]
        metrics["sic_exclusion_applied"] = True
    metrics["after_sic_exclusions"] = int(len(out))

    out = compute_features(out)
    metrics["missing_rdq_rows"] = int(out["rdq_missing"].sum())
    metrics["missing_rdq_percent"] = (
        float(out["rdq_missing"].mean() * 100) if len(out) else 0.0
    )
    out = out.dropna(subset=["gvkey", "datadate", "rdq"])
    metrics["after_required_dates"] = int(len(out))

    before_dedup = len(out)
    metrics["exact_duplicate_rows"] = int(out.duplicated().sum())
    duplicate_keys = out.loc[
        out.duplicated(["gvkey", "datadate"], keep=False),
        ["gvkey", "datadate"],
    ]
    metrics["duplicate_gvkey_datadate_keys"] = int(
        duplicate_keys.drop_duplicates().shape[0]
    )
    sort_cols = [col for col in ["gvkey", "datadate", "rdq", "datafqtr", "datacqtr"] if col in out.columns]
    out = out.sort_values(sort_cols)
    out = out.drop_duplicates(["gvkey", "datadate"], keep="last")
    metrics["duplicate_observations_dropped"] = int(before_dedup - len(out))

    keep_cols = [col for col in OUTPUT_COLUMNS if col in out.columns]
    return out[keep_cols], metrics


def load_feature_observations(raw_start_year: int, end_year: int) -> tuple[pd.DataFrame, list[dict]]:
    frames = []
    partition_metrics = []

    for path in yearly_files(raw_start_year, end_year):
        year = int(path.stem.rsplit("_", 1)[-1])
        log(f"Loading Compustat quarterly partition {year}")
        raw = read_raw_partition(path)
        processed, metrics = apply_filters(raw)
        metrics["year"] = year
        metrics["output_rows"] = int(len(processed))
        partition_metrics.append(metrics)
        if not processed.empty:
            frames.append(processed)

    if not frames:
        raise RuntimeError("No Compustat observations remained after filtering")

    observations = pd.concat(frames, ignore_index=True)
    observations = observations.sort_values(["gvkey", "rdq", "datadate"])
    return observations, partition_metrics


def quarter_ends(start_year: int, end_year: int) -> list[pd.Timestamp]:
    periods = pd.period_range(
        start=f"{start_year}Q1",
        end=f"{end_year}Q4",
        freq="Q",
    )
    return [p.to_timestamp(freq="D", how="end").normalize() for p in periods]


def aligned_quarter(observations: pd.DataFrame, quarter_end: pd.Timestamp) -> pd.DataFrame:
    eligible = observations[
        (observations["rdq"] <= quarter_end)
        & (observations["datadate"] <= quarter_end)
    ]
    if eligible.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    aligned = eligible.drop_duplicates("gvkey", keep="last").copy()
    aligned.insert(1, "quarter_end", quarter_end)
    aligned["accounting_age_days"] = (
        aligned["quarter_end"] - aligned["rdq"]
    ).dt.days
    return aligned[[col for col in OUTPUT_COLUMNS if col in aligned.columns]]


def write_aligned_panel(
    observations: pd.DataFrame,
    start_year: int,
    end_year: int,
    output_path: Path,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    writer: pq.ParquetWriter | None = None
    rows_by_year: dict[str, int] = {}
    total_rows = 0

    try:
        for quarter_end in quarter_ends(start_year, end_year):
            aligned = aligned_quarter(observations, quarter_end)
            if aligned.empty:
                rows_by_year.setdefault(str(quarter_end.year), 0)
                continue
            table = pa.Table.from_pandas(aligned, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema)
            else:
                table = table.cast(writer.schema)
            writer.write_table(table)
            row_count = len(aligned)
            rows_by_year[str(quarter_end.year)] = rows_by_year.get(str(quarter_end.year), 0) + row_count
            total_rows += row_count
            log(f"Aligned Compustat features for {quarter_end.date()}: {row_count:,} rows")
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise RuntimeError("No aligned Compustat rows were written")

    os.replace(tmp_path, output_path)
    return {"row_count": total_rows, "rows_by_year": rows_by_year}


def validate_output(path: Path, start_year: int, end_year: int) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Output file was not created: {path}")

    keys = pd.read_parquet(path, columns=["gvkey", "quarter_end"])
    keys["quarter_end"] = pd.to_datetime(keys["quarter_end"], errors="coerce")
    missing_dates = int(keys["quarter_end"].isna().sum())
    duplicate_keys = int(keys.duplicated(["gvkey", "quarter_end"]).sum())
    start = pd.Timestamp(f"{start_year}-01-01")
    end = pd.Timestamp(f"{end_year}-12-31")
    outside = int(((keys["quarter_end"] < start) | (keys["quarter_end"] > end)).sum())

    if missing_dates:
        raise ValueError(f"Output contains {missing_dates:,} missing quarter_end values")
    if duplicate_keys:
        raise ValueError(f"Output contains {duplicate_keys:,} duplicate gvkey-quarter rows")
    if outside:
        raise ValueError(f"Output contains {outside:,} rows outside requested date range")

    parquet_file = pq.ParquetFile(path)
    return {
        "validated_rows": int(len(keys)),
        "distinct_gvkeys": int(keys["gvkey"].nunique(dropna=True)),
        "min_quarter_end": keys["quarter_end"].min().date().isoformat(),
        "max_quarter_end": keys["quarter_end"].max().date().isoformat(),
        "row_groups": parquet_file.num_row_groups,
        "file_size": path.stat().st_size,
    }


def is_valid_output(path: Path, start_year: int, end_year: int) -> bool:
    if not path.exists():
        return False
    try:
        validate_output(path, start_year, end_year)
        return True
    except Exception:
        return False


def process_compustat(
    *,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    force: bool = False,
) -> dict:
    if start_year < 2000:
        raise ValueError("start_year must be 2000 or later for the modeling panel")
    if end_year < start_year:
        raise ValueError("end_year must be greater than or equal to start_year")

    if OUTPUT_PATH.exists() and not force and is_valid_output(OUTPUT_PATH, start_year, end_year):
        log(f"Using existing Compustat feature file: {OUTPUT_PATH}")
        validation = validate_output(OUTPUT_PATH, start_year, end_year)
        return {
            "status": "skipped_existing",
            "output_path": str(OUTPUT_PATH),
            **validation,
        }

    raw_start_year = min(RAW_LOOKBACK_YEAR, start_year - 1)
    observations, partition_metrics = load_feature_observations(raw_start_year, end_year)
    log(f"Filtered Compustat observations: {len(observations):,}")

    write_stats = write_aligned_panel(
        observations,
        start_year,
        end_year,
        OUTPUT_PATH,
    )
    validation = validate_output(OUTPUT_PATH, start_year, end_year)

    sic_applied = any(m["sic_exclusion_applied"] for m in partition_metrics)
    missing_rdq_rows = sum(m["missing_rdq_rows"] or 0 for m in partition_metrics)
    rows_after_sic = sum(m["after_sic_exclusions"] or 0 for m in partition_metrics)
    missing_rdq_percent = (
        float(missing_rdq_rows / rows_after_sic * 100) if rows_after_sic else 0.0
    )
    manifest = {
        "status": "success",
        "created_at_utc": utc_now(),
        "input_directory": str(QUARTERLY_DIR),
        "output_path": str(OUTPUT_PATH),
        "start_year": start_year,
        "end_year": end_year,
        "standard_filters": STANDARD_FILTERS,
        "point_in_time_policy": "Rows with missing rdq are excluded. rdq is never imputed; accounting observations are eligible only when rdq <= quarter_end.",
        "winsorization_note": "No winsorization is performed during preprocessing. Any clipping/scaling must be estimated within each training fold during model training.",
        "missing_rdq_rows": int(missing_rdq_rows),
        "missing_rdq_percent": missing_rdq_percent,
        "missing_rdq_by_year": {
            str(m["year"]): {
                "missing_rdq_rows": m["missing_rdq_rows"],
                "missing_rdq_percent": m["missing_rdq_percent"],
            }
            for m in partition_metrics
        },
        "sic_exclusion_applied": sic_applied,
        "sic_exclusion_note": None
        if sic_applied
        else "No sic/sich column was present in the local Compustat quarterly partitions; SIC exclusions must be applied later from another validated local source.",
        "sic_sources_by_year": {
            str(m["year"]): m["sic_source"] for m in partition_metrics
        },
        "partition_metrics": partition_metrics,
        **write_stats,
        **validation,
    }
    write_json_atomic(MANIFEST_PATH, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Compustat quarterly features from local raw parquet partitions.")
    parser.add_argument("--process-compustat", action="store_true", help="Run local Compustat preprocessing.")
    parser.add_argument("--force", action="store_true", help="Rebuild output even if a valid file already exists.")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.process_compustat:
        log("Nothing to do. Pass --process-compustat to build Compustat features.")
        return
    manifest = process_compustat(
        start_year=args.start_year,
        end_year=args.end_year,
        force=args.force,
    )
    log(f"Compustat preprocessing finished with status: {manifest['status']}")
    log(f"Output: {manifest['output_path']}")


if __name__ == "__main__":
    main()
