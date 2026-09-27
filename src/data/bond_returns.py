import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.wrds import WRDSClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WRDS_RAW_DIR = PROJECT_ROOT / "data/raw/wrds"

CANDIDATE_SOURCES = (
    ("wrdsapps_bondret", "bondret_std"),
    ("wrdsapps_bondret", "bondret"),
)

DATE_COLUMN_PREFERENCES = (
    "date",
    "caldt",
    "trd_exctn_dt",
    "trade_date",
    "tradedate",
    "return_date",
    "ret_date",
    "month",
    "eom",
)


@dataclass
class BondReturnSource:
    schema: str
    table: str
    columns: list[str]
    metadata: pd.DataFrame
    date_column: str | None
    min_date: pd.Timestamp | None
    max_date: pd.Timestamp | None

    @property
    def source(self) -> str:
        return f"{self.schema}.{self.table}"

    @property
    def output_dir(self) -> Path:
        return WRDS_RAW_DIR / self.table

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "extraction_manifest.json"

    @property
    def column_audit_path(self) -> Path:
        return self.output_dir / "column_audit.json"


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


def load_manifest(path: Path) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def query_with_retry(db: WRDSClient, sql: str, max_attempts: int = 3) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return db.query(sql)
        except Exception as exc:
            last_error = exc
            try:
                db._db.connection.rollback()
            except Exception:
                pass
            if attempt == max_attempts:
                break
            sleep_seconds = 2 ** (attempt - 1)
            log(f"WRDS query failed on attempt {attempt}; retrying in {sleep_seconds}s")
            time.sleep(sleep_seconds)
    raise RuntimeError(
        f"WRDS query failed after {max_attempts} attempts: {last_error}"
    ) from last_error


def safe_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


def table_metadata(db: WRDSClient, schema: str, table: str) -> pd.DataFrame:
    sql = f"""
        SELECT
            column_name,
            data_type,
            ordinal_position
        FROM information_schema.columns
        WHERE table_schema = '{schema}'
          AND table_name = '{table}'
        ORDER BY ordinal_position
    """
    try:
        metadata = query_with_retry(db, sql)
    except Exception as exc:
        log(f"Could not inspect metadata for {schema}.{table}: {exc}; trying LIMIT 0")
        columns = query_with_retry(db, f"SELECT * FROM {schema}.{table} LIMIT 0").columns
        return pd.DataFrame(
            {
                "column_name": columns.tolist(),
                "data_type": [None] * len(columns),
                "ordinal_position": list(range(1, len(columns) + 1)),
            }
        )
    if metadata.empty:
        log(f"No metadata columns returned for {schema}.{table}; trying LIMIT 0")
        columns = query_with_retry(db, f"SELECT * FROM {schema}.{table} LIMIT 0").columns
        return pd.DataFrame(
            {
                "column_name": columns.tolist(),
                "data_type": [None] * len(columns),
                "ordinal_position": list(range(1, len(columns) + 1)),
            }
        )
    return metadata


def choose_date_column(metadata: pd.DataFrame) -> str | None:
    columns = metadata["column_name"].tolist()
    lower_to_original = {col.lower(): col for col in columns}
    for preferred in DATE_COLUMN_PREFERENCES:
        if preferred in lower_to_original:
            return lower_to_original[preferred]

    for _, row in metadata.iterrows():
        data_type = "" if pd.isna(row.get("data_type")) else str(row["data_type"]).lower()
        if "date" in data_type or "timestamp" in data_type:
            return str(row["column_name"])

    for col in columns:
        lowered = col.lower()
        if lowered.endswith("_dt") or "date" in lowered:
            return col
    return None


def source_date_bounds(
    db: WRDSClient,
    source: str,
    date_column: str,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    bounds = query_with_retry(
        db,
        f"""
        SELECT MIN({date_column}) AS min_date, MAX({date_column}) AS max_date
        FROM {source}
        """,
    )
    if bounds.empty:
        return None, None
    min_date = pd.to_datetime(bounds.loc[0, "min_date"], errors="coerce")
    max_date = pd.to_datetime(bounds.loc[0, "max_date"], errors="coerce")
    min_date = None if pd.isna(min_date) else min_date
    max_date = None if pd.isna(max_date) else max_date
    return min_date, max_date


def validate_year_file(
    path: Path,
    expected_rows: int | None,
    date_column: str,
    year: int,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected output file was not created: {path}")
    df = pd.read_parquet(path)
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(
            f"Saved row count mismatch for {path}: expected {expected_rows}, got {len(df)}"
        )
    if date_column not in df.columns:
        raise ValueError(f"{path} missing date column: {date_column}")
    dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year + 1}-01-01")
    if not dates.empty and ((dates < start) | (dates >= end)).any():
        raise ValueError(f"{path} contains {date_column} values outside {year}")
    return df


def is_valid_year_file(path: Path, date_column: str, year: int) -> bool:
    if not path.exists():
        return False
    try:
        validate_year_file(path, None, date_column, year)
        return True
    except Exception:
        return False


def manifest_record(
    *,
    dataset: str,
    source: str,
    path: Path | None,
    status: str,
    date_column: str | None,
    start_date: str | None = None,
    end_date: str | None = None,
    df: pd.DataFrame | None = None,
    error: str | None = None,
) -> dict:
    record = {
        "dataset": dataset,
        "source_schema_table": source,
        "status": status,
        "row_count": None,
        "column_count": None,
        "min_date": None,
        "max_date": None,
        "date_column": date_column,
        "query_date_bounds": {"start": start_date, "end_exclusive": end_date},
        "output_filepath": str(path) if path is not None else None,
        "file_size": path.stat().st_size if path is not None and path.exists() else None,
        "error_message": error,
        "extraction_timestamp_utc": utc_now(),
    }
    if df is not None:
        record["row_count"] = int(len(df))
        record["column_count"] = int(len(df.columns))
        if date_column and date_column in df.columns:
            dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
            if not dates.empty:
                record["min_date"] = dates.min().date().isoformat()
                record["max_date"] = dates.max().date().isoformat()
    return record


class BondReturnsExtractor:
    def __init__(self, db: WRDSClient, force: bool):
        self.db = db
        self.force = force

    def discover_accessible_sources(self) -> list[BondReturnSource]:
        accessible = []
        for schema, table in CANDIDATE_SOURCES:
            source = f"{schema}.{table}"
            try:
                metadata = table_metadata(self.db, schema, table)
                columns = metadata["column_name"].tolist()
                query_with_retry(self.db, f"SELECT * FROM {source} LIMIT 1")
                date_column = choose_date_column(metadata)
                min_date = None
                max_date = None
                if date_column:
                    min_date, max_date = source_date_bounds(self.db, source, date_column)
                log(f"Selected WRDS bond returns source: {source}")
                if date_column:
                    log(
                        f"Using {source}.{date_column} for date bounds "
                        f"({min_date} to {max_date})"
                    )
                else:
                    log(f"No date-like column found for {source}; full-table archive only")
                accessible.append(
                    BondReturnSource(
                        schema=schema,
                        table=table,
                        columns=columns,
                        metadata=metadata,
                        date_column=date_column,
                        min_date=min_date,
                        max_date=max_date,
                    )
                )
            except Exception as exc:
                log(f"WRDS bond returns candidate unavailable: {source} ({exc})")
        if not accessible:
            raise ValueError("No accessible WRDS bond returns table found")
        return accessible

    def write_column_audit(self, source: BondReturnSource) -> None:
        write_json_atomic(
            source.column_audit_path,
            {
                "source_table": source.source,
                "available_columns": source.columns,
                "selected_columns": source.columns,
                "missing_columns": [],
                "date_column": source.date_column,
                "extraction_timestamp_utc": utc_now(),
            },
        )

    def update_manifest(
        self,
        manifest: dict,
        source: BondReturnSource,
        key: str,
        status: str,
        path: Path,
        start_date: str | None,
        end_date: str | None,
        df: pd.DataFrame,
    ) -> None:
        manifest[key] = manifest_record(
            dataset=key,
            source=source.source,
            path=path,
            status=status,
            date_column=source.date_column,
            start_date=start_date,
            end_date=end_date,
            df=df,
        )
        write_json_atomic(source.manifest_path, manifest)

    def record_failure(
        self,
        manifest: dict,
        source: BondReturnSource,
        key: str,
        error: Exception,
        path: Path | None,
        start_date: str | None,
        end_date: str | None,
    ) -> None:
        log(f"FAILED {key}: {error}")
        manifest[key] = manifest_record(
            dataset=key,
            source=source.source,
            path=path,
            status="failed",
            date_column=source.date_column,
            start_date=start_date,
            end_date=end_date,
            error=str(error),
        )
        write_json_atomic(source.manifest_path, manifest)

    def extract_year(
        self,
        source: BondReturnSource,
        manifest: dict,
        year: int,
    ) -> tuple[str, int, str | None, str | None]:
        if not source.date_column:
            raise ValueError("Year extraction requires a date column")

        start = f"{year}-01-01"
        end = f"{year + 1}-01-01"
        key = f"{source.table}_{year}"
        path = source.output_dir / f"{key}.parquet"
        try:
            if not self.force and is_valid_year_file(path, source.date_column, year):
                cached = validate_year_file(path, None, source.date_column, year)
                log(f"Skipping valid cache: {path}")
                self.update_manifest(
                    manifest,
                    source,
                    key,
                    "skipped",
                    path,
                    start,
                    end,
                    cached,
                )
                return (
                    "success",
                    int(len(cached)),
                    manifest[key]["min_date"],
                    manifest[key]["max_date"],
                )

            log(f"Extracting {source.source} for {year}")
            sql = f"""
                SELECT {", ".join(source.columns)}
                FROM {source.source}
                WHERE {source.date_column} >= DATE '{start}'
                  AND {source.date_column} < DATE '{end}'
                ORDER BY {source.date_column}
            """
            df = query_with_retry(self.db, sql)
            df[source.date_column] = pd.to_datetime(
                df[source.date_column],
                errors="coerce",
            )
            df = df.sort_values(source.date_column).reset_index(drop=True)
            safe_write_parquet(df, path)
            saved = validate_year_file(path, len(df), source.date_column, year)
            self.update_manifest(
                manifest,
                source,
                key,
                "success",
                path,
                start,
                end,
                saved,
            )
            return (
                "success",
                int(len(saved)),
                manifest[key]["min_date"],
                manifest[key]["max_date"],
            )
        except Exception as exc:
            self.record_failure(manifest, source, key, exc, path, start, end)
            return "failed", 0, None, None

    def extract_full_table_without_date(
        self,
        source: BondReturnSource,
        manifest: dict,
    ) -> tuple[str, int, str | None, str | None]:
        key = source.table
        path = source.output_dir / f"{source.table}.parquet"
        try:
            if not self.force and path.exists():
                cached = pd.read_parquet(path)
                log(f"Skipping valid cache: {path}")
                self.update_manifest(
                    manifest,
                    source,
                    key,
                    "skipped",
                    path,
                    None,
                    None,
                    cached,
                )
                return "success", int(len(cached)), None, None

            log(f"Extracting full WRDS bond returns table without date chunks: {source.source}")
            df = query_with_retry(self.db, f"SELECT {', '.join(source.columns)} FROM {source.source}")
            safe_write_parquet(df, path)
            saved = pd.read_parquet(path)
            if len(saved) != len(df):
                raise ValueError(
                    f"Saved row count mismatch for {path}: expected {len(df)}, got {len(saved)}"
                )
            self.update_manifest(
                manifest,
                source,
                key,
                "success",
                path,
                None,
                None,
                saved,
            )
            return "success", int(len(saved)), None, None
        except Exception as exc:
            self.record_failure(manifest, source, key, exc, path, None, None)
            return "failed", 0, None, None

    def write_summary(
        self,
        source: BondReturnSource,
        manifest: dict,
        results: dict[int | str, tuple[str, int, str | None, str | None]],
    ) -> None:
        successful_keys = [
            key for key, (status, _, _, _) in results.items() if status == "success"
        ]
        failed_keys = [
            key for key, (status, _, _, _) in results.items() if status != "success"
        ]
        total_rows = sum(rows for status, rows, _, _ in results.values() if status == "success")
        min_dates = [min_date for status, _, min_date, _ in results.values() if status == "success" and min_date]
        max_dates = [max_date for status, _, _, max_date in results.values() if status == "success" and max_date]
        if len(successful_keys) == len(results):
            status = "success"
        elif successful_keys:
            status = "partial"
        else:
            status = "failed"

        manifest[f"{source.table}_summary"] = {
            "dataset": f"{source.table}_summary",
            "source_schema_table": source.source,
            "status": status,
            "date_column": source.date_column,
            "source_min_date": source.min_date.date().isoformat() if source.min_date is not None else None,
            "source_max_date": source.max_date.date().isoformat() if source.max_date is not None else None,
            "successful_chunks": successful_keys,
            "failed_chunks": failed_keys,
            "total_rows": int(total_rows),
            "overall_min_date": min(min_dates) if min_dates else None,
            "overall_max_date": max(max_dates) if max_dates else None,
            "selected_columns": source.columns,
            "missing_columns": [],
            "extraction_timestamp_utc": utc_now(),
        }
        write_json_atomic(source.manifest_path, manifest)

    def extract_source(self, source: BondReturnSource) -> None:
        source.output_dir.mkdir(parents=True, exist_ok=True)
        self.write_column_audit(source)
        manifest = load_manifest(source.manifest_path)

        if source.date_column and source.min_date is not None and source.max_date is not None:
            start_year = int(source.min_date.year)
            end_year = int(source.max_date.year)
            results = {
                year: self.extract_year(source, manifest, year)
                for year in range(start_year, end_year + 1)
            }
        else:
            results = {"full": self.extract_full_table_without_date(source, manifest)}
        self.write_summary(source, manifest, results)

    def run(self) -> None:
        sources = self.discover_accessible_sources()
        for source in sources:
            self.extract_source(source)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DynamicPD-CLO WRDS Bond Returns raw extractor")
    parser.add_argument("--extract-bond-returns", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.extract_bond_returns:
        return

    with WRDSClient() as db:
        extractor = BondReturnsExtractor(db=db, force=args.force)
        extractor.run()


if __name__ == "__main__":
    main()
