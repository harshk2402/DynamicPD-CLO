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
IBES_RAW_DIR = PROJECT_ROOT / "data/raw/wrds/ibes"
DEFAULT_START_YEAR = 1998
DEFAULT_END_YEAR = 2024

IBES_TABLES = (
    ("ibes", "statsum_epsus"),
    ("ibes", "act_epsus"),
    ("ibes", "actu_epsus"),
    ("ibes", "recdsum"),
)

DATE_COLUMN_PREFERENCES = (
    "anndats",
    "anndats_act",
    "actdats",
    "statpers",
    "fpedats",
    "anndats_est",
    "date",
)
TABLE_DATE_PREFERENCES = {
    "statsum_epsus": ("statpers", "fpedats", "anndats_act"),
    "recdsum": ("statpers", "anndats"),
    "act_epsus": ("anndats", "actdats", "anndats_act"),
    "actu_epsus": ("anndats", "actdats", "anndats_act"),
}


@dataclass
class IBESSource:
    schema: str
    table: str
    columns: list[str]
    metadata: pd.DataFrame
    date_column: str | None
    min_date: pd.Timestamp | None
    max_date: pd.Timestamp | None
    total_source_rows: int | None
    null_date_rows: int | None
    archive_window_rows: int | None

    @property
    def source(self) -> str:
        return f"{self.schema}.{self.table}"

    @property
    def output_dir(self) -> Path:
        return IBES_RAW_DIR / self.table

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


def choose_date_column(table: str, metadata: pd.DataFrame) -> str | None:
    columns = metadata["column_name"].tolist()
    lower_to_original = {col.lower(): col for col in columns}
    preferences = TABLE_DATE_PREFERENCES.get(table, DATE_COLUMN_PREFERENCES)
    for preferred in preferences:
        if preferred in lower_to_original:
            return lower_to_original[preferred]

    for _, row in metadata.iterrows():
        data_type = "" if pd.isna(row.get("data_type")) else str(row["data_type"]).lower()
        if "date" in data_type or "timestamp" in data_type:
            return str(row["column_name"])
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


def source_row_counts(
    db: WRDSClient,
    source: str,
    date_column: str | None,
    archive_start_date: str,
    archive_end_exclusive: str,
) -> tuple[int | None, int | None, int | None]:
    if not date_column:
        count = query_with_retry(db, f"SELECT COUNT(*) AS n_rows FROM {source}")
        return int(count.loc[0, "n_rows"]), None, int(count.loc[0, "n_rows"])
    counts = query_with_retry(
        db,
        f"""
        SELECT
            COUNT(*) AS n_rows,
            SUM(CASE WHEN {date_column} IS NULL THEN 1 ELSE 0 END) AS null_date_rows,
            SUM(
                CASE
                    WHEN {date_column} >= DATE '{archive_start_date}'
                     AND {date_column} < DATE '{archive_end_exclusive}'
                    THEN 1
                    ELSE 0
                END
            ) AS archive_window_rows
        FROM {source}
        """,
    )
    null_date_rows = counts.loc[0, "null_date_rows"]
    null_date_rows = 0 if pd.isna(null_date_rows) else int(null_date_rows)
    archive_window_rows = counts.loc[0, "archive_window_rows"]
    archive_window_rows = 0 if pd.isna(archive_window_rows) else int(archive_window_rows)
    return int(counts.loc[0, "n_rows"]), null_date_rows, archive_window_rows


def validate_period_file(
    path: Path,
    expected_rows: int | None,
    date_column: str,
    start: str,
    end: str,
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
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if not dates.empty and ((dates < start_ts) | (dates >= end_ts)).any():
        raise ValueError(f"{path} contains {date_column} values outside {start} to {end}")
    return df


def is_valid_period_file(path: Path, date_column: str, start: str, end: str) -> bool:
    if not path.exists():
        return False
    try:
        validate_period_file(path, None, date_column, start, end)
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


def extraction_periods(
    source: IBESSource,
    archive_start_date: str,
    archive_end_date: str,
) -> list[tuple[str, str, str]]:
    if source.date_column is None or source.min_date is None or source.max_date is None:
        return []
    start_bound = pd.Timestamp(archive_start_date)
    end_bound = pd.Timestamp(archive_end_date)
    start = max(pd.Timestamp(source.min_date).normalize(), start_bound)
    end_inclusive = min(pd.Timestamp(source.max_date).normalize(), end_bound - pd.Timedelta(days=1))
    if start > end_inclusive:
        return []

    return [
        (
            str(year),
            max(pd.Timestamp(f"{year}-01-01"), start).date().isoformat(),
            min(pd.Timestamp(f"{year + 1}-01-01"), end_bound).date().isoformat(),
        )
        for year in range(int(start.year), int(end_inclusive.year) + 1)
    ]


class IBESExtractor:
    def __init__(self, db: WRDSClient, force: bool, start_year: int, end_year: int):
        self.db = db
        self.force = force
        self.start_year = start_year
        self.end_year = end_year
        self.archive_start_date = f"{start_year}-01-01"
        self.archive_end_date = f"{end_year}-12-31"
        self.archive_end_exclusive = f"{end_year + 1}-01-01"

    def discover_source(self, schema: str, table: str) -> IBESSource:
        source = f"{schema}.{table}"
        metadata = table_metadata(self.db, schema, table)
        columns = metadata["column_name"].tolist()
        query_with_retry(self.db, f"SELECT * FROM {source} LIMIT 1")
        date_column = choose_date_column(table, metadata)
        min_date = None
        max_date = None
        total_source_rows = None
        null_date_rows = None
        archive_window_rows = None
        if date_column:
            min_date, max_date = source_date_bounds(self.db, source, date_column)
            total_source_rows, null_date_rows, archive_window_rows = source_row_counts(
                self.db,
                source,
                date_column,
                self.archive_start_date,
                self.archive_end_exclusive,
            )
            log(f"Selected IBES date column for {source}: {date_column}")
            log(f"{source} has {null_date_rows} rows with NULL {date_column}")
        else:
            total_source_rows, null_date_rows, archive_window_rows = source_row_counts(
                self.db,
                source,
                None,
                self.archive_start_date,
                self.archive_end_exclusive,
            )
            log(f"No usable date column found for {source}; full-table archive only")
        return IBESSource(
            schema=schema,
            table=table,
            columns=columns,
            metadata=metadata,
            date_column=date_column,
            min_date=min_date,
            max_date=max_date,
            total_source_rows=total_source_rows,
            null_date_rows=null_date_rows,
            archive_window_rows=archive_window_rows,
        )

    def write_column_audit(self, source: IBESSource) -> None:
        write_json_atomic(
            source.column_audit_path,
            {
                "source_table": source.source,
                "available_columns": source.columns,
                "selected_columns": source.columns,
                "missing_columns": [],
                "date_column": source.date_column,
                "total_source_rows": source.total_source_rows,
                "null_date_rows": source.null_date_rows,
                "archive_window_rows": source.archive_window_rows,
                "archive_start_date": self.archive_start_date,
                "archive_end_date": self.archive_end_date,
                "archive_end_exclusive": self.archive_end_exclusive,
                "extraction_timestamp_utc": utc_now(),
            },
        )

    def update_manifest(
        self,
        manifest: dict,
        source: IBESSource,
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
        source: IBESSource,
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

    def extract_period(
        self,
        source: IBESSource,
        manifest: dict,
        label: str,
        start: str,
        end: str,
    ) -> tuple[str, int, str | None, str | None]:
        if not source.date_column:
            raise ValueError("Period extraction requires a date column")
        key = f"{source.table}_{label}"
        path = source.output_dir / f"{key}.parquet"
        try:
            if not self.force and is_valid_period_file(
                path,
                source.date_column,
                start,
                end,
            ):
                cached = validate_period_file(
                    path,
                    None,
                    source.date_column,
                    start,
                    end,
                )
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

            log(f"Extracting {source.source} for {label}")
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
            saved = validate_period_file(path, len(df), source.date_column, start, end)
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

    def extract_null_date_rows(
        self,
        source: IBESSource,
        manifest: dict,
    ) -> tuple[str, int, str | None, str | None]:
        if not source.date_column:
            raise ValueError("Null-date extraction requires a date column")
        key = f"{source.table}_null_{source.date_column}"
        path = source.output_dir / f"{key}.parquet"
        try:
            if source.null_date_rows == 0:
                empty = pd.DataFrame(columns=source.columns)
                safe_write_parquet(empty, path)
                saved = pd.read_parquet(path)
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
                return "success", 0, None, None

            if not self.force and path.exists():
                cached = pd.read_parquet(path)
                if source.date_column not in cached.columns:
                    raise ValueError(f"{path} missing date column: {source.date_column}")
                non_null_dates = cached[source.date_column].notna().sum()
                if non_null_dates:
                    raise ValueError(f"{path} contains non-null {source.date_column} rows")
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

            log(f"Extracting {source.source} rows with NULL {source.date_column}")
            sql = f"""
                SELECT {", ".join(source.columns)}
                FROM {source.source}
                WHERE {source.date_column} IS NULL
            """
            df = query_with_retry(self.db, sql)
            safe_write_parquet(df, path)
            saved = pd.read_parquet(path)
            if len(saved) != len(df):
                raise ValueError(
                    f"Saved row count mismatch for {path}: expected {len(df)}, got {len(saved)}"
                )
            if source.date_column in saved.columns and saved[source.date_column].notna().any():
                raise ValueError(f"{path} contains non-null {source.date_column} rows")
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

    def extract_full_table_without_date(
        self,
        source: IBESSource,
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

            log(f"Extracting full IBES table without date chunks: {source.source}")
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
        source: IBESSource,
        manifest: dict,
        results: dict[int | str, tuple[str, int, str | None, str | None]],
    ) -> None:
        successful_years = [
            key for key, (status, _, _, _) in results.items() if status == "success"
        ]
        failed_years = [
            key for key, (status, _, _, _) in results.items() if status != "success"
        ]
        archived_rows = sum(rows for status, rows, _, _ in results.values() if status == "success")
        expected_archive_rows = None
        if source.date_column is None:
            expected_archive_rows = source.total_source_rows
        elif source.archive_window_rows is not None and source.null_date_rows is not None:
            expected_archive_rows = int(source.archive_window_rows) + int(source.null_date_rows)
        extracted_row_difference = (
            None if expected_archive_rows is None else int(expected_archive_rows) - int(archived_rows)
        )
        if len(successful_years) == len(results):
            status = "success"
        elif successful_years:
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
            "archive_start_date": self.archive_start_date,
            "archive_end_date": self.archive_end_date,
            "archive_end_exclusive": self.archive_end_exclusive,
            "successful_years": successful_years,
            "failed_years": failed_years,
            "chunk_frequency": "full" if source.date_column is None else "yearly",
            "total_source_rows": source.total_source_rows,
            "null_date_rows": source.null_date_rows,
            "archive_window_rows": source.archive_window_rows,
            "expected_archive_rows": expected_archive_rows,
            "total_rows": int(archived_rows),
            "archived_rows": int(archived_rows),
            "extracted_row_difference": extracted_row_difference,
            "selected_columns": source.columns,
            "extraction_timestamp_utc": utc_now(),
        }
        write_json_atomic(source.manifest_path, manifest)

    def extract_source(self, source: IBESSource) -> None:
        source.output_dir.mkdir(parents=True, exist_ok=True)
        self.write_column_audit(source)
        manifest = load_manifest(source.manifest_path)
        if source.date_column and source.min_date is not None and source.max_date is not None:
            periods = extraction_periods(
                source,
                self.archive_start_date,
                self.archive_end_exclusive,
            )
            results = {
                label: self.extract_period(source, manifest, label, start, end)
                for label, start, end in periods
            }
            null_key = f"null_{source.date_column}"
            results[null_key] = self.extract_null_date_rows(source, manifest)
        else:
            results = {"full": self.extract_full_table_without_date(source, manifest)}
        self.write_summary(source, manifest, results)

    def run(self) -> None:
        for schema, table in IBES_TABLES:
            source_text = f"{schema}.{table}"
            try:
                log(f"Preparing IBES raw archive for {source_text}")
                source = self.discover_source(schema, table)
                self.extract_source(source)
            except Exception as exc:
                output_dir = IBES_RAW_DIR / table
                output_dir.mkdir(parents=True, exist_ok=True)
                manifest_path = output_dir / "extraction_manifest.json"
                manifest = load_manifest(manifest_path)
                manifest[f"{table}_summary"] = {
                    "dataset": f"{table}_summary",
                    "source_schema_table": source_text,
                    "status": "failed",
                    "date_column": None,
                    "source_min_date": None,
                    "source_max_date": None,
                    "archive_start_date": self.archive_start_date,
                    "archive_end_date": self.archive_end_date,
                    "archive_end_exclusive": self.archive_end_exclusive,
                    "successful_years": [],
                    "failed_years": ["discovery"],
                    "total_rows": 0,
                    "archived_rows": 0,
                    "selected_columns": [],
                    "error_message": str(exc),
                    "extraction_timestamp_utc": utc_now(),
                }
                write_json_atomic(manifest_path, manifest)
                log(f"FAILED {source_text}: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DynamicPD-CLO IBES raw extractor")
    parser.add_argument("--extract-ibes", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.extract_ibes:
        return
    if args.end_year < args.start_year:
        raise ValueError("--end-year must be greater than or equal to --start-year")

    with WRDSClient() as db:
        extractor = IBESExtractor(
            db=db,
            force=args.force,
            start_year=args.start_year,
            end_year=args.end_year,
        )
        extractor.run()


if __name__ == "__main__":
    main()
