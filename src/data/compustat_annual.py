import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.wrds import WRDSClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data/raw/wrds/compustat_annual"
MANIFEST_PATH = OUTPUT_DIR / "extraction_manifest.json"
COLUMN_AUDIT_PATH = OUTPUT_DIR / "column_audit.json"
DEFAULT_START_YEAR = 1998
DEFAULT_END_YEAR = 2024

SOURCE_CANDIDATES = (
    ("comp", "funda"),
    ("comp_na_daily_all", "funda"),
    ("comp_na_annual_all", "funda"),
)

REQUIRED_COLUMNS = ("gvkey", "datadate")
REQUESTED_COLUMNS = (
    "gvkey",
    "datadate",
    "fyear",
    "fyr",
    "consol",
    "indfmt",
    "datafmt",
    "popsrc",
    "curcd",
    "costat",
    "tic",
    "cusip",
    "cik",
    "conm",
    "at",
    "act",
    "che",
    "rect",
    "invt",
    "ppent",
    "intan",
    "gdwl",
    "ao",
    "lt",
    "lct",
    "dlc",
    "dltt",
    "txp",
    "ap",
    "lo",
    "ceq",
    "seq",
    "teq",
    "pstk",
    "pstkrv",
    "pstkl",
    "csho",
    "sale",
    "revt",
    "cogs",
    "xsga",
    "oibdp",
    "oiadp",
    "ib",
    "ni",
    "pi",
    "xint",
    "txt",
    "dp",
    "epspx",
    "epsfi",
    "oancf",
    "capx",
    "ivncf",
    "fincf",
    "dltis",
    "dltr",
    "sstk",
    "prstkc",
    "dvt",
    "xrd",
    "xad",
    "xido",
    "spi",
    "mib",
    "emp",
    "mkvalt",
    "prcc_f",
)


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def write_manifest(manifest: dict) -> None:
    write_json_atomic(MANIFEST_PATH, manifest)


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


def validate_year_file(
    path: Path,
    expected_rows: int | None,
    year: int,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected output file was not created: {path}")
    df = pd.read_parquet(path)
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(
            f"Saved row count mismatch for {path}: expected {expected_rows}, got {len(df)}"
        )
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {', '.join(missing)}")
    dates = pd.to_datetime(df["datadate"], errors="coerce")
    dates = dates.dropna()
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year + 1}-01-01")
    if not dates.empty and ((dates < start) | (dates >= end)).any():
        raise ValueError(f"{path} contains datadate values outside {year}")
    return df


def is_valid_year_file(path: Path, year: int) -> bool:
    if not path.exists():
        return False
    try:
        validate_year_file(path, None, year)
        return True
    except Exception:
        return False


def manifest_record(
    *,
    dataset: str,
    source: str,
    path: Path | None,
    status: str,
    start_date: str | None,
    end_date: str | None,
    df: pd.DataFrame | None = None,
    error: str | None = None,
) -> dict:
    record = {
        "dataset": dataset,
        "source_schema_table": source,
        "status": status,
        "row_count": None,
        "column_count": None,
        "distinct_gvkey_count": None,
        "min_date": None,
        "max_date": None,
        "query_date_bounds": {"start": start_date, "end_exclusive": end_date},
        "output_filepath": str(path) if path is not None else None,
        "file_size": path.stat().st_size if path is not None and path.exists() else None,
        "error_message": error,
        "extraction_timestamp_utc": utc_now(),
    }
    if df is not None:
        record["row_count"] = int(len(df))
        record["column_count"] = int(len(df.columns))
        if "gvkey" in df.columns:
            record["distinct_gvkey_count"] = int(df["gvkey"].nunique(dropna=True))
        if "datadate" in df.columns:
            dates = pd.to_datetime(df["datadate"], errors="coerce").dropna()
            if not dates.empty:
                record["min_date"] = dates.min().date().isoformat()
                record["max_date"] = dates.max().date().isoformat()
    return record


def discover_funda_tables(db: WRDSClient) -> list[tuple[str, str]]:
    sql = """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_name = 'funda'
          AND table_schema ILIKE 'comp%'
        ORDER BY table_schema, table_name
    """
    tables = query_with_retry(db, sql)
    return list(tables[["table_schema", "table_name"]].itertuples(index=False, name=None))


class CompustatAnnualExtractor:
    def __init__(self, db: WRDSClient, start_year: int, end_year: int, force: bool):
        self.db = db
        self.start_year = start_year
        self.end_year = end_year
        self.force = force
        self.manifest = load_manifest()
        self.source: str | None = None
        self.available_columns: list[str] = []
        self.selected_columns: list[str] = []
        self.missing_columns: list[str] = []
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def table_columns(self, schema: str, table: str) -> list[str]:
        sql = f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = '{schema}'
              AND table_name = '{table}'
            ORDER BY ordinal_position
        """
        try:
            cols = query_with_retry(self.db, sql)
        except Exception as exc:
            log(f"Could not inspect metadata for {schema}.{table}: {exc}; trying LIMIT 0")
            return query_with_retry(
                self.db,
                f"SELECT * FROM {schema}.{table} LIMIT 0",
            ).columns.tolist()
        if cols.empty:
            log(f"No metadata columns returned for {schema}.{table}; trying LIMIT 0")
            return query_with_retry(
                self.db,
                f"SELECT * FROM {schema}.{table} LIMIT 0",
            ).columns.tolist()
        return cols["column_name"].tolist()

    def discover_source(self) -> None:
        candidates = list(SOURCE_CANDIDATES)
        try:
            for discovered in discover_funda_tables(self.db):
                if discovered not in candidates:
                    candidates.append(discovered)
        except Exception as exc:
            log(f"Could not enumerate Compustat funda tables from metadata: {exc}")

        checked = []
        for schema, table in candidates:
            source = f"{schema}.{table}"
            try:
                columns = self.table_columns(schema, table)
                checked.append(source)
            except Exception as exc:
                log(f"Compustat annual candidate unavailable: {source} ({exc})")
                continue

            lower_columns = {col.lower() for col in columns}
            if not all(col in lower_columns for col in REQUIRED_COLUMNS):
                log(f"Compustat annual candidate lacks gvkey/datadate: {source}")
                continue

            try:
                query_with_retry(
                    self.db,
                    f"SELECT gvkey, datadate FROM {source} LIMIT 1",
                )
            except Exception as exc:
                log(f"Compustat annual candidate failed access test: {source} ({exc})")
                continue

            self.source = source
            self.available_columns = columns
            log(f"Selected Compustat annual source: {source}")
            return

        raise ValueError(
            "No accessible Compustat annual table found. Checked: "
            f"{', '.join(checked) or 'none'}"
        )

    def audit_columns(self) -> None:
        available_lookup = {col.lower(): col for col in self.available_columns}
        selected = []
        missing = []
        for col in REQUESTED_COLUMNS:
            if col.lower() in available_lookup:
                selected.append(available_lookup[col.lower()])
            else:
                missing.append(col)

        selected_lower = {col.lower() for col in selected}
        missing_required = [col for col in REQUIRED_COLUMNS if col not in selected_lower]
        if missing_required:
            raise ValueError(
                "Required Compustat annual columns unavailable: "
                f"{', '.join(missing_required)}"
            )
        if missing:
            log(f"Missing compustat annual optional columns omitted: {', '.join(missing)}")

        available_order = {col.lower(): i for i, col in enumerate(self.available_columns)}
        selected = sorted(selected, key=lambda col: available_order.get(col.lower(), 10**9))
        self.selected_columns = selected
        self.missing_columns = missing
        write_json_atomic(
            COLUMN_AUDIT_PATH,
            {
                "source_table": self.source,
                "requested_columns": list(REQUESTED_COLUMNS),
                "available_columns": self.available_columns,
                "selected_columns": self.selected_columns,
                "missing_columns": self.missing_columns,
                "extraction_timestamp_utc": utc_now(),
            },
        )

    def update_manifest(
        self,
        key: str,
        status: str,
        path: Path,
        start_date: str,
        end_date: str,
        df: pd.DataFrame,
    ) -> None:
        self.manifest[key] = manifest_record(
            dataset=key,
            source=self.source or "",
            path=path,
            status=status,
            start_date=start_date,
            end_date=end_date,
            df=df,
        )
        write_manifest(self.manifest)

    def record_failure(
        self,
        key: str,
        error: Exception,
        path: Path,
        start_date: str,
        end_date: str,
    ) -> None:
        log(f"FAILED {key}: {error}")
        self.manifest[key] = manifest_record(
            dataset=key,
            source=self.source or "",
            path=path,
            status="failed",
            start_date=start_date,
            end_date=end_date,
            error=str(error),
        )
        write_manifest(self.manifest)

    def extract_year(self, year: int) -> tuple[str, int, str | None, str | None]:
        start = f"{year}-01-01"
        end = f"{year + 1}-01-01"
        key = f"compustat_annual_{year}"
        path = OUTPUT_DIR / f"{key}.parquet"
        try:
            if not self.force and is_valid_year_file(path, year):
                cached = validate_year_file(path, None, year)
                log(f"Skipping valid cache: {path}")
                self.update_manifest(key, "skipped", path, start, end, cached)
                min_date = self.manifest[key]["min_date"]
                max_date = self.manifest[key]["max_date"]
                return "success", int(len(cached)), min_date, max_date

            log(f"Extracting Compustat annual fundamentals for {year}")
            sql = f"""
                SELECT {", ".join(self.selected_columns)}
                FROM {self.source}
                WHERE datadate >= DATE '{start}'
                  AND datadate < DATE '{end}'
                ORDER BY gvkey, datadate
            """
            df = query_with_retry(self.db, sql)
            df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
            df = df.sort_values(["gvkey", "datadate"]).reset_index(drop=True)
            safe_write_parquet(df, path)
            saved = validate_year_file(path, len(df), year)
            self.update_manifest(key, "success", path, start, end, saved)
            min_date = self.manifest[key]["min_date"]
            max_date = self.manifest[key]["max_date"]
            return "success", int(len(saved)), min_date, max_date
        except Exception as exc:
            self.record_failure(key, exc, path, start, end)
            return "failed", 0, None, None

    def write_summary(self, results: dict[int, tuple[str, int, str | None, str | None]]) -> None:
        requested_years = list(range(self.start_year, self.end_year + 1))
        successful_years = [
            year for year, (status, _, _, _) in results.items() if status == "success"
        ]
        failed_years = [
            year for year, (status, _, _, _) in results.items() if status != "success"
        ]
        total_rows = sum(rows for status, rows, _, _ in results.values() if status == "success")
        min_dates = [min_date for status, _, min_date, _ in results.values() if status == "success" and min_date]
        max_dates = [max_date for status, _, _, max_date in results.values() if status == "success" and max_date]
        if len(successful_years) == len(requested_years):
            status = "success"
        elif successful_years:
            status = "partial"
        else:
            status = "failed"
        self.manifest["compustat_annual_summary"] = {
            "requested_years": requested_years,
            "successful_years": successful_years,
            "failed_years": failed_years,
            "total_rows": int(total_rows),
            "overall_min_date": min(min_dates) if min_dates else None,
            "overall_max_date": max(max_dates) if max_dates else None,
            "source_schema_table": self.source,
            "selected_columns": self.selected_columns,
            "missing_columns": self.missing_columns,
            "status": status,
            "extraction_timestamp_utc": utc_now(),
        }
        write_manifest(self.manifest)

    def run(self) -> None:
        self.discover_source()
        self.audit_columns()
        results = {}
        for year in range(self.start_year, self.end_year + 1):
            results[year] = self.extract_year(year)
        self.write_summary(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DynamicPD-CLO Compustat annual raw extractor")
    parser.add_argument("--extract-compustat-annual", action="store_true")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.extract_compustat_annual:
        return
    if args.end_year < args.start_year:
        raise ValueError("--end-year must be greater than or equal to --start-year")

    with WRDSClient() as db:
        extractor = CompustatAnnualExtractor(
            db=db,
            start_year=args.start_year,
            end_year=args.end_year,
            force=args.force,
        )
        extractor.run()


if __name__ == "__main__":
    main()
