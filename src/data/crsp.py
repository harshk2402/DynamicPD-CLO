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
CRSP_RAW_DIR = PROJECT_ROOT / "data/raw/wrds/crsp"
MANIFEST_PATH = CRSP_RAW_DIR / "extraction_manifest.json"
DEFAULT_START_YEAR = 1999
DEFAULT_END_YEAR = 2024

DATASETS = (
    "monthly",
    "daily",
    "delistings",
    "names",
    "ccm",
    "market_monthly",
    "market_daily",
)
CRSP_MONTHLY_CANDIDATES = (
    ("crsp", "msf"),
    ("crsp_a_stock", "msf"),
)
CRSP_DAILY_CANDIDATES = (
    ("crsp", "dsf"),
    ("crsp_a_stock", "dsf"),
)
CRSP_DELISTING_CANDIDATES = (
    ("crsp", "msedelist"),
    ("crsp_a_stock", "msedelist"),
)
CRSP_MARKET_MONTHLY_CANDIDATES = (
    ("crsp", "msi"),
    ("crsp_m_indexes", "msi"),
    ("crsp_a_indexes", "msi"),
)
CRSP_MARKET_DAILY_CANDIDATES = (
    ("crsp", "dsi"),
    ("crsp_d_indexes", "dsi"),
    ("crsp_a_indexes", "dsi"),
)
CRSP_NAMES_CANDIDATES = (
    ("crsp", "msenames"),
    ("crsp_a_stock", "msenames"),
)
CRSP_CCM_CANDIDATES = (
    ("crsp", "ccmxpf_linktable"),
    ("crsp", "ccmxpf_lnkhist"),
    ("crsp_a_ccm", "ccmxpf_linktable"),
    ("crsp_a_ccm", "ccmxpf_lnkhist"),
)

MONTHLY_REQUIRED_COLUMNS = ("permno", "date")
MONTHLY_REQUESTED_COLUMNS = (
    "permno",
    "permco",
    "date",
    "ret",
    "retx",
    "prc",
    "altprc",
    "bidlo",
    "askhi",
    "vol",
    "shrout",
    "cfacpr",
    "cfacshr",
    "dlret",
    "dlstcd",
    "hexcd",
    "hsiccd",
)
DAILY_REQUIRED_COLUMNS = ("permno", "date")
DAILY_REQUESTED_COLUMNS = (
    "permno",
    "permco",
    "date",
    "ret",
    "retx",
    "prc",
    "bidlo",
    "askhi",
    "vol",
    "shrout",
    "cfacpr",
    "cfacshr",
)
DELISTING_REQUIRED_COLUMNS = ("permno",)
DELISTING_REQUESTED_COLUMNS = (
    "permno",
    "dlstdt",
    "dlret",
    "dlretx",
    "dlprc",
    "dlstcd",
    "dlamt",
    "nextdt",
    "nextprc",
    "nwperm",
    "nwcomp",
)
MARKET_MONTHLY_REQUIRED_COLUMNS = ("date",)
MARKET_MONTHLY_REQUESTED_COLUMNS = (
    "date",
    "vwretd",
    "vwretx",
    "ewretd",
    "ewretx",
    "sprtrn",
    "spindx",
    "totval",
    "totcnt",
    "usdval",
    "usdcnt",
)
MARKET_DAILY_REQUIRED_COLUMNS = ("date",)
MARKET_DAILY_REQUESTED_COLUMNS = (
    "date",
    "vwretd",
    "vwretx",
    "ewretd",
    "ewretx",
    "sprtrn",
    "spindx",
    "totval",
    "totcnt",
    "usdval",
    "usdcnt",
)
NAMES_REQUIRED_COLUMNS = ("permno",)
NAMES_REQUESTED_COLUMNS = (
    "permno",
    "permco",
    "namedt",
    "nameendt",
    "ticker",
    "comnam",
    "ncusip",
    "cusip",
    "shrcd",
    "exchcd",
    "siccd",
    "naics",
    "primexch",
    "trdstat",
    "secstat",
)
CCM_REQUIRED_COLUMNS = ("gvkey", "lpermno")
CCM_REQUESTED_COLUMNS = (
    "gvkey",
    "lpermno",
    "lpermco",
    "linktype",
    "linkprim",
    "linkdt",
    "linkenddt",
    "liid",
    "linkid",
    "conm",
    "tic",
    "cusip",
    "cik",
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


def write_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = MANIFEST_PATH.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp_path, MANIFEST_PATH)


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


def validate_saved_parquet(
    path: Path, expected_rows: int | None = None
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected output file was not created: {path}")
    df = pd.read_parquet(path)
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(
            f"Saved row count mismatch for {path}: expected {expected_rows}, got {len(df)}"
        )
    return df


def is_valid_nonempty_parquet(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return len(pd.read_parquet(path)) > 0
    except Exception:
        return False


def manifest_record(
    *,
    source: str,
    path: Path | None,
    status: str,
    start_date: str | None = None,
    end_date: str | None = None,
    df: pd.DataFrame | None = None,
    date_column: str | None = None,
    error: str | None = None,
) -> dict:
    record = {
        "extraction_timestamp_utc": utc_now(),
        "source_schema_table": source,
        "query_date_bounds": {"start": start_date, "end_exclusive": end_date},
        "row_count": None,
        "column_count": None,
        "distinct_permno_count": None,
        "distinct_permco_count": None,
        "distinct_gvkey_count": None,
        "output_filepath": str(path) if path is not None else None,
        "file_size": (
            path.stat().st_size if path is not None and path.exists() else None
        ),
        "min_date": None,
        "max_date": None,
        "status": status,
        "error_message": error,
    }
    if df is not None:
        record["row_count"] = int(len(df))
        record["column_count"] = int(len(df.columns))
        if "permno" in df.columns:
            record["distinct_permno_count"] = int(df["permno"].nunique(dropna=True))
        if "permco" in df.columns:
            record["distinct_permco_count"] = int(df["permco"].nunique(dropna=True))
        if "gvkey" in df.columns:
            record["distinct_gvkey_count"] = int(df["gvkey"].nunique(dropna=True))
        if date_column and date_column in df.columns and df[date_column].notna().any():
            dates = pd.to_datetime(df[date_column])
            record["min_date"] = dates.min().date().isoformat()
            record["max_date"] = dates.max().date().isoformat()
    return record


def query_table_columns(db: WRDSClient, schema: str, table: str) -> list[str]:
    df = query_with_retry(db, f"SELECT * FROM {schema}.{table} LIMIT 0")
    return df.columns.tolist()


def discover_tables(
    db: WRDSClient, table_names: Iterable[str]
) -> list[tuple[str, str]]:
    table_sql = ", ".join(f"'{name}'" for name in table_names)
    sql = f"""
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_name IN ({table_sql})
          AND table_schema ILIKE 'crsp%'
        ORDER BY table_schema, table_name
    """
    tables = query_with_retry(db, sql)
    return list(
        tables[["table_schema", "table_name"]].itertuples(index=False, name=None)
    )


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


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


class CRSPExtractor:
    def __init__(self, db: WRDSClient, start_year: int, end_year: int, force: bool):
        self.db = db
        self.start_year = start_year
        self.end_year = end_year
        self.force = force
        self.manifest = load_manifest()
        CRSP_RAW_DIR.mkdir(parents=True, exist_ok=True)
        self.selected_sources: dict[str, str] = {}

    def update_manifest(
        self,
        key: str,
        source: str,
        path: Path,
        status: str,
        df: pd.DataFrame,
        date_column: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> None:
        self.manifest[key] = manifest_record(
            source=source,
            path=path,
            status=status,
            start_date=start_date,
            end_date=end_date,
            df=df,
            date_column=date_column,
        )
        write_manifest(self.manifest)

    def record_failure(
        self,
        key: str,
        source: str,
        error: Exception,
        path: Path | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> None:
        log(f"FAILED {key}: {error}")
        self.manifest[key] = manifest_record(
            source=source,
            path=path,
            status="failed",
            start_date=start_date,
            end_date=end_date,
            error=str(error),
        )
        write_manifest(self.manifest)

    def discover_source(
        self,
        dataset: str,
        preferred: tuple[tuple[str, str], ...],
        table_names: Iterable[str],
        required: tuple[str, ...],
    ) -> tuple[str, str, list[str]]:
        candidates = list(preferred)
        try:
            for discovered in discover_tables(self.db, table_names):
                if discovered not in candidates:
                    candidates.append(discovered)
        except Exception as exc:
            log(f"Could not enumerate CRSP {dataset} tables from metadata: {exc}")

        checked = []
        for schema, table in candidates:
            source = f"{schema}.{table}"
            try:
                cols = query_table_columns(self.db, schema, table)
                checked.append(source)
            except Exception as exc:
                log(f"CRSP {dataset} candidate unavailable: {source} ({exc})")
                continue

            lower_cols = {col.lower() for col in cols}
            if not all(col in lower_cols for col in required):
                log(f"CRSP {dataset} candidate lacks required columns: {source}")
                continue

            try:
                query_with_retry(
                    self.db,
                    f"SELECT {', '.join(required)} FROM {source} LIMIT 1",
                )
            except Exception as exc:
                log(f"CRSP {dataset} candidate failed access test: {source} ({exc})")
                continue

            log(f"Selected CRSP {dataset} source: {source}")
            self.selected_sources[dataset] = source
            return schema, table, cols

        raise ValueError(
            f"No accessible CRSP {dataset} table found. Checked: "
            f"{', '.join(checked) or 'none'}"
        )

    def selected_columns(
        self, available: list[str], requested: tuple[str, ...]
    ) -> list[str]:
        available_lower = {col.lower(): col for col in available}
        selected = []
        missing = []
        for col in dedupe_preserve_order(requested):
            if col.lower() in available_lower:
                selected.append(available_lower[col.lower()])
            else:
                missing.append(col)
        if missing:
            log(f"Missing optional columns omitted: {', '.join(missing)}")
        return selected

    def validate_year_file(
        self,
        path: Path,
        expected_rows: int | None,
        year: int,
        required_columns: tuple[str, ...] = MONTHLY_REQUIRED_COLUMNS,
    ) -> pd.DataFrame:
        df = validate_saved_parquet(path, expected_rows)
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            raise ValueError(
                f"{path} missing required columns: {', '.join(missing_required)}"
            )
        dates = pd.to_datetime(df["date"])
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year + 1}-01-01")
        if not dates.empty and ((dates < start) | (dates >= end)).any():
            raise ValueError(f"{path} contains dates outside {year}")
        return df

    def extract_monthly(self) -> None:
        out_dir = CRSP_RAW_DIR / "monthly"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            schema, table, available = self.discover_source(
                "monthly",
                CRSP_MONTHLY_CANDIDATES,
                ("msf",),
                MONTHLY_REQUIRED_COLUMNS,
            )
            source = f"{schema}.{table}"
            selected = self.selected_columns(available, MONTHLY_REQUESTED_COLUMNS)
            for year in range(self.start_year, self.end_year + 1):
                start = f"{year}-01-01"
                end = f"{year + 1}-01-01"
                path = out_dir / f"crsp_monthly_{year}.parquet"
                key = f"monthly_{year}"
                try:
                    if not self.force and path.exists():
                        cached = self.validate_year_file(path, None, year)
                        log(f"Skipping valid cache: {path}")
                        self.update_manifest(
                            key,
                            source,
                            path,
                            "skipped",
                            cached,
                            date_column="date",
                            start_date=start,
                            end_date=end,
                        )
                        continue

                    log(f"Extracting CRSP monthly stock file for {year}")
                    sql = f"""
                        SELECT {", ".join(selected)}
                        FROM {source}
                        WHERE date >= DATE '{start}'
                          AND date < DATE '{end}'
                        ORDER BY permno, date
                    """
                    df = query_with_retry(self.db, sql)
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.sort_values(["permno", "date"]).reset_index(drop=True)
                    safe_write_parquet(df, path)
                    saved = self.validate_year_file(path, len(df), year)
                    self.update_manifest(
                        key,
                        source,
                        path,
                        "success",
                        saved,
                        date_column="date",
                        start_date=start,
                        end_date=end,
                    )
                except Exception as exc:
                    self.record_failure(
                        key,
                        source,
                        exc,
                        path=path,
                        start_date=start,
                        end_date=end,
                    )
        except Exception as exc:
            self.record_failure(
                "monthly", "CRSP monthly source discovery/extraction", exc, path=out_dir
            )

    def extract_daily(self) -> None:
        out_dir = CRSP_RAW_DIR / "daily"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            schema, table, available = self.discover_source(
                "daily",
                CRSP_DAILY_CANDIDATES,
                ("dsf",),
                DAILY_REQUIRED_COLUMNS,
            )
            source = f"{schema}.{table}"
            selected = self.selected_columns(available, DAILY_REQUESTED_COLUMNS)
            for year in range(self.start_year, self.end_year + 1):
                start = f"{year}-01-01"
                end = f"{year + 1}-01-01"
                path = out_dir / f"crsp_daily_{year}.parquet"
                key = f"daily_{year}"
                try:
                    if not self.force and path.exists():
                        cached = self.validate_year_file(
                            path,
                            None,
                            year,
                            required_columns=DAILY_REQUIRED_COLUMNS,
                        )
                        log(f"Skipping valid cache: {path}")
                        self.update_manifest(
                            key,
                            source,
                            path,
                            "skipped",
                            cached,
                            date_column="date",
                            start_date=start,
                            end_date=end,
                        )
                        continue

                    log(f"Extracting CRSP daily stock file for {year}")
                    sql = f"""
                        SELECT {", ".join(selected)}
                        FROM {source}
                        WHERE date >= DATE '{start}'
                          AND date < DATE '{end}'
                        ORDER BY permno, date
                    """
                    df = query_with_retry(self.db, sql)
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.sort_values(["permno", "date"]).reset_index(drop=True)
                    safe_write_parquet(df, path)
                    saved = self.validate_year_file(
                        path,
                        len(df),
                        year,
                        required_columns=DAILY_REQUIRED_COLUMNS,
                    )
                    self.update_manifest(
                        key,
                        source,
                        path,
                        "success",
                        saved,
                        date_column="date",
                        start_date=start,
                        end_date=end,
                    )
                except Exception as exc:
                    self.record_failure(
                        key,
                        source,
                        exc,
                        path=path,
                        start_date=start,
                        end_date=end,
                    )
        except Exception as exc:
            self.record_failure(
                "daily", "CRSP daily source discovery/extraction", exc, path=out_dir
            )

    def extract_market_yearly(
        self,
        *,
        dataset: str,
        candidates: tuple[tuple[str, str], ...],
        table_names: tuple[str, ...],
        required_columns: tuple[str, ...],
        requested_columns: tuple[str, ...],
        output_dir: str,
        filename_prefix: str,
        log_label: str,
    ) -> None:
        out_dir = CRSP_RAW_DIR / output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            schema, table, available = self.discover_source(
                dataset,
                candidates,
                table_names,
                required_columns,
            )
            source = f"{schema}.{table}"
            selected = self.selected_columns(available, requested_columns)
            for year in range(self.start_year, self.end_year + 1):
                start = f"{year}-01-01"
                end = f"{year + 1}-01-01"
                path = out_dir / f"{filename_prefix}_{year}.parquet"
                key = f"{dataset}_{year}"
                try:
                    if not self.force and path.exists():
                        cached = self.validate_year_file(
                            path,
                            None,
                            year,
                            required_columns=required_columns,
                        )
                        log(f"Skipping valid cache: {path}")
                        self.update_manifest(
                            key,
                            source,
                            path,
                            "skipped",
                            cached,
                            date_column="date",
                            start_date=start,
                            end_date=end,
                        )
                        continue

                    log(f"Extracting {log_label} for {year}")
                    sql = f"""
                        SELECT {", ".join(selected)}
                        FROM {source}
                        WHERE date >= DATE '{start}'
                          AND date < DATE '{end}'
                        ORDER BY date
                    """
                    df = query_with_retry(self.db, sql)
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.sort_values(["date"]).reset_index(drop=True)
                    safe_write_parquet(df, path)
                    saved = self.validate_year_file(
                        path,
                        len(df),
                        year,
                        required_columns=required_columns,
                    )
                    self.update_manifest(
                        key,
                        source,
                        path,
                        "success",
                        saved,
                        date_column="date",
                        start_date=start,
                        end_date=end,
                    )
                except Exception as exc:
                    self.record_failure(
                        key,
                        source,
                        exc,
                        path=path,
                        start_date=start,
                        end_date=end,
                    )
        except Exception as exc:
            self.record_failure(
                dataset,
                f"CRSP {dataset} source discovery/extraction",
                exc,
                path=out_dir,
            )

    def extract_market_monthly(self) -> None:
        self.extract_market_yearly(
            dataset="market_monthly",
            candidates=CRSP_MARKET_MONTHLY_CANDIDATES,
            table_names=("msi",),
            required_columns=MARKET_MONTHLY_REQUIRED_COLUMNS,
            requested_columns=MARKET_MONTHLY_REQUESTED_COLUMNS,
            output_dir="market_monthly",
            filename_prefix="crsp_market_monthly",
            log_label="CRSP monthly market-index data",
        )

    def extract_market_daily(self) -> None:
        self.extract_market_yearly(
            dataset="market_daily",
            candidates=CRSP_MARKET_DAILY_CANDIDATES,
            table_names=("dsi",),
            required_columns=MARKET_DAILY_REQUIRED_COLUMNS,
            requested_columns=MARKET_DAILY_REQUESTED_COLUMNS,
            output_dir="market_daily",
            filename_prefix="crsp_market_daily",
            log_label="CRSP daily market-index data",
        )

    def extract_delistings(self) -> None:
        path = CRSP_RAW_DIR / "crsp_delistings.parquet"
        key = "delistings"
        try:
            schema, table, available = self.discover_source(
                "delistings",
                CRSP_DELISTING_CANDIDATES,
                ("msedelist",),
                DELISTING_REQUIRED_COLUMNS,
            )
            source = f"{schema}.{table}"
            if not self.force and is_valid_nonempty_parquet(path):
                cached = validate_saved_parquet(path)
                if "permno" not in cached.columns:
                    raise ValueError(f"{path} missing required column: permno")
                log(f"Skipping valid cache: {path}")
                self.update_manifest(
                    key, source, path, "skipped", cached, date_column="dlstdt"
                )
                return

            selected = self.selected_columns(available, DELISTING_REQUESTED_COLUMNS)
            log("Extracting CRSP delisting history")
            sql = f"""
                SELECT {", ".join(selected)}
                FROM {source}
            """
            df = query_with_retry(self.db, sql)
            for col in ("dlstdt", "nextdt"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            sort_cols = [col for col in ("permno", "dlstdt") if col in df.columns]
            if sort_cols:
                df = df.sort_values(sort_cols).reset_index(drop=True)
            safe_write_parquet(df, path)
            saved = validate_saved_parquet(path, len(df))
            if "permno" not in saved.columns:
                raise ValueError(f"{path} missing required column: permno")
            self.update_manifest(
                key, source, path, "success", saved, date_column="dlstdt"
            )
        except Exception as exc:
            self.record_failure(
                key, "CRSP delistings source discovery/extraction", exc, path=path
            )

    def extract_names(self) -> None:
        path = CRSP_RAW_DIR / "crsp_names.parquet"
        key = "names"
        try:
            schema, table, available = self.discover_source(
                "names",
                CRSP_NAMES_CANDIDATES,
                ("msenames",),
                NAMES_REQUIRED_COLUMNS,
            )
            source = f"{schema}.{table}"
            if not self.force and is_valid_nonempty_parquet(path):
                cached = validate_saved_parquet(path)
                log(f"Skipping valid cache: {path}")
                self.update_manifest(
                    key, source, path, "skipped", cached, date_column="namedt"
                )
                return

            selected = self.selected_columns(available, NAMES_REQUESTED_COLUMNS)
            order_cols = [col for col in ("permno", "namedt") if col in selected]
            order_sql = f"ORDER BY {', '.join(order_cols)}" if order_cols else ""
            log("Extracting CRSP security names/history")
            sql = f"""
                SELECT {", ".join(selected)}
                FROM {source}
                {order_sql}
            """
            df = query_with_retry(self.db, sql)
            for col in ("namedt", "nameendt"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            sort_cols = [col for col in ("permno", "namedt") if col in df.columns]
            if sort_cols:
                df = df.sort_values(sort_cols).reset_index(drop=True)
            safe_write_parquet(df, path)
            saved = validate_saved_parquet(path, len(df))
            self.update_manifest(
                key, source, path, "success", saved, date_column="namedt"
            )
        except Exception as exc:
            self.record_failure(
                key, "CRSP names source discovery/extraction", exc, path=path
            )

    def extract_ccm(self) -> None:
        path = CRSP_RAW_DIR / "crsp_compustat_link.parquet"
        key = "ccm"
        try:
            schema, table, available = self.discover_source(
                "ccm",
                CRSP_CCM_CANDIDATES,
                ("ccmxpf_linktable", "ccmxpf_lnkhist"),
                CCM_REQUIRED_COLUMNS,
            )
            source = f"{schema}.{table}"
            if not self.force and is_valid_nonempty_parquet(path):
                cached = validate_saved_parquet(path)
                log(f"Skipping valid cache: {path}")
                self.update_manifest(
                    key, source, path, "skipped", cached, date_column="linkdt"
                )
                return

            selected = self.selected_columns(available, CCM_REQUESTED_COLUMNS)
            log("Extracting CRSP-Compustat link table")
            sql = f"""
                SELECT {", ".join(selected)}
                FROM {source}
                ORDER BY gvkey
            """
            df = query_with_retry(self.db, sql)
            for col in ("linkdt", "linkenddt"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            sort_cols = [
                col for col in ("gvkey", "lpermno", "linkdt") if col in df.columns
            ]
            if sort_cols:
                df = df.sort_values(sort_cols).reset_index(drop=True)
            safe_write_parquet(df, path)
            saved = validate_saved_parquet(path, len(df))
            self.update_manifest(
                key, source, path, "success", saved, date_column="linkdt"
            )
        except Exception as exc:
            self.record_failure(
                key, "CRSP CCM source discovery/extraction", exc, path=path
            )

    def run(self, only: set[str]) -> None:
        if "monthly" in only:
            self.extract_monthly()
        if "daily" in only:
            self.extract_daily()
        if "delistings" in only:
            self.extract_delistings()
        if "names" in only:
            self.extract_names()
        if "ccm" in only:
            self.extract_ccm()
        if "market_monthly" in only:
            self.extract_market_monthly()
        if "market_daily" in only:
            self.extract_market_daily()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DynamicPD-CLO CRSP raw extractor")
    parser.add_argument("--extract-crsp", action="store_true")
    parser.add_argument(
        "--only",
        default="all",
        help=(
            "monthly, daily, delistings, names, ccm, market_monthly, "
            "market_daily, or all"
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.extract_crsp:
        return
    if args.end_year < args.start_year:
        raise ValueError("--end-year must be greater than or equal to --start-year")

    only = normalize_only(args.only)
    with WRDSClient() as db:
        extractor = CRSPExtractor(
            db=db,
            start_year=args.start_year,
            end_year=args.end_year,
            force=args.force,
        )
        extractor.run(only)
        if extractor.selected_sources:
            log("Selected CRSP source tables:")
            for dataset, source in extractor.selected_sources.items():
                log(f"  {dataset}: {source}")


if __name__ == "__main__":
    main()
