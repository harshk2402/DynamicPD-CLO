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
WRDS_RAW_DIR = PROJECT_ROOT / "data/raw/wrds"
MANIFEST_PATH = WRDS_RAW_DIR / "extraction_manifest.json"
DEFAULT_START_YEAR = 2000
DEFAULT_COMPUSTAT_START_YEAR = 1999
DEFAULT_END_YEAR = 2024
RATING_TYPES = ("MR", "SPR", "FR")
ID_CHUNK_SIZE = 5000
DATASETS = (
    "fisd_rating_hist",
    "fisd_rated_issue_master",
    "fisd_rated_issuer_master",
    "fisd_issue_issuer_map",
    "bondcrsp_link",
    "fisd_rating",
    "fisd_ratings",
    "compustat_quarterly",
)
DATASET_ALIASES = {
    "compustat": "compustat_quarterly",
}
DEFAULT_BOUND = object()
COMPUSTAT_SOURCE_CANDIDATES = (
    ("comp", "fundq"),
    ("comp_na_daily_all", "fundq"),
    ("comp_na_annual_all", "fundq"),
)
COMPUSTAT_REQUIRED_IDENTIFIERS = ("gvkey", "datadate")
COMPUSTAT_REQUESTED_COLUMNS = (
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
    "curcdq",
    "costat",
    "atq",
    "actq",
    "cheq",
    "rectq",
    "invtq",
    "ppentq",
    "intanq",
    "gdwlq",
    "aoq",
    "ltq",
    "lctq",
    "dlcq",
    "dlttq",
    "txpq",
    "apq",
    "loq",
    "ceqq",
    "seqq",
    "teqq",
    "pstkq",
    "pstkrq",
    "cshoq",
    "saleq",
    "revtq",
    "cogsq",
    "xsgaq",
    "oibdpq",
    "oiadpq",
    "ibq",
    "niq",
    "piq",
    "xintq",
    "txtq",
    "dpq",
    "epspxq",
    "epsfiq",
    "oancfy",
    "capxy",
    "ivncfy",
    "fincfy",
    "dltisy",
    "dltry",
    "sstky",
    "prstkcy",
    "dvty",
    "xrdq",
    "xadq",
    "xidoq",
    "spiqq",
    "mibq",
    "empq",
    "mkvaltq",
    "prccq",
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


def is_valid_nonempty_parquet(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return len(pd.read_parquet(path)) > 0
    except Exception:
        return False


def validate_saved_parquet(path: Path, expected_rows: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected output file was not created: {path}")
    saved = pd.read_parquet(path)
    if len(saved) != expected_rows:
        raise ValueError(
            f"Saved row count mismatch for {path}: expected {expected_rows}, got {len(saved)}"
        )
    return saved


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
        "distinct_issue_count": None,
        "distinct_issuer_count": None,
        "distinct_gvkey_count": None,
        "output_filepath": str(path) if path is not None else None,
        "file_size": path.stat().st_size if path is not None and path.exists() else None,
        "min_date": None,
        "max_date": None,
        "status": status,
        "error_message": error,
    }
    if df is not None:
        record["row_count"] = int(len(df))
        record["column_count"] = int(len(df.columns))
        if "issue_id" in df.columns:
            record["distinct_issue_count"] = int(df["issue_id"].nunique(dropna=True))
        if "issuer_id" in df.columns:
            record["distinct_issuer_count"] = int(df["issuer_id"].nunique(dropna=True))
        if "gvkey" in df.columns:
            record["distinct_gvkey_count"] = int(df["gvkey"].nunique(dropna=True))
        if date_column and date_column in df.columns and df[date_column].notna().any():
            dates = pd.to_datetime(df[date_column])
            record["min_date"] = dates.min().date().isoformat()
            record["max_date"] = dates.max().date().isoformat()
    return record


def record_failure(
    manifest: dict,
    dataset_key: str,
    source: str,
    error: Exception,
    start_date: str | None = None,
    end_date: str | None = None,
    path: Path | None = None,
) -> None:
    log(f"FAILED {dataset_key}: {error}")
    manifest[dataset_key] = manifest_record(
        source=source,
        path=path,
        status="failed",
        start_date=start_date,
        end_date=end_date,
        error=str(error),
    )
    write_manifest(manifest)


def available_columns(db: WRDSClient, schema: str, table: str) -> set[str]:
    sql = f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = '{schema}'
          AND table_name = '{table}'
    """
    cols = query_with_retry(db, sql)
    return set(cols["column_name"].str.lower())


def table_columns(db: WRDSClient, schema: str, table: str) -> list[str]:
    sql = f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = '{schema}'
          AND table_name = '{table}'
        ORDER BY ordinal_position
    """
    cols = query_with_retry(db, sql)
    return cols["column_name"].tolist()


def query_table_columns(db: WRDSClient, schema: str, table: str) -> list[str]:
    df = query_with_retry(db, f"SELECT * FROM {schema}.{table} LIMIT 0")
    return df.columns.tolist()


def discover_fundq_tables(db: WRDSClient) -> list[tuple[str, str]]:
    sql = """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_name = 'fundq'
          AND table_schema ILIKE 'comp%'
        ORDER BY table_schema, table_name
    """
    tables = query_with_retry(db, sql)
    return list(tables[["table_schema", "table_name"]].itertuples(index=False, name=None))


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def select_existing_columns(
    db: WRDSClient,
    schema: str,
    table: str,
    requested: Iterable[str],
    alias: str,
) -> tuple[list[str], list[str]]:
    present = available_columns(db, schema, table)
    selected = []
    missing = []
    for col in requested:
        if col.lower() in present:
            selected.append(f"{alias}.{col}")
        else:
            missing.append(col)
    if missing:
        log(f"Missing {schema}.{table} columns omitted: {', '.join(missing)}")
    return selected, missing


def sql_id_list(values: Iterable) -> str:
    cleaned = []
    for value in values:
        if pd.isna(value):
            continue
        if isinstance(value, str):
            escaped = value.replace("'", "''")
            cleaned.append(f"'{escaped}'")
        else:
            cleaned.append(str(int(value)))
    return ", ".join(cleaned)


def chunked(values: list, size: int = ID_CHUNK_SIZE) -> Iterable[list]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


class WRDSFISDExtractor:
    def __init__(
        self,
        db: WRDSClient,
        start_year: int,
        end_year: int,
        force: bool,
        compustat_start_year: int = DEFAULT_COMPUSTAT_START_YEAR,
    ):
        self.db = db
        self.start_year = start_year
        self.end_year = end_year
        self.force = force
        self.compustat_start_year = compustat_start_year
        self.start_date = f"{start_year}-01-01"
        self.end_exclusive = f"{end_year + 1}-01-01"
        self.manifest = load_manifest()
        WRDS_RAW_DIR.mkdir(parents=True, exist_ok=True)

    def update_manifest(
        self,
        dataset_key: str,
        source: str,
        path: Path,
        status: str,
        df: pd.DataFrame,
        date_column: str | None = None,
        start_date: str | None | object = DEFAULT_BOUND,
        end_date: str | None | object = DEFAULT_BOUND,
    ) -> None:
        manifest_start = self.start_date if start_date is DEFAULT_BOUND else start_date
        manifest_end = self.end_exclusive if end_date is DEFAULT_BOUND else end_date
        self.manifest[dataset_key] = manifest_record(
            source=source,
            path=path,
            status=status,
            start_date=manifest_start,
            end_date=manifest_end,
            df=df,
            date_column=date_column,
        )
        write_manifest(self.manifest)

    def extract_rating_hist(self) -> None:
        out_dir = WRDS_RAW_DIR / "fisd_rating_hist"
        out_dir.mkdir(parents=True, exist_ok=True)
        rating_type_sql = ", ".join(f"'{x}'" for x in RATING_TYPES)
        for year in range(self.start_year, self.end_year + 1):
            start = f"{year}-01-01"
            end = f"{year + 1}-01-01"
            path = out_dir / f"fisd_rating_hist_{year}.parquet"
            key = f"fisd_rating_hist_{year}"
            try:
                if not self.force and is_valid_nonempty_parquet(path):
                    df = pd.read_parquet(path)
                    log(f"Skipping valid cache: {path}")
                    self.update_manifest(
                        key,
                        "fisd.fisd_rating_hist",
                        path,
                        "skipped",
                        df,
                        date_column="rating_date",
                        start_date=start,
                        end_date=end,
                    )
                    continue

                log(f"Extracting FISD rating history for {year}")
                sql = f"""
                    SELECT
                        issue_id,
                        rating_type,
                        rating_date,
                        rating,
                        rating_status,
                        reason
                    FROM fisd.fisd_rating_hist
                    WHERE rating_date >= '{start}'
                      AND rating_date < '{end}'
                      AND rating_type IN ({rating_type_sql})
                    ORDER BY issue_id, rating_type, rating_date
                """
                df = query_with_retry(self.db, sql)
                if "rating_date" in df.columns:
                    df["rating_date"] = pd.to_datetime(df["rating_date"])
                df = df.sort_values(["issue_id", "rating_type", "rating_date"])
                safe_write_parquet(df, path)
                saved = validate_saved_parquet(path, len(df))
                self.update_manifest(
                    key,
                    "fisd.fisd_rating_hist",
                    path,
                    "success",
                    saved,
                    date_column="rating_date",
                    start_date=start,
                    end_date=end,
                )
            except Exception as exc:
                record_failure(
                    self.manifest,
                    key,
                    "fisd.fisd_rating_hist",
                    exc,
                    start_date=start,
                    end_date=end,
                    path=path,
                )

    def load_issue_universe(self) -> list:
        issue_ids = []
        rating_dir = WRDS_RAW_DIR / "fisd_rating_hist"
        for year in range(self.start_year, self.end_year + 1):
            path = rating_dir / f"fisd_rating_hist_{year}.parquet"
            if path.exists():
                df = pd.read_parquet(path, columns=["issue_id"])
                issue_ids.extend(df["issue_id"].dropna().tolist())
        issue_ids = sorted({int(x) for x in issue_ids if not pd.isna(x)})
        if not issue_ids:
            raise ValueError("No issue_id universe found from cached FISD rating history")
        return issue_ids

    def extract_rated_issue_master(self) -> None:
        path = WRDS_RAW_DIR / "fisd_rated_issue_master.parquet"
        key = "fisd_rated_issue_master"
        try:
            if not self.force and is_valid_nonempty_parquet(path):
                df = pd.read_parquet(path)
                log(f"Skipping valid cache: {path}")
                self.update_manifest(key, "fisd.fisd_issue+fisd.fisd_mergedissue", path, "skipped", df)
                return

            issue_ids = self.load_issue_universe()
            issue_cols = [
                "issue_id",
                "issuer_id",
                "prospectus_issuer_name",
                "issuer_cusip",
                "issue_cusip",
            ]
            merged_cols = [
                "offering_date",
                "delivery_date",
                "dated_date",
                "maturity",
                "security_level",
                "security_pledge",
                "enhancement",
                "convertible",
                "preferred_security",
                "bond_type",
                "offering_amt",
                "principal_amt",
                "amount_outstanding",
                "defeased_date",
                "refunding_date",
                "effective_date",
                "as_of_date",
                "change_date",
            ]
            issue_select, _ = select_existing_columns(
                self.db, "fisd", "fisd_issue", issue_cols, "i"
            )
            merged_select, _ = select_existing_columns(
                self.db, "fisd", "fisd_mergedissue", merged_cols, "m"
            )
            if "i.issue_id" not in issue_select:
                raise ValueError("fisd.fisd_issue.issue_id is required but unavailable")

            frames = []
            for i, chunk in enumerate(chunked(issue_ids), 1):
                log(f"Extracting rated issue master chunk {i}")
                cols = ",\n                        ".join(issue_select + merged_select)
                sql = f"""
                    SELECT {cols}
                    FROM fisd.fisd_issue i
                    LEFT JOIN fisd.fisd_mergedissue m
                      ON i.issue_id = m.issue_id
                    WHERE i.issue_id IN ({sql_id_list(chunk)})
                """
                frames.append(query_with_retry(self.db, sql))

            df = pd.concat(frames, ignore_index=True)
            if df["issue_id"].duplicated().any():
                raise ValueError("Rated issue master is not unique by issue_id")
            safe_write_parquet(df, path)
            saved = validate_saved_parquet(path, len(df))
            self.update_manifest(
                key,
                "fisd.fisd_issue+fisd.fisd_mergedissue",
                path,
                "success",
                saved,
            )
        except Exception as exc:
            record_failure(
                self.manifest,
                key,
                "fisd.fisd_issue+fisd.fisd_mergedissue",
                exc,
                path=path,
            )

    def load_issue_master(self) -> pd.DataFrame:
        path = WRDS_RAW_DIR / "fisd_rated_issue_master.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing required issue master: {path}")
        return pd.read_parquet(path)

    def extract_rated_issuer_master(self) -> None:
        path = WRDS_RAW_DIR / "fisd_rated_issuer_master.parquet"
        key = "fisd_rated_issuer_master"
        try:
            if not self.force and is_valid_nonempty_parquet(path):
                df = pd.read_parquet(path)
                log(f"Skipping valid cache: {path}")
                self.update_manifest(key, "fisd.fisd_issuer", path, "skipped", df)
                return

            issue_master = self.load_issue_master()
            issuer_ids = sorted(
                {int(x) for x in issue_master["issuer_id"].dropna().tolist()}
            )
            if not issuer_ids:
                raise ValueError("No issuer_id values found in rated issue master")

            cols = [
                "issuer_id",
                "agent_id",
                "cusip_name",
                "industry_group",
                "industry_code",
                "esop",
                "in_bankruptcy",
                "parent_id",
                "naics_code",
                "country_domicile",
            ]
            selected, _ = select_existing_columns(
                self.db, "fisd", "fisd_issuer", cols, "fi"
            )
            if "fi.issuer_id" not in selected:
                raise ValueError("fisd.fisd_issuer.issuer_id is required but unavailable")

            frames = []
            for i, chunk in enumerate(chunked(issuer_ids), 1):
                log(f"Extracting rated issuer master chunk {i}")
                sql = f"""
                    SELECT {", ".join(selected)}
                    FROM fisd.fisd_issuer fi
                    WHERE fi.issuer_id IN ({sql_id_list(chunk)})
                """
                frames.append(query_with_retry(self.db, sql))
            df = pd.concat(frames, ignore_index=True)
            if df["issuer_id"].duplicated().any():
                raise ValueError("Rated issuer master is not unique by issuer_id")
            safe_write_parquet(df, path)
            saved = validate_saved_parquet(path, len(df))
            self.update_manifest(key, "fisd.fisd_issuer", path, "success", saved)
        except Exception as exc:
            record_failure(self.manifest, key, "fisd.fisd_issuer", exc, path=path)

    def extract_issue_issuer_map(self) -> None:
        path = WRDS_RAW_DIR / "fisd_issue_issuer_map.parquet"
        key = "fisd_issue_issuer_map"
        try:
            if not self.force and is_valid_nonempty_parquet(path):
                df = pd.read_parquet(path)
                log(f"Skipping valid cache: {path}")
                self.update_manifest(key, "fisd_rated_issue_master", path, "skipped", df)
                return

            issue_master = self.load_issue_master()
            cols = [
                "issue_id",
                "issuer_id",
                "issue_cusip",
                "issuer_cusip",
                "prospectus_issuer_name",
            ]
            available = [col for col in cols if col in issue_master.columns]
            df = issue_master[available].drop_duplicates()
            ambiguous = (
                df.dropna(subset=["issuer_id"])
                .groupby("issue_id")["issuer_id"]
                .nunique()
            )
            ambiguous = ambiguous[ambiguous > 1]
            if not ambiguous.empty:
                raise ValueError(
                    f"Ambiguous issue_id to issuer_id mapping for {len(ambiguous)} issues"
                )
            df = df.drop_duplicates("issue_id", keep="first")
            safe_write_parquet(df, path)
            saved = validate_saved_parquet(path, len(df))
            self.update_manifest(key, "fisd_rated_issue_master", path, "success", saved)
        except Exception as exc:
            record_failure(self.manifest, key, "fisd_rated_issue_master", exc, path=path)

    def extract_bondcrsp_link(self) -> None:
        path = WRDS_RAW_DIR / "bondcrsp_link.parquet"
        key = "bondcrsp_link"
        try:
            if not self.force and is_valid_nonempty_parquet(path):
                df = pd.read_parquet(path)
                log(f"Skipping valid cache: {path}")
                self.update_manifest(
                    key,
                    "wrdsapps.bondcrsp_link",
                    path,
                    "skipped",
                    df,
                    start_date=None,
                    end_date=None,
                )
                return

            sql = """
                SELECT
                    cusip,
                    permno,
                    permco,
                    trace_startdt,
                    trace_enddt,
                    crsp_startdt,
                    crsp_enddt,
                    link_startdt,
                    link_enddt
                FROM wrdsapps.bondcrsp_link
            """
            df = query_with_retry(self.db, sql)
            for col in [
                "trace_startdt",
                "trace_enddt",
                "crsp_startdt",
                "crsp_enddt",
                "link_startdt",
                "link_enddt",
            ]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            safe_write_parquet(df, path)
            saved = validate_saved_parquet(path, len(df))
            self.update_manifest(
                key,
                "wrdsapps.bondcrsp_link",
                path,
                "success",
                saved,
                date_column="link_startdt",
                start_date=None,
                end_date=None,
            )
        except Exception as exc:
            record_failure(self.manifest, key, "wrdsapps.bondcrsp_link", exc, path=path)

    def extract_supplemental_rating_table(self, table: str) -> None:
        path = WRDS_RAW_DIR / f"{table}.parquet"
        key = table
        source = f"fisd.{table}"
        try:
            if not self.force and is_valid_nonempty_parquet(path):
                df = pd.read_parquet(path)
                log(f"Skipping valid cache: {path}")
                self.update_manifest(
                    key,
                    source,
                    path,
                    "skipped",
                    df,
                    start_date=None,
                    end_date=None,
                )
                return
            log(f"Extracting supplemental table {source}")
            df = query_with_retry(self.db, f"SELECT * FROM {source}")
            safe_write_parquet(df, path)
            saved = validate_saved_parquet(path, len(df))
            self.update_manifest(
                key,
                source,
                path,
                "success",
                saved,
                start_date=None,
                end_date=None,
            )
        except Exception as exc:
            record_failure(self.manifest, key, source, exc, path=path)

    def discover_compustat_source(self) -> tuple[str, str, list[str]]:
        candidates = list(COMPUSTAT_SOURCE_CANDIDATES)
        try:
            for discovered in discover_fundq_tables(self.db):
                if discovered not in candidates:
                    candidates.append(discovered)
        except Exception as exc:
            log(f"Could not enumerate Compustat fundq tables from metadata: {exc}")

        checked = []
        for schema, table in candidates:
            try:
                try:
                    cols = table_columns(self.db, schema, table)
                except Exception as metadata_exc:
                    log(
                        f"Metadata column discovery failed for {schema}.{table}; "
                        f"trying direct LIMIT 0 query ({metadata_exc})"
                    )
                    cols = query_table_columns(self.db, schema, table)
                checked.append(f"{schema}.{table}")
            except Exception as exc:
                log(f"Compustat source candidate unavailable: {schema}.{table} ({exc})")
                continue

            lower_cols = {col.lower() for col in cols}
            if all(col in lower_cols for col in COMPUSTAT_REQUIRED_IDENTIFIERS):
                try:
                    query_with_retry(
                        self.db,
                        f"""
                        SELECT gvkey, datadate
                        FROM {schema}.{table}
                        LIMIT 1
                        """,
                    )
                except Exception as exc:
                    log(
                        f"Compustat source candidate failed access test: "
                        f"{schema}.{table} ({exc})"
                    )
                    continue
                log(f"Selected Compustat quarterly source: {schema}.{table}")
                return schema, table, cols

            log(
                f"Compustat source candidate lacks required identifiers: "
                f"{schema}.{table}"
            )

        raise ValueError(
            "No accessible Compustat quarterly fundamentals table with gvkey and "
            f"datadate found. Checked: {', '.join(checked) or 'none'}"
        )

    def write_compustat_column_audit(
        self,
        source_table: str,
        requested_columns: list[str],
        available_cols: list[str],
        selected_columns: list[str],
        missing_columns: list[str],
    ) -> None:
        path = WRDS_RAW_DIR / "compustat_quarterly_columns.json"
        audit = {
            "source_table": source_table,
            "requested_columns": requested_columns,
            "selected_columns": selected_columns,
            "available_columns": available_cols,
            "missing_columns": missing_columns,
            "extraction_timestamp_utc": utc_now(),
        }
        tmp_path = path.with_suffix(".json.tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2, sort_keys=True)
        os.replace(tmp_path, path)

    def validate_compustat_year_file(
        self,
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
        missing_required = [
            col for col in COMPUSTAT_REQUIRED_IDENTIFIERS if col not in df.columns
        ]
        if missing_required:
            raise ValueError(
                f"{path} is missing required columns: {', '.join(missing_required)}"
            )
        dates = pd.to_datetime(df["datadate"])
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year + 1}-01-01")
        if not dates.empty and ((dates < start) | (dates >= end)).any():
            raise ValueError(f"{path} contains datadate values outside {year}")
        return df

    def extract_compustat_quarterly(self) -> None:
        key = "compustat_quarterly"
        out_dir = WRDS_RAW_DIR / "compustat_quarterly"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            schema, table, available_cols = self.discover_compustat_source()
            source = f"{schema}.{table}"
            requested = dedupe_preserve_order(COMPUSTAT_REQUESTED_COLUMNS)
            available_lower = {col.lower() for col in available_cols}
            selected = [col for col in requested if col.lower() in available_lower]
            missing = [col for col in requested if col.lower() not in available_lower]
            if missing:
                log(f"Missing {source} columns omitted: {', '.join(missing)}")
            required_missing = [
                col for col in COMPUSTAT_REQUIRED_IDENTIFIERS if col not in selected
            ]
            if required_missing:
                raise ValueError(
                    f"{source} lacks required identifiers: {', '.join(required_missing)}"
                )
            self.write_compustat_column_audit(
                source,
                requested,
                available_cols,
                selected,
                missing,
            )

            invalid_years = []
            for year in range(self.compustat_start_year, self.end_year + 1):
                start = f"{year}-01-01"
                end = f"{year + 1}-01-01"
                path = out_dir / f"compustat_quarterly_{year}.parquet"
                year_key = f"compustat_quarterly_{year}"
                try:
                    if not self.force and path.exists():
                        cached = self.validate_compustat_year_file(path, None, year)
                        log(f"Skipping valid cache: {path}")
                        self.update_manifest(
                            year_key,
                            source,
                            path,
                            "skipped",
                            cached,
                            date_column="datadate",
                            start_date=start,
                            end_date=end,
                        )
                        continue

                    log(f"Extracting Compustat quarterly fundamentals for {year}")
                    cols = ", ".join(selected)
                    sql = f"""
                        SELECT {cols}
                        FROM {source}
                        WHERE datadate >= DATE '{start}'
                          AND datadate < DATE '{end}'
                        ORDER BY gvkey, datadate
                    """
                    df = query_with_retry(self.db, sql)
                    df["datadate"] = pd.to_datetime(df["datadate"])
                    if "rdq" in df.columns:
                        df["rdq"] = pd.to_datetime(df["rdq"])
                    df = df.sort_values(["gvkey", "datadate"]).reset_index(drop=True)
                    safe_write_parquet(df, path)
                    saved = self.validate_compustat_year_file(path, len(df), year)
                    self.update_manifest(
                        year_key,
                        source,
                        path,
                        "success",
                        saved,
                        date_column="datadate",
                        start_date=start,
                        end_date=end,
                    )
                except Exception as exc:
                    invalid_years.append(year)
                    record_failure(
                        self.manifest,
                        year_key,
                        source,
                        exc,
                        start_date=start,
                        end_date=end,
                        path=path,
                    )

            invalid_after_run = []
            for year in range(self.compustat_start_year, self.end_year + 1):
                path = out_dir / f"compustat_quarterly_{year}.parquet"
                try:
                    self.validate_compustat_year_file(path, None, year)
                except Exception:
                    invalid_after_run.append(year)

            if invalid_after_run:
                raise ValueError(
                    "Compustat quarterly extraction incomplete for years: "
                    + ", ".join(str(year) for year in invalid_after_run)
                )

            self.manifest[key] = {
                "extraction_timestamp_utc": utc_now(),
                "source_schema_table": source,
                "query_date_bounds": {
                    "start": f"{self.compustat_start_year}-01-01",
                    "end_exclusive": f"{self.end_year + 1}-01-01",
                },
                "status": "success",
                "error_message": None,
                "output_filepath": str(out_dir),
                "file_size": None,
                "row_count": None,
                "column_count": len(selected),
                "distinct_issue_count": None,
                "distinct_issuer_count": None,
                "distinct_gvkey_count": None,
                "min_date": None,
                "max_date": None,
                "completed_years": list(
                    range(self.compustat_start_year, self.end_year + 1)
                ),
                "failed_years": sorted(set(invalid_years)),
            }
            write_manifest(self.manifest)
        except Exception as exc:
            record_failure(
                self.manifest,
                key,
                "Compustat quarterly source discovery/extraction",
                exc,
                start_date=f"{self.compustat_start_year}-01-01",
                end_date=f"{self.end_year + 1}-01-01",
                path=out_dir,
            )

    def run(self, only: set[str] | None = None) -> None:
        selected = only or set(DATASETS)
        steps = [
            ("fisd_rating_hist", self.extract_rating_hist),
            ("fisd_rated_issue_master", self.extract_rated_issue_master),
            ("fisd_rated_issuer_master", self.extract_rated_issuer_master),
            ("fisd_issue_issuer_map", self.extract_issue_issuer_map),
            ("bondcrsp_link", self.extract_bondcrsp_link),
            ("fisd_rating", lambda: self.extract_supplemental_rating_table("fisd_rating")),
            ("fisd_ratings", lambda: self.extract_supplemental_rating_table("fisd_ratings")),
            ("compustat_quarterly", self.extract_compustat_quarterly),
        ]
        for name, func in steps:
            if name in selected:
                func()


def parse_only(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    selected = set()
    for value in values:
        for part in value.split(","):
            dataset = part.strip()
            if dataset:
                selected.add(DATASET_ALIASES.get(dataset, dataset))
    unknown = selected.difference(DATASETS)
    if unknown:
        raise ValueError(f"Unknown dataset(s) for --only: {', '.join(sorted(unknown))}")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DynamicPD-CLO macro and WRDS extraction tools")
    parser.add_argument("--extract-wrds", action="store_true")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", action="append", help="Dataset name or comma-separated names")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.extract_wrds:
        return
    only = parse_only(args.only)
    compustat_only = only == {"compustat_quarterly"}
    default_start_year = (
        DEFAULT_COMPUSTAT_START_YEAR if compustat_only else DEFAULT_START_YEAR
    )
    start_year = args.start_year if args.start_year is not None else default_start_year
    end_year = args.end_year if args.end_year is not None else DEFAULT_END_YEAR
    compustat_start_year = (
        args.start_year if args.start_year is not None else DEFAULT_COMPUSTAT_START_YEAR
    )
    if end_year < start_year:
        raise ValueError("--end-year must be greater than or equal to --start-year")

    with WRDSClient() as db:
        extractor = WRDSFISDExtractor(
            db=db,
            start_year=start_year,
            end_year=end_year,
            force=args.force,
            compustat_start_year=compustat_start_year,
        )
        extractor.run(only=only)


if __name__ == "__main__":
    main()
