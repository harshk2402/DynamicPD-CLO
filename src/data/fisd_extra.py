import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.wrds import WRDSClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WRDS_RAW_DIR = PROJECT_ROOT / "data/raw/wrds"
FISD_EXTRA_DIR = WRDS_RAW_DIR / "fisd_extra"
MANIFEST_PATH = FISD_EXTRA_DIR / "extraction_manifest.json"
RATED_ISSUE_MASTER_PATH = WRDS_RAW_DIR / "fisd_rated_issue_master.parquet"
ID_CHUNK_SIZE = 5000
MISSING_ISSUE_ID_SAMPLE_SIZE = 100

DATASETS = (
    "rating",
    "ratings",
    "issue_default",
    "issue_affected",
    "related_issues",
    "issue_enhancement",
    "mergedissue_full",
)

SOURCE_TABLES = {
    "rating": ("fisd", "fisd_rating"),
    "ratings": ("fisd", "fisd_ratings"),
    "issue_default": ("fisd", "fisd_issue_default"),
    "issue_affected": ("fisd", "fisd_issue_affected"),
    "related_issues": ("fisd", "fisd_related_issues"),
    "issue_enhancement": ("fisd", "fisd_issue_enhancement"),
    "mergedissue_full": ("fisd", "fisd_mergedissue"),
}

OUTPUT_FILES = {
    "rating": "fisd_rating.parquet",
    "ratings": "fisd_ratings.parquet",
    "issue_default": "fisd_issue_default.parquet",
    "issue_affected": "fisd_issue_affected.parquet",
    "related_issues": "fisd_related_issues.parquet",
    "issue_enhancement": "fisd_issue_enhancement.parquet",
    "mergedissue_full": "fisd_mergedissue_rated_full.parquet",
}

EXPECTED_KEY_COLUMNS = {
    "issue_default": "issue_id",
    "issue_affected": "issue_id",
    "related_issues": "issue_id",
    "issue_enhancement": "issue_id",
    "mergedissue_full": "issue_id",
}

PRIMARY_DATE_CANDIDATES = {
    "issue_default": (
        "default_date",
        "default_dt",
        "bankruptcy_date",
        "bankruptcy_dt",
        "filing_date",
        "filing_dt",
    ),
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
    path: Path,
    expected_rows: int | None = None,
    required_column: str | None = None,
    require_nonempty: bool = False,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected output file was not created: {path}")
    df = pd.read_parquet(path)
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(
            f"Saved row count mismatch for {path}: expected {expected_rows}, got {len(df)}"
        )
    if required_column and required_column not in df.columns:
        raise ValueError(f"{path} missing required column: {required_column}")
    if require_nonempty and df.empty:
        raise ValueError(f"{path} is empty")
    return df


def validate_issue_id_parquet(
    path: Path,
    expected_rows: int | None = None,
    require_nonempty: bool = False,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected output file was not created: {path}")
    df = pd.read_parquet(path, columns=["issue_id"])
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(
            f"Saved row count mismatch for {path}: expected {expected_rows}, got {len(df)}"
        )
    if require_nonempty and df.empty:
        raise ValueError(f"{path} is empty")
    return df


def parquet_shape(path: Path) -> tuple[int, int]:
    metadata = pq.ParquetFile(path).metadata
    return int(metadata.num_rows), int(metadata.num_columns)


def is_valid_nonempty_parquet(
    path: Path,
    required_column: str | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        validate_saved_parquet(path, required_column=required_column, require_nonempty=True)
        return True
    except Exception:
        return False


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


def sql_literal(value) -> str:
    if pd.isna(value):
        raise ValueError("Cannot render NULL as SQL literal for IN clause")
    if isinstance(value, str):
        return f"'{value.replace(chr(39), chr(39) + chr(39))}'"
    try:
        if float(value).is_integer():
            return str(int(value))
    except (TypeError, ValueError):
        pass
    return f"'{str(value).replace(chr(39), chr(39) + chr(39))}'"


def sql_id_list(values: Iterable) -> str:
    return ", ".join(sql_literal(value) for value in values if not pd.isna(value))


def chunked(values: list, size: int = ID_CHUNK_SIZE) -> Iterable[list]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def canonical_id(value) -> str:
    if pd.isna(value):
        return ""
    try:
        if float(value).is_integer():
            return str(int(value))
    except (TypeError, ValueError):
        pass
    return str(value)


def summarize_ids(values: Iterable[str]) -> tuple[int, list[str]]:
    ids = sorted({value for value in values if value})
    return len(ids), ids[:MISSING_ISSUE_ID_SAMPLE_SIZE]


def arrow_type_from_db_type(data_type, udt_name) -> pa.DataType:
    data_type = "" if pd.isna(data_type) else str(data_type).lower()
    udt_name = "" if pd.isna(udt_name) else str(udt_name).lower()
    type_text = f"{data_type} {udt_name}"
    if "timestamp" in type_text or data_type == "date":
        return pa.timestamp("ns")
    if "bool" in type_text:
        return pa.bool_()
    if any(token in type_text for token in ("smallint", "integer", "bigint", "int2", "int4", "int8")):
        return pa.int64()
    if any(
        token in type_text
        for token in (
            "numeric",
            "decimal",
            "double",
            "float",
            "real",
            "number",
            "float4",
            "float8",
        )
    ):
        return pa.float64()
    return pa.string()


def build_arrow_schema(metadata: pd.DataFrame) -> pa.Schema:
    fields = []
    for _, row in metadata.iterrows():
        fields.append(
            pa.field(
                str(row["column_name"]),
                arrow_type_from_db_type(row.get("data_type"), row.get("udt_name")),
                nullable=True,
            )
        )
    return pa.schema(fields)


def normalize_batch_to_schema(batch: pd.DataFrame, schema: pa.Schema) -> pd.DataFrame:
    normalized = pd.DataFrame(index=batch.index)
    for field in schema:
        if field.name in batch.columns:
            series = batch[field.name]
        else:
            series = pd.Series(pd.NA, index=batch.index)

        if pa.types.is_string(field.type):
            normalized[field.name] = series.astype("string")
        elif pa.types.is_integer(field.type):
            normalized[field.name] = pd.to_numeric(series, errors="coerce").astype("Int64")
        elif pa.types.is_floating(field.type):
            normalized[field.name] = pd.to_numeric(series, errors="coerce").astype("float64")
        elif pa.types.is_timestamp(field.type):
            normalized[field.name] = pd.to_datetime(series, errors="coerce")
        elif pa.types.is_boolean(field.type):
            normalized[field.name] = series.astype("boolean")
        else:
            normalized[field.name] = series.astype("string")
    return normalized


def choose_date_column(dataset: str, columns: Iterable[str]) -> str | None:
    column_lookup = {col.lower(): col for col in columns}
    for candidate in PRIMARY_DATE_CANDIDATES.get(dataset, ()):
        if candidate.lower() in column_lookup:
            return column_lookup[candidate.lower()]
    return None


def manifest_record(
    *,
    dataset: str,
    source: str,
    path: Path | None,
    status: str,
    df: pd.DataFrame | None = None,
    date_column: str | None = None,
    error: str | None = None,
    extra: dict | None = None,
) -> dict:
    record = {
        "dataset": dataset,
        "source_schema_table": source,
        "status": status,
        "row_count": None,
        "column_count": None,
        "distinct_issue_count": None,
        "distinct_issuer_count": None,
        "min_date": None,
        "max_date": None,
        "output_filepath": str(path) if path is not None else None,
        "file_size": path.stat().st_size if path is not None and path.exists() else None,
        "error_message": error,
        "extraction_timestamp_utc": utc_now(),
    }
    if df is not None:
        record["row_count"] = int(len(df))
        record["column_count"] = int(len(df.columns))
        if "issue_id" in df.columns:
            record["distinct_issue_count"] = int(df["issue_id"].nunique(dropna=True))
        if "issuer_id" in df.columns:
            record["distinct_issuer_count"] = int(df["issuer_id"].nunique(dropna=True))
        if date_column and date_column in df.columns and df[date_column].notna().any():
            dates = pd.to_datetime(df[date_column], errors="coerce")
            dates = dates.dropna()
            if not dates.empty:
                record["min_date"] = dates.min().date().isoformat()
                record["max_date"] = dates.max().date().isoformat()
    if extra:
        record.update(extra)
    return record


class FISDExtraExtractor:
    def __init__(self, db: WRDSClient, force: bool):
        self.db = db
        self.force = force
        self.manifest = load_manifest()
        FISD_EXTRA_DIR.mkdir(parents=True, exist_ok=True)

    def update_manifest(
        self,
        dataset: str,
        source: str,
        path: Path,
        status: str,
        df: pd.DataFrame,
        date_column: str | None = None,
        extra: dict | None = None,
    ) -> None:
        self.manifest[dataset] = manifest_record(
            dataset=dataset,
            source=source,
            path=path,
            status=status,
            df=df,
            date_column=date_column,
            extra=extra,
        )
        write_manifest(self.manifest)

    def record_failure(
        self,
        dataset: str,
        source: str,
        error: Exception,
        path: Path | None = None,
        extra: dict | None = None,
    ) -> None:
        log(f"FAILED {dataset}: {error}")
        self.manifest[dataset] = manifest_record(
            dataset=dataset,
            source=source,
            path=path,
            status="failed",
            error=str(error),
            extra=extra,
        )
        write_manifest(self.manifest)

    def table_column_metadata(self, schema: str, table: str) -> pd.DataFrame:
        sql = f"""
            SELECT
                column_name,
                data_type,
                udt_name,
                ordinal_position
            FROM information_schema.columns
            WHERE table_schema = '{schema}'
              AND table_name = '{table}'
            ORDER BY ordinal_position
        """
        try:
            metadata = query_with_retry(self.db, sql)
        except Exception as exc:
            log(f"Could not inspect metadata for {schema}.{table}: {exc}; trying LIMIT 0")
            limit_zero = query_with_retry(self.db, f"SELECT * FROM {schema}.{table} LIMIT 0")
            return pd.DataFrame(
                {
                    "column_name": limit_zero.columns.tolist(),
                    "data_type": [None] * len(limit_zero.columns),
                    "udt_name": [None] * len(limit_zero.columns),
                    "ordinal_position": list(range(1, len(limit_zero.columns) + 1)),
                }
            )
        if metadata.empty:
            log(f"No metadata columns returned for {schema}.{table}; trying LIMIT 0")
            limit_zero = query_with_retry(self.db, f"SELECT * FROM {schema}.{table} LIMIT 0")
            return pd.DataFrame(
                {
                    "column_name": limit_zero.columns.tolist(),
                    "data_type": [None] * len(limit_zero.columns),
                    "udt_name": [None] * len(limit_zero.columns),
                    "ordinal_position": list(range(1, len(limit_zero.columns) + 1)),
                }
            )
        return metadata

    def table_columns(self, schema: str, table: str) -> list[str]:
        metadata = self.table_column_metadata(schema, table)
        return metadata["column_name"].tolist()

    def discover_source(self, dataset: str) -> tuple[str, list[str]]:
        schema, table = SOURCE_TABLES[dataset]
        source = f"{schema}.{table}"
        columns = self.table_columns(schema, table)
        query_with_retry(self.db, f"SELECT * FROM {source} LIMIT 1")
        log(f"Selected FISD {dataset} source: {source}")
        return source, columns

    def extract_full_table(self, dataset: str) -> None:
        path = FISD_EXTRA_DIR / OUTPUT_FILES[dataset]
        source = ".".join(SOURCE_TABLES[dataset])
        required_column = EXPECTED_KEY_COLUMNS.get(dataset)
        try:
            source, columns = self.discover_source(dataset)
            date_column = choose_date_column(dataset, columns)
            if not self.force and is_valid_nonempty_parquet(path, required_column):
                cached = validate_saved_parquet(path, required_column=required_column)
                log(f"Skipping valid cache: {path}")
                self.update_manifest(dataset, source, path, "skipped", cached, date_column)
                return

            log(f"Extracting full FISD table: {source}")
            sql = f"SELECT * FROM {source}"
            df = query_with_retry(self.db, sql)
            if required_column and required_column not in df.columns:
                raise ValueError(f"{source} missing expected key column: {required_column}")
            if date_column and date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
            if required_column and required_column in df.columns:
                df = df.sort_values(required_column).reset_index(drop=True)

            safe_write_parquet(df, path)
            saved = validate_saved_parquet(
                path,
                expected_rows=len(df),
                required_column=required_column,
                require_nonempty=not df.empty,
            )
            self.update_manifest(dataset, source, path, "success", saved, date_column)
        except Exception as exc:
            self.record_failure(dataset, source, exc, path=path)

    def load_rated_issue_ids(self) -> list:
        if not RATED_ISSUE_MASTER_PATH.exists():
            raise FileNotFoundError(
                f"Missing local rated issue master: {RATED_ISSUE_MASTER_PATH}"
            )
        issue_df = pd.read_parquet(RATED_ISSUE_MASTER_PATH, columns=["issue_id"])
        if "issue_id" not in issue_df.columns:
            raise ValueError(f"{RATED_ISSUE_MASTER_PATH} missing issue_id")
        issue_ids = [
            value
            for value in issue_df["issue_id"].dropna().drop_duplicates().tolist()
            if canonical_id(value)
        ]
        issue_ids = sorted(issue_ids, key=lambda value: canonical_id(value))
        if not issue_ids:
            raise ValueError("No issue_id values found in local rated issue master")
        return issue_ids

    def validate_mergedissue_output(
        self,
        path: Path,
        expected_rows: int,
        requested_ids: list,
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        saved = validate_issue_id_parquet(
            path,
            expected_rows=expected_rows,
        )
        duplicated = saved.loc[saved["issue_id"].duplicated(), "issue_id"].tolist()
        duplicate_ids = sorted({canonical_id(value) for value in duplicated})
        if duplicate_ids:
            raise ValueError(
                "fisd_mergedissue_rated_full is not unique by issue_id; "
                f"duplicates: {duplicate_ids[:20]}"
            )
        returned_ids = {canonical_id(value) for value in saved["issue_id"].dropna()}
        requested_id_set = {canonical_id(value) for value in requested_ids}
        missing_ids = sorted(requested_id_set.difference(returned_ids))
        return saved, missing_ids, duplicate_ids

    def write_mergedissue_batches(
        self,
        *,
        source: str,
        arrow_schema: pa.Schema,
        issue_ids: list,
        path: Path,
    ) -> tuple[int, int, set[str]]:
        tmp_path = path.with_name(f".{path.name}.tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        writer = None
        schema = None
        row_count = 0
        returned_ids: set[str] = set()
        duplicate_ids: set[str] = set()
        batch_count = 0
        try:
            for batch_count, issue_chunk in enumerate(
                chunked(issue_ids, ID_CHUNK_SIZE),
                start=1,
            ):
                log(
                    "Extracting fisd_mergedissue rated issue batch "
                    f"{batch_count} ({len(issue_chunk)} issue IDs)"
                )
                sql = f"""
                    SELECT *
                    FROM {source}
                    WHERE issue_id IN ({sql_id_list(issue_chunk)})
                    ORDER BY issue_id
                """
                batch = query_with_retry(self.db, sql)
                if "issue_id" not in batch.columns:
                    raise ValueError(f"{source} returned no issue_id column")
                if batch.empty:
                    continue

                current_ids = [canonical_id(value) for value in batch["issue_id"]]
                duplicated_in_batch = {
                    canonical_id(value)
                    for value in batch.loc[batch["issue_id"].duplicated(), "issue_id"]
                }
                duplicated_across_batches = set(current_ids).intersection(returned_ids)
                duplicate_ids.update(duplicated_in_batch)
                duplicate_ids.update(duplicated_across_batches)
                if duplicate_ids:
                    raise ValueError(
                        "fisd_mergedissue returned duplicate issue_id values: "
                        f"{sorted(duplicate_ids)[:20]}"
                    )

                returned_ids.update(current_ids)
                batch = batch.sort_values("issue_id").reset_index(drop=True)
                batch = normalize_batch_to_schema(batch, arrow_schema)
                table = pa.Table.from_pandas(
                    batch,
                    schema=arrow_schema,
                    preserve_index=False,
                )
                if writer is None:
                    schema = arrow_schema
                    writer = pq.ParquetWriter(tmp_path, schema)
                writer.write_table(table)
                row_count += len(batch)

            if writer is not None:
                writer.close()
                writer = None
                os.replace(tmp_path, path)
            else:
                empty_arrays = [
                    pa.array([], type=field.type) for field in arrow_schema
                ]
                empty_table = pa.Table.from_arrays(
                    empty_arrays,
                    schema=arrow_schema,
                )
                pq.write_table(empty_table, tmp_path)
                os.replace(tmp_path, path)
        except Exception:
            if writer is not None:
                writer.close()
            if tmp_path.exists():
                tmp_path.unlink()
            raise

        return row_count, batch_count, returned_ids

    def extract_mergedissue_full(self) -> None:
        dataset = "mergedissue_full"
        path = FISD_EXTRA_DIR / OUTPUT_FILES[dataset]
        source = ".".join(SOURCE_TABLES[dataset])
        try:
            source, columns = self.discover_source(dataset)
            if "issue_id" not in {col.lower() for col in columns}:
                raise ValueError(f"{source} missing required issue_id column")
            schema_name, table_name = SOURCE_TABLES[dataset]
            column_metadata = self.table_column_metadata(schema_name, table_name)
            arrow_schema = build_arrow_schema(column_metadata)

            if not self.force and path.exists():
                cached = validate_issue_id_parquet(path, require_nonempty=True)
                if cached["issue_id"].duplicated().any():
                    raise ValueError(f"{path} contains duplicate issue_id values")
                log(f"Skipping valid cache: {path}")
                parquet_rows, parquet_columns = parquet_shape(path)
                extra = {
                    "total_local_rated_issue_ids_requested": None,
                    "total_rows_extracted": int(parquet_rows),
                    "missing_issue_id_count": None,
                    "missing_issue_id_sample": None,
                    "duplicate_issue_ids": [],
                    "number_of_query_batches": None,
                    "batch_size": int(ID_CHUNK_SIZE),
                    "column_count": int(parquet_columns),
                }
                try:
                    issue_ids = self.load_rated_issue_ids()
                    returned_ids = {
                        canonical_id(value) for value in cached["issue_id"].dropna()
                    }
                    requested_ids = {canonical_id(value) for value in issue_ids}
                    missing_count, missing_sample = summarize_ids(
                        requested_ids.difference(returned_ids)
                    )
                    extra.update(
                        {
                            "total_local_rated_issue_ids_requested": int(len(issue_ids)),
                            "missing_issue_id_count": int(missing_count),
                            "missing_issue_id_sample": missing_sample,
                        }
                    )
                except Exception as exc:
                    extra["local_rated_issue_id_audit_error"] = str(exc)
                self.update_manifest(dataset, source, path, "skipped", cached, extra=extra)
                return

            issue_ids = self.load_rated_issue_ids()
            row_count, batch_count, returned_ids = self.write_mergedissue_batches(
                source=source,
                arrow_schema=arrow_schema,
                issue_ids=issue_ids,
                path=path,
            )
            saved, missing_ids, duplicate_ids = self.validate_mergedissue_output(
                path,
                row_count,
                issue_ids,
            )
            parquet_rows, parquet_columns = parquet_shape(path)
            missing_count, missing_sample = summarize_ids(missing_ids)
            extra = {
                "total_local_rated_issue_ids_requested": int(len(issue_ids)),
                "total_rows_extracted": int(parquet_rows),
                "missing_issue_id_count": int(missing_count),
                "missing_issue_id_sample": missing_sample,
                "duplicate_issue_ids": duplicate_ids,
                "number_of_query_batches": int(batch_count),
                "batch_size": int(ID_CHUNK_SIZE),
                "column_count": int(parquet_columns),
            }
            self.update_manifest(dataset, source, path, "success", saved, extra=extra)
            if missing_ids:
                log(
                    "fisd_mergedissue rated full extract missing "
                    f"{len(missing_ids)} requested issue IDs"
                )
        except Exception as exc:
            self.record_failure(dataset, source, exc, path=path)

    def run(self, only: set[str]) -> None:
        for dataset in DATASETS:
            if dataset not in only:
                continue
            if dataset == "mergedissue_full":
                self.extract_mergedissue_full()
            else:
                self.extract_full_table(dataset)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DynamicPD-CLO FISD extra raw extractor")
    parser.add_argument("--extract-fisd-extra", action="store_true")
    parser.add_argument(
        "--only",
        default="all",
        help=(
            "rating, ratings, issue_default, issue_affected, related_issues, "
            "issue_enhancement, mergedissue_full, or all"
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.extract_fisd_extra:
        return

    only = normalize_only(args.only)
    with WRDSClient() as db:
        extractor = FISDExtraExtractor(db=db, force=args.force)
        extractor.run(only)


if __name__ == "__main__":
    main()
