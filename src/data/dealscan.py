import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import decimal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from src.data.wrds import WRDSClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEALSCAN_RAW_DIR = PROJECT_ROOT / "data/raw/wrds/dealscan"
MANIFEST_PATH = DEALSCAN_RAW_DIR / "extraction_manifest.json"

DEALSCAN_TABLES = (
    "borrowerbase",
    "chars",
    "company",
    "currfacpricing",
    "dealamendment",
    "dealpurposecomment",
    "dealscan",
    "facility",
    "facilityamendment",
    "facilitydates",
    "facilityguarantor",
    "facilitypaymentschedule",
    "facilityrepaymentcomment",
    "facilitysecurity",
    "facilitysponsor",
    "financialcovenant",
    "financialratios",
    "lendershares",
    "lins",
    "lpc_loanconnector_company_id_map",
    "marketsegment",
    "networthcovenant",
    "organizationtype",
    "package",
    "packageassignmentcomment",
    "packageprepaymentcomment",
    "performancepricing",
    "performancepricingcomments",
    "sublimits",
    "wrds_financial_covenants",
    "wrds_loanconnector_ids",
)

LARGE_TABLE_WARN_ROWS = 2_000_000


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    os.replace(tmp_path, path)


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"tables": {}}


NUMERIC_PG_TYPES = {"numeric", "double precision", "real", "money"}
INTEGER_PG_TYPES = {"integer", "bigint", "smallint"}
DATE_PG_TYPES = {"date", "timestamp without time zone", "timestamp with time zone"}


def fetch_column_types(db: WRDSClient, schema: str, table: str) -> dict:
    """Postgres column types, queried once per table. Pyarrow infers a type per batch from
    whatever values happen to be present (decimal precision, or 'null' for an all-null batch),
    and that inferred type drifts batch to batch -- breaking ParquetWriter's fixed schema. Instead
    of trusting per-batch inference, we fetch the real column types up front and force every
    batch to match them, so the schema can never drift."""
    result = db.query(
        f"""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = '{schema}' AND table_name = '{table}'
        ORDER BY ordinal_position
        """
    )
    return dict(zip(result["column_name"], result["data_type"]))


def normalize_chunk(chunk: pd.DataFrame, column_types: dict) -> pd.DataFrame:
    for col in chunk.columns:
        pg_type = column_types.get(col, "")
        if pg_type in NUMERIC_PG_TYPES:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("float64")
        elif pg_type in INTEGER_PG_TYPES:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("Int64")
        elif pg_type == "boolean":
            chunk[col] = chunk[col].astype("boolean")
        elif pg_type in DATE_PG_TYPES:
            chunk[col] = pd.to_datetime(chunk[col], errors="coerce")
        else:
            chunk[col] = chunk[col].astype("string")
    return chunk


def pull_table(db: WRDSClient, table: str, refresh: bool, batch_size: int = 50_000) -> dict:
    output_path = DEALSCAN_RAW_DIR / f"{table}.parquet"
    tmp_path = output_path.with_suffix(".parquet.tmp")

    if output_path.exists() and not refresh:
        log(f"{table}: cached at {output_path}, skipping (use --refresh to re-pull)")
        return {"status": "cached", "path": str(output_path)}

    try:
        expected_rows = db.count_rows("dealscan", table)
    except Exception as exc:
        log(f"{table}: FAILED on row count — {exc}")
        return {"status": "error", "error": str(exc)}

    log(f"{table}: {expected_rows:,} rows expected, streaming in batches of {batch_size:,} ...")
    start = time.monotonic()
    DEALSCAN_RAW_DIR.mkdir(parents=True, exist_ok=True)

    rows = 0
    columns: list[str] = []
    writer = None
    column_types = fetch_column_types(db, "dealscan", table)

    try:
        with tqdm(total=expected_rows, desc=table, unit="rows", unit_scale=True) as bar:
            for chunk in db.stream_query(f"SELECT * FROM dealscan.{table}", batch_size=batch_size):
                chunk = normalize_chunk(chunk, column_types)
                if writer is None:
                    columns = list(chunk.columns)
                    table_schema = pa.Table.from_pandas(chunk, preserve_index=False).schema
                    writer = pq.ParquetWriter(tmp_path, table_schema)
                writer.write_table(pa.Table.from_pandas(chunk, preserve_index=False))
                rows += len(chunk)
                bar.update(len(chunk))
    except Exception as exc:
        if writer is not None:
            writer.close()
        if tmp_path.exists():
            tmp_path.unlink()
        log(f"{table}: FAILED — {exc}")
        return {"status": "error", "error": str(exc)}
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        # Empty table: no rows at all.
        pq.write_table(pa.table({}), tmp_path)

    tmp_path.replace(output_path)
    elapsed = time.monotonic() - start

    log(f"{table}: saved {rows:,} rows x {len(columns)} cols to {output_path} ({elapsed:.1f}s)")

    return {
        "status": "ok",
        "path": str(output_path),
        "rows": int(rows),
        "expected_rows": expected_rows,
        "columns": columns,
        "elapsed_seconds": round(elapsed, 1),
        "extracted_at_utc": utc_now(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull all dealscan.* tables from WRDS to local parquet.")
    parser.add_argument("--refresh", action="store_true", help="Re-pull tables even if cached locally.")
    parser.add_argument("--tables", nargs="*", default=None, help="Subset of tables to pull (default: all).")
    args = parser.parse_args()

    tables = args.tables if args.tables else list(DEALSCAN_TABLES)
    unknown = [t for t in tables if t not in DEALSCAN_TABLES]
    if unknown:
        raise SystemExit(f"Unknown table(s) not in DEALSCAN_TABLES: {unknown}")

    manifest = load_manifest()
    manifest.setdefault("tables", {})

    with WRDSClient() as db:
        for table in tables:
            result = pull_table(db, table, refresh=args.refresh)
            manifest["tables"][table] = result
            write_json_atomic(MANIFEST_PATH, manifest)  # persist progress after each table

    ok = sum(1 for r in manifest["tables"].values() if r.get("status") == "ok")
    cached = sum(1 for r in manifest["tables"].values() if r.get("status") == "cached")
    failed = sum(1 for r in manifest["tables"].values() if r.get("status") == "error")
    log(f"Done. ok={ok} cached={cached} failed={failed} total={len(manifest['tables'])}")
    if failed:
        failed_tables = [t for t, r in manifest["tables"].items() if r.get("status") == "error"]
        log(f"Failed tables (re-run with --tables to retry just these): {failed_tables}")


if __name__ == "__main__":
    main()
