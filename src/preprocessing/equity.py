import argparse
import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CRSP_DIR = PROJECT_ROOT / "data/raw/wrds/crsp"
MONTHLY_DIR = CRSP_DIR / "monthly"
DAILY_DIR = CRSP_DIR / "daily"
MARKET_MONTHLY_DIR = CRSP_DIR / "market_monthly"
MARKET_DAILY_DIR = CRSP_DIR / "market_daily"
NAMES_PATH = CRSP_DIR / "crsp_names.parquet"
DELISTINGS_PATH = CRSP_DIR / "crsp_delistings.parquet"
CCM_PATH = PROJECT_ROOT / "data/raw/merton/ccm_links.parquet"
COMPUSTAT_PATH = PROJECT_ROOT / "data/processed/compustat/compustat_features.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data/raw/equity"
OUTPUT_PATH = OUTPUT_DIR / "equity_features.parquet"
MANIFEST_PATH = OUTPUT_DIR / "equity_features_manifest.json"

DEFAULT_START_YEAR = 2000
DEFAULT_END_YEAR = 2024
RAW_LOOKBACK_YEAR = 1999

MONTHLY_COLUMNS = (
    "permno",
    "permco",
    "date",
    "ret",
    "retx",
    "prc",
    "altprc",
    "vol",
    "shrout",
    "hexcd",
    "hsiccd",
)
MARKET_MONTHLY_COLUMNS = (
    "date",
    "vwretd",
    "vwretx",
    "ewretd",
    "ewretx",
    "sprtrn",
    "spindx",
)
DELISTING_COLUMNS = ("permno", "dlstdt", "dlret", "dlretx", "dlprc", "dlstcd")
CCM_COLUMNS = ("gvkey", "lpermno", "lpermco", "linkdt", "linkenddt", "linktype", "linkprim")
COMPUSTAT_COLUMNS = ("gvkey", "quarter_end", "atq")
OUTPUT_COLUMNS = (
    "gvkey",
    "permno",
    "permco",
    "ticker",
    "comnam",
    "quarter_end",
    "month_end",
    "ret",
    "dlret",
    "ret_with_dl",
    "ret_12m",
    "market_ret_vw",
    "market_ret_ew",
    "market_ret_12m_vw",
    "market_ret_12m_ew",
    "prc",
    "shrout",
    "mktcap",
    "mktcap_mil",
    "atq",
    "market_to_book",
    "hexcd",
    "hsiccd",
    "shrcd",
    "exchcd",
    "siccd",
    "linktype",
    "linkprim",
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


def read_parquet(path: Path, columns: list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=list(columns) if columns else None)
    except OSError as exc:
        if "Repetition level histogram size mismatch" not in str(exc):
            raise
        log(f"PyArrow could not read {path.name}; retrying with fastparquet")
        return pd.read_parquet(path, columns=list(columns) if columns else None, engine="fastparquet")


def parquet_columns(path: Path) -> list[str]:
    return pq.read_schema(path).names


def yearly_files(directory: Path, stem: str, raw_start_year: int, end_year: int) -> list[Path]:
    files = []
    for year in range(raw_start_year, end_year + 1):
        path = directory / f"{stem}_{year}.parquet"
        if path.exists():
            files.append(path)
        else:
            log(f"Missing expected CRSP file for {year}: {path}")
    return files


def month_end(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    return dates.dt.to_period("M").dt.to_timestamp("M")


def quarter_end(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    return dates.dt.to_period("Q").dt.to_timestamp("Q")


def compound_return(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    return float(np.prod(1.0 + clean) - 1.0)


def load_monthly(raw_start_year: int, end_year: int) -> pd.DataFrame:
    frames = []
    for path in yearly_files(MONTHLY_DIR, "crsp_monthly", raw_start_year, end_year):
        available = parquet_columns(path)
        missing = [col for col in ("permno", "date", "ret", "prc", "shrout") if col not in available]
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        selected = [col for col in MONTHLY_COLUMNS if col in available]
        log(f"Loading CRSP monthly partition {path.stem.rsplit('_', 1)[-1]}")
        frames.append(read_parquet(path, selected))
    if not frames:
        raise RuntimeError("No CRSP monthly partitions found")

    df = pd.concat(frames, ignore_index=True)
    df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
    if "permco" in df.columns:
        df["permco"] = pd.to_numeric(df["permco"], errors="coerce").astype("Int64")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month_end"] = month_end(df["date"])
    for col in ("ret", "retx", "prc", "altprc", "vol", "shrout"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("hexcd", "hsiccd"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df.dropna(subset=["permno", "month_end"])


def load_market_monthly(raw_start_year: int, end_year: int) -> pd.DataFrame:
    frames = []
    for path in yearly_files(MARKET_MONTHLY_DIR, "crsp_market_monthly", raw_start_year, end_year):
        available = parquet_columns(path)
        selected = [col for col in MARKET_MONTHLY_COLUMNS if col in available]
        log(f"Loading CRSP market monthly partition {path.stem.rsplit('_', 1)[-1]}")
        frames.append(read_parquet(path, selected))
    if not frames:
        raise RuntimeError("No CRSP market monthly partitions found")

    market = pd.concat(frames, ignore_index=True)
    market["month_end"] = month_end(market["date"])
    market["market_ret_vw"] = pd.to_numeric(market.get("vwretd"), errors="coerce")
    market["market_ret_ew"] = pd.to_numeric(market.get("ewretd"), errors="coerce")
    market = market.sort_values("month_end")
    market["market_ret_12m_vw"] = (
        market["market_ret_vw"].rolling(12, min_periods=9).apply(lambda x: np.prod(1.0 + x) - 1.0, raw=True)
    )
    market["market_ret_12m_ew"] = (
        market["market_ret_ew"].rolling(12, min_periods=9).apply(lambda x: np.prod(1.0 + x) - 1.0, raw=True)
    )
    return market[
        ["month_end", "market_ret_vw", "market_ret_ew", "market_ret_12m_vw", "market_ret_12m_ew"]
    ].drop_duplicates("month_end", keep="last")


def load_delistings() -> tuple[pd.DataFrame, dict]:
    if not DELISTINGS_PATH.exists():
        return pd.DataFrame(columns=["permno", "delist_date", "month_end", "dlret", "dlprc", "dlstcd"]), {
            "status": "missing",
            "path": str(DELISTINGS_PATH),
        }
    available = parquet_columns(DELISTINGS_PATH)
    selected = [col for col in DELISTING_COLUMNS if col in available]
    dl = read_parquet(DELISTINGS_PATH, selected)
    dl["permno"] = pd.to_numeric(dl["permno"], errors="coerce").astype("Int64")
    dl["delist_date"] = pd.to_datetime(dl["dlstdt"], errors="coerce")
    dl["month_end"] = month_end(dl["dlstdt"])
    dl["dlret"] = pd.to_numeric(dl.get("dlret"), errors="coerce")
    if "dlprc" in dl.columns:
        dl["dlprc"] = pd.to_numeric(dl["dlprc"], errors="coerce")
    if "dlstcd" in dl.columns:
        dl["dlstcd"] = pd.to_numeric(dl["dlstcd"], errors="coerce").astype("Int64")
    dl = dl.dropna(subset=["permno", "month_end"])
    dl = dl.sort_values(["permno", "month_end"]).drop_duplicates(["permno", "month_end"], keep="last")
    keep = [col for col in ["permno", "delist_date", "month_end", "dlret", "dlprc", "dlstcd"] if col in dl.columns]
    return dl[keep], {
        "status": "loaded",
        "path": str(DELISTINGS_PATH),
        "rows": int(len(dl)),
        "non_null_dlret_rows": int(dl["dlret"].notna().sum()),
    }


def load_names() -> tuple[pd.DataFrame, dict]:
    if not NAMES_PATH.exists():
        return pd.DataFrame(columns=["permno", "namedt", "nameendt"]), {
            "status": "missing",
            "path": str(NAMES_PATH),
        }
    available = parquet_columns(NAMES_PATH)
    selected = [
        col
        for col in (
            "permno",
            "permco",
            "namedt",
            "nameendt",
            "ticker",
            "comnam",
            "shrcd",
            "exchcd",
            "siccd",
        )
        if col in available
    ]
    names = read_parquet(NAMES_PATH, selected)
    names["permno"] = pd.to_numeric(names["permno"], errors="coerce").astype("Int64")
    if "permco" in names.columns:
        names["permco"] = pd.to_numeric(names["permco"], errors="coerce").astype("Int64")
    names["namedt"] = pd.to_datetime(names["namedt"], errors="coerce")
    names["nameendt"] = pd.to_datetime(names["nameendt"], errors="coerce")
    for col in ("shrcd", "exchcd", "siccd"):
        if col in names.columns:
            names[col] = pd.to_numeric(names[col], errors="coerce").astype("Int64")
    names = names.dropna(subset=["permno", "namedt"])
    return names, {
        "status": "loaded",
        "path": str(NAMES_PATH),
        "rows": int(len(names)),
    }


def apply_names_history(monthly: pd.DataFrame, names: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if names.empty:
        return monthly, {"names_rows_after_date_windows": 0, "monthly_rows_without_names": int(len(monthly))}

    monthly = monthly.copy()
    monthly["_crsp_row_id"] = np.arange(len(monthly))
    with_names = monthly.merge(
        names,
        on="permno",
        how="left",
        suffixes=("", "_name"),
    )
    open_end = pd.Timestamp("2099-12-31")
    valid_name = (
        (with_names["month_end"] >= with_names["namedt"])
        & (with_names["month_end"] <= with_names["nameendt"].fillna(open_end))
    )
    matched = with_names.loc[valid_name].copy()
    matched = matched.sort_values(["_crsp_row_id", "namedt"])
    matched = matched.drop_duplicates("_crsp_row_id", keep="last")

    metadata_cols = [
        col
        for col in (
            "_crsp_row_id",
            "ticker",
            "comnam",
            "shrcd",
            "exchcd",
            "siccd",
        )
        if col in matched.columns
    ]
    enriched = monthly.merge(
        matched[metadata_cols],
        on="_crsp_row_id",
        how="left",
        suffixes=("", "_name_valid"),
    )
    missing_count = int(enriched["ticker"].isna().sum()) if "ticker" in enriched.columns else int(len(enriched))
    return enriched.drop(columns=["_crsp_row_id"]), {
        "names_rows_after_date_windows": int(len(matched)),
        "monthly_rows_without_names": missing_count,
    }


def apply_delisting_returns(monthly: pd.DataFrame, delistings: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if delistings.empty:
        out = monthly.copy()
        out["dlret"] = np.nan
        out["dlstcd"] = pd.NA
        out["ret_with_dl"] = pd.to_numeric(out["ret"], errors="coerce")
        return out, {
            "delisting_rows_matched_to_monthly": 0,
            "delisting_only_rows_appended": 0,
            "delisting_rows_with_non_null_ret_used": 0,
        }

    out = monthly.merge(delistings, on=["permno", "month_end"], how="left")
    ret = pd.to_numeric(out["ret"], errors="coerce")
    dlret = pd.to_numeric(out["dlret"], errors="coerce")
    out["ret_with_dl"] = ret
    both = ret.notna() & dlret.notna()
    out.loc[both, "ret_with_dl"] = (1.0 + ret.loc[both]) * (1.0 + dlret.loc[both]) - 1.0
    only_dl = ret.isna() & dlret.notna()
    out.loc[only_dl, "ret_with_dl"] = dlret.loc[only_dl]

    monthly_keys = monthly[["permno", "month_end"]].drop_duplicates()
    unmatched = delistings.merge(
        monthly_keys,
        on=["permno", "month_end"],
        how="left",
        indicator=True,
    )
    unmatched = unmatched[unmatched["_merge"] == "left_only"].drop(columns=["_merge"])
    if not unmatched.empty:
        append_payload = {}
        for col in out.columns:
            if col in unmatched.columns:
                append_payload[col] = unmatched[col]
        append = pd.DataFrame(append_payload, index=unmatched.index)
        append["date"] = unmatched["delist_date"]
        append["ret"] = np.nan
        append["retx"] = np.nan
        append["ret_with_dl"] = pd.to_numeric(append["dlret"], errors="coerce")
        if "dlprc" in unmatched.columns:
            append["prc"] = pd.to_numeric(unmatched["dlprc"], errors="coerce")
        append = append.reindex(columns=out.columns)
        append = append.dropna(axis=1, how="all")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
                category=FutureWarning,
            )
            out = pd.concat([out, append], ignore_index=True)

    return out, {
        "delisting_rows_matched_to_monthly": int(out["dlret"].notna().sum() - unmatched["dlret"].notna().sum()),
        "delisting_only_rows_appended": int(len(unmatched)),
        "delisting_rows_with_non_null_ret_used": int(out["dlret"].notna().sum()),
    }


def load_ccm_links() -> pd.DataFrame:
    links = read_parquet(CCM_PATH, CCM_COLUMNS)
    missing = [col for col in CCM_COLUMNS if col not in links.columns]
    if missing:
        raise ValueError(f"{CCM_PATH} missing required CCM columns: {missing}")
    links = links.copy()
    links["gvkey"] = links["gvkey"].astype("string")
    links["lpermno"] = pd.to_numeric(links["lpermno"], errors="coerce").astype("Int64")
    links["lpermco"] = pd.to_numeric(links["lpermco"], errors="coerce").astype("Int64")
    links["linkdt"] = pd.to_datetime(links["linkdt"], errors="coerce")
    links["linkenddt"] = pd.to_datetime(links["linkenddt"], errors="coerce")
    links = links.dropna(subset=["gvkey", "lpermno", "linkdt"])
    links = links[links["linktype"].isin(["LC", "LU"])]
    links = links[links["linkprim"].isin(["P", "C"])]
    return links


def link_to_gvkey(monthly: pd.DataFrame, links: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    linked = monthly.merge(
        links,
        left_on="permno",
        right_on="lpermno",
        how="inner",
        suffixes=("", "_ccm"),
    )
    open_end = pd.Timestamp("2099-12-31")
    valid_window = (
        (linked["date"] >= linked["linkdt"])
        & (linked["date"] <= linked["linkenddt"].fillna(open_end))
    )
    linked = linked.loc[valid_window].copy()

    linked["mktcap"] = linked["prc"].abs() * linked["shrout"]
    linked["mktcap_mil"] = linked["mktcap"] / 1000.0
    linked["linkprim_rank"] = linked["linkprim"].map({"P": 0, "C": 1}).fillna(9)
    linked["linktype_rank"] = linked["linktype"].map({"LC": 0, "LU": 1}).fillna(9)
    linked = linked.sort_values(
        ["gvkey", "month_end", "mktcap", "linkprim_rank", "linktype_rank"],
        ascending=[True, True, False, True, True],
    )
    duplicate_gvkey_months = int(linked.duplicated(["gvkey", "month_end"]).sum())
    linked = linked.drop_duplicates(["gvkey", "month_end"], keep="first")
    return linked, {
        "linked_rows_after_date_windows": int(len(linked)),
        "duplicate_gvkey_month_rows_dropped": duplicate_gvkey_months,
    }


def add_trailing_returns(linked: pd.DataFrame) -> pd.DataFrame:
    out = linked.sort_values(["gvkey", "month_end"]).copy()
    out["ret_12m"] = (
        out.groupby("gvkey", group_keys=False)["ret_with_dl"]
        .rolling(12, min_periods=9)
        .apply(lambda x: np.prod(1.0 + x) - 1.0, raw=True)
        .reset_index(level=0, drop=True)
    )
    return out


def load_compustat_atq() -> pd.DataFrame:
    comp = read_parquet(COMPUSTAT_PATH, COMPUSTAT_COLUMNS)
    comp["gvkey"] = comp["gvkey"].astype("string")
    comp["quarter_end"] = pd.to_datetime(comp["quarter_end"], errors="coerce")
    comp["atq"] = pd.to_numeric(comp["atq"], errors="coerce")
    comp = comp.dropna(subset=["gvkey", "quarter_end"])
    return comp.sort_values(["gvkey", "quarter_end"]).drop_duplicates(["gvkey", "quarter_end"], keep="last")


def build_quarterly_features(linked: pd.DataFrame, market: pd.DataFrame, compustat: pd.DataFrame) -> pd.DataFrame:
    quarter_months = linked[linked["month_end"].dt.month.isin([3, 6, 9, 12])].copy()
    quarter_months["quarter_end"] = quarter_end(quarter_months["month_end"])
    quarter_months = quarter_months.merge(market, on="month_end", how="left")
    quarter_months = quarter_months.merge(compustat, on=["gvkey", "quarter_end"], how="left")
    quarter_months["market_to_book"] = np.where(
        quarter_months["atq"] > 0,
        quarter_months["mktcap_mil"] / quarter_months["atq"],
        np.nan,
    )
    quarter_months = quarter_months.sort_values(["gvkey", "quarter_end", "mktcap"], ascending=[True, True, False])
    quarter_months = quarter_months.drop_duplicates(["gvkey", "quarter_end"], keep="first")
    return quarter_months[[col for col in OUTPUT_COLUMNS if col in quarter_months.columns]]


def validate_output(path: Path, start_year: int, end_year: int) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Output file was not created: {path}")
    keys = read_parquet(path, ["gvkey", "quarter_end"])
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


def process_equity(
    *,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    force: bool = False,
) -> dict:
    if start_year < 2000:
        raise ValueError("start_year must be 2000 or later for the modeling panel")
    if end_year < start_year:
        raise ValueError("end_year must be greater than or equal to start_year")

    if OUTPUT_PATH.exists() and not force:
        validation = validate_output(OUTPUT_PATH, start_year, end_year)
        manifest = {"status": "skipped_existing", "output_path": str(OUTPUT_PATH), **validation}
        log(f"Using existing equity feature file: {OUTPUT_PATH}")
        return manifest

    raw_start_year = min(RAW_LOOKBACK_YEAR, start_year - 1)
    monthly_files = yearly_files(MONTHLY_DIR, "crsp_monthly", raw_start_year, end_year)
    daily_files = yearly_files(DAILY_DIR, "crsp_daily", raw_start_year, end_year)
    market_monthly_files = yearly_files(MARKET_MONTHLY_DIR, "crsp_market_monthly", raw_start_year, end_year)
    market_daily_files = yearly_files(MARKET_DAILY_DIR, "crsp_market_daily", raw_start_year, end_year)

    monthly = load_monthly(raw_start_year, end_year)
    raw_monthly_rows = int(len(monthly))
    market = load_market_monthly(raw_start_year, end_year)
    delistings, delisting_manifest = load_delistings()
    monthly, delisting_metrics = apply_delisting_returns(monthly, delistings)
    monthly_rows_after_delistings = int(len(monthly))
    names, names_manifest = load_names()
    monthly, names_metrics = apply_names_history(monthly, names)
    links = load_ccm_links()
    linked, link_metrics = link_to_gvkey(monthly, links)
    linked = add_trailing_returns(linked)
    compustat = load_compustat_atq()
    output = build_quarterly_features(linked, market, compustat)
    output = output[
        (output["quarter_end"] >= pd.Timestamp(f"{start_year}-01-01"))
        & (output["quarter_end"] <= pd.Timestamp(f"{end_year}-12-31"))
    ].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = OUTPUT_PATH.with_name(f".{OUTPUT_PATH.name}.tmp")
    output.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, OUTPUT_PATH)
    validation = validate_output(OUTPUT_PATH, start_year, end_year)

    manifest = {
        "status": "success",
        "created_at_utc": utc_now(),
        "output_path": str(OUTPUT_PATH),
        "start_year": start_year,
        "end_year": end_year,
        "monthly_input_directory": str(MONTHLY_DIR),
        "daily_input_directory": str(DAILY_DIR),
        "market_monthly_input_directory": str(MARKET_MONTHLY_DIR),
        "market_daily_input_directory": str(MARKET_DAILY_DIR),
        "names_path": str(NAMES_PATH),
        "names": names_manifest,
        "delistings": delisting_manifest,
        "ccm_path": str(CCM_PATH),
        "compustat_path": str(COMPUSTAT_PATH),
        "monthly_files": len(monthly_files),
        "daily_files_present": len(daily_files),
        "market_monthly_files": len(market_monthly_files),
        "market_daily_files_present": len(market_daily_files),
        "raw_monthly_rows": raw_monthly_rows,
        "monthly_rows_after_delistings": monthly_rows_after_delistings,
        "monthly_rows_after_names_windows": int(len(monthly)),
        "ccm_links_used": int(len(links)),
        "link_policy": "CRSP rows are joined to CCM on permno and retained only when linkdt <= CRSP observation date <= linkenddt. Linktypes LC/LU and linkprim P/C are retained.",
        "security_selection_policy": "When multiple securities map to a gvkey-month, keep the largest market capitalization security, with primary/current links as tiebreakers.",
        "delisting_return_policy": "ret_with_dl compounds CRSP monthly ret with dlret when both are present, and uses dlret when ret is missing.",
        "names_policy": "CRSP names rows are joined by permno when namedt <= month_end <= nameendt. Names are metadata enrichment only; missing names do not drop CRSP observations.",
        "market_to_book_units": "mktcap_mil is abs(prc) * shrout / 1000 because CRSP shrout is in thousands and Compustat atq is in millions.",
        **delisting_metrics,
        **names_metrics,
        **link_metrics,
        **validation,
    }
    write_json_atomic(MANIFEST_PATH, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CRSP equity features from local parquet archives.")
    parser.add_argument("--process-equity", action="store_true", help="Run local CRSP equity preprocessing.")
    parser.add_argument("--force", action="store_true", help="Rebuild output even if a valid file already exists.")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.process_equity:
        log("Nothing to do. Pass --process-equity to build equity features.")
        return
    manifest = process_equity(
        start_year=args.start_year,
        end_year=args.end_year,
        force=args.force,
    )
    log(f"Equity preprocessing finished with status: {manifest['status']}")
    log(f"Output: {manifest['output_path']}")


if __name__ == "__main__":
    main()
