import argparse
import difflib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CRSP_NAMES_PATH = PROJECT_ROOT / "data/raw/wrds/crsp/crsp_names.parquet"
CCM_LINKS_PATH = PROJECT_ROOT / "data/raw/merton/ccm_links.parquet"
COMPUSTAT_ANNUAL_DIR = PROJECT_ROOT / "data/raw/wrds/compustat_annual"
DEALSCAN_RAW_DIR = PROJECT_ROOT / "data/raw/wrds/dealscan"
DEALSCAN_COMPANY_PATH = DEALSCAN_RAW_DIR / "company.parquet"
DEALSCAN_WIDE_PATH = DEALSCAN_RAW_DIR / "dealscan.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/dealscan"
OUTPUT_PATH = OUTPUT_DIR / "dealscan_gvkey_link.parquet"
MANIFEST_PATH = OUTPUT_DIR / "dealscan_gvkey_link_manifest.json"

CCM_COLUMNS = ("gvkey", "lpermno", "lpermco", "linkdt", "linkenddt", "linktype", "linkprim")
COMPUSTAT_START_YEAR = 1998
COMPUSTAT_END_YEAR = 2024
FUZZY_THRESHOLD = 0.92
FUZZY_MIN_CORE_LEN = 6
FUZZY_BLOCK_CHARS = 4

# Legal-form tokens carry no identifying information and are written inconsistently across
# vendors ("Inc" vs "Incorporated"), so they are standardised before matching and dropped
# entirely when building the more permissive "core" name key.
SUFFIX_CANON = {
    "INCORPORATED": "INC",
    "CORPORATION": "CORP",
    "COMPANY": "CO",
    "COMPANIES": "CO",
    "LIMITED": "LTD",
    "HOLDINGS": "HLDGS",
    "HOLDING": "HLDGS",
    "INTERNATIONAL": "INTL",
    "INDUSTRIES": "IND",
    "TECHNOLOGIES": "TECH",
    "TECHNOLOGY": "TECH",
    "SERVICES": "SVCS",
    "SERVICE": "SVCS",
    "GROUP": "GRP",
    "SYSTEMS": "SYS",
    "PARTNERS": "PTNRS",
    "RESOURCES": "RES",
    "AND": "&",
}
DROPPABLE_SUFFIXES = {
    "INC", "CORP", "CO", "LTD", "LLC", "LP", "LLP", "PLC", "SA", "NV", "AG", "GMBH",
    "SPA", "AB", "AS", "OY", "BV", "PTY", "ULC", "TRUST", "HLDGS", "GRP", "CL",
}


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    tmp.replace(path)


def normalize_name(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).upper()
    text = re.sub(r"[^A-Z0-9& ]+", " ", text)
    tokens = [SUFFIX_CANON.get(tok, tok) for tok in text.split()]
    return " ".join(tokens).strip()


def core_key(normalized: str) -> str:
    """Normalized name with trailing legal-form tokens removed and spacing collapsed, so
    'ACME HOLDINGS INC' and 'Acme Holdings, Incorporated' converge on the same key."""
    if not normalized:
        return ""
    tokens = normalized.split()
    while tokens and tokens[-1] in DROPPABLE_SUFFIXES:
        tokens.pop()
    return "".join(tokens)


def add_name_keys(frame: pd.DataFrame, source_col: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["name_norm"] = frame[source_col].map(normalize_name)
    frame["name_core"] = frame["name_norm"].map(core_key)
    return frame


def load_dealscan_borrowers() -> pd.DataFrame:
    """Borrower universe = legacy company table plus distinct borrowers appearing only in the
    wide dealscan table. The wide table is the live source (the legacy tables stop in 2020) and
    about a quarter of its borrower ids are absent from the legacy company table, so linking on
    the company table alone leaves those borrowers permanently unmatchable."""
    company = pd.read_parquet(
        DEALSCAN_COMPANY_PATH, columns=["companyid", "company", "ticker", "publicprivate", "primarysiccode"]
    )
    company["companyid"] = pd.to_numeric(company["companyid"], errors="coerce").astype("Int64")
    company = company.dropna(subset=["companyid"]).rename(columns={"company": "name"})
    company["source"] = "company_table"
    log(f"legacy company table borrowers: {len(company):,}")

    seen = set(company["companyid"].dropna().astype("int64").tolist())
    extra = []
    for batch in pq.ParquetFile(DEALSCAN_WIDE_PATH).iter_batches(
        batch_size=200_000, columns=["borrower_id", "borrower_name", "ticker"]
    ):
        df = batch.to_pandas()
        df["companyid"] = pd.to_numeric(df["borrower_id"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["companyid"]).drop_duplicates("companyid")
        df = df[~df["companyid"].astype("int64").isin(seen)]
        if len(df):
            seen.update(df["companyid"].astype("int64").tolist())
            extra.append(
                df[["companyid", "borrower_name", "ticker"]].rename(columns={"borrower_name": "name"})
            )
    if extra:
        extra = pd.concat(extra, ignore_index=True).drop_duplicates("companyid")
        extra["publicprivate"] = pd.Series([pd.NA] * len(extra), dtype=company["publicprivate"].dtype)
        extra["primarysiccode"] = pd.Series([pd.NA] * len(extra), dtype=company["primarysiccode"].dtype)
        extra["source"] = "wide_table_only"
        log(f"additional borrowers found only in the wide table: {len(extra):,}")
        borrowers = pd.concat([company, extra], ignore_index=True)
    else:
        borrowers = company

    borrowers = borrowers.drop_duplicates("companyid").reset_index(drop=True)
    borrowers["ticker_norm"] = borrowers["ticker"].astype("string").str.upper().str.strip()
    return add_name_keys(borrowers, "name")


def load_compustat_identity() -> pd.DataFrame:
    """gvkey identity table from the annual Compustat archive: every distinct company name and
    ticker a gvkey has carried, so name changes over the sample do not cost us a match."""
    frames = []
    for year in range(COMPUSTAT_START_YEAR, COMPUSTAT_END_YEAR + 1):
        path = COMPUSTAT_ANNUAL_DIR / f"compustat_annual_{year}.parquet"
        if not path.exists():
            continue
        available = pq.ParquetFile(path).schema_arrow.names
        wanted = [c for c in ("gvkey", "conm", "tic", "sic") if c in available]
        frames.append(pd.read_parquet(path, columns=wanted))
    identity = pd.concat(frames, ignore_index=True)
    identity["gvkey"] = identity["gvkey"].astype("string")
    identity = identity.dropna(subset=["gvkey", "conm"]).drop_duplicates(["gvkey", "conm"])
    identity["tic_norm"] = identity["tic"].astype("string").str.upper().str.strip()
    log(f"compustat identity rows: {len(identity):,} | unique gvkeys: {identity['gvkey'].nunique():,}")
    return add_name_keys(identity, "conm")


def load_crsp_ticker_map() -> pd.DataFrame:
    names = pd.read_parquet(CRSP_NAMES_PATH, columns=["permno", "ticker", "comnam"])
    names = names.dropna(subset=["permno", "ticker"])
    names["permno"] = pd.to_numeric(names["permno"], errors="coerce").astype("Int64")
    names["ticker"] = names["ticker"].astype("string").str.upper().str.strip()
    names["comnam"] = names["comnam"].astype("string").str.upper().str.strip()
    return names.drop_duplicates(["ticker", "permno", "comnam"])


def load_ccm_links() -> pd.DataFrame:
    links = pd.read_parquet(CCM_LINKS_PATH, columns=list(CCM_COLUMNS))
    links["gvkey"] = links["gvkey"].astype("string")
    links["lpermno"] = pd.to_numeric(links["lpermno"], errors="coerce").astype("Int64")
    links = links.dropna(subset=["gvkey", "lpermno"])
    links = links[links["linktype"].isin(["LC", "LU"])]
    links = links[links["linkprim"].isin(["P", "C"])]
    links["linkprim_rank"] = links["linkprim"].map({"P": 0, "C": 1})
    links["linktype_rank"] = links["linktype"].map({"LC": 0, "LU": 1})
    links = links.sort_values(["lpermno", "linkprim_rank", "linktype_rank"])
    links = links.drop_duplicates("lpermno", keep="first")
    return links[["lpermno", "gvkey"]].rename(columns={"lpermno": "permno"})


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def candidate_map(frame: pd.DataFrame, key: str) -> dict:
    """key -> list of (gvkey, name_norm). A key resolving to several gvkeys is ambiguous and is
    never silently collapsed to one."""
    out = {}
    for key_value, gvkey, name_norm, name_core in zip(
        frame[key], frame["gvkey"], frame["name_norm"], frame["name_core"]
    ):
        if not isinstance(key_value, str) or not key_value:
            continue
        if not isinstance(name_norm, str):
            name_norm = ""
        if not isinstance(name_core, str):
            name_core = ""
        out.setdefault(key_value, set()).add((gvkey, name_norm, name_core))
    return {k: sorted(v) for k, v in out.items()}


def resolve(candidates, borrower_core: str, match_type: str) -> dict:
    """Pick a gvkey from the candidates for one borrower, using name similarity only to break
    ties among an already-keyed candidate set. Similarity is measured on the suffix-stripped core
    name so the reported score is comparable across layers."""
    gvkeys = {c[0] for c in candidates}
    if len(gvkeys) == 1:
        gvkey, _, cand_core = candidates[0]
        return {"gvkey": gvkey, "match_type": match_type, "name_similarity": similarity(borrower_core, cand_core)}
    scored = sorted(((similarity(borrower_core, c[2]), c[0]) for c in candidates), reverse=True)
    best_score, best_gvkey = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    if best_score >= 0.90 and best_score - runner_up >= 0.05:
        return {"gvkey": best_gvkey, "match_type": f"{match_type}_resolved", "name_similarity": best_score}
    return {"gvkey": pd.NA, "match_type": "ambiguous_needs_review", "name_similarity": best_score}


def build_link(borrowers: pd.DataFrame, identity: pd.DataFrame, use_fuzzy: bool) -> pd.DataFrame:
    # Layer 1: DealScan ticker -> CRSP ticker -> permno -> CCM -> gvkey (highest confidence).
    crsp = load_crsp_ticker_map().merge(load_ccm_links(), on="permno", how="inner")
    crsp = crsp.dropna(subset=["gvkey"]).rename(columns={"comnam": "name_norm"})
    crsp["name_norm"] = crsp["name_norm"].fillna("").map(normalize_name)
    crsp["name_core"] = crsp["name_norm"].map(core_key)
    crsp_by_ticker = candidate_map(crsp, "ticker")

    # A DealScan-ticker -> Compustat-tic layer was tried and removed: Compustat `tic` holds only
    # the CURRENT ticker, and tickers are recycled after a delisting, so historical borrowers got
    # assigned to whoever holds the ticker today (e.g. Jacuzzi Corp -> an ETF). Ticker matching is
    # only safe against CRSP's names history, which pairs a ticker with the firm that held it.
    # Layers 2 and 3: name-based keys against Compustat company names.
    comp_by_norm = candidate_map(identity, "name_norm")
    comp_by_core = candidate_map(identity, "name_core")

    fuzzy_blocks = {}
    if use_fuzzy:
        for key, cands in comp_by_core.items():
            if len(key) >= FUZZY_MIN_CORE_LEN:
                fuzzy_blocks.setdefault(key[:FUZZY_BLOCK_CHARS], []).append(key)

    rows = []
    for row in borrowers.itertuples(index=False):
        ticker = row.ticker_norm if isinstance(row.ticker_norm, str) else ""
        result = {
            "companyid": row.companyid,
            "company": row.name,
            "ticker": row.ticker,
            "publicprivate": row.publicprivate,
            "primarysiccode": row.primarysiccode,
            "borrower_source": row.source,
            "gvkey": pd.NA,
            "match_type": "unmatched",
            "name_similarity": np.nan,
        }

        for key, table, label in (
            (ticker, crsp_by_ticker, "ticker_crsp"),
            (row.name_norm, comp_by_norm, "exact_name"),
            (row.name_core, comp_by_core, "core_name"),
        ):
            if not key or key not in table:
                continue
            resolved = resolve(table[key], row.name_core, label)
            if not pd.isna(resolved["gvkey"]):
                result.update(resolved)
                break
            # Keep the ambiguity on record but let a later, different key try to resolve it.
            result.update(resolved)

        if use_fuzzy and pd.isna(result["gvkey"]) and len(row.name_core) >= FUZZY_MIN_CORE_LEN:
            block = fuzzy_blocks.get(row.name_core[:FUZZY_BLOCK_CHARS], [])
            close = difflib.get_close_matches(row.name_core, block, n=1, cutoff=FUZZY_THRESHOLD)
            if close:
                resolved = resolve(comp_by_core[close[0]], row.name_core, "fuzzy_name")
                if not pd.isna(resolved["gvkey"]):
                    result.update(resolved)

        rows.append(result)

    return pd.DataFrame(rows)


def summarize(link: pd.DataFrame) -> dict:
    return {
        "total_dealscan_borrowers": int(len(link)),
        "matched_to_gvkey": int(link["gvkey"].notna().sum()),
        "unique_gvkeys": int(link["gvkey"].dropna().nunique()),
        "by_match_type": {k: int(v) for k, v in link["match_type"].value_counts().items()},
        "by_borrower_source": {k: int(v) for k, v in link["borrower_source"].value_counts().items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Link DealScan borrower companies to Compustat gvkey.")
    parser.add_argument(
        "--fuzzy",
        action="store_true",
        help="Add a fuzzy name-matching pass for borrowers no deterministic layer matched (slower).",
    )
    args = parser.parse_args()

    borrowers = load_dealscan_borrowers()
    identity = load_compustat_identity()
    log(f"matching {len(borrowers):,} borrowers (fuzzy={args.fuzzy}) ...")
    link = build_link(borrowers, identity, use_fuzzy=args.fuzzy)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    link.to_parquet(OUTPUT_PATH, index=False)
    stats = summarize(link)
    write_json_atomic(
        MANIFEST_PATH,
        {"created_at_utc": utc_now(), "output_path": str(OUTPUT_PATH), "fuzzy_enabled": args.fuzzy, **stats},
    )
    log(f"saved {OUTPUT_PATH}")
    log(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
