import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RATINGS_PATH = PROJECT_ROOT / "data/raw/wrds/fisd_extra/fisd_ratings.parquet"
ISSUE_MASTER_PATH = PROJECT_ROOT / "data/raw/wrds/fisd_rated_issue_master.parquet"
ISSUE_DEFAULT_PATH = PROJECT_ROOT / "data/raw/wrds/fisd_extra/fisd_issue_default.parquet"
BONDCRSP_LINK_PATH = PROJECT_ROOT / "data/raw/wrds/bondcrsp_link.parquet"
CCM_LINKS_PATH = PROJECT_ROOT / "data/raw/merton/ccm_links.parquet"
COMPUSTAT_ANNUAL_DIR = PROJECT_ROOT / "data/raw/wrds/compustat_annual"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/downgrades"
EVENTS_PATH = OUTPUT_DIR / "downgrade_events.parquet"
AGENCY_PANEL_PATH = OUTPUT_DIR / "issuer_quarter_ratings_by_agency.parquet"
MANIFEST_PATH = OUTPUT_DIR / "downgrade_events_manifest.json"

START_QUARTER = "2000Q1"
END_QUARTER = "2024Q4"
AGENCIES = ("SPR", "MR", "FR")

# S&P and Fitch share notation; Moody's does not. Mapping per agency rather than through one
# merged dictionary removes any chance of a notation collision.
SP_FITCH_SCALE = {
    "AAA": 1, "AA+": 2, "AA": 3, "AA-": 4, "A+": 5, "A": 6, "A-": 7,
    "BBB+": 8, "BBB": 9, "BBB-": 10, "BB+": 11, "BB": 12, "BB-": 13,
    "B+": 14, "B": 15, "B-": 16, "CCC+": 17, "CCC": 18, "CCC-": 19,
    "CC": 20, "C": 21, "D": 22, "DD": 22, "DDD": 22, "RD": 22, "SD": 22,
}
MOODY_SCALE = {
    "Aaa": 1, "Aa1": 2, "Aa2": 3, "Aa3": 4, "A1": 5, "A2": 6, "A3": 7,
    "Baa1": 8, "Baa2": 9, "Baa3": 10, "Ba1": 11, "Ba2": 12, "Ba3": 13,
    "B1": 14, "B2": 15, "B3": 16, "Caa1": 17, "Caa2": 18, "Caa3": 19,
    "Ca": 20, "C": 21, "D": 22,
}
# A withdrawn or absent rating ends a spell. This is the practical substitute for the
# redemption data FISD does not carry (defeased_date and refunding_date are ~100% null).
TERMINATOR_CODES = {"NR", "NR/NR", "SUSP", "NAV", "PIF"}
# Short-term / commercial paper scale - not comparable to the long-term ordinal.
EXCLUDED_CODES = {"P-1", "P-2", "P-3"}
# Sovereign, preferred stock, trust-preferred and treasury/agency paper. The gvkey linkage drops
# most of this independently; excluding it explicitly makes the filter visible rather than implicit.
EXCLUDED_BOND_TYPES = {
    "FGOV",                      # sovereign
    "PSTK", "PS", "TPCS",        # preferred and trust-preferred securities
    "ADEB", "AMTN", "ARNT",      # agency paper: 87.4%, 91.8% and 98.0% AAA respectively
    "USNT",                      # treasury: 99.0% AAA
}
SENIOR_UNSECURED = "SEN"
CCM_COLUMNS = ("gvkey", "lpermno", "linkdt", "linkenddt", "linktype", "linkprim")
OPEN_END = pd.Timestamp("2099-12-31")


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


def quarter_ends() -> pd.DatetimeIndex:
    return pd.period_range(START_QUARTER, END_QUARTER, freq="Q").to_timestamp(how="end").normalize()


def to_quarter_end(values: pd.Series) -> pd.Series:
    return values.dt.to_period("Q").dt.to_timestamp(how="end").dt.normalize()


def load_issue_master() -> pd.DataFrame:
    cols = ["issue_id", "issuer_cusip", "issue_cusip", "offering_date", "maturity",
            "offering_amt", "security_level", "bond_type"]
    m = pd.read_parquet(ISSUE_MASTER_PATH, columns=cols)
    m["issue_id"] = pd.to_numeric(m["issue_id"], errors="coerce").astype("int64")
    m["offering_date"] = pd.to_datetime(m["offering_date"], errors="coerce")
    m["maturity"] = pd.to_datetime(m["maturity"], errors="coerce")
    m["offering_amt"] = pd.to_numeric(m["offering_amt"], errors="coerce")
    m["cusip8"] = (
        m["issuer_cusip"].astype("string").str.upper().str.strip()
        + m["issue_cusip"].astype("string").str.upper().str.strip()
    ).str[:8]
    m["cusip6"] = m["issuer_cusip"].astype("string").str.upper().str.strip().str[:6]
    before = len(m)
    m = m[~m["bond_type"].astype("string").isin(EXCLUDED_BOND_TYPES)]
    log(f"issue master: {before:,} issues, {len(m):,} after bond-type exclusion")
    return m.dropna(subset=["offering_date"])


def load_issue_defaults() -> pd.DataFrame:
    d = pd.read_parquet(ISSUE_DEFAULT_PATH, columns=["issue_id", "default_date"])
    d["issue_id"] = pd.to_numeric(d["issue_id"], errors="coerce").astype("int64")
    d["default_date"] = pd.to_datetime(d["default_date"], errors="coerce")
    return d.dropna(subset=["default_date"]).groupby("issue_id", as_index=False)["default_date"].min()


def load_ccm_links() -> pd.DataFrame:
    links = pd.read_parquet(CCM_LINKS_PATH, columns=list(CCM_COLUMNS))
    links["gvkey"] = links["gvkey"].astype("string")
    links["lpermno"] = pd.to_numeric(links["lpermno"], errors="coerce").astype("Int64")
    links["linkdt"] = pd.to_datetime(links["linkdt"], errors="coerce")
    links["linkenddt"] = pd.to_datetime(links["linkenddt"], errors="coerce").fillna(OPEN_END)
    links = links.dropna(subset=["gvkey", "lpermno", "linkdt"])
    links = links[links["linktype"].isin(["LC", "LU"]) & links["linkprim"].isin(["P", "C"])]
    return links.rename(columns={"lpermno": "permno"})[["permno", "gvkey", "linkdt", "linkenddt"]]


def link_issues_to_gvkey(master: pd.DataFrame) -> pd.DataFrame:
    """Two independent routes, unioned with provenance. Route A is the bond-level chain; route B
    is an issuer-CUSIP6 match against Compustat. Neither alone reaches the whole universe."""
    ccm = load_ccm_links()

    bcl = pd.read_parquet(BONDCRSP_LINK_PATH, columns=["cusip", "permno", "link_startdt", "link_enddt"])
    bcl["cusip8"] = bcl["cusip"].astype("string").str.upper().str.strip().str[:8]
    bcl["permno"] = pd.to_numeric(bcl["permno"], errors="coerce").astype("Int64")
    bcl = bcl.dropna(subset=["permno"])
    route_a = master[["issue_id", "cusip8", "offering_date"]].merge(bcl[["cusip8", "permno"]], on="cusip8", how="inner")
    route_a = route_a.merge(ccm, on="permno", how="inner")
    # CCM windows are respected against the issue's offering date rather than ignored.
    route_a = route_a[(route_a["offering_date"] >= route_a["linkdt"]) & (route_a["offering_date"] <= route_a["linkenddt"])]
    route_a = route_a[["issue_id", "gvkey"]].drop_duplicates("issue_id").assign(link_route="bond_cusip8")

    frames = []
    for path in sorted(COMPUSTAT_ANNUAL_DIR.glob("compustat_annual_*.parquet")):
        frames.append(pd.read_parquet(path, columns=["gvkey", "cusip"]))
    cs = pd.concat(frames, ignore_index=True).dropna(subset=["cusip"])
    cs["gvkey"] = cs["gvkey"].astype("string")
    cs["cusip6"] = cs["cusip"].astype("string").str.upper().str[:6]
    cs = cs.drop_duplicates(["cusip6", "gvkey"])[["cusip6", "gvkey"]]
    dup6 = cs["cusip6"].duplicated(keep=False)
    cs = cs[~dup6]  # a CUSIP6 mapping to several gvkeys is ambiguous; leave it unmatched
    route_b = master[["issue_id", "cusip6"]].merge(cs, on="cusip6", how="inner")
    route_b = route_b[["issue_id", "gvkey"]].drop_duplicates("issue_id").assign(link_route="issuer_cusip6")

    merged = route_a.merge(route_b, on="issue_id", how="outer", suffixes=("_a", "_b"))
    conflict = merged["gvkey_a"].notna() & merged["gvkey_b"].notna() & (merged["gvkey_a"] != merged["gvkey_b"])
    merged["gvkey"] = merged["gvkey_a"].fillna(merged["gvkey_b"])
    merged["link_route"] = np.where(
        merged["gvkey_a"].notna() & merged["gvkey_b"].notna(), "both",
        np.where(merged["gvkey_a"].notna(), "bond_cusip8", "issuer_cusip6"),
    )
    merged.loc[conflict, "link_route"] = "conflict"
    log(f"linkage: route A {route_a['issue_id'].nunique():,} issues, route B {route_b['issue_id'].nunique():,}, "
        f"union {merged['gvkey'].notna().sum():,}, conflicts {int(conflict.sum()):,}")
    out = merged.loc[~conflict, ["issue_id", "gvkey", "link_route"]].dropna(subset=["gvkey"])
    return out


def load_and_clean_ratings(issue_ids: set) -> pd.DataFrame:
    r = pd.read_parquet(RATINGS_PATH, columns=["issue_id", "rating_type", "rating_date", "rating"])
    r = r[r["rating_type"].isin(AGENCIES)]
    r["rating_date"] = pd.to_datetime(r["rating_date"], errors="coerce")
    r = r.dropna(subset=["rating_date"])
    r["issue_id"] = pd.to_numeric(r["issue_id"], errors="coerce").astype("int64")
    r = r[r["issue_id"].isin(issue_ids)]
    return clean_ratings(r)


def clean_ratings(r: pd.DataFrame) -> pd.DataFrame:
    """Normalise rating codes to a common ordinal and strip records that carry no event.
    Separated from the file read so the rules are directly testable."""
    r = r.copy()
    code = r["rating"].astype("string").str.strip()
    r = r[~code.isin(EXCLUDED_CODES)]
    code = code[~code.isin(EXCLUDED_CODES)]
    r["ord"] = np.where(r["rating_type"].eq("MR"), code.map(MOODY_SCALE), code.map(SP_FITCH_SCALE))
    r["ord"] = pd.to_numeric(r["ord"], errors="coerce")
    r["is_terminator"] = code.isin(TERMINATOR_CODES)
    r = r[r["ord"].notna() | r["is_terminator"]]

    # Several records can share a day for one issue and agency; take the most severe so the
    # construction never looks better than the worst thing the agency said that day.
    r = r.sort_values(["issue_id", "rating_type", "rating_date", "ord"])
    r = r.drop_duplicates(["issue_id", "rating_type", "rating_date"], keep="last")

    # Drop affirmations: consecutive identical grades carry no event. Terminators are kept, since
    # a withdrawal is a state change even when the preceding grade repeats.
    prev = r.groupby(["issue_id", "rating_type"])["ord"].shift()
    prev_term = r.groupby(["issue_id", "rating_type"])["is_terminator"].shift().astype("boolean").fillna(False).astype(bool)
    keep = r["is_terminator"] | prev.isna() | (prev != r["ord"]) | prev_term
    before = len(r)
    r = r[keep]
    log(f"ratings: {before:,} records -> {len(r):,} after same-day resolution and affirmation dedup")
    return r.sort_values(["issue_id", "rating_type", "rating_date"]).reset_index(drop=True)


def issue_quarter_ratings(ratings: pd.DataFrame, master: pd.DataFrame, defaults: pd.DataFrame,
                          agency: str) -> pd.DataFrame:
    """Rating in force for each issue at each quarter-end, within the issue's active window.
    Built one agency at a time to keep the intermediate frame small."""
    r = ratings[ratings["rating_type"] == agency]
    if r.empty:
        return pd.DataFrame()
    quarters = quarter_ends()
    qidx = pd.Series(range(len(quarters)), index=quarters)

    spans = master.merge(defaults, on="issue_id", how="left")
    spans = spans[spans["issue_id"].isin(r["issue_id"].unique())].copy()
    # Active window: offered, not yet matured, and not past a default. The downgrade *into*
    # default still lands, because the window closes only after the default quarter.
    spans["start_q"] = to_quarter_end(spans["offering_date"])
    end = pd.to_datetime(spans["maturity"]).fillna(OPEN_END)
    default_end = pd.to_datetime(spans["default_date"]).fillna(OPEN_END)
    end = np.minimum(end, default_end)
    spans["end_q"] = to_quarter_end(pd.Series(end, index=spans.index))
    spans = spans[spans["end_q"] >= quarters[0]]
    spans["start_q"] = spans["start_q"].clip(lower=quarters[0])
    spans["end_q"] = spans["end_q"].clip(upper=quarters[-1])
    spans = spans[spans["start_q"] <= spans["end_q"]]

    si = spans["start_q"].map(qidx).to_numpy()
    ei = spans["end_q"].map(qidx).to_numpy()
    counts = (ei - si + 1).astype(int)
    issue_ids = np.repeat(spans["issue_id"].to_numpy(), counts)
    offsets = np.concatenate([np.arange(c) for c in counts]) if len(counts) else np.array([], dtype=int)
    q_positions = np.repeat(si, counts) + offsets
    frame = pd.DataFrame({"issue_id": issue_ids.astype("int64"), "quarter_end": quarters[q_positions]})

    frame = frame.sort_values(["quarter_end", "issue_id"])
    r = r.sort_values(["rating_date", "issue_id"])
    frame = pd.merge_asof(frame, r[["issue_id", "rating_date", "ord", "is_terminator"]],
                          left_on="quarter_end", right_on="rating_date", by="issue_id", direction="backward")
    is_term = frame["is_terminator"].astype("boolean").fillna(True).astype(bool)
    frame = frame[frame["rating_date"].notna() & ~is_term]
    return frame[["issue_id", "quarter_end", "ord"]]


def issuer_quarter_ratings(iq: pd.DataFrame, master: pd.DataFrame, links: pd.DataFrame,
                           agency: str) -> pd.DataFrame:
    """Collapse issue-quarters to issuer-quarters under both the pre-specified primary hierarchy
    and the worst-active robustness rule."""
    d = iq.merge(master[["issue_id", "security_level", "offering_amt"]], on="issue_id", how="left")
    d = d.merge(links[["issue_id", "gvkey"]], on="issue_id", how="inner")
    d["is_senior_unsecured"] = d["security_level"].astype("string").eq(SENIOR_UNSECURED)

    # Primary: representative senior unsecured issue by largest original offering amount;
    # fall back to the largest active issue when the firm has no senior unsecured bond.
    d = d.sort_values(["gvkey", "quarter_end", "is_senior_unsecured", "offering_amt"],
                      ascending=[True, True, False, False])
    primary = d.drop_duplicates(["gvkey", "quarter_end"], keep="first")
    primary = primary.rename(columns={"ord": "rating_primary"})
    primary["used_senior_unsecured"] = primary["is_senior_unsecured"]

    worst = d.groupby(["gvkey", "quarter_end"])["ord"].max().rename("rating_worst").reset_index()
    n_issues = d.groupby(["gvkey", "quarter_end"])["issue_id"].size().rename("n_active_issues").reset_index()

    out = (primary[["gvkey", "quarter_end", "rating_primary", "used_senior_unsecured"]]
           .merge(worst, on=["gvkey", "quarter_end"], how="left")
           .merge(n_issues, on=["gvkey", "quarter_end"], how="left"))
    out["agency"] = agency
    return out


def add_downgrade_labels(panel: pd.DataFrame, rating_col: str, prefix: str) -> pd.DataFrame:
    """Downgrade when the issuer-quarter rating worsens against the previous quarter, plus the
    forward-looking horizons. Horizons that run past the sample end are NaN, never a silent zero."""
    panel = panel.sort_values(["gvkey", "quarter_end"]).copy()
    prev = panel.groupby("gvkey")[rating_col].shift()
    panel[f"{prefix}_downgrade"] = ((panel[rating_col] > prev) & prev.notna()).astype("float")
    panel.loc[prev.isna(), f"{prefix}_downgrade"] = np.nan

    last_q = panel["quarter_end"].max()
    g = panel.groupby("gvkey")[f"{prefix}_downgrade"]
    for h in (1, 4, 8):
        fwd = g.transform(lambda s: s[::-1].rolling(h, min_periods=1).max()[::-1].shift(-1))
        horizon_end = panel["quarter_end"] + pd.offsets.QuarterEnd(h)
        panel[f"{prefix}_downgrade_{h}q"] = np.where(horizon_end > last_q, np.nan, fwd)

    def quarters_to_next(s: pd.Series) -> pd.Series:
        arr = s.to_numpy()
        out = np.full(len(arr), np.nan)
        nxt = np.nan
        for i in range(len(arr) - 1, -1, -1):
            out[i] = nxt
            if arr[i] == 1:
                nxt = 1
            elif not np.isnan(nxt):
                nxt = nxt + 1
        return pd.Series(out, index=s.index)

    panel[f"{prefix}_quarters_to_next_downgrade"] = (
        panel.groupby("gvkey")[f"{prefix}_downgrade"].transform(quarters_to_next))
    return panel


def build() -> tuple:
    master = load_issue_master()
    defaults = load_issue_defaults()
    links = link_issues_to_gvkey(master)
    master = master[master["issue_id"].isin(set(links["issue_id"]))]
    log(f"issues carrying a gvkey: {len(master):,} across {links['gvkey'].nunique():,} firms")

    ratings = load_and_clean_ratings(set(master["issue_id"]))

    per_agency = []
    for agency in AGENCIES:
        iq = issue_quarter_ratings(ratings, master, defaults, agency)
        if iq.empty:
            continue
        ia = issuer_quarter_ratings(iq, master, links, agency)
        log(f"  {agency}: {len(iq):,} issue-quarters -> {len(ia):,} issuer-quarters, {ia['gvkey'].nunique():,} firms")
        per_agency.append(ia)
    agency_panel = pd.concat(per_agency, ignore_index=True)

    # Cross-agency: worst rating across agencies present, plus dispersion and coverage.
    combined = agency_panel.groupby(["gvkey", "quarter_end"]).agg(
        rating_primary=("rating_primary", "max"),
        rating_primary_best=("rating_primary", "min"),
        rating_worst=("rating_worst", "max"),
        n_agencies=("agency", "nunique"),
        n_active_issues=("n_active_issues", "max"),
        used_senior_unsecured=("used_senior_unsecured", "max"),
    ).reset_index()
    combined["rating_dispersion"] = combined["rating_primary"] - combined["rating_primary_best"]

    combined = add_downgrade_labels(combined, "rating_primary", "primary")
    combined = add_downgrade_labels(combined, "rating_worst", "worst")
    agency_panel = pd.concat(
        [add_downgrade_labels(d, "rating_primary", "primary") for _, d in agency_panel.groupby("agency")],
        ignore_index=True)
    return combined, agency_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FISD issuer-quarter ratings and downgrade labels.")
    parser.parse_args()

    combined, agency_panel = build()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(EVENTS_PATH, index=False)
    agency_panel.to_parquet(AGENCY_PANEL_PATH, index=False)

    stats = {
        "issuer_quarters": int(len(combined)),
        "firms": int(combined["gvkey"].nunique()),
        "quarter_range": [str(combined["quarter_end"].min().date()), str(combined["quarter_end"].max().date())],
        "senior_unsecured_share": float(combined["used_senior_unsecured"].mean()),
        "primary_downgrade_rate": float(combined["primary_downgrade"].mean()),
        "worst_downgrade_rate": float(combined["worst_downgrade"].mean()),
        "primary_vs_worst_disagreement": float((combined["rating_primary"] != combined["rating_worst"]).mean()),
        "mean_n_agencies": float(combined["n_agencies"].mean()),
    }
    for h in (1, 4, 8):
        stats[f"primary_downgrade_{h}q_rate"] = float(combined[f"primary_downgrade_{h}q"].mean())
    write_json_atomic(MANIFEST_PATH, {"created_at_utc": utc_now(), **stats})
    log(f"saved {EVENTS_PATH} and {AGENCY_PANEL_PATH}")
    log(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
