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
ENHANCEMENT_PATH = PROJECT_ROOT / "data/raw/wrds/fisd_extra/fisd_issue_enhancement.parquet"
BONDCRSP_LINK_PATH = PROJECT_ROOT / "data/raw/wrds/bondcrsp_link.parquet"
CCM_LINKS_PATH = PROJECT_ROOT / "data/raw/merton/ccm_links.parquet"
COMPUSTAT_ANNUAL_DIR = PROJECT_ROOT / "data/raw/wrds/compustat_annual"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/downgrades"
EVENTS_PATH = OUTPUT_DIR / "downgrade_events.parquet"
AGENCY_PANEL_PATH = OUTPUT_DIR / "issuer_quarter_ratings_by_agency.parquet"
EPISODES_PATH = OUTPUT_DIR / "default_episodes.parquet"
MANIFEST_PATH = OUTPUT_DIR / "downgrade_events_manifest.json"

START_QUARTER = "2000Q1"
END_QUARTER = "2024Q4"
AGENCIES = ("SPR", "MR", "FR")
AGENCY_NAMES = {"SPR": "sp", "MR": "moodys", "FR": "fitch"}
HORIZONS = (1, 4, 8)

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
# Third-party insurance and letters of credit: the rating reflects the backer, not the issuer.
EXTERNAL_BACKING_TYPES = {"INS", "LOC"}
# Ratings are used unconverted and labelled by the kind of bond they came from; the model learns what a
# subordinated or secured rating means. Unclassified issues (NON) are never used: against S&P company
# ratings they sit a median 3 notches better, consistent with insured or structured paper.
SOURCE_OF_LEVEL = {
    "SEN": "senior_unsecured",
    "SS": "secured",
    "SENS": "subordinated", "SUB": "subordinated", "JUNS": "subordinated", "JUN": "subordinated",
}
TIER_RANK = {"senior_unsecured": 0, "secured": 1, "subordinated": 2}
LEVEL_RANK = {"SEN": 0, "SS": 0, "SENS": 0, "SUB": 1, "JUNS": 2, "JUN": 3}
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
    enh = pd.read_parquet(ENHANCEMENT_PATH, columns=["issue_id", "enh_type"])
    backed = pd.to_numeric(enh.loc[enh["enh_type"].isin(EXTERNAL_BACKING_TYPES), "issue_id"], errors="coerce")
    m = m[~m["issue_id"].isin(set(backed.dropna().astype("int64")))]
    log(f"issue master: {before:,} issues, {len(m):,} after bond-type and third-party-backing exclusion")
    return m.dropna(subset=["offering_date"])


def load_default_records() -> pd.DataFrame:
    d = pd.read_parquet(ISSUE_DEFAULT_PATH, columns=["issue_id", "default_date", "reinstated_date"])
    d["issue_id"] = pd.to_numeric(d["issue_id"], errors="coerce").astype("int64")
    d["default_date"] = pd.to_datetime(d["default_date"], errors="coerce")
    d["reinstated_date"] = pd.to_datetime(d["reinstated_date"], errors="coerce")
    return d.dropna(subset=["default_date"])


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


def quarter_index(values: pd.Series) -> pd.Series:
    """Consecutive integer per calendar quarter, so quarter gaps and horizons are plain arithmetic."""
    return values.dt.year * 4 + (values.dt.month - 1) // 3


def enrich_issue_quarters(iq: pd.DataFrame, master: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Attach issuer, seniority and the issue's own rating in the previous quarter. prev_ord is set only
    when the same agency rated the same issue at t-1, so a new, maturing or re-rated issue can never
    look like a cut."""
    d = iq.merge(master[["issue_id", "security_level", "offering_amt"]], on="issue_id")
    d = d.merge(links[["issue_id", "gvkey"]], on="issue_id")
    d["security_level"] = d["security_level"].astype("string")
    d = d[d["security_level"].isin(SOURCE_OF_LEVEL)].copy()
    d["rating_source"] = d["security_level"].map(SOURCE_OF_LEVEL)
    d["q"] = quarter_index(d["quarter_end"])
    d = d.sort_values(["issue_id", "q"])
    g = d.groupby("issue_id")
    d["prev_ord"] = g["ord"].shift().where(g["q"].shift() == d["q"] - 1)
    return d.reset_index(drop=True)


def reference_ratings(d: pd.DataFrame) -> pd.DataFrame:
    """One rating per issuer-quarter for one agency: the representative senior unsecured issue, else the
    most senior other issue. The rating is used unconverted and labelled with its source."""
    d = d.assign(tier=d["rating_source"].map(TIER_RANK), level_rank=d["security_level"].map(LEVEL_RANK))
    ref = d.sort_values(["gvkey", "quarter_end", "tier", "level_rank", "offering_amt", "issue_id"],
                        ascending=[True, True, True, True, False, True])
    ref = ref.drop_duplicates(["gvkey", "quarter_end"])
    ref = ref.rename(columns={"ord": "rating_ord", "issue_id": "ref_issue_id"})
    g = d.groupby(["gvkey", "quarter_end"])
    extra = pd.DataFrame({"rating_worst": g["ord"].max(), "n_active_issues": g["issue_id"].size()}).reset_index()
    return ref[["gvkey", "quarter_end", "q", "rating_ord", "rating_source", "ref_issue_id"]].merge(
        extra, on=["gvkey", "quarter_end"], how="left")


def bond_cut_events(d: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Issuer-quarters in which the agency cut at least one issue it also rated in the previous quarter.
    Primary: a cut on an issue of the issuer's source tier at t-1 (senior unsecured when it had one).
    Strict: a cut on the t-1 reference issue, or on a majority of that tier's issues.
    Any bond: a cut on any issue of any seniority."""
    prev_ref = ref[["gvkey", "q", "rating_source", "ref_issue_id"]].rename(
        columns={"rating_source": "prev_source", "ref_issue_id": "prev_ref_issue"})
    prev_ref = prev_ref.assign(q=prev_ref["q"] + 1)
    b = d[d["prev_ord"].notna()].merge(prev_ref, on=["gvkey", "q"], how="inner")
    b["cut"] = b["ord"] > b["prev_ord"]
    b["eligible"] = b["rating_source"] == b["prev_source"]
    b["eligible_cut"] = b["cut"] & b["eligible"]
    b["ref_cut"] = b["cut"] & (b["issue_id"] == b["prev_ref_issue"])
    g = b.groupby(["gvkey", "quarter_end", "q"])
    out = g.agg(n_eligible=("eligible", "sum"), n_eligible_cut=("eligible_cut", "sum"),
                ref_cut=("ref_cut", "any"), any_cut=("cut", "any")).reset_index()
    out["cut_primary"] = out["n_eligible_cut"] > 0
    out["cut_strict"] = out["ref_cut"] | (out["n_eligible_cut"] * 2 > out["n_eligible"])
    out["cut_anybond"] = out["any_cut"]
    return out[["gvkey", "quarter_end", "q", "cut_primary", "cut_strict", "cut_anybond"]]


def default_episodes(records: pd.DataFrame, links: pd.DataFrame, master: pd.DataFrame,
                     ratings: pd.DataFrame) -> pd.DataFrame:
    """One row per issuer default episode: the quarter of the first default and the quarter the issuer
    re-emerged, taken as the earlier of a FISD reinstatement or the first rating on debt issued after the
    default. Defaults inside an episode belong to it; the next episode starts after re-emergence."""
    rec = records.merge(links[["issue_id", "gvkey"]], on="issue_id")
    first_rating = (ratings[ratings["ord"].notna()].groupby("issue_id")["rating_date"].min()
                    .rename("first_rating").reset_index())
    new_debt = (master[["issue_id", "offering_date"]].merge(first_rating, on="issue_id")
                .merge(links[["issue_id", "gvkey"]], on="issue_id"))
    new_debt = new_debt[new_debt["gvkey"].isin(set(rec["gvkey"]))]

    rows = []
    for gvkey, r in rec.groupby("gvkey"):
        dates = np.sort(r["default_date"].unique())
        debt = new_debt[new_debt["gvkey"] == gvkey]
        i = 0
        while i < len(dates):
            start = pd.Timestamp(dates[i])
            reinstated = r.loc[(r["default_date"] >= start) & (r["reinstated_date"] > start), "reinstated_date"]
            fresh = debt.loc[(debt["offering_date"] > start) & (debt["first_rating"] > start), "first_rating"]
            candidates = [x for x in (reinstated.min(), fresh.min()) if pd.notna(x)]
            emerge = min(candidates) if candidates else pd.NaT
            rows.append((gvkey, start, emerge))
            if pd.isna(emerge):
                break
            later = np.nonzero(dates > np.datetime64(emerge))[0]
            if len(later) == 0:
                break
            i = int(later[0])

    ep = pd.DataFrame(rows, columns=["gvkey", "default_date", "emerge_date"])
    ep["default_q"] = quarter_index(pd.to_datetime(ep["default_date"]))
    ep["emerge_q"] = quarter_index(pd.to_datetime(ep["emerge_date"]))
    return ep


def post_default_mask(frame: pd.DataFrame, episodes: pd.DataFrame) -> np.ndarray:
    """True for issuer-quarters strictly after a default quarter and before re-emergence. A defaulted
    issuer cannot be downgraded further, so these quarters carry no early-warning content."""
    if episodes.empty or frame.empty:
        return np.zeros(len(frame), dtype=bool)
    m = frame[["gvkey", "q"]].reset_index().merge(episodes[["gvkey", "default_q", "emerge_q"]], on="gvkey")
    end = m["emerge_q"].fillna(np.inf)
    hit = m.loc[(m["q"] > m["default_q"]) & (m["q"] < end), "index"]
    return frame.index.isin(hit)


def combine_agencies(refs: pd.DataFrame) -> pd.DataFrame:
    """Issuer-quarter rating across agencies: the most senior source tier any agency has, then the middle
    of three / lower of two / single rating among the agencies at that tier."""
    refs = refs.assign(tier=refs["rating_source"].map(TIER_RANK))
    best = refs.groupby(["gvkey", "quarter_end"])["tier"].transform("min")
    at_best = refs[refs["tier"] == best].sort_values(["gvkey", "quarter_end", "rating_ord"])
    g = at_best.groupby(["gvkey", "quarter_end"])
    pick = g.cumcount() == g["rating_ord"].transform("size") // 2
    out = at_best.loc[pick, ["gvkey", "quarter_end", "q", "rating_ord", "rating_source"]].copy()
    spread = g["rating_ord"].agg(lambda s: s.max() - s.min()).rename("rating_dispersion")

    rg = refs.groupby(["gvkey", "quarter_end"])
    extra = pd.DataFrame({
        "rating_worst": rg["rating_worst"].max(),
        "n_active_issues": rg["n_active_issues"].max(),
        "n_agencies": rg["agency"].nunique(),
    })
    presence = pd.crosstab([refs["gvkey"], refs["quarter_end"]], refs["agency"]).gt(0)
    for agency, name in AGENCY_NAMES.items():
        extra[f"has_{name}"] = presence[agency] if agency in presence.columns else False
    out = out.merge(spread.reset_index(), on=["gvkey", "quarter_end"]).merge(
        extra.reset_index(), on=["gvkey", "quarter_end"])
    out["senior_unsecured_source"] = out["rating_source"] == "senior_unsecured"
    return out.reset_index(drop=True)


def make_labels(rows: pd.DataFrame, events: pd.DataFrame, observable: pd.Series, name: str,
                with_countdown: bool = False) -> pd.DataFrame:
    """Contemporaneous flag and forward labels from a set of event quarters. The flag at t is NaN when the
    issuer was not observed at t-1 and no default occurred, since no cut could have been seen. Horizons that
    run past the sample end are NaN, never a silent zero. Labels are calendar-based, so a gap in an issuer's
    rows cannot stretch a 4-quarter window."""
    ev = events[["gvkey", "q"]].astype({"gvkey": "string", "q": "int64"}).drop_duplicates()
    key = pd.MultiIndex.from_frame(rows[["gvkey", "q"]].astype({"gvkey": "string", "q": "int64"}))
    is_event = key.isin(pd.MultiIndex.from_frame(ev))
    out = pd.DataFrame(index=rows.index)
    out[name] = np.where(is_event, 1.0, np.where(observable.to_numpy(), 0.0, np.nan))

    left = rows[["gvkey", "q"]].astype({"gvkey": "string", "q": "int64"}).reset_index().sort_values("q")
    right = ev.rename(columns={"q": "next_q"}).sort_values("next_q")
    nxt = pd.merge_asof(left, right, left_on="q", right_on="next_q", by="gvkey",
                        direction="forward", allow_exact_matches=False).set_index("index")
    gap = (nxt["next_q"] - nxt["q"]).reindex(rows.index)

    last_q = quarter_index(pd.Series([quarter_ends()[-1]])).iloc[0]
    for h in HORIZONS:
        val = (gap <= h).astype(float)
        val[rows["q"] + h > last_q] = np.nan
        out[f"{name}_{h}q"] = val
    if with_countdown:
        out[f"quarters_to_next_{name}"] = gap
    return out


def observed_last_quarter(rows: pd.DataFrame) -> pd.Series:
    key = pd.MultiIndex.from_frame(rows[["gvkey", "q"]])
    prev = pd.MultiIndex.from_arrays([rows["gvkey"], rows["q"] - 1])
    return pd.Series(prev.isin(key), index=rows.index)


def label_panel(rows: pd.DataFrame, event_sets: dict, episodes: pd.DataFrame) -> pd.DataFrame:
    """event_sets: {column name: events frame (gvkey, q)}. Events inside a post-default window are ignored,
    then post-default rows are dropped."""
    observable = observed_last_quarter(rows)
    parts = [rows]
    for name, ev in event_sets.items():
        ev = ev.reset_index(drop=True)
        ev = ev[~post_default_mask(ev, episodes)]
        parts.append(make_labels(rows, ev, observable, name, with_countdown=(name == "downgrade")))
    out = pd.concat(parts, axis=1)
    return out[~post_default_mask(out, episodes)].reset_index(drop=True)


def build() -> tuple:
    master = load_issue_master()
    records = load_default_records()
    links = link_issues_to_gvkey(master)
    master = master[master["issue_id"].isin(set(links["issue_id"]))]
    log(f"issues carrying a gvkey: {len(master):,} across {links['gvkey'].nunique():,} firms")

    ratings = load_and_clean_ratings(set(master["issue_id"]))
    records = records[records["issue_id"].isin(set(master["issue_id"]))]
    window_defaults = records.groupby("issue_id", as_index=False)["default_date"].min()
    episodes = default_episodes(records, links, master, ratings)
    default_events = episodes[["gvkey", "default_q"]].rename(columns={"default_q": "q"})
    log(f"default episodes: {len(episodes):,} across {episodes['gvkey'].nunique():,} issuers")

    refs, cuts = [], []
    for agency in AGENCIES:
        iq = issue_quarter_ratings(ratings, master, window_defaults, agency)
        if iq.empty:
            continue
        d = enrich_issue_quarters(iq, master, links)
        ref = reference_ratings(d)
        cut = bond_cut_events(d, ref)
        refs.append(ref.assign(agency=agency))
        cuts.append(cut.assign(agency=agency))
        log(f"  {agency}: {len(ref):,} issuer-quarters, {int(cut['cut_primary'].sum()):,} primary cuts")
    refs = pd.concat(refs, ignore_index=True)
    cuts = pd.concat(cuts, ignore_index=True)

    def events(flag: str, agency: str | None = None) -> pd.DataFrame:
        c = cuts if agency is None else cuts[cuts["agency"] == agency]
        return pd.concat([c.loc[c[flag], ["gvkey", "q"]], default_events], ignore_index=True)

    combined = combine_agencies(refs)
    event_sets = {"downgrade": events("cut_primary"),
                  "downgrade_strict": events("cut_strict"),
                  "downgrade_anybond": events("cut_anybond")}
    for agency, name in AGENCY_NAMES.items():
        event_sets[f"downgrade_{name}"] = events("cut_primary", agency)
    combined = label_panel(combined, event_sets, episodes)
    # Per-agency labels describe the rows that agency actually rates.
    for agency, name in AGENCY_NAMES.items():
        cols = [c for c in combined.columns if c.startswith(f"downgrade_{name}")]
        combined.loc[~combined[f"has_{name}"], cols] = np.nan
    combined["default_event"] = pd.MultiIndex.from_frame(combined[["gvkey", "q"]]).isin(
        pd.MultiIndex.from_frame(default_events)).astype(int)

    agency_panel = []
    for agency, name in AGENCY_NAMES.items():
        rows = refs[refs["agency"] == agency].reset_index(drop=True)
        agency_panel.append(label_panel(rows, {"downgrade": events("cut_primary", agency)}, episodes))
    agency_panel = pd.concat(agency_panel, ignore_index=True)
    return combined, agency_panel, episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FISD issuer-quarter ratings and downgrade labels.")
    parser.parse_args()

    combined, agency_panel, episodes = build()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(EVENTS_PATH, index=False)
    agency_panel.to_parquet(AGENCY_PANEL_PATH, index=False)
    episodes.to_parquet(EPISODES_PATH, index=False)

    stats = {
        "issuer_quarters": int(len(combined)),
        "firms": int(combined["gvkey"].nunique()),
        "quarter_range": [str(combined["quarter_end"].min().date()), str(combined["quarter_end"].max().date())],
        "rating_source_shares": combined["rating_source"].value_counts(normalize=True).round(4).to_dict(),
        "default_episodes": int(len(episodes)),
        "mean_n_agencies": float(combined["n_agencies"].mean()),
    }
    for name in ["downgrade", "downgrade_strict", "downgrade_anybond",
                 *[f"downgrade_{n}" for n in AGENCY_NAMES.values()]]:
        stats[f"{name}_rate"] = float(combined[name].mean())
        for h in HORIZONS:
            stats[f"{name}_{h}q_rate"] = float(combined[f"{name}_{h}q"].mean())
    write_json_atomic(MANIFEST_PATH, {"created_at_utc": utc_now(), **stats})
    log(f"saved {EVENTS_PATH}, {AGENCY_PANEL_PATH} and {EPISODES_PATH}")
    log(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
