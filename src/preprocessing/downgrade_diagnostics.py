import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.downgrades import (
    AGENCY_NAMES,
    AGENCY_PANEL_PATH,
    CCM_LINKS_PATH,
    EPISODES_PATH,
    EVENTS_PATH,
    SP_FITCH_SCALE,
    log,
    quarter_index,
    to_quarter_end,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SP_ISSUER_PATH = PROJECT_ROOT / "data/raw/wrds/sp_issuer_ratings.parquet"
DELISTINGS_PATH = PROJECT_ROOT / "data/raw/wrds/crsp/crsp_delistings.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data/output/tables"
REPORT_PATH = OUTPUT_DIR / "downgrade_diagnostics.md"
UNMATCHED_PATH = OUTPUT_DIR / "downgrade_unmatched_sp_sample.csv"
# CRSP delisting code for bankruptcy / insolvency, used to check the FISD default table for gaps.
BANKRUPTCY_DLSTCD = 574
SP_VALIDATION_END = pd.Timestamp("2016-12-31")


def pct(x: float) -> str:
    return f"{x:.1%}" if pd.notna(x) else "n/a"


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def rates_by_year(panel: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    cols = ["downgrade", "downgrade_strict", "downgrade_anybond", *[f"downgrade_{n}" for n in AGENCY_NAMES.values()]]
    g = panel.groupby(panel["quarter_end"].dt.year)
    out = pd.DataFrame({"firms": g["gvkey"].nunique(),
                        "senior unsecured source": g["senior_unsecured_source"].mean().map(pct)})
    for c in cols:
        out[c] = g[c].mean().map(pct)
    # Counted from episodes, not rows: a default where every rating was withdrawn has no issuer-quarter row.
    out["default episodes"] = episodes.groupby(episodes["default_date"].dt.year).size().reindex(out.index).fillna(0).astype(int)
    return out.reset_index().rename(columns={"quarter_end": "year"})


def default_checks(panel: pd.DataFrame, episodes: pd.DataFrame) -> dict:
    ep = episodes[episodes["default_date"].between("2000-01-01", "2024-12-31")].copy()
    rows = panel.set_index(["gvkey", "q"])
    rated_at_default = ep.apply(lambda e: (e["gvkey"], e["default_q"]) in rows.index, axis=1)
    rated_rows = rows.reindex(list(zip(ep["gvkey"], ep["default_q"])))
    at_bottom = rated_rows["rating_ord"].to_numpy() >= 20

    events = panel.loc[panel["downgrade"] == 1, ["gvkey", "q"]]
    prior = ep.merge(events, on="gvkey")
    prior = prior[(prior["q"] < prior["default_q"]) & (prior["q"] >= prior["default_q"] - 4)]
    with_prior = ep.set_index(["gvkey", "default_q"]).index.isin(prior.set_index(["gvkey", "default_q"]).index)

    # CRSP bankruptcy delistings of panel firms, matched to gvkey through date-valid CCM links
    dl = pd.read_parquet(DELISTINGS_PATH, columns=["permno", "dlstdt", "dlstcd"])
    dl = dl[pd.to_numeric(dl["dlstcd"], errors="coerce") == BANKRUPTCY_DLSTCD].copy()
    dl["dlstdt"] = pd.to_datetime(dl["dlstdt"])
    dl["permno"] = pd.to_numeric(dl["permno"], errors="coerce").astype("Int64")
    ccm = pd.read_parquet(CCM_LINKS_PATH, columns=["gvkey", "lpermno", "linkdt", "linkenddt"])
    ccm["permno"] = pd.to_numeric(ccm["lpermno"], errors="coerce").astype("Int64")
    ccm["linkdt"] = pd.to_datetime(ccm["linkdt"])
    ccm["linkenddt"] = pd.to_datetime(ccm["linkenddt"]).fillna(pd.Timestamp("2099-12-31"))
    dl = dl.merge(ccm[["permno", "gvkey", "linkdt", "linkenddt"]], on="permno")
    dl = dl[(dl["dlstdt"] >= dl["linkdt"]) & (dl["dlstdt"] <= dl["linkenddt"] + pd.Timedelta(days=365))]
    dl["gvkey"] = dl["gvkey"].astype("string")
    dl = dl[dl["gvkey"].isin(set(panel["gvkey"])) & dl["dlstdt"].between("2000-01-01", "2024-12-31")]
    dl = dl.drop_duplicates(["gvkey", "dlstdt"])
    dl["dq"] = quarter_index(dl["dlstdt"])
    m = dl.merge(episodes[["gvkey", "default_q"]], on="gvkey", how="left")
    m["near"] = (m["default_q"] - m["dq"]).abs() <= 4
    matched = m.groupby(["gvkey", "dlstdt"])["near"].any()

    return {
        "episodes": len(ep),
        "issuers": ep["gvkey"].nunique(),
        "no rated row at default quarter (all ratings withdrawn)": pct(1 - rated_at_default.mean()),
        "rated row at default quarter, rating Ca/C/D or worse": pct(np.nanmean(at_bottom[rated_at_default.to_numpy()])),
        "downgrade event in the 4 quarters before default": pct(with_prior.mean()),
        "CRSP bankruptcy delistings of panel firms": len(matched),
        "... with a FISD default within 4 quarters": pct(matched.mean()),
    }


def sp_validation(agency_panel: pd.DataFrame) -> tuple:
    sp = pd.read_parquet(SP_ISSUER_PATH)
    sp = sp[sp["datadate"].dt.month.isin([3, 6, 9, 12])]
    sp["quarter_end"] = to_quarter_end(sp["datadate"])
    sp["sp_ord"] = sp["splticrm"].map(SP_FITCH_SCALE)
    sp = sp.dropna(subset=["sp_ord"]).drop_duplicates(["gvkey", "quarter_end"])[["gvkey", "quarter_end", "sp_ord"]]
    sp["gvkey"] = sp["gvkey"].astype("string")

    ours = agency_panel[agency_panel["agency"] == "SPR"].copy()
    ours["gvkey"] = ours["gvkey"].astype("string")
    m = ours.merge(sp, on=["gvkey", "quarter_end"])
    m = m[m["quarter_end"] <= SP_VALIDATION_END].sort_values(["gvkey", "quarter_end"])

    diff = m["rating_ord"] - m["sp_ord"]
    level = []
    for source, sel in [("all", m.index == m.index), *[(s, m["rating_source"] == s) for s in
                                                     ("senior_unsecured", "subordinated", "secured")]]:
        dd = diff[sel]
        level.append({"rating source": source, "issuer-quarters": f"{len(dd):,}",
                      "exact": pct((dd == 0).mean()), "off by 1": pct((dd.abs() == 1).mean()),
                      "off by 2+": pct((dd.abs() >= 2).mean()), "mean (ours - S&P)": f"{dd.mean():+.2f}"})

    g = m.groupby("gvkey")
    consecutive = g["q"].shift() == m["q"] - 1
    m["sp_dg"] = (m["sp_ord"] > g["sp_ord"].shift()) & consecutive
    m["ours"] = m["downgrade"].eq(1) & consecutive
    m = m[consecutive]
    gg = m.groupby("gvkey")

    def near(col):
        return m[col] | gg[col].shift(1, fill_value=False) | gg[col].shift(-1, fill_value=False)

    n_sp, n_ours = int(m["sp_dg"].sum()), int(m["ours"].sum())
    events = {
        "consecutive S&P issuer-quarter pairs": f"{len(m):,}",
        "S&P company downgrades": f"{n_sp:,}",
        "our S&P events": f"{n_ours:,}",
        "S&P downgrades we catch, same quarter": pct((m["ours"] & m["sp_dg"]).sum() / n_sp),
        "S&P downgrades we catch, within 1 quarter": pct((near("ours") & m["sp_dg"]).sum() / n_sp),
        "our events matching an S&P downgrade, same quarter": pct((m["ours"] & m["sp_dg"]).sum() / n_ours),
        "our events matching an S&P downgrade, within 1 quarter": pct((m["ours"] & near("sp_dg")).sum() / n_ours),
    }
    unmatched = m[m["ours"] & ~near("sp_dg")]
    sample = unmatched.sample(min(40, len(unmatched)), random_state=0)[
        ["gvkey", "quarter_end", "rating_ord", "rating_source", "n_active_issues", "sp_ord"]]
    return pd.DataFrame(level), events, sample, len(unmatched)


def run() -> None:
    panel = pd.read_parquet(EVENTS_PATH)
    agency_panel = pd.read_parquet(AGENCY_PANEL_PATH)
    episodes = pd.read_parquet(EPISODES_PATH)
    for f in (panel, agency_panel, episodes):
        f["gvkey"] = f["gvkey"].astype("string")

    sources = panel["rating_source"].value_counts(normalize=True).map(pct).rename("share").reset_index()
    labels = pd.DataFrame([{"label": c, **{f"{h}q": pct(panel[f"{c}_{h}q"].mean()) for h in (1, 4, 8)},
                            "events": int(panel[c].eq(1).sum())}
                           for c in ["downgrade", "downgrade_strict", "downgrade_anybond",
                                     *[f"downgrade_{n}" for n in AGENCY_NAMES.values()]]])
    defaults = default_checks(panel, episodes)
    level, events, sample, n_unmatched = sp_validation(agency_panel)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_csv(UNMATCHED_PATH, index=False)
    report = [
        "# Chunk 5 downgrade label diagnostics", "",
        f"Issuer-quarters: {len(panel):,}; firms: {panel['gvkey'].nunique():,}; "
        f"{panel['quarter_end'].min().date()} to {panel['quarter_end'].max().date()}.", "",
        "## Rating source", "", md_table(sources), "",
        "## Label rates (share of issuer-quarters with a known label)", "", md_table(labels), "",
        "## Contemporaneous event rate by year", "", md_table(rates_by_year(panel, episodes)), "",
        "## Defaults (episodes starting 2000-2024)", "",
        md_table(pd.DataFrame(list(defaults.items()), columns=["check", "value"])), "",
        "## Agreement with S&P company ratings (comp.adsprate), 2000-2016", "",
        "Rating level, our S&P reference rating vs S&P's company rating:", "", md_table(level), "",
        "Events:", "", md_table(pd.DataFrame(list(events.items()), columns=["check", "value"])), "",
        f"{n_unmatched:,} of our S&P events have no S&P company downgrade within one quarter; "
        f"a random sample of 40 is in `{UNMATCHED_PATH.relative_to(PROJECT_ROOT)}`.", "",
    ]
    REPORT_PATH.write_text("\n".join(report))
    log(f"saved {REPORT_PATH}")


def main() -> None:
    argparse.ArgumentParser(description="Diagnostics for the Chunk 5 downgrade labels.").parse_args()
    run()


if __name__ == "__main__":
    main()
