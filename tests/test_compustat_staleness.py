import numpy as np
import pandas as pd

from src.preprocessing.compustat import (
    DEFAULT_STALENESS_CAP_DAYS,
    FEATURE_COLUMNS,
    STALE_NULLED_COLUMNS,
    aligned_quarter,
)

QE = pd.Timestamp("2010-12-31")


def observations(rows):
    """rows: (gvkey, datadate, rdq). Accounting values are dummies; only timing matters here."""
    df = pd.DataFrame(rows, columns=["gvkey", "datadate", "rdq"])
    df["datadate"] = pd.to_datetime(df["datadate"])
    df["rdq"] = pd.to_datetime(df["rdq"])
    for col in (*STALE_NULLED_COLUMNS, *FEATURE_COLUMNS):
        df[col] = 1.0
    for col in ("fyearq", "fqtr", "datafqtr", "datacqtr"):
        df[col] = 0
    for col in ("rdq_missing", "debt_component_missing", "coverage_missing"):
        df[col] = False
    return df.sort_values(["gvkey", "rdq", "datadate"])


def test_fresh_filing_is_untouched():
    out = aligned_quarter(observations([("A", "2010-09-30", "2010-11-05")]), QE)
    assert not out["accounting_stale"].iloc[0]
    assert (out[list(FEATURE_COLUMNS)].iloc[0] == 1.0).all()


def test_stale_row_is_kept_not_dropped():
    out = aligned_quarter(observations([("A", "2005-09-30", "2005-11-05")]), QE)
    assert len(out) == 1


def test_stale_row_has_accounting_nulled_and_flag_set():
    out = aligned_quarter(observations([("A", "2005-09-30", "2005-11-05")]), QE)
    row = out.iloc[0]
    assert row["accounting_stale"]
    assert row[list(STALE_NULLED_COLUMNS) + list(FEATURE_COLUMNS)].isna().all()


def test_stale_row_keeps_filing_identifiers_so_age_is_traceable():
    out = aligned_quarter(observations([("A", "2005-09-30", "2005-11-05")]), QE)
    row = out.iloc[0]
    assert row["rdq"] == pd.Timestamp("2005-11-05")
    assert row["accounting_age_days"] == (QE - pd.Timestamp("2005-11-05")).days


def test_cap_boundary_is_inclusive_of_the_cap():
    rdq = QE - pd.Timedelta(days=DEFAULT_STALENESS_CAP_DAYS)
    out = aligned_quarter(observations([("A", "2010-01-01", rdq)]), QE)
    assert not out["accounting_stale"].iloc[0]          # exactly at the cap: still usable
    out = aligned_quarter(observations([("A", "2010-01-01", rdq - pd.Timedelta(days=1))]), QE)
    assert out["accounting_stale"].iloc[0]              # one day past: nulled


def test_cap_is_configurable():
    obs = observations([("A", "2010-01-31", "2010-03-15")])  # ~291 days old at QE
    assert aligned_quarter(obs, QE, staleness_cap_days=270)["accounting_stale"].iloc[0]
    assert not aligned_quarter(obs, QE, staleness_cap_days=450)["accounting_stale"].iloc[0]


def test_firms_are_judged_independently():
    out = aligned_quarter(observations([
        ("A", "2010-09-30", "2010-11-05"),   # fresh
        ("B", "2005-09-30", "2005-11-05"),   # stale
    ]), QE).set_index("gvkey")
    assert not out.loc["A", "accounting_stale"] and out.loc["B", "accounting_stale"]
    assert out.loc["A", "leverage"] == 1.0 and np.isnan(out.loc["B", "leverage"])


def test_filings_after_quarter_end_are_never_used():
    # the point-in-time rule still holds: a later filing cannot rescue a stale quarter
    out = aligned_quarter(observations([
        ("A", "2005-09-30", "2005-11-05"),
        ("A", "2010-12-31", "2011-02-10"),   # reported after QE
    ]), QE)
    assert out["rdq"].iloc[0] == pd.Timestamp("2005-11-05")
    assert out["accounting_stale"].iloc[0]
