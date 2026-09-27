import numpy as np
import pandas as pd
import pytest

from src.preprocessing.downgrades import (
    MOODY_SCALE,
    SP_FITCH_SCALE,
    add_downgrade_labels,
    clean_ratings,
    issue_quarter_ratings,
    issuer_quarter_ratings,
    quarter_ends,
)

Q = pd.Timestamp


def ratings_frame(rows):
    """rows: (issue_id, agency, date, rating)"""
    df = pd.DataFrame(rows, columns=["issue_id", "rating_type", "rating_date", "rating"])
    df["issue_id"] = df["issue_id"].astype("int64")
    df["rating_date"] = pd.to_datetime(df["rating_date"])
    return df


def master_frame(rows):
    """rows: (issue_id, offering_date, maturity, offering_amt, security_level)"""
    df = pd.DataFrame(rows, columns=["issue_id", "offering_date", "maturity", "offering_amt", "security_level"])
    df["issue_id"] = df["issue_id"].astype("int64")
    df["offering_date"] = pd.to_datetime(df["offering_date"])
    df["maturity"] = pd.to_datetime(df["maturity"])
    return df


class TestOrdinalMapping:
    def test_scales_are_monotone_and_share_anchors(self):
        assert SP_FITCH_SCALE["AAA"] < SP_FITCH_SCALE["BBB"] < SP_FITCH_SCALE["CCC"]
        assert MOODY_SCALE["Aaa"] < MOODY_SCALE["Baa2"] < MOODY_SCALE["Caa2"]
        # equivalent notches must land on the same ordinal across agencies
        assert SP_FITCH_SCALE["BBB"] == MOODY_SCALE["Baa2"]
        assert SP_FITCH_SCALE["BB+"] == MOODY_SCALE["Ba1"]

    def test_mapping_is_agency_specific(self):
        r = clean_ratings(ratings_frame([
            (1, "SPR", "2005-01-01", "A"),     # S&P single A -> 6
            (2, "MR", "2005-01-01", "A2"),     # Moody's A2 -> 6
        ]))
        assert set(r["ord"]) == {6.0}

    def test_default_grades_map_to_bottom_not_dropped(self):
        r = clean_ratings(ratings_frame([
            (1, "SPR", "2005-01-01", "BBB"),
            (1, "SPR", "2006-01-01", "D"),
            (2, "FR", "2005-01-01", "DDD"),
            (3, "MR", "2005-01-01", "Ca"),
        ]))
        assert (r.loc[r["rating"].isin(["D", "DDD"]), "ord"] == 22).all()
        assert (r.loc[r["rating"] == "Ca", "ord"] == 20).all()

    def test_short_term_scale_excluded(self):
        r = clean_ratings(ratings_frame([
            (1, "MR", "2005-01-01", "P-1"),
            (1, "MR", "2006-01-01", "Baa2"),
        ]))
        assert len(r) == 1 and r.iloc[0]["rating"] == "Baa2"

    def test_withdrawal_codes_flagged_as_terminators(self):
        r = clean_ratings(ratings_frame([
            (1, "SPR", "2005-01-01", "BBB"),
            (1, "SPR", "2006-01-01", "NR"),
        ]))
        assert r["is_terminator"].tolist() == [False, True]
        assert pd.isna(r.loc[r["is_terminator"], "ord"]).all()


class TestCleaning:
    def test_affirmations_are_dropped(self):
        r = clean_ratings(ratings_frame([
            (1, "SPR", "2005-01-01", "BBB"),
            (1, "SPR", "2005-06-01", "BBB"),   # affirmation, no event
            (1, "SPR", "2006-01-01", "BBB-"),
        ]))
        assert r["rating"].tolist() == ["BBB", "BBB-"]

    def test_withdrawal_kept_even_when_grade_repeats(self):
        # grade -> NR -> same grade: the withdrawal is a state change and must survive
        r = clean_ratings(ratings_frame([
            (1, "SPR", "2005-01-01", "BBB"),
            (1, "SPR", "2006-01-01", "NR"),
            (1, "SPR", "2007-01-01", "BBB"),
        ]))
        assert len(r) == 3

    def test_same_day_records_resolve_to_most_severe(self):
        r = clean_ratings(ratings_frame([
            (1, "SPR", "2005-01-01", "BBB"),
            (1, "SPR", "2005-01-01", "BB"),    # worse, same day
        ]))
        assert len(r) == 1 and r.iloc[0]["ord"] == SP_FITCH_SCALE["BB"]

    def test_agencies_are_deduped_independently(self):
        r = clean_ratings(ratings_frame([
            (1, "SPR", "2005-01-01", "BBB"),
            (1, "MR", "2005-01-01", "Baa2"),   # same notch, different agency: both kept
        ]))
        assert len(r) == 2


class TestActiveWindow:
    def _run(self, ratings, master, defaults=None):
        defaults = defaults if defaults is not None else pd.DataFrame(columns=["issue_id", "default_date"])
        return issue_quarter_ratings(clean_ratings(ratings), master, defaults, "SPR")

    def test_rating_is_carried_forward_until_maturity(self):
        iq = self._run(
            ratings_frame([(1, "SPR", "2005-02-15", "BBB")]),
            master_frame([(1, "2005-01-01", "2006-12-31", 100.0, "SEN")]),
        )
        # Q1 2005 predates the rating; Q1 2005 excluded, Q1 2006..Q4 2006 carry it
        assert iq["quarter_end"].min() == Q("2005-03-31")
        assert iq["quarter_end"].max() == Q("2006-12-31")
        assert (iq["ord"] == SP_FITCH_SCALE["BBB"]).all()

    def test_no_rows_after_maturity(self):
        iq = self._run(
            ratings_frame([(1, "SPR", "2005-02-15", "BBB")]),
            master_frame([(1, "2005-01-01", "2006-06-30", 100.0, "SEN")]),
        )
        assert iq["quarter_end"].max() <= Q("2006-06-30")

    def test_no_rows_before_first_rating(self):
        iq = self._run(
            ratings_frame([(1, "SPR", "2006-02-15", "BBB")]),
            master_frame([(1, "2005-01-01", "2008-12-31", 100.0, "SEN")]),
        )
        assert iq["quarter_end"].min() == Q("2006-03-31")

    def test_terminator_ends_the_spell(self):
        iq = self._run(
            ratings_frame([(1, "SPR", "2005-02-15", "BBB"), (1, "SPR", "2006-02-15", "NR")]),
            master_frame([(1, "2005-01-01", "2010-12-31", 100.0, "SEN")]),
        )
        assert iq["quarter_end"].max() == Q("2005-12-31")

    def test_grade_after_terminator_restarts_the_spell(self):
        iq = self._run(
            ratings_frame([
                (1, "SPR", "2005-02-15", "BBB"),
                (1, "SPR", "2006-02-15", "NR"),
                (1, "SPR", "2008-02-15", "BB"),
            ]),
            master_frame([(1, "2005-01-01", "2009-12-31", 100.0, "SEN")]),
        )
        quarters = set(iq["quarter_end"])
        assert Q("2006-06-30") not in quarters      # inside the withdrawn gap
        assert Q("2008-06-30") in quarters          # after the new grade
        assert iq.loc[iq["quarter_end"] == Q("2008-06-30"), "ord"].iloc[0] == SP_FITCH_SCALE["BB"]

    def test_downgrade_into_default_still_lands(self):
        # the window closes only after the default quarter, so the move to D is observable
        iq = self._run(
            ratings_frame([(1, "SPR", "2005-02-15", "BBB"), (1, "SPR", "2006-02-15", "D")]),
            master_frame([(1, "2005-01-01", "2010-12-31", 100.0, "SEN")]),
            defaults=pd.DataFrame({"issue_id": [1], "default_date": [pd.Timestamp("2006-02-15")]}),
        )
        assert iq["ord"].max() == 22
        assert iq["quarter_end"].max() == Q("2006-03-31")


class TestIssuerAggregation:
    def _issuer(self, master, iq_rows):
        iq = pd.DataFrame(iq_rows, columns=["issue_id", "quarter_end", "ord"])
        iq["issue_id"] = iq["issue_id"].astype("int64")
        links = pd.DataFrame({"issue_id": master["issue_id"], "gvkey": "001234"})
        return issuer_quarter_ratings(iq, master, links, "SPR")

    def test_prefers_largest_senior_unsecured(self):
        master = master_frame([
            (1, "2005-01-01", "2010-12-31", 500.0, "SEN"),
            (2, "2005-01-01", "2010-12-31", 900.0, "SUB"),   # bigger but subordinated
        ])
        out = self._issuer(master, [(1, Q("2006-03-31"), 9.0), (2, Q("2006-03-31"), 14.0)])
        assert out["rating_primary"].iloc[0] == 9.0
        assert bool(out["used_senior_unsecured"].iloc[0]) is True
        assert out["rating_worst"].iloc[0] == 14.0          # robustness rule still sees the worst

    def test_falls_back_to_largest_when_no_senior_unsecured(self):
        master = master_frame([
            (1, "2005-01-01", "2010-12-31", 500.0, "SUB"),
            (2, "2005-01-01", "2010-12-31", 900.0, "SUB"),
        ])
        out = self._issuer(master, [(1, Q("2006-03-31"), 9.0), (2, Q("2006-03-31"), 14.0)])
        assert out["rating_primary"].iloc[0] == 14.0        # the larger issue
        assert bool(out["used_senior_unsecured"].iloc[0]) is False


class TestLabels:
    def _panel(self, ords, start="2005-03-31"):
        quarters = pd.date_range(start, periods=len(ords), freq="QE")
        p = pd.DataFrame({"gvkey": "001234", "quarter_end": quarters, "rating_primary": ords})
        return add_downgrade_labels(p, "rating_primary", "primary")

    def test_downgrade_fires_on_worsening_only(self):
        out = self._panel([9.0, 9.0, 10.0, 9.0])
        assert pd.isna(out["primary_downgrade"].iloc[0])     # no prior quarter
        assert out["primary_downgrade"].tolist()[1:] == [0.0, 1.0, 0.0]  # upgrade is not a downgrade

    def test_forward_label_is_the_next_quarter(self):
        out = self._panel([9.0, 9.0, 10.0, 10.0])
        # dg_1q[t] must equal dg[t+1]
        assert out["primary_downgrade_1q"].iloc[1] == out["primary_downgrade"].iloc[2]

    def test_no_look_ahead_in_contemporaneous_flag(self):
        out = self._panel([9.0, 10.0, 10.0, 10.0])
        # the flag at t depends only on t-1 -> t, so a later downgrade cannot change it
        assert out["primary_downgrade"].iloc[2] == 0.0

    def test_horizon_beyond_sample_end_is_nan_not_zero(self):
        out = self._panel([9.0] * 6)
        assert pd.isna(out["primary_downgrade_4q"].iloc[-1])
        assert pd.isna(out["primary_downgrade_8q"].iloc[-1])
        # a silent zero here would understate late-sample rates
        assert out["primary_downgrade_4q"].isna().sum() >= 4

    def test_quarters_to_next_downgrade_counts_down(self):
        out = self._panel([9.0, 9.0, 9.0, 10.0])
        assert out["primary_quarters_to_next_downgrade"].tolist()[:3] == [3.0, 2.0, 1.0]

    def test_quarters_to_next_is_nan_when_none_remain(self):
        out = self._panel([9.0, 10.0, 10.0, 10.0])
        assert pd.isna(out["primary_quarters_to_next_downgrade"].iloc[-1])

    def test_firms_are_labelled_independently(self):
        quarters = pd.date_range("2005-03-31", periods=3, freq="QE")
        p = pd.DataFrame({
            "gvkey": ["A"] * 3 + ["B"] * 3,
            "quarter_end": list(quarters) * 2,
            "rating_primary": [9.0, 10.0, 10.0, 12.0, 12.0, 12.0],
        })
        out = add_downgrade_labels(p, "rating_primary", "primary").sort_values(["gvkey", "quarter_end"])
        b = out[out["gvkey"] == "B"]
        assert pd.isna(b["primary_downgrade"].iloc[0])       # B's first quarter, not carried from A
        assert b["primary_downgrade"].tolist()[1:] == [0.0, 0.0]


def test_quarter_grid_spans_the_sample():
    q = quarter_ends()
    assert q[0] == Q("2000-03-31") and q[-1] == Q("2024-12-31")
    assert len(q) == 100
