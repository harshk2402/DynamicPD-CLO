import numpy as np
import pandas as pd
import pytest

from src.preprocessing.downgrades import (
    MOODY_SCALE,
    SP_FITCH_SCALE,
    bond_cut_events,
    clean_ratings,
    combine_agencies,
    default_episodes,
    enrich_issue_quarters,
    issue_quarter_ratings,
    label_panel,
    make_labels,
    observed_last_quarter,
    post_default_mask,
    quarter_ends,
    quarter_index,
    reference_ratings,
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


def qe(s):
    return Q(s)


def enriched(issues, iq_rows, gvkey="A"):
    """issues: (issue_id, security_level, offering_amt[, gvkey]); iq_rows: (issue_id, quarter_end, ord)"""
    master = pd.DataFrame([i[:3] for i in issues], columns=["issue_id", "security_level", "offering_amt"])
    master["issue_id"] = master["issue_id"].astype("int64")
    links = pd.DataFrame({"issue_id": master["issue_id"],
                          "gvkey": [i[3] if len(i) > 3 else gvkey for i in issues]})
    iq = pd.DataFrame(iq_rows, columns=["issue_id", "quarter_end", "ord"])
    iq["issue_id"] = iq["issue_id"].astype("int64")
    iq["quarter_end"] = pd.to_datetime(iq["quarter_end"])
    return enrich_issue_quarters(iq, master, links)


def cuts_for(issues, iq_rows):
    d = enriched(issues, iq_rows)
    return bond_cut_events(d, reference_ratings(d))


class TestReferenceRating:
    def test_prefers_senior_unsecured_over_larger_subordinated(self):
        d = enriched([(1, "SEN", 500.0), (2, "SUB", 900.0)],
                     [(1, "2006-03-31", 9.0), (2, "2006-03-31", 14.0)])
        ref = reference_ratings(d).iloc[0]
        assert ref["rating_ord"] == 9.0 and ref["rating_source"] == "senior_unsecured"
        assert ref["rating_worst"] == 14.0             # robustness level still sees the worst issue

    def test_secured_before_subordinated_and_unconverted(self):
        d = enriched([(1, "SUB", 900.0), (2, "SS", 100.0)],
                     [(1, "2006-03-31", 15.0), (2, "2006-03-31", 11.0)])
        ref = reference_ratings(d).iloc[0]
        assert ref["rating_source"] == "secured" and ref["rating_ord"] == 11.0   # no notching offset

    def test_senior_subordinated_before_subordinated(self):
        d = enriched([(1, "SUB", 900.0), (2, "SENS", 100.0)],
                     [(1, "2006-03-31", 16.0), (2, "2006-03-31", 15.0)])
        ref = reference_ratings(d).iloc[0]
        assert ref["rating_source"] == "subordinated" and ref["rating_ord"] == 15.0

    def test_unclassified_issues_are_never_used(self):
        d = enriched([(1, "NON", 900.0), (2, "SENS", 100.0)],
                     [(1, "2006-03-31", 3.0), (2, "2006-03-31", 15.0)])
        assert set(d["issue_id"]) == {2}


class TestBondCutEvents:
    def test_switching_representative_issue_is_not_an_event(self):
        # issue 1 matures after Q1; the larger issue 2 is newly rated worse in Q2. Level jumps, nothing was cut.
        cuts = cuts_for([(1, "SEN", 500.0), (2, "SEN", 900.0)],
                        [(1, "2006-03-31", 9.0), (2, "2006-06-30", 12.0)])
        assert not cuts["cut_primary"].any()

    def test_cut_on_a_non_representative_senior_issue_counts(self):
        cuts = cuts_for([(1, "SEN", 900.0), (2, "SEN", 500.0), (3, "SEN", 100.0)],
                        [(1, "2006-03-31", 9.0), (2, "2006-03-31", 9.0), (3, "2006-03-31", 9.0),
                         (1, "2006-06-30", 9.0), (2, "2006-06-30", 9.0), (3, "2006-06-30", 10.0)])
        row = cuts[cuts["quarter_end"] == qe("2006-06-30")].iloc[0]
        assert row["cut_primary"]
        assert not row["cut_strict"]                  # neither the reference issue nor a majority

    def test_strict_rule_fires_on_majority(self):
        cuts = cuts_for([(1, "SEN", 900.0), (2, "SEN", 500.0), (3, "SEN", 100.0)],
                        [(1, "2006-03-31", 9.0), (2, "2006-03-31", 9.0), (3, "2006-03-31", 9.0),
                         (1, "2006-06-30", 9.0), (2, "2006-06-30", 10.0), (3, "2006-06-30", 10.0)])
        assert cuts.loc[cuts["quarter_end"] == qe("2006-06-30"), "cut_strict"].iloc[0]

    def test_strict_rule_fires_on_reference_issue(self):
        cuts = cuts_for([(1, "SEN", 900.0), (2, "SEN", 500.0), (3, "SEN", 100.0)],
                        [(1, "2006-03-31", 9.0), (2, "2006-03-31", 9.0), (3, "2006-03-31", 9.0),
                         (1, "2006-06-30", 10.0), (2, "2006-06-30", 9.0), (3, "2006-06-30", 9.0)])
        assert cuts.loc[cuts["quarter_end"] == qe("2006-06-30"), "cut_strict"].iloc[0]

    def test_subordinated_cut_counts_only_for_any_bond_when_senior_exists(self):
        cuts = cuts_for([(1, "SEN", 900.0), (2, "SUB", 500.0)],
                        [(1, "2006-03-31", 9.0), (2, "2006-03-31", 11.0),
                         (1, "2006-06-30", 9.0), (2, "2006-06-30", 12.0)])
        row = cuts[cuts["quarter_end"] == qe("2006-06-30")].iloc[0]
        assert not row["cut_primary"] and row["cut_anybond"]

    def test_fallback_issuer_uses_cuts_on_its_own_tier(self):
        cuts = cuts_for([(1, "SENS", 900.0)], [(1, "2006-03-31", 14.0), (1, "2006-06-30", 15.0)])
        assert cuts["cut_primary"].any()

    def test_upgrade_is_not_an_event(self):
        cuts = cuts_for([(1, "SEN", 900.0)], [(1, "2006-03-31", 10.0), (1, "2006-06-30", 9.0)])
        assert not cuts["cut_primary"].any()

    def test_cut_across_a_rating_gap_is_not_an_event(self):
        # rated Q1, withdrawn Q2, re-rated worse Q3: not a cut observed quarter to quarter
        cuts = cuts_for([(1, "SEN", 900.0)], [(1, "2006-03-31", 9.0), (1, "2006-09-30", 12.0)])
        assert not cuts["cut_primary"].any()


def agency_refs(rows):
    """rows: (gvkey, quarter_end, agency, rating_ord, rating_source)"""
    r = pd.DataFrame(rows, columns=["gvkey", "quarter_end", "agency", "rating_ord", "rating_source"])
    r["quarter_end"] = pd.to_datetime(r["quarter_end"])
    r["q"] = quarter_index(r["quarter_end"])
    r["rating_worst"] = r["rating_ord"]
    r["n_active_issues"] = 1
    return r


class TestCombineAgencies:
    def test_middle_of_three(self):
        out = combine_agencies(agency_refs([
            ("A", "2006-03-31", "SPR", 9.0, "senior_unsecured"),
            ("A", "2006-03-31", "MR", 11.0, "senior_unsecured"),
            ("A", "2006-03-31", "FR", 10.0, "senior_unsecured")]))
        assert out["rating_ord"].iloc[0] == 10.0 and out["rating_dispersion"].iloc[0] == 2.0

    def test_lower_of_two(self):
        out = combine_agencies(agency_refs([
            ("A", "2006-03-31", "SPR", 9.0, "senior_unsecured"),
            ("A", "2006-03-31", "MR", 11.0, "senior_unsecured")]))
        assert out["rating_ord"].iloc[0] == 11.0

    def test_most_senior_source_tier_wins(self):
        out = combine_agencies(agency_refs([
            ("A", "2006-03-31", "SPR", 9.0, "senior_unsecured"),
            ("A", "2006-03-31", "MR", 14.0, "subordinated")]))
        row = out.iloc[0]
        assert row["rating_ord"] == 9.0 and row["rating_source"] == "senior_unsecured"
        assert row["n_agencies"] == 2 and row["has_sp"] and row["has_moodys"] and not row["has_fitch"]


def panel_rows(quarters, gvkey="A"):
    r = pd.DataFrame({"gvkey": gvkey, "quarter_end": pd.to_datetime(quarters)})
    r["q"] = quarter_index(r["quarter_end"])
    return r


def events_at(quarters, gvkey="A"):
    return pd.DataFrame({"gvkey": gvkey, "q": quarter_index(pd.Series(pd.to_datetime(quarters)))})


def no_episodes():
    return pd.DataFrame(columns=["gvkey", "default_date", "emerge_date", "default_q", "emerge_q"])


class TestLabels:
    def _labels(self, quarters, event_quarters):
        rows = panel_rows(quarters)
        return make_labels(rows, events_at(event_quarters), observed_last_quarter(rows), "downgrade",
                           with_countdown=True)

    def test_flag_and_forward_labels(self):
        qs = pd.date_range("2005-03-31", periods=6, freq="QE")
        out = self._labels(qs, [qs[3]])
        assert pd.isna(out["downgrade"].iloc[0])                  # first quarter: nothing observed before
        assert out["downgrade"].tolist()[1:] == [0.0, 0.0, 1.0, 0.0, 0.0]
        assert out["downgrade_1q"].iloc[2] == 1.0 and out["downgrade_1q"].iloc[3] == 0.0
        assert out["downgrade_4q"].iloc[0] == 1.0

    def test_countdown(self):
        qs = pd.date_range("2005-03-31", periods=5, freq="QE")
        out = self._labels(qs, [qs[3]])
        assert out["quarters_to_next_downgrade"].tolist()[:3] == [3.0, 2.0, 1.0]
        assert pd.isna(out["quarters_to_next_downgrade"].iloc[4])

    def test_horizon_is_calendar_based_across_row_gaps(self):
        # rows skip two years; an event 9 quarters after the first row must not reach its 4q label
        rows = ["2005-03-31", "2007-03-31", "2007-06-30"]
        out = self._labels(rows, ["2007-06-30"])
        assert out["downgrade_4q"].iloc[0] == 0.0 and out["downgrade_8q"].iloc[0] == 0.0
        assert out["downgrade_1q"].iloc[1] == 1.0
        assert pd.isna(out["downgrade"].iloc[1])                  # t-1 unobserved and no event

    def test_event_without_a_row_still_labels_earlier_quarters(self):
        # a default when every rating had been withdrawn: no row at the default quarter
        out = self._labels(["2005-03-31", "2005-06-30"], ["2005-09-30"])
        assert out["downgrade_1q"].iloc[1] == 1.0

    def test_horizon_beyond_sample_end_is_nan(self):
        out = self._labels(quarter_ends()[-3:], [])
        assert out["downgrade_4q"].isna().all() and out["downgrade_1q"].iloc[:2].eq(0).all()


def default_inputs(records, issues, ratings=()):
    """records: (issue_id, default_date, reinstated_date); issues: (issue_id, gvkey, offering_date);
    ratings: (issue_id, rating_date, ord)"""
    rec = pd.DataFrame(records, columns=["issue_id", "default_date", "reinstated_date"])
    rec[["default_date", "reinstated_date"]] = rec[["default_date", "reinstated_date"]].apply(pd.to_datetime)
    iss = pd.DataFrame(issues, columns=["issue_id", "gvkey", "offering_date"])
    iss["offering_date"] = pd.to_datetime(iss["offering_date"])
    r = pd.DataFrame(list(ratings), columns=["issue_id", "rating_date", "ord"])
    r["rating_date"] = pd.to_datetime(r["rating_date"])
    return default_episodes(rec, iss[["issue_id", "gvkey"]], iss[["issue_id", "offering_date"]], r)


class TestDefaults:
    def test_reinstatement_ends_the_episode(self):
        ep = default_inputs([(1, "2006-02-15", "2007-05-01")], [(1, "A", "2003-01-01")])
        assert len(ep) == 1
        assert ep["default_q"].iloc[0] == quarter_index(pd.Series([qe("2006-03-31")])).iloc[0]
        assert ep["emerge_q"].iloc[0] == quarter_index(pd.Series([qe("2007-06-30")])).iloc[0]

    def test_new_debt_ends_the_episode_when_earlier(self):
        ep = default_inputs([(1, "2006-02-15", None)],
                            [(1, "A", "2003-01-01"), (2, "A", "2006-10-01")],
                            [(2, "2006-11-01", 15.0)])
        assert ep["emerge_date"].iloc[0] == qe("2006-11-01")

    def test_defaults_before_reemergence_join_the_episode(self):
        ep = default_inputs([(1, "2006-02-15", "2008-01-10"), (2, "2006-08-01", "2008-01-10"),
                             (1, "2010-05-01", None)],
                            [(1, "A", "2003-01-01"), (2, "A", "2004-01-01")])
        assert len(ep) == 2 and pd.isna(ep["emerge_date"].iloc[1])

    def test_post_default_quarters_are_masked(self):
        ep = default_inputs([(1, "2006-02-15", "2007-05-01")], [(1, "A", "2003-01-01")])
        rows = panel_rows(pd.date_range("2005-12-31", "2007-09-30", freq="QE"))
        mask = post_default_mask(rows, ep)
        kept = rows.loc[~mask, "quarter_end"].tolist()
        assert qe("2006-03-31") in kept                          # the default quarter itself stays
        assert qe("2006-06-30") not in kept and qe("2007-03-31") not in kept
        assert qe("2007-06-30") in kept                          # re-emergence quarter re-enters

    def test_default_with_withdrawn_rating_is_an_event_and_later_rows_drop(self):
        ep = default_inputs([(1, "2006-02-15", None)], [(1, "A", "2003-01-01")])
        rows = panel_rows(pd.date_range("2005-06-30", "2006-12-31", freq="QE"))
        default_ev = ep[["gvkey", "default_q"]].rename(columns={"default_q": "q"})
        out = label_panel(rows, {"downgrade": default_ev}, ep)
        assert out["quarter_end"].max() == qe("2006-03-31")
        assert out.loc[out["quarter_end"] == qe("2005-12-31"), "downgrade_1q"].iloc[0] == 1.0

    def test_events_inside_the_post_default_window_are_ignored(self):
        ep = default_inputs([(1, "2006-02-15", "2008-01-10")], [(1, "A", "2003-01-01")])
        rows = panel_rows(pd.date_range("2005-06-30", "2008-12-31", freq="QE"))
        ev = pd.concat([ep[["gvkey", "default_q"]].rename(columns={"default_q": "q"}),
                        events_at(["2006-09-30"])])
        out = label_panel(rows, {"downgrade": ev}, ep)
        # 2006-Q3 is within four quarters of the default quarter, so only the filter keeps this at zero
        assert out.loc[out["quarter_end"] == qe("2006-03-31"), "downgrade_4q"].iloc[0] == 0.0


def test_quarter_grid_spans_the_sample():
    q = quarter_ends()
    assert q[0] == Q("2000-03-31") and q[-1] == Q("2024-12-31")
    assert len(q) == 100
