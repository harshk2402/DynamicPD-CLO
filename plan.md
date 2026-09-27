# Implementation Plan — DynamicPD-CLO

## Overview

22 sequential chunks. Each chunk must be confirmed complete before the next begins. No random splits anywhere — walk-forward only. Strict 2000–2018 train / 2019–2024 test wall enforced from Chunk 9 onward.

**Pivot note (2026-06-20):** Project shifted from default prediction to downgrade prediction. Defaults are too rare (~1–2% of firm-quarters) to train reliably on ~150 borrowers × 100 quarters. Rating downgrades (~5–10% of firm-quarters) provide a much richer signal of credit deterioration with the same feature framework intact. Ground truth is now confirmed primary FISD rating downgrade history rather than realized defaults. The CLO component shifts from copula Monte Carlo pricing to portfolio-level collateral quality aggregation. Research question is unchanged: do ML signals precede traditional credit assessments?

**Data audit (2026-09-18):** The archived raw data has now been audited end to end. Verified figures and
the corrections they force are recorded in the affected chunks below. Headline results: the downgrade base
rate is 5.5% of issuer-quarters (vs 1-2% for defaults), confirming the pivot; the trainable panel is ~196k
firm-quarters across 2,440 rated firms; the legacy DealScan normalised tables are frozen at mid-2020 and
must not be used as the loan source; and the DealScan-Compustat borrower link had to be built locally
because no WRDS-hosted crosswalk exists. Numbers quoted below come from diagnostic scripts using a
worst-active-rating rule and a CUSIP8 bond link, not the full Chunk 5 spec - treat them as reliable
magnitudes, not final sample counts.

**Coverage note:** Earlier DealScan/FISD borrower counts are retained only as preliminary historical benchmarks. Final sample counts, issuer-quarter coverage, and downgrade-event frequencies must be recomputed from the locally archived raw datasets after applying validated issue, issuer, CCM, Compustat, CRSP, IBES, and DealScan linkage rules.

**Code organization:** `src/data/` is reserved for immutable raw-data extractors and connection/client utilities. All local preprocessing, cleaning, linkage application, timing alignment, feature engineering, and master-panel construction from archived raw data must live under `src/preprocessing/`, with one Python module per chunk/domain as needed.

---

## Chunk 1 — Environment Setup and Connections

**Goal:** Verify all credentials and connections work before touching data.

**Steps:**

1. Create directory structure: `/data/`, `/models/`, `/clo/`, `/eval/`, `/agentic/`, `/utils/`
2. Use `.env` at project root — loads env vars: `WRDS_USERNAME`, `FRED_API_KEY`, paths for `data/raw/`, `data/processed/`, `data/output/`
3. Create `utils/connections.py`:
   - `get_wrds_connection()` — returns authenticated `wrds.Connection()`
   - `get_fred_client()` — returns `fredapi.Fred(api_key=...)`

**Files created:** `utils/connections.py`

---

## Chunk 2 — Macro Data Pipeline

**Goal:** Use the completed immutable raw FRED macro archive for 2000–2024 and prepare the confirmed macro feature sources for later point-in-time feature construction.

**Steps:**

1. Maintain the raw macro extractor in `src/data/fred.py`.
2. Use the locally archived immutable raw files under `data/raw/macro/`:
   - `treasury_rates.parquet`: `DGS3MO` → `treasury_3m`, `DGS10` → `treasury_10y`, `term_spread = treasury_10y - treasury_3m`
   - `gdp_growth.parquet`: `A191RL1Q225SBEA` → `gdp_growth_qoq`
   - `fed_funds.parquet`: `FEDFUNDS` → `fed_funds_rate`
   - `unemployment.parquet`: `UNRATE` → `unemployment_rate`, `unemployment_change`
   - `cpi.parquet`: `CPIAUCSL` → `cpi`, `cpi_yoy`
   - `credit_spreads.parquet`: `AAA10Y` → `aaa10y`, `BAA10Y` → `baa10y`
   - `vix.parquet`: `VIXCLS` → `vix`
3. Exclude historical ICE/BofA OAS series from the primary feature set because accessible coverage is unavailable for the full 2000–2024 sample.
4. Use CRSP market-index returns as the canonical market-return source; do not use FRED S&P 500 as a required source.
5. Keep Chunk 2 as immutable raw extraction/provenance only. Quarterly timing-safe alignment, release-aware transformations, forward-fill rules, lags, z-scores, and aggregation belong in the feature-engineering/master-panel stages.

**Files created:** `src/data/fred.py`, `data/raw/macro/treasury_rates.parquet`, `data/raw/macro/gdp_growth.parquet`, `data/raw/macro/fed_funds.parquet`, `data/raw/macro/unemployment.parquet`, `data/raw/macro/cpi.parquet`, `data/raw/macro/credit_spreads.parquet`, `data/raw/macro/vix.parquet`

---

## Chunk 3 — Compustat Firm Financials Pipeline

**Goal:** Load local Compustat raw Parquet partitions, apply final filters, enforce point-in-time availability, and construct quarterly accounting features.

**Steps:**

1. Create `src/preprocessing/compustat.py`
2. Use locally archived Compustat raw partitions:
   - Quarterly: `data/raw/wrds/compustat_quarterly/`, 1999–2024
   - Annual: `data/raw/wrds/compustat_annual/`, 1998–2024
3. Load quarterly fields from the archived `comp.fundq` partitions:
   - Fields: `gvkey`, `datadate`, `atq` (total assets), `ltq` (total liabilities), `dlttq`+`dlcq` (total debt), `xintq` (interest expense), `oibdpq` (EBIT proxy), `niq` (net income), `actq` (current assets), `lctq` (current liabilities), `cheq` (cash), `sic`
   - Filter: `indfmt='INDL'`, `datafmt='STD'`, `popsrc='D'`, `consol='C'`
   - Exclude SIC 6000–6999 (financials) and 4900–4999 (utilities)
4. Resolve duplicate format observations after applying standard Compustat filters.
5. Enforce accounting timing:
   - `datadate` is the fiscal-period date
   - `rdq` is the availability date
   - Features may only enter a prediction quarter once `rdq <= prediction_date`
   - Never align accounting information solely by `datadate`
6. Compute features:
   - `leverage` = total debt / atq
   - `coverage` = oibdpq / xintq (NaN if xintq ≤ 0)
   - `profitability` = niq / atq
   - `liquidity` = actq / lctq
   - `log_assets` = log(atq)
   - `cash_ratio` = cheq / atq
7. Winsorize each feature at 1st/99th percentile (within training period only — compute percentiles on 2000–2018, apply to all)
8. Align available observations to prediction quarter-ends using `rdq`; keep one obs per `gvkey` per quarter
9. Cap staleness. The built output currently forward-fills the last available filing indefinitely:
   `accounting_age_days` has median 504, 75th percentile 3,318, and maximum 9,450 days (26 years), so firms
   that stopped reporting still carry old fundamentals into recent quarters. Quarterly filers report within
   roughly 90 days, so accounting older than a stated cap must not be carried forward.
   **Decision (2026-09-27): cap at 270 days; null the values rather than drop the row.** Verified on the
   uncapped build: for live filers (CRSP price present at quarter-end, 540,694 rows) `accounting_age_days` has
   median 56, p95 78, p99 146, p99.5 203 and p99.9 494, so 270 days tolerates one missed filing but not two,
   and affects only 0.3% of live-filer rows (0.1% at 450, so the choice between the two barely matters for
   them). Beyond the cap the row is kept, flagged `accounting_stale`, and the accounting values and features are
   set to NaN, because labels, ratings and market data may still be live - e.g. an LBO'd or private firm with
   public bonds. `accounting_age_days` is retained as a model feature. On the rated panel 18.3% of
   issuer-quarters exceed the cap (16.8% at 450), concentrated in firms without CRSP equity (59% of their
   rows): these keep their labels but carry no current accounting, which Chunk 9 must handle explicitly.
10. Save to `data/processed/compustat/compustat_features.parquet`

**Files created:** `src/preprocessing/compustat.py`, `data/processed/compustat/compustat_features.parquet`

---

## Chunk 4 — CRSP Equity Data Pipeline

**Goal:** Load local CRSP raw history and construct timing-safe equity features for 2000–2024. Merton Distance-to-Default and daily equity volatility now come from Chunk 8.

**Steps:**

1. Create `src/preprocessing/equity.py`
2. Use locally archived CRSP raw files:
   - Monthly stock files: `data/raw/wrds/crsp/monthly/`
   - Daily stock files: `data/raw/wrds/crsp/daily/`
   - Names history: `data/raw/wrds/crsp/crsp_names.parquet`
   - Delistings: `data/raw/wrds/crsp/crsp_delistings.parquet`
   - CCM links: `data/raw/merton/ccm_links.parquet`
   - Monthly market index: `data/raw/wrds/crsp/market_monthly/`
   - Daily market index: `data/raw/wrds/crsp/market_daily/`
3. Use validated CCM fields: `gvkey`, `lpermno`, `lpermco`, `linkdt`, `linkenddt`, `linktype`, `linkprim`
4. Apply valid CCM link-date windows for all CRSP-to-Compustat joins.
5. Compute from CRSP monthly:
   - Trailing 12-month firm stock return using CRSP monthly returns (product of monthly returns), matching the DSW one-year firm-return covariate
   - Market capitalization = `abs(prc) * shrout`
   - Market-to-book = mktcap / (atq from Compustat — merge on gvkey + quarter)
   - CRSP value-weighted market return
   - CRSP equal-weighted market return
   - Trailing 12-month CRSP market return
6. Incorporate delisting returns where applicable; do not silently drop delisting events.
7. Daily equity volatility and estimated Merton Distance-to-Default come from Chunk 8.
8. Align to quarter-end using information available through quarter-end only; save to `data/raw/equity/equity_features.parquet`

**Files created:** `src/preprocessing/equity.py`, `data/raw/equity/equity_features.parquet`

---

## Chunk 5 — Rating Downgrade Events Pipeline

**Goal:** Construct binary downgrade indicators at 1, 4, and 8 quarter horizons for all firms.

**Pivot rationale:** Defaults are too rare to train on. Rating downgrades are ~5–10× more frequent and capture a broader spectrum of credit deterioration while remaining economically meaningful.

**Steps:**

1. Create `src/preprocessing/downgrades.py`
2. Primary source — FISD issue-level rating history and issue metadata:
   - Build the complete FISD issue-level and issuer-level rating history first
   - Construct issuer-quarter ratings using FISD `issue_id → issuer_id`
   - Link the reconstructed issuer universe to Compustat/CRSP using the validated local linkage chain where needed
   - Restrict to DealScan borrowers later when the modeling universe is finalized
   - Fields: `issue_id`, `rating_date`, `rating`, `rating_agency`, issue seniority/security fields, `offering_date`, `maturity`, `offering_amt`, and validated retirement/default/redemption fields
   - Confirmed bond linkage chain where applicable: FISD bond CUSIP → `bondcrsp_link` → PERMNO → CCM → GVKEY
   - Build FISD bond CUSIP from `issuer_cusip + issue_cusip`, then normalize to CUSIP8
   - Bond-CRSP and CCM links must respect their date-validity fields during final panel construction
3. Clean issue-level rating histories before issuer aggregation:
   - Normalize S&P, Moody's, and Fitch ratings onto a common ordinal scale
   - Remove consecutive duplicate ratings within each issue × agency history so surveillance updates, affirmations, and vendor backfills are not treated as rating events
   - Treat the Fitch 2013–2016 volume spike as mainly unchanged-rating administrative/backfill activity handled through consecutive-rating deduplication
   - Define an active issue at quarter-end when `offering_date <= quarter_end < maturity` and the issue has not been retired, defaulted, or redeemed according to validated FISD fields
   - Do not use `amount_outstanding` for historical weighting because the available field is a sparse/current snapshot rather than a validated historical series
4. Reconstruct issuer-quarter ratings from active rated issues using a pre-specified hierarchy:
   - Primary methodology, representative senior unsecured issue: at each quarter-end, identify all active rated issues; if one or more senior unsecured issues exist, select the senior unsecured issue with the largest original `offering_amt`; this becomes the issuer rating for that quarter
   - Fallback methodology, estimated senior rating: if no senior unsecured issue exists, take the most senior remaining active rated issue and shift its rating by the cross-sectional notching offset for its security level. Superseded the original largest-offering fallback on 2026-09-26; see "Fallback rating decision" below for the method and validation
   - Robustness methodology, worst active issue rating: separately reconstruct issuer-quarter ratings using the worst rating among all active rated issues
   - Ratings are reconstructed per agency. The combined issuer rating level is the middle of three agency ratings, or the lower of two, or the single rating present
5. Construct downgrade events from actual rating cuts on bonds, never from changes in a reconstructed rating level (revised 2026-09-27; see "Label construction decisions" below):
   - Per agency: a downgrade in quarter t when the agency lowered its rating on at least one of the issuer's active senior unsecured issues that it also rated in quarter t-1; issuers with no senior unsecured issue use the fallback reference issue
   - A default recorded in `fisd_issue_default` on any of the issuer's issues is a downgrade event in the quarter of the earliest default date, for every agency, whatever the ratings did
   - Primary label: any agency downgraded in the quarter. Per-agency labels are kept for robustness
   - After a default, drop the issuer until it re-emerges (`reinstated_date`) or has newly rated debt
6. Capital IQ ratings access was denied and is not required; do not plan to supplement FISD using CIQ.
7. For each firm-quarter observation, create primary-label targets:
   - `downgrade_1q` = 1 if any downgrade event within next 1 quarter
   - `downgrade_4q` = 1 if any downgrade event within next 4 quarters
   - `downgrade_8q` = 1 if any downgrade event within next 8 quarters
8. Document coverage rate (% of final matched issuer-quarters with at least one active rating) and recompute downgrade-event frequencies after final FISD → Bond-CRSP → CCM → Compustat linkage and DealScan restriction.
9. Report issuer-rating construction diagnostics:
   - Percentage of issuer-quarters using the representative senior unsecured rule
   - Percentage of issuer-quarters using the estimated-senior-rating fallback
   - Disagreement rate between the primary construction and the worst-rating construction
   - Downgrade counts under each methodology and each robustness label
   - Defaults: count, share with a rating cut vs a withdrawal in the default quarter, share with a prior cut, and a completeness cross-check against CRSP bankruptcy delisting codes
   - Agreement of the rebuilt S&P rating and events with `comp.adsprate` over 2000-2016
10. Save to `data/raw/downgrades/downgrade_events.parquet`

**Phase 0 field verification (2026-09-18) - read before implementing:**

- **Use `fisd.fisd_ratings`, not the archived `fisd_rating_hist` partitions.** WRDS exposes three rating
  tables and they are not interchangeable: `fisd_rating` is a current-snapshot (~one row per issue per
  agency), `fisd_rating_hist` is a curated subset, and `fisd_ratings` is the full history. Restricted to the
  same 298,562 issues, the same 2000-2024 window and the same three agencies, `fisd_ratings` holds 2,784,849
  records against `fisd_rating_hist`'s 1,943,184. The archived partitions are a strict subset (nothing in
  them is absent from `fisd_ratings`), and **724,524 of the 841,665 missing records are genuine rating level
  changes, not affirmations** - so the archived extract carries only 58% of the available rating changes.
  The shortfall is worst in the recent years that make up the test period (45,747 missing level changes in
  2024 alone), which is the main reason observed downgrade rates looked implausibly low late in the sample.
  `fisd_ratings` also carries two columns the partitions lack: `rating_status_date` and `investment_grade`.
  Re-extract from `fisd.fisd_ratings` before building labels; all previously quoted base rates are understated
  and must be recomputed.
- **`security_level` supports the primary hierarchy as specified.** Zero nulls; `SEN` (senior unsecured)
  covers 237,531 issues (79.6%), then `NON` 47,232, `SS` 6,590, `SENS` 4,637, `SUB` 2,116, `JUNS` 317,
  `JUN` 13. 8,460 of 11,908 issuers (71.0%) have at least one senior unsecured issue, so 3,448 issuers take
  the largest-offering fallback. `offering_amt` has zero nulls, so the representative-issue selection works.
- **The retirement fields the active-issue rule assumed are empty:** `defeased_date` is 100% null,
  `refunding_date` 99.9% null, `security_pledge` 100% null. Only `offering_date` (0.1% null), `maturity`
  (0.6% null) and `fisd_issue_default` (5,709 rows over 5,604 issues, `default_date` fully populated, with
  `reinstated`/`reinstated_date` for emergences) are usable. There is no call, tender or redemption date
  anywhere in the archive, so an issue called early will look active until its maturity date. Use `NR`
  records as the practical spell terminator (see below) and record this as a stated limitation.
- **Rating codes outside the ordinal scale:** `NR` dominates at 568,640 records and marks a withdrawn or
  absent rating - it must terminate a rating spell rather than carry forward, which is the mechanism that
  substitutes for the missing redemption data. `D`/`DD`/`DDD`/`RD`/`SD` are genuine bottom-of-scale default
  grades and must be mapped, not dropped. `P-1` is a short-term/commercial-paper scale and must be excluded.
  `PIF`, `SUSP`, `NAV`, `NR/NR` are terminators. Determine grade status from the `rating` value itself;
  `rating_status` (`WNEG`, `WPOS`, `WNOT`, `WOFF`, `Off`, `Neg`, `Pos`, `WUND`) is watch/lifecycle metadata
  and belongs in the static benchmark specification, never in the ML feature set.
- **`bond_type` needs an exclusion list, and the obvious reading of the codes is wrong.** The AAA mass is
  concentrated in `ADEB` (404,434 records, 87.4% AAA), `AMTN` (187,302, 91.8%), `ARNT` (45,523, 98.0%) and
  `USNT` (25,619, 99.0%), while `CDEB` (476,990, 0.7% AAA), `CMTN` (290,762, 2.6%) and `RNT` (283,254, 1.4%)
  show corporate-like rating distributions. The A-prefixed types are therefore agency paper rather than
  "American corporate", so any exclusion list that keeps `ADEB`/`AMTN` while dropping smaller codes is
  backwards. Verify empirically during the build - agency and sovereign paper carries no Compustat gvkey, so
  the linkage step should drop it independently, and the two filters should agree. Exclude sovereign
  (`FGOV`), preferred stock (`PSTK`, `PS`), trust preferred (`TPCS`) and treasury/agency types explicitly
  rather than relying on linkage alone.
- Duff & Phelps (`DPR`) appears on 33,524 records but only for 2000-2002, before it was absorbed into Fitch.
  Exclude it: a fourth agency that vanishes two years into the sample creates exactly the coverage
  discontinuity the agency-availability control is meant to prevent.

**Verified against the archive (2026-09-18):**

- Label supply is ample: 1.94M raw rating records (Fitch 864k, Moody's 681k, S&P 398k). On issues linked to a
  gvkey this leaves 412,230 genuine rating changes after consecutive-rating dedup, across 79,395 rated issues
  and 2,483 issuers.
- Base rates on the linked panel: 5.5% of issuer-quarters carry a downgrade; 5.6% within 1q, 17.5% within 4q,
  26.7% within 8q. These validate the pivot (defaults were 1-2%) and need no imbalance correction.
- The series shows correctly timed cyclical peaks (12.8% in 2001, 12.1% in 2002, 10.1% in 2009, 5.0% in 2020),
  which is evidence the construction behaves like real credit data.
- FISD to gvkey: build the bond CUSIP8 and go through `bondcrsp_link` -> PERMNO -> CCM. This route reaches
  1,636 issuers with a rated bond outstanding at 2019-Q1 versus 1,269 via an issuer-CUSIP6-to-Compustat
  shortcut; the union is 2,014. Use both routes and union them rather than relying on either alone.
- **The active-issue rule in step 3 is load-bearing, not housekeeping.** A diagnostic that forward-filled
  ratings without it produced a panel whose firm count rises monotonically (1,096 in 2000 to 2,483 in 2024)
  with no exit in any year, because issuers whose bonds matured never leave the denominator. That depresses
  later-year downgrade rates (9.9% in 2000 falling to 1.4% in 2024) and would understate the test period.
  Requiring `offering_date <= quarter_end < maturity` and excluding retired/defaulted/redeemed issues is what
  prevents this.
- **Agency coverage drifts across the sample and must be controlled.** S&P records fall from 38k in 2000 to
  4.4k in 2024, while Moody's stays roughly stable, so an "earliest downgrade across agencies" rule silently
  changes meaning over time (S&P-informed early, Moody's-dominated late). Record per-issuer-quarter which
  agencies are present and carry agency-availability indicators into the panel.
- The Fitch 2013-2016 backfill spike is confirmed: 31,862 records in 2012 rising to 191,029 in 2016 and
  falling back to 24,631 in 2017. Consecutive-rating dedup absorbs the unchanged-rating bulk; verify the
  residual rather than assuming it is fully handled.

**Label construction decisions (2026-09-27):**

The unit of analysis is the issuer-quarter (gvkey x quarter-end). Every feature is issuer-level; bonds enter
only through the label and the rating-level feature, because FISD ratings exist only at issue level.
Modelling at bond level would repeat identical issuer features once per bond, weight issuers by how much debt
they issue, and mostly relearn mechanical notching between bond classes.

The first build (2026-09-18) defined a downgrade as a worsening of the reconstructed rating level: per agency
from the representative issue, then across agencies as the worst agency rating. Checks on that build showed
the level is the wrong object to difference, for three reasons.

*1. Across agencies, detect events within each agency, never from a combined score.* A combined rating moves
when an agency starts or stops covering the issuer, with no rating action. On the first build, 174 of the
7,015 combined downgrades had no agency cutting anything, and 171 of those coincided with a change in agency
coverage. The worst-of-agencies rule also missed 4,339 issuer-quarters in which an agency did cut, because
the cutting agency still sat above the worst one - 39% of all quarters with an agency downgrade. The primary
label is therefore "any agency downgraded" (6.4% of issuer-quarters against 4.0% under the first build),
which matches the original intent of "earliest downgrade across agencies" and makes the static benchmark the
hardest one to beat: the first agency to move. The combined rating level, needed only as a feature and for
the static benchmark buckets, uses the regulatory convention of middle-of-three, lower-of-two.

*2. Within an agency, detect events from cuts on bonds, not from the representative issue's level.* The
representative issue changes when a bond matures or a larger one is issued, and the recorded level then jumps
with no rating action. The representative switches in about 4% of issuer-agency-quarters but those switches
produce 14-23% of level-based downgrades, about 70% of which had no bond actually cut: 1,719 spurious events
across the three agencies. The level rule also missed 3,244 real cuts on senior unsecured issues other than the
representative one. Under the adopted rule a bond counts only if the agency rated it in both t-1 and t, so
new issuance, maturity and switching can never create an event. Agency issuer downgrades move all senior
unsecured issues together, so the cut appears whichever issue is representative.

Validation against S&P's company rating (`comp.adsprate.splticrm`, 2000-2016, 91,629 issuer-quarters,
4,133 S&P company downgrades):

| Event rule | Our events | S&P downgrades caught, same quarter / within 1q | Our events matching an S&P downgrade, same / within 1q |
| --- | --- | --- | --- |
| Level-based (first build) | 4,059 | 78.3% / 82.1% | 79.7% / 83.3% |
| **Bond cut (adopted)** | 4,553 | **86.5% / 89.1%** | 78.5% / 82.5% |

The bond-cut rule catches 8 points more real downgrades at essentially the same precision. About 20% of our
events still have no matching S&P company downgrade (likely issue-specific actions or subsidiary bonds) and
about 11% of S&P downgrades are not caught; both are to be examined in the Chunk 5 diagnostics.

*3. Defaults are events even when the rating is withdrawn.* Of 2,206 linked issues defaulting in 2000-2024
(524 issuers), 43-48% had their rating withdrawn (NR) rather than cut to D by the end of the default quarter,
so the bond-cut rule alone would label about half of defaults as "no downgrade". 95% of defaulted issues had
at least one cut in the four quarters before default, so the 4q and 8q labels mostly survive, but the 1q label,
lead-time timing and sudden defaults with no prior cut would be lost. Withdrawals in general cannot be counted,
since most reflect redemption, acquisition or issuer request, so `fisd_issue_default` supplies the event. The
label is therefore "downgrade or default", which matches the rating-transition literature (default as a
transition state). Post-default quarters are dropped until re-emergence, following Moody's practice of
removing defaulted entities from study cohorts, because a defaulted issuer cannot be downgraded further and
its quarters carry no early-warning content. Distressed exchanges are defaults and count.

*Agency availability.* S&P coverage falls sharply across the sample, so the panel carries per-agency presence
flags (`has_sp`, `has_moodys`, `has_fitch`) alongside `n_agencies`.

*Robustness labels (pre-specified, never selected on model performance):*
- Each agency's labels on their own
- Strict bond-cut rule: the cut must hit the representative issue at t-1 or a majority of the issuer's active
  senior unsecured issues, to test whether single-issue actions drive results
- Events from cuts on any active issue, not senior unsecured only
- Senior-unsecured-only sample (see the fallback decision below)
- Worst-active-issue rating level, as originally pre-specified

These decisions were made from agreement with an external rating and from the mechanics of the data, before
any model was fit.

**Fallback rating decision (2026-09-26):**

*Problem.* Agencies rate bonds, not companies: each bond's rating is the company rating shifted up or down
for its place in the repayment order (secured above, subordinated below). Senior unsecured bonds carry the
company rating almost exactly, so they are the primary representative. About 17% of issuer-quarters (19% of
train, 9% of test) have no active rated senior unsecured bond, and the original fallback - largest remaining
bond, rating used as-is - inherits that bond's notching. Fallback firms are disproportionately high yield
(70% of fallback issuer-quarters vs 43% of senior unsecured ones), so the error concentrates in the
leveraged-borrower population the project targets.

*Method: cross-sectional notching offset, measured against senior unsecured bonds.*
1. Reference issue: the most senior active rated issue, in the order senior secured (`SS`), senior
   subordinated (`SENS`), subordinated (`SUB`), junior subordinated (`JUNS`), junior (`JUN`); largest
   original `offering_amt` breaks ties. Unclassified issues (`NON`) are never used as a reference: against
   S&P company ratings they sit a median 3 notches better (mean 4), consistent with insured or structured
   paper.
2. Offsets: across every issuer-quarter where a firm has both an active senior unsecured issue and an active
   issue of level L, record `rating(L) - rating(SEN)` on the same firm, same quarter and same agency. The
   offset for (agency, L, IG/HY) is the median of these gaps across all firms. Measuring each gap inside one
   firm and quarter isolates the effect of the security level from differences between firms.
3. Estimated senior rating = reference issue rating - offset, clipped to the ordinal scale.
4. Offsets are estimated on 2000-2018 only and applied unchanged to 2019-2024, per the train/test wall, and
   estimated separately for S&P, Moody's and Fitch, so no agency borrows another's notching practice.
5. Carry a `rating_estimated` flag on every issuer-quarter whose level comes from this fallback.

*Scope.* The fallback sets only the rating level (rating feature, static benchmark buckets). Downgrade
events do not depend on it.

*Relation to the literature.* This is a simplified form of Moody's Senior Ratings Algorithm (SRA), which
Moody's uses to give every entity a senior unsecured rating for its default and transition studies. The
original SRA (Hamilton 2005; Gupta, Parwani & Emery 2009) notched non-senior ratings by fixed rules. The
redesigned SRA (Kanthan, Ou, Agarwal & Irfan, "Moody's Revised Senior Ratings Algorithm", Special Comment,
Sept 2017; in use since Oct 2015) replaced these with rules inferred from data - the same core step adopted
here. The redesigned SRA's steps are: (1) drop ineligible credits, including externally backed ones;
(2) group each entity's debt by class, seniority, backing and currency, taking the median-worst rating within
a group; (3) set each group's notching rule to the modal gap to the senior unsecured rating among entities
holding both, conditioned on time, the group's rating level, region and sector, and formed only if at least
50% of entities and at least 10 entities share the mode; (4) take as reference the group whose rule is most
consistent, with more targeted rules winning ties; (5) smooth artificial rating changes caused by
switching reference group or rule; (6) drop periods with no rated debt. The adopted method keeps the core of
step 3 but estimates one median offset per (agency, seniority, IG/HY) pooled over the training period, and
uses a fixed seniority order in step 4. Why the coarser rules were chosen is set out in the comparison
below. The redesigned SRA contains no step that reuses a firm's own historical gap; that idea was tested
separately and rejected (below).

*Validation.* Estimated S&P ratings were compared with S&P's own company rating (`comp.adsprate.splticrm`,
quarter-end months). That table ends in Feb 2017, which is why it can validate the construction but cannot
replace it as the label source. Offsets were estimated on 2000-2010 and scored on the 2,877 fallback
issuer-quarters of 2011-2016:

| Method | Exact | Off by 1 notch | Off by 2+ | Mean error |
| --- | --- | --- | --- | --- |
| No shift (original fallback) | 34% | 30% | 36% | -0.30 |
| Cross-sectional offset, measured against S&P company ratings | 43% | 35% | 22% | -0.55 |
| **Cross-sectional offset, measured against senior unsecured bonds (adopted)** | 27% | 56% | **17%** | **+0.06** |
| Firm's own past gap first, then adopted offset | 27% | 53% | 20% | +0.13 |
| *Yardstick: senior unsecured issue as-is, 2000-2016* | *72%* | *16%* | *12%* | *+0.20* |

Mean error is estimate minus S&P ordinal; positive means the estimate is worse than the true rating. The
adopted method has the fewest misses of two notches or more and essentially no bias. The S&P-measured
offset hits exactly more often but leans half a notch optimistic, which would put risky firms in safer
static-benchmark buckets, weaken the benchmark and flatter the ML model; a pessimistic lean would err in the
conservative direction. Reusing a firm's own past gap applied to only 13% of fallback issuer-quarters
without look-ahead (31% if future history is allowed, which a point-in-time feature cannot use), and where it
applied it scored 59% within one notch against 84% for the cross-sectional offset. The likely reason:
notching reflects the debt ranked ahead of an issue, so once the senior unsecured bond is gone the firm's old
gap is stale.

*Comparison with Moody's redesigned SRA (2026-09-26).* Two tests. Test 1 (all agencies, the SRA paper's own
accuracy measure): for issuer-quarters that do have a senior unsecured rating, hide it, estimate it from the
firm's other debt, and compare, 2011-2018. Test 2 (S&P only): the real fallback issuer-quarters against S&P's
company rating, 2011-2016. Rules were learned on data before the evaluation window only.

- *Exact SRA* (steps 1-4 as specified; sector = SIC division from CRSP; trailing 20-quarter window; step 5
  omitted because it changes levels over time rather than per-quarter accuracy and is not fully specified).
  Rules almost never form at this sample size: estimates exist for 45 of 7,358 S&P, 186 of 7,640 Moody's and
  421 of 5,878 Fitch Test 1 cases, and 15 of 2,900 Test 2 cases. Where they form they are somewhat more
  accurate (e.g. Moody's ratings: 51% exact and 5% off 2+ against 44% and 18% for the adopted method on the
  same 186 cases), but they are the data-rich, regular cases. Moody's applies the SRA to its full global rated
  universe including loans and entity-level ratings; a US bond-only sample cannot populate its cells.
- *Relaxed SRA* (modal gap with the 50%/10-entity consensus, rules pooled over 2000-2010, at three
  granularities: IG/HY, letter grade, letter grade x sector; adopted method where no rule forms). Rules form
  for 26-40% of Test 1 cases but only 7-10% of Test 2 cases, because real fallback firms have unusual capital
  structures. Where rules form they are 1-6 points more often exact. Over all cases the gain is about one
  point: best Test 2 variant 28% exact / 16% off 2+ / +0.02 against 27% / 17% / +0.05 for the adopted method;
  Test 1 gains are 0-2 points exact. Granularity makes no material difference.

Decision: keep the adopted method. The exact SRA cannot produce estimates at this sample size, and the relaxed
SRA's one-point gain does not justify a two-stage rule-formation step. Test 1 also shows the method is less
precise for firms holding both senior unsecured and other debt (25-36% off 2+) than for the real fallback
population (17%); this does not affect its use but is a stated caveat on precision.

This choice was made on agreement with an external rating before any model was fit, not on model
performance, so it respects the pre-specification rule.

*Robustness.* Re-run on the senior-unsecured-only sample. Dropping fallback quarters keeps 147,239 of
177,446 issuer-quarters (83%; 2,854 of 3,490 firms; 986 of 1,091 test-period downgrades), but removes about
26% of high-yield DealScan-borrower issuer-quarters, so it tilts the sample toward investment grade and is a
robustness check rather than the primary sample.

*Limitations.* The fallback remains less precise than the primary rule (83% within one notch vs 88%). The
external validation covers S&P only; Moody's and Fitch offsets are estimated the same way but cannot be
checked against a company-level rating.

**Files created:** `src/preprocessing/downgrades.py`, `data/raw/downgrades/downgrade_events.parquet`

---

## Chunk 6 — IBES Analyst Data Pipeline

**Goal:** Load local IBES raw archives, link analyst data to the issuer universe, and compute point-in-time analyst features.

**Steps:**

1. Create `src/preprocessing/ibes.py`
2. Use locally archived IBES raw datasets:
   - `ibes.statsum_epsus`
   - `ibes.act_epsus`
   - `ibes.actu_epsus`
   - `ibes.recdsum`
3. Use `statsum_epsus` as the main forecast source:
   - `statpers` = consensus snapshot date
   - `fpedats` = forecast-period end
   - `fpi` = forecast horizon
   - `meanest`, `medest`, `stdev`, `numest`, high/low estimates where available
4. Use `act_epsus` and `actu_epsus` for actual EPS data. Their actual schemas include fields such as `ticker`, `cusip`, `oftic`, `pends`, `measure`, `pdicity`, `anndats`, `actdats`, and `value`; do not assume these files contain `fpedats` or `fpi`.
5. Use the validated IBES-to-issuer linkage:
   - Primary match: exact normalized CUSIP8 between IBES and Compustat
   - In the actual FISD–Bond-CRSP–CCM–Compustat modeling universe, exact CUSIP8 matching covers approximately 92.8% of linked issuers
   - 4,185 of 4,510 linked issuers matched exactly
   - Unmatched issuers remain in the sample with missing analyst variables and missingness indicators
   - Ticker/CRSP historical fallback matching is optional and must be separately validated before use
6. Compute analyst features only after timing alignment:
   - Consensus level
   - Revision direction and magnitude
   - Dispersion
   - Analyst count
   - Actual-vs-consensus surprise if included
7. Enforce timing: analyst features may use only snapshots with `statpers <= prediction_date`.
8. Save to `data/raw/ibes/ibes_features.parquet`

**Files created:** `src/preprocessing/ibes.py`, `data/raw/ibes/ibes_features.parquet`

---

## Chunk 7 — DealScan Leveraged Loan Pipeline

**Goal:** Build the leveraged-loan borrower restriction after validated firm, rating, equity, and analyst inputs are available.

**Steps:**

**Schema corrections (2026-09-18).** The original spec for this chunk was written against assumed field
names and does not match the data. Every item below was checked against the archive.

1. Create `src/preprocessing/dealscan.py`
2. **Source: the wide `dealscan.dealscan` table, not the normalised `facility`/`package` tables.** The legacy
   normalised tables are frozen at mid-2020 (the `facility` table holds 1 row dated 2021 and 2 dated 2024),
   so building on them would leave the 2019-2024 test period with loan data for 2019 and half of 2020 only.
   The wide table runs 1998-2026 with 500-1,500 Term Loan B tranches every year. The legacy tables remain
   useful for covenant detail and cross-checks, within their coverage window.
3. Field names, as they actually exist:
   - tranche type is `tranche_type` on the wide table (`loantype` on legacy `facility`); there is no
     `facilitytype` anywhere. Filter on `tranche_type` containing 'Term Loan B'.
   - spread is `all_in_spread_drawn_bps`; there is no `allindrawnbps` field. Quality is good: 11.3% null on
     TLB tranches, median 350bps, mean 380bps, which matches the leveraged loan market.
   - size is `tranche_amount` (0.1% null); maturity is `tranche_maturity_date` / `tenor_maturity` (4% null).
   - borrower is `borrower_id`, which shares the ID space of legacy `company.companyid` (77% overlap).
4. **Grain: dedupe on `lpc_tranche_id` before anything else.** The wide table is exploded by lender - 3.18M
   rows resolve to 431,127 unique tranches and 275,247 deals, roughly 7 rows per tranche. Aggregating without
   deduping would weight every loan by its number of syndicate members.
5. Link `borrower_id` -> `gvkey` using `data/processed/dealscan/dealscan_gvkey_link.parquet`, built by
   `src/preprocessing/dealscan_link.py`. **The Chava-Roberts crosswalk is not available**: WRDS hosts no
   DealScan-Compustat link library (its own linking matrix has no such entry) and the contributed-data file
   that is hosted is Schwert's *lender*-side table, which is a different thing. The local link matches on
   ticker via CRSP names history plus exact and suffix-stripped company name against Compustat `conm`,
   covering 17,184 of 176,020 borrowers. A Compustat-`tic` layer was tried and removed because `tic` holds
   only the current ticker and tickers are recycled after delisting, producing false matches.
6. Date-validate the ticker match here, where `tranche_active_date` is available: the identity link cannot
   tell a ticker reuse from a rename, so a borrower whose ticker matched a PERMNO outside the loan's active
   window should be treated as unmatched.
7. Keep DealScan as a sample restriction only. Do not make DealScan the starting point for FISD or CRSP
   extraction; firm, rating, equity and analyst pipelines are built on the broader linked issuer universe
   first (see the revised Chunk 9 step 6).
8. Per firm, keep the most recent active tranche as of each quarter-end:
   - `loan_spread` = `all_in_spread_drawn_bps`
   - `loan_size` = log(`tranche_amount`)
   - `loan_maturity` = years from quarter-end to `tranche_maturity_date`
   - `cov_lite` = **unresolved, see below**
9. **Covenant-lite is an open problem and must not be taken from the `covenants` field as-is.** That field
   reads "No" on 82% of TLB tranches, which would mean 82% of all TLBs since 1998 were covenant-lite - false,
   since the structure barely existed before the mid-2000s. The giveaway is that `all_covenants_financial` is
   null on 84.8% of the same tranches, almost exactly matching, so `covenants` tracks whether DealScan
   recorded anything rather than whether covenants exist. Using it directly would inject a proxy for vendor
   reporting completeness that trends with time the same way genuine cov-lite incidence does, so it would
   look predictive in-sample while measuring nothing. A defensible derivation needs a per-tranche coverage
   indicator built from the covenant tables (`financialcovenant` and `networthcovenant`, both legacy and
   frozen at 2020; `wrds_financial_covenants`, 695,867 rows on the newer ID space), reading absence as
   cov-lite only where covenant data was observable. Validate any derived series against the documented
   market profile - near zero pre-2005, a pre-crisis rise, a 2009 collapse, then a climb above 80% of new
   institutional issuance by the late 2010s. If it cannot reproduce that shape it is measuring coverage:
   drop the feature or restrict it to the covered period with an explicit missingness indicator.
10. Country: TLB tranches are 72% United States, then UK, France, Germany, Netherlands. Non-US borrowers are
   not in Compustat North America and drop out at the link stage; make the filter explicit rather than
   letting it happen silently.
11. Save the unique eligible borrower GVKEY universe to `data/processed/leveraged_loan_borrowers.parquet` for
   the Merton Distance-to-Default pipeline in Chunk 8.
12. Save to `data/raw/dealscan/dealscan_features.parquet`

**Verified counts:** 25,925 unique TLB tranches; 11,919 unique TLB borrowers over the full history, 4,091
active at 2019-Q1, of which 600 carry a gvkey and 140 also have a rated bond outstanding - enough for the
100-firm portfolio in Chunk 12 with headroom for the industry/spread stratification.

**Files created:** `src/preprocessing/dealscan.py`, `data/raw/dealscan/dealscan_features.parquet`, `data/processed/leveraged_loan_borrowers.parquet`

---

## Chunk 8 — Merton Distance-to-Default Pipeline

**Goal:** Reproduce a public-data Merton/KMV-style Distance-to-Default measure following Bharath and Shumway (2008) for the final linked DealScan borrower GVKEY/PERMNO universe, using local CRSP, Compustat, CCM, and FRED 3-month Treasury data.

**Reason for order:** DD is only a feature for the leveraged-loan borrower sample. Complete CRSP daily history is already archived, so Chunk 8 restricts local DD processing to the final linked DealScan borrower universe to reduce computation.

**Steps:**

1. Create `src/preprocessing/dd.py`
2. Load the final linked DealScan borrower GVKEY/PERMNO universe from Chunk 7.
3. Load local raw inputs:
   - `data/raw/merton/ccm_links.parquet`
   - Local Compustat quarterly debt partitions
   - Local CRSP daily partitions from `data/raw/wrds/crsp/daily/`
   - FRED 3-month Treasury series from `data/raw/macro/treasury_rates.parquet`
4. Apply date-valid CCM links before selecting borrower PERMNOs; do not estimate Distance-to-Default for the full Compustat universe.
5. Construct the Bharath–Shumway default point:
   - `default_point = dlcq + 0.5 × dlttq`
   - Convert Compustat debt from millions to thousands to match CRSP market-equity units
6. For each firm-month, use the prior 12 months of daily CRSP observations to estimate equity volatility and initialize asset volatility
7. Iteratively solve for asset value and asset volatility until convergence or an iteration cap
8. Compute:
   - Asset value
   - Asset volatility
   - Expected asset return
   - Distance-to-Default
   - Merton-implied expected default frequency
9. Aggregate monthly estimates to quarter-end without look-ahead
10. Save:
   - `data/processed/merton_dd_monthly.parquet`
   - `data/processed/merton_dd_quarterly.parquet`
11. Validate convergence, missingness, outliers, and the expected inverse relationship between Distance-to-Default and downgrade incidence

**Execution order:** borrower universe → date-valid CCM → borrower-filtered Compustat debt → FRED risk-free rate → restricted local CRSP daily processing.

**Scope decision:** Distance-to-Default is a model feature rather than a standalone research output. Restricting estimation to the leveraged-loan borrower universe produces the same required feature while avoiding unnecessary CRSP daily processing and Merton iterations for thousands of firms that never enter the modeling sample.

**Important:** This is a public-data Merton/KMV-style estimate, not Moody's proprietary KMV EDF. It is used as the direct DSW Distance-to-Default analogue.

**Files created:** `src/preprocessing/dd.py`, `data/raw/merton/ccm_compustat_matched.parquet`, `data/processed/merton_dd_monthly.parquet`, `data/processed/merton_dd_quarterly.parquet`

---

## Chunk 9 — Master Feature Dataset Construction

**Goal:** Merge all pipeline outputs into a single panel; enforce train/test wall.

**Steps:**

1. Create `src/preprocessing/master.py`
2. Base panel: all unique `gvkey` × `quarter_end` combinations from Compustat (2000–2024)
3. Build the master panel from timing-safe local processing outputs in this order: Compustat → Equity/CRSP → Merton DD → Downgrades → IBES → DealScan → Macro/market series
4. Use the primary issuer-rating construction for the master-panel downgrade labels; retain the worst-active-issue robustness construction as separate fields for later sensitivity analysis
5. Left-join analyst data and retain firms without IBES coverage; add analyst-missingness indicators rather than requiring IBES coverage for sample inclusion
6. **Training universe: rated firms that have any DealScan loan - not Term Loan B borrowers only.** The
   original instruction ("only leveraged loan borrowers in final modeling scope") is ambiguous, and read
   strictly it discards most of the labels. Measured on the archive:

   | Training universe | Firm-quarters | Firms | Train | Test |
   | --- | --- | --- | --- | --- |
   | All rated firms | 195,757 | 2,440 | 137,969 | 57,788 |
   | Rated + any DealScan loan | 166,009 | 2,018 | 117,946 | 48,063 |
   | Rated + TLB active at 2019-Q1 | 14,826 | 183 | 10,443 | 4,383 |

   Restricting to any DealScan loan costs 15% of the panel and keeps the leveraged-loan framing honest.
   Restricting to TLB borrowers costs 92% and leaves roughly 2,000 positive events in training. The portfolio
   is an application of the model, not its training universe, so the TLB restriction belongs in Chunk 12
   only. Train on rated + any DealScan loan; report the TLB-only fit as a robustness check if desired.
7. Enforce exclusions: SIC 6000–6999, 4900–4999 (double-check after merge)
8. Apply winsorization bounds (computed on train set 2000–2018 only) to any features not already winsorized
9. Add `in_train` flag: 1 if quarter_end ≤ 2018-Q4, 0 otherwise
10. Enforce point-in-time joins:
   - Compustat by `rdq`
   - IBES by `statpers`
   - CCM by `linkdt`/`linkenddt`
   - Ratings by known rating date
   - CRSP through quarter-end only
   - Macro series using observations available by quarter-end
11. Preserve the complete but targeted candidate feature set before model selection/elimination. Organize it into:
   - DSW-analogue core: estimated Merton Distance-to-Default from Chunk 8, trailing 12-month firm stock return, current 3M Treasury yield, and trailing 12-month S&P 500 return
   - Additional borrower fundamentals: interest coverage, profitability, liquidity, cash ratio, and market-to-book
   - Analyst variables: EPS consensus level, revision direction/magnitude, dispersion, analyst count, surprise if included, and missingness indicators
   - Loan variables: loan spread, covenant-lite flag, loan size, and remaining maturity
   - Targeted macro/credit extensions: current 3M Treasury yield, 10Y Treasury yield, term spread, Fed funds rate and/or change, GDP growth, unemployment level/change, CPI YoY, `AAA10Y`, `BAA10Y`, VIX, CRSP value-weighted market return, CRSP equal-weighted market return, and trailing 12-month CRSP market return
   Do not include `BAA10Y - AAA10Y` by default in the raw feature set; it may be tested later as a derived feature, but it is a linear combination and should not be treated as inherently new information.
12. Save full panel to `data/processed/master_panel.parquet`
13. Save separate `data/processed/train.parquet` and `data/processed/test.parquet`

**Files created:** `src/preprocessing/master.py`, `data/processed/master_panel.parquet`, `data/processed/train.parquet`, `data/processed/test.parquet`

---

## Chunk 10 — Downgrade Prediction Model Training and Validation

**Goal:** Train XGBoost and LightGBM downgrade prediction models at 1q, 4q, 8q horizons with walk-forward CV; benchmark against logistic regression.

**Steps:**

1. Create `src/models/downgrade_model.py`
2. Walk-forward CV setup:
   - Expanding window: train on all data up to year Y, validate on year Y+1
   - Folds: 2000–2005 train / 2006 val, ..., 2000–2017 train / 2018 val
   - Never use data after 2018 for any model selection decision
3. For each horizon (1q, 4q, 8q) × each model (XGBoost, LightGBM, LogisticRegression):
   - Target: `downgrade_{horizon}` binary flag from Chunk 5
   - Tune hyperparameters via CV (learning rate, max depth, n_estimators for GBM; C for logistic)
   - Initial feature set: all candidate columns in train excluding target columns and identifiers
   - No feature selection step may use 2019–2024 test data
4. Benchmark specification:
   - DSW-analogue baseline features: estimated Merton Distance-to-Default from Chunk 8, trailing 12-month firm stock return, current 3M Treasury yield, and trailing 12-month S&P 500 return
   - Distance-to-Default is estimated in Chunk 8 using a public-data Merton/KMV-style implementation following Bharath and Shumway (2008); this is not Moody's proprietary KMV EDF.
   - Expanded DynamicPD-CLO features: DSW-analogue baseline plus the additional borrower fundamentals, analyst variables, loan variables, `AAA10Y`, `BAA10Y`, VIX, CRSP market returns, and targeted macro/rate extensions defined in Chunk 9
   - Compare the DSW-analogue baseline against the expanded DynamicPD-CLO specification using identical walk-forward validation and report the incremental predictive improvement
5. Feature selection / elimination procedure:
   - Train baseline models on the full candidate feature set using walk-forward CV only
   - Use cross-validated performance, SHAP importance for tree models, missingness diagnostics, and stability of feature importance across folds to identify weak or redundant variables
   - Remove variables only if they are consistently low-importance, unstable, mostly missing, or economically duplicative with stronger alternatives
   - Consider redundancy between correlated rate/spread variables, missingness, stability across walk-forward folds, and economic interpretability before eliminating features
   - For the limited correlated macro/credit variables retained, compare raw-feature models against PCA-factor variants as a robustness check rather than automatically dropping variables
   - Save the selected feature list and eliminated feature list with reasons to `data/output/tables/feature_selection_log.csv`
6. Final model: retrain on full 2000–2018 with best hyperparams and selected features; predict on 2019–2024 test set
7. Evaluation metrics per model × horizon:
   - AUC-ROC (quarterly and annually)
   - Precision-Recall AUC
   - Brier score
   - Out-of-sample log-likelihood
8. Save trained models to `models/dg_xgb_{horizon}.pkl`, `models/dg_lgbm_{horizon}.pkl`, `models/dg_logit_{horizon}.pkl`
9. Save predictions to `data/output/downgrade_predictions.parquet` — columns: `gvkey`, `quarter_end`, `dg_xgb_1q`, `dg_xgb_4q`, `dg_xgb_8q`, `dg_lgbm_*`, `dg_logit_*`
10. Rating aggregation robustness analysis:
   - Rebuild issuer ratings using the worst-active-issue methodology from Chunk 5
   - Rebuild downgrade labels
   - Rerun the final trained pipeline
   - Compare AUC, PR-AUC, Brier score, log-likelihood, calibration, and lead-time metrics
   - The aggregation methodology is pre-specified before model estimation. The robustness construction is used only as a sensitivity analysis and must never be selected based on model performance.

**Files created:** `src/models/downgrade_model.py`, `models/dg_*.pkl`, `data/output/downgrade_predictions.parquet`, `data/output/tables/feature_selection_log.csv`

---

## Chunk 11 — SHAP Analysis

**Goal:** Compute SHAP values for the best-performing downgrade model; produce feature importance and waterfall plots.

**Steps:**

1. Create `src/models/shap_analysis.py`
2. Use `shap.TreeExplainer` on the XGBoost 4q downgrade model (primary model)
3. Compute SHAP values on full test set (2019–2024) using the final selected feature set from Chunk 10
4. Compare final selected-feature SHAP rankings against the pre-elimination baseline feature importances saved during Chunk 10 to document whether removed variables were genuinely low-information or redundant
5. Output:
   - Global feature importance bar chart → `data/output/figures/shap_importance.png`
   - Summary beeswarm plot → `data/output/figures/shap_summary.png`
   - Waterfall plots for top 5 at-risk firms in each stress episode (COVID 2020, rate shock 2022, SVB 2023) → `data/output/figures/shap_waterfall_{firm}_{date}.png`
6. Save SHAP values to `data/output/shap_values.parquet`

**Files created:** `src/models/shap_analysis.py`, `data/output/shap_values.parquet`, figure files

---

## Chunk 12 — Leveraged Loan Portfolio Construction

**Goal:** Select 100 Term Loan B borrowers to form the representative portfolio.

**Steps:**

1. Create `src/clo/portfolio.py`
2. Selection criteria from DealScan + Compustat intersection:
   - Active Term Loan B facility as of 2019-Q1 (start of test period)
   - Non-financial, non-utility
   - Has required core accounting, rating, equity, macro, and loan inputs in the master panel
   - Missing optional analyst features are allowed and retained with missingness indicators
   - Has at least one rating record in downgrade events dataset
   - If > 100 candidates: stratify by industry (2-digit SIC) and loan spread to ensure diversity
3. Equal notional weights (1% each)
4. Save portfolio manifest to `data/processed/portfolio.parquet` — columns: `gvkey`, `company_name`, `industry`, `loan_spread`, `weight`

**Feasibility verified (2026-09-18):** 140 borrowers satisfy TLB-active-at-2019-Q1 + linked gvkey + a rated
bond outstanding, against the 100 required, leaving headroom for the financial/utility exclusion and the
stratification in step 2. This was not a given: the original ticker-only borrower link yielded just 25
candidates, and the pool only reached 140 after the link was rebuilt with name matching and the FISD side was
linked through both the bond-CUSIP8 and issuer-CUSIP6 routes. If the portfolio filters are tightened further,
re-check this count before assuming 100 names are available.

**Files created:** `src/clo/portfolio.py`, `data/processed/portfolio.parquet`

---

## Chunk 13 — Portfolio Credit Quality Aggregation

**Goal:** Construct portfolio-level downgrade risk signal each quarter using ML predictions vs static ratings.

**Steps:**

1. Create `src/clo/portfolio_quality.py`
2. For each quarter-end in test period (2019-Q1 through 2024-Q4):
   - Pull 4q-ahead downgrade probability for each of the 100 firms from ML model
   - Static rating proxy: map the current primary reconstructed issuer rating to the historical 4-quarter downgrade frequency for that rating bucket, computed exclusively from 2000–2018 training observations.
   - Pool downgrade signal = equal-weighted average predicted downgrade probability across portfolio
3. Repeat the portfolio evaluation using the worst-rating construction as a robustness check and report it separately from the primary static benchmark
4. Track concentration: count of firms above 75th percentile threshold (`n_elevated`) per quarter
5. Save to `data/output/portfolio_quality.parquet` — columns: `quarter_end`, `signal_type` (ml/static), `pool_dg_prob`, `n_elevated`

**Files created:** `src/clo/portfolio_quality.py`, `data/output/portfolio_quality.parquet`

---

## Chunk 14 — [REMOVED]

Copula Monte Carlo pricing removed following pivot to downgrade prediction. Portfolio-level aggregation is handled in Chunk 13.

---

## Chunk 15 — [REMOVED]

TRACE DM pipeline removed. DM widening is no longer the primary evaluation ground truth. Early warning tests now compare ML signals directly against actual downgrade dates (Chunk 18).

---

## Chunk 16 — Rating Actions Pipeline (Portfolio Firms, Evaluation Period)

**Goal:** Construct the portfolio evaluation dataset from the downgrade history already built in Chunk 5.

**Note:** Chunk 5 is the master downgrade event pipeline. Chunk 16 filters the FISD-derived primary and robustness downgrade histories to the final 100-firm portfolio and creates lead-time evaluation datasets.

**Steps:**

1. Create `src/preprocessing/ratings.py`
2. Load downgrade history produced in Chunk 5.
3. Restrict the FISD-derived primary and robustness downgrade histories to the final 100-firm portfolio.
4. Per firm-quarter in test period (2019-Q1 to 2024-Q4):
   - `downgrade_flag` = 1 if any downgrade in that quarter
   - `quarters_to_next_downgrade` = forward-looking count (NaN if none in remaining test window)
5. Document coverage: % of portfolio firm-quarters with active rating — flag if < 80%
6. Save to `data/raw/ratings/rating_actions.parquet`

**Files created:** `src/preprocessing/ratings.py`, `data/raw/ratings/rating_actions.parquet`

---

## Chunk 17 — ML Stress Signal Construction

**Goal:** Construct the pool-level ML stress signal from firm-level downgrade predictions.

**Steps:**

1. Create `src/eval/stress_signal.py`
2. For each quarter-end in test period:
   - Pool stress signal = equal-weighted average of ML 4q downgrade probability across the 100 portfolio firms
   - Standardize: z-score relative to 2000–2018 mean and std (train period only)
3. Flag elevated stress: `stress_flag` = 1 if z-score > 2.0
4. Identify top 5 contributors when flag = 1 (highest individual downgrade probability)
5. Save to `data/output/stress_signal.parquet` — columns: `quarter_end`, `pool_dg_ml`, `pool_dg_static`, `stress_zscore`, `stress_flag`, `top5_gvkeys`

**Files created:** `src/eval/stress_signal.py`, `data/output/stress_signal.parquet`

---

## Chunk 18 — Early Warning: Downgrade Lead Time

**Goal:** Measure how many quarters ML downgrade signal leads actual rating downgrades vs static ratings.

**Steps:**

1. Create `src/eval/downgrade_lead_time.py`
2. For each downgrade event in the 100 portfolio firms (from Chunk 16):
   - Find the first quarter where firm's ML 4q downgrade probability crossed a threshold (top quartile of train-period distribution, computed on 2000–2018 only)
   - Compute ML lead time = downgrade quarter − first elevated ML signal quarter
   - Static lead time: find first quarter where static rating-implied downgrade probability crossed same threshold
   - Compute static lead time = downgrade quarter − first elevated static signal quarter
3. Compare distribution of lead times: ML vs static
   - Summary stats: mean, median, % with lead > 0
   - Wilcoxon signed-rank test: ML lead time > static lead time
4. Repeat for three stress episodes separately: COVID 2020, rate shock 2022, SVB 2023
5. Save to `data/output/downgrade_lead_time.parquet` and `data/output/tables/downgrade_lead_time_summary.csv`

**Files created:** `src/eval/downgrade_lead_time.py`, `data/output/downgrade_lead_time.parquet`, `data/output/tables/downgrade_lead_time_summary.csv`

---

## Chunk 19 — [REMOVED]

DM lead time analysis removed. TRACE DM data is no longer the primary evaluation target. Early warning is evaluated directly against downgrade dates (Chunk 18).

---

## Chunk 20 — Regression Test: Incremental Information

**Goal:** Test whether ML downgrade signal predicts forward realized downgrades incrementally over static ratings.

**Steps:**

1. Create `src/eval/regression_test.py`
2. Regression specification (panel, firm × quarter):
   - Dependent variable: `downgrade_flag_{t+1}` (realized downgrade in next quarter)
   - Model 1: downgrade\_{t+1} = α + β₁ · static_dg_prob_t + ε
   - Model 2: downgrade\_{t+1} = α + β₁ · ml_dg_prob_t + ε
   - Model 3: downgrade\_{t+1} = α + β₁ · static_dg_prob_t + β₂ · ml_dg_prob_t + ε (horse race)
3. Standard errors: Newey-West with 4-lag bandwidth (`statsmodels` `sandwich_covariance`)
4. Run for full test period and for each stress episode separately
5. Report: coefficients, t-stats, pseudo-R², whether β₂ significant in Model 3 (incremental information test)
6. Save regression tables to `data/output/tables/regression_results.csv`

**Files created:** `src/eval/regression_test.py`, `data/output/tables/regression_results.csv`

---

## Chunk 21 — Agentic Explanation Layer

**Goal:** When pool stress signal > 2 std, retrieve earnings transcripts and generate risk narrative for top 5 flagged firms.

**Steps:**

1. Create `src/agentic/stress_explainer.py`
2. Trigger condition: `stress_flag == 1` in stress signal (Chunk 17)
3. For each triggered quarter:
   a. Retrieve top 5 firms by downgrade probability contribution
   b. Retrieve earnings-call transcripts, earnings releases, or relevant SEC filings from the prior two quarters.
   c. Retrieve the most relevant excerpts related to credit deterioration, leverage, liquidity, refinancing risk, covenant pressure, or earnings weakness.
   d. Combine retrieved excerpts with SHAP explanations from the downgrade model.
   e. Generate a structured downgrade-risk narrative containing: key risk themes, supporting excerpts, and a qualitative severity assessment (low/medium/high).
4. Log output to `data/output/agentic_logs/stress_{quarter}.json` — human review required before use

**Files created:** `src/agentic/stress_explainer.py`, log files

---

## Chunk 22 — Results Compilation and Visualization

**Goal:** Produce all tables and figures for the paper.

**Steps:**

1. Create `src/utils/visualize.py` and `src/utils/results.py`
2. Figures:
   - Fig 1: Pool ML downgrade probability vs static rating-implied probability over time (2019–2024) with stress episodes shaded
   - Fig 2: Portfolio credit quality evolution — ML vs static signal, `n_elevated` firms per quarter
   - Fig 3: SHAP feature importance (from Chunk 11)
   - Fig 4: Lead time distributions — ML vs static, downgrade events by stress episode
   - Fig 5: Regression coefficients with confidence intervals across stress episodes
3. Tables:
   - Table 1: Model performance (AUC, Brier, log-likelihood) by model × horizon, including the DSW-analogue baseline versus the expanded DynamicPD-CLO feature set
   - Table 2: Downgrade lead time summary statistics
   - Table 3: Regression results (Models 1–3, full period + stress episodes)
4. All figures saved to `data/output/figures/` as PNG (300 dpi) and PDF
5. All tables saved to `data/output/tables/` as CSV and LaTeX (`.tex`)

**Files created:** `src/utils/visualize.py`, `src/utils/results.py`, all figure/table files

---

## Key Constraints (Enforce Throughout)

- **Train/test wall**: winsorization bounds, z-score normalization, downgrade probability threshold for lead time, static rating bucket rates — all computed on 2000–2018 only, applied to 2019–2024
- **Walk-forward CV only** — no random splits at any stage
- **Issuer-level rating aggregation is pre-specified**: The primary hierarchy is: 1. representative senior unsecured issue; 2. estimated senior rating from the most senior remaining issue, shifted by the cross-sectional notching offset (Chunk 5, "Fallback rating decision"). The worst active issue construction is robustness only. Never choose the aggregation rule based on model performance or test-set results. Never use current `amount_outstanding` as a historical aggregation weight.
- **Point-in-time data timing**: Compustat features use `rdq`; IBES consensus features use `statpers`; CCM and Bond-CRSP links must respect valid link dates; ratings use known rating dates; CRSP uses data through quarter-end only; macro observations must be available by quarter-end.
- **IBES linkage and missingness**: Exact normalized CUSIP8 is the confirmed primary IBES linkage. Unmatched IBES firms remain in the sample with missing analyst variables and missingness indicators.
- **Macro/credit scope**: HY OAS and other truncated ICE OAS histories are excluded from the 2000–2024 primary model because accessible histories do not cover the full primary sample.
- **Accounting staleness is capped**: forward-filling the last filing indefinitely put the 75th percentile of
  `accounting_age_days` at 3,318 days. Beyond a 270-day cap, accounting values are nulled and flagged
  `accounting_stale` (the row is kept), and `accounting_age_days` is carried as a feature.
- **Downgrade events come from rating actions, not level changes**: an event is a cut on a bond the agency
  rated in both quarters, or a recorded default; any agency counts. Changes in agency coverage or in the
  representative issue must never create or hide an event (Chunk 5, "Label construction decisions").
- **Issuer ratings use active issues only**: `offering_date <= quarter_end < maturity`, excluding retired,
  defaulted and redeemed issues. Without this the rated universe never loses a firm and later-year downgrade
  rates are understated.
- **Agency availability is a control, not an assumption**: S&P coverage in FISD falls from 38k records in 2000
  to 4.4k in 2024 while Moody's holds steady, so cross-agency rules change meaning over the sample. Carry
  per-quarter agency-presence indicators.
- **Loan data comes from the wide `dealscan` table**: the normalised DealScan tables are frozen at mid-2020
  and cannot cover the test period.
- **The borrower-gvkey link is locally built and imperfect**: no WRDS DealScan-Compustat crosswalk exists.
  Ticker matches must be date-validated against loan dates, and `ambiguous_needs_review` rows stay unmatched
  rather than being guessed.
- **One chunk at a time** — confirm complete before proceeding
- **Flag data gaps immediately** — especially FISD rating coverage, valid linkage coverage, Compustat/CRSP timing gaps, and IBES exact-CUSIP8 coverage in the final matched universe
