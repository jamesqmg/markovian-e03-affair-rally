<!-- Public copy of the locked pre-registration (2026-11-12). Internal file paths and the name of the viral reel's creator are redacted; nothing else is edited. -->
---
title: E03 "Is infidelity priced in?" — pre-registration
status: LOCKED 2026-11-12 (approved by the author). Any change after this goes in an erratum section with a date; no edits in place.
date: 2026-11-12
---

# E03 pre-registration

## 0. Disclosure: what had been seen before this file

- **Names and dates only** for the viral reel's 15 cases (from his reel screenshots). No returns.
- **Raw daily closes, Nov 3–11 2026**, for NVDA and 26 US chip peers. These were used to fact-check the hook line ("every chip stock went up except one").
  On Nov 9: NVDA −4.27%, 26 of 27 peers up, SOX +6.85% (Yahoo closes, not redistributed).
  So the sign and rough size of Nvidia's raw move is **known**, and the Nvidia arm is not blind. Its contribution is the counterfactual and the
  placebo rank, not the direction.
- Nothing from the Gentry data and no historical event returns have been seen.

## 1. Claims, verbatim

- **C1, the viral claim** ("affair rally"): when a CEO's affair goes public, the stock goes **up**. Source: as tested by the viral reel's creator,
  "Is there Alpha behind a Horny CEO?", Instagram, 2026-11-09.
- **C2, the literature:** shareholders lose about 1.8% at the news, and 3.8% when the CEO is the one accused (Cline, Walkling & Yore, 2013 draft p.4;
  *JFE* 127(2), 2018). This is the direction we expect, and it's stated before the test.

## 2. Arms (fixed now and reported separately; never pooled after the fact)

| Arm | Events | Role |
|---|---|---|
| **A, claim** | Bezos/AMZN 2019-01-09, Gates/MSFT 2021-05-16, Brin/GOOGL 2013-08-28, Chey/SK Hynix 2015-12-29, as the claim lists them | Descriptive only (n = 4). Gates and Brin were **not CEOs** at the time (`fact-check.md` #8), so we report them flagged as well as excluded |
| **B, the viral reel** | Its 15 cases exactly as listed in the reel, its dates | Replication |
| **C, systematic (primary)** | Gentry et al. CEO Dismissal DB, latest Zenodo version, **`departure_code == 4`** ("Involuntary – CEO dismissed for legal violations or concerns", codebook v10Oct2024) | **The test.** Two cuts: **C-all** (every code 4) and **C-sex** (code 4 where `notes` matches, case-insensitive: `affair, relationship, romantic, consensual, sexual, harassment, misconduct, personal conduct`). C-sex is primary, because it's the claim's own subject; C-all is secondary |
| **D, Nvidia** | 2026-11-08 letter → first trading day **2026-11-09** | Single-firm case |

**Event date (arms B–C).** Use `date_departure_announced` when it exists. If it's missing, use the earliest dated news or 8-K (Item 5.02) found by a
**fixed search**: EDGAR full-text search, then a single news search. If neither turns up a date, **exclude the event and count it**. The codebook says
announcement dates "are not checked randomly", so we report N with and without dates, and the share of dated events by year. If more than 30% of C-sex is
undated, the verdict is labelled "selection risk".

**Exclusions** (from `selection.md`): flag any earnings or M&A within ±3 trading days and report results with and without those events.
Events with fewer than 120 trading days of pre-event history are excluded. Non-US listings are reported separately using French's regional factors.

## 3. Primary metric and test (arm C-sex)

- **Abnormal return:** FF5 daily (Ken French), with OLS betas from an estimation window of trading days [−250, −30] (at least 120 valid days).
  The market model (CRSP-VW proxy = French Mkt) is secondary.
- **Primary window: CAR[0, +1].** Secondary windows: [−1, +5] and [0, +20].
- **Primary test:** Kolari–Pynnönen-adjusted BMP, two-sided, α = 0.05. Secondary tests: Corrado rank and the generalized sign test.
- **Power:** compute the minimum detectable CAR[0, +1] at 80% power from the realised N and the cross-sectional σ, and report it with the result.

**Verdict on C1** (the words used in the episode, mapped to the endings already written in `script_v3.md` L19):

| Result (C-sex, CAR[0, +1]) | Verdict | Ending |
|---|---|---|
| CAR > 0, p < 0.05 | **Held** | Positive |
| 95% CI entirely below 0 | **Contradicted** (the paper's direction) | Negative |
| CI includes 0, and MDE ≤ 2% | **Not supported** | Nothing |
| CI includes 0, and MDE > 2% | **Can't tell** (too few events) | A fourth ending, written now: "so honestly, with this few scandals nobody can tell yet, including the internet" |

## 4. Arm D: Nvidia

- **Primary:** synthetic control (CausalPy, Bayesian weights on the simplex). The donor pool is the 26 US chip stocks in the fact-check file;
  ETFs and indices are excluded.
- **Pre-period:** the 250 trading days ending **2026-10-28**, which stops before the first disclosure stage (NYT, Oct 29).
- **Effect:** the cumulative gap over **Nov 9–11** (event day + 2). Secondary windows: Nov 9–13; side windows starting Oct 29 and Nov 5. Nothing from Nov 17 on
  (earnings Nov 17 after the close; hearing Nov 18).
- **Inference:** placebo-in-space. Run the identical method with each donor treated in turn, then rank Nvidia's post/pre RMSPE ratio among the 27.
  Nvidia is "outside the fan" if it ranks 1st of 27 (p = 1/27 ≈ 0.037). Placebo-in-time (every Tuesday-to-Thursday triple in the pre-period) is secondary.
- **Baseline shown alongside:** market-model abnormal returns against SMH and against the S&P 500.
- **Confound:** same-day Megaspeed reporting (CNBC, Nov 9) can't be separated from the affair. The verdict covers "the Nov 8–9 news package". That's said on screen (L18).
- French daily factors end around Sep 2026, so the Nvidia arm uses no French factors.

## 5. Replication (arm B)

the viral reel's specification as far as it's visible: monthly FF5 (French factors in USD, regional where needed), `r = α + δ·Post + β·FF5`,
36 months before vs 36 after, Newey-West with 3 lags, his 15 dates. Any choice we can't see is written into the notebook as an **assumption** *before* the run.
- **"Replicated"** means our δ has the same sign as his, and the same significance verdict at 5%, if his estimate is visible in his reel.
- If his number isn't visible, we report "his spec, our run" and say so.

## 6. Secondary analyses (descriptive; they don't change the verdict)

- Calendar-time portfolio (12/36 months, FF5 + UMD, Newey-West), on C-all and C-sex.
- Fama–MacBeth on S&P 1500 monthly returns: a "code-4 exit in the last k ∈ {1, 12} months" dummy plus size, B/M, profitability, investment and momentum characteristics.
- Bayesian hierarchical model of CAR[0, +1] (Student-t, partial pooling by arm and by `ceo_left`), reporting P(CAR > 0).
- GARCH(1,1)-t standardized ARs and abnormal volume.
- Leave-one-out on arms A and B. 1,000 random placebo dates for the same firms.

## 7. Multiple testing

There is **one primary cell** (C-sex, CAR[0, +1]) and one primary Nvidia statistic (the placebo rank), both reported unadjusted.
Every secondary cell (arms × windows × tests) is reported in full with a Holm adjustment, and none of them can change an episode verdict.
This is the first test in this series (count of tests run at lock: 1).

## 8. Data and code

- Factors: Ken French Data Library (daily and monthly; FF5 + Mom; US + regional).
- Prices: Yahoo (yfinance), cross-checked on 3 names against Tiingo.
- Events: Gentry DB (Zenodo, CC BY 4.0, cite the version) plus the arm A/B dates with source URLs in `03-TESTS/E03-affair-rally/events.csv`.
- Code: this repo, in a `uv` env on Python 3.12 (pandas, statsmodels, linearmodels, arch, pymc, causalpy, yfinance).
- Every on-screen number goes to `results_card.json` with its notebook cell and label.
- No raw vendor data is in the public repo: only fetch scripts and checksums.

## 9. What would make us wrong (written before the run)

- If the code-4 sample turns out to be mostly expense and HR cases, C-sex may be small and C-all may not be "affairs". The script says "fired over conduct" for that reason.
- Staged leaks, where markets learn before the dated report, push the effect into the pre-window. [−1, +5] and the Nvidia side windows cover this partly.
- Survivorship in Execucomp (S&P 1500 only): small caps, where scandals may matter more, are missing. This is stated as a scope limit.

---

## Erratum E1 (2026-11-12, written after dates were built and **before any return was computed**)

1. **Data version:** Zenodo record 23328926 (v7, files dated 26 Nov 2025). Code 4: N = 237. The locked keyword rule gives C-sex N = 34.
2. **"Fixed search" for undated events, as run:**
   - Candidate dates: an 8-K listed in the dataset's own `eight_ks` column with Item 5.02 (Item 5 before 2004), and dated citations in its `sources` column.
   - Both must fall within **±7 calendar days of `left_ceo_office`**. The earliest candidate wins.
   - A ±60/+10-day window was tried first. It picked up unrelated 5.02 filings (e.g. Priceline's Mar 10 for an Apr 28 exit), so it was tightened before any return was computed.
   - Result: **27 of 34 C-sex events dated** (20.6% undated, under the 30% line, so no "selection risk" label).
3. **Prices:**
   - Yahoo is primary.
   - **Tiingo adjusted closes** are used only for firms Yahoo lacks (delisted or renamed): GRA, WDR, CRR, VIAC (= CBS Corp), IGTE, ADEA (= Tessera/Xperi), WWE, BKS.
   - No price source exists for Riggs, Enesco, Kemet or Comverse, so they are excluded and counted.
   - RH (fired before its Nov 2012 IPO) fails the 120-day history rule.
4. **Keyword false positives** (e.g. Riggs: "business relationships" means money laundering): the locked rule stands for the primary result. A manual reading of each `notes` field is reported as a **sensitivity check only**.
5. **C-all tickers** come from the SEC CIK map only (60 of 237). C-all is secondary and is reported with its coverage.
6. **Day 0** = the first trading day on or after the event date.
7. **Known oddity, kept by rule:** Johnson Controls (Molinaroli). The dataset's own announcement date is 2016-01-27, against an exit on 2017-09-01. Flagged in the audit.

---

## Erratum E2 (2026-11-12): disclosure of everything after E1

**Order of events, stated plainly:**
1. Arm C was run **once on an engine that had not been reviewed yet** (C-sex FF5 CAR[0,+1] = −1.92%, iid CI [−3.72, −0.13], KP p = 0.087).
2. The reviewer then found implementation defects:
   - relative days counted on the stock's own rows instead of the trading calendar;
   - partial windows not rejected;
   - KP rbar not clamped at 0 (could give NaN);
   - the sign-test p̂ taken from daily rather than window sums;
   - a normal-quantile MDE;
   - an iid CI next to a KP-adjusted test.
3. All of these were fixed, and tests were added, including a null-size Monte Carlo. The run was repeated.
4. The fixes moved **no** C-sex date and left the primary estimate at −1.92%. **The verdict rule was not changed.** Both runs are reported.

**Other changes after E1:**
- **Yahoo batch downloads dropped live tickers at random** (CBOE, AMZN, TXN). A per-ticker retry was added. SNCR and GES (delisted 2026) moved to Tiingo under the E1 §3 rule.
- **Data-layer fixes:**
  - the Item 5 cutoff is now **2004-08-23** (the date Item 5.02 began), not the calendar year 2004;
  - pre-2001 `.txt` 8-K URLs are parsed;
  - failed EDGAR fetches are no longer cached.
  - Effect: 5 C-all dates changed (HCA, Stewart & Stevenson, Genentech, Reynolds & Reynolds, Advanced Marketing Services). **0 C-sex dates changed.**
- **Which CI the verdict uses:** §3 says "95% CI" without saying which one. The verdict uses the **KP-widened CI**, the one consistent with the primary test. The iid CI is reported next to it. Both exclude 0 for C-sex CAR[0, +1].
- **Manual audit** (E1 §4), reported as a sensitivity check only (`03-TESTS/E03-affair-rally/out/csex_manual_audit.csv`). Each of the 24 usable events was classed from the dataset's own `notes`:
  - **S**, clearly sex or relationship: 16;
  - **U**, conduct not specified: 7;
  - **N**, not sexual: 1 (FirstEnergy, a bribery probe; the keyword hit was "External Affairs").
- **Arm B assumptions**, fixed in `e03/viral_test.py` before its first run:
  - A1: the event month is excluded;
  - A2: returns are month-end;
  - A3: BP uses Europe FF5, and SK Hynix is converted KRW → USD with Asia-Pac ex Japan FF5;
  - A4: post windows are cut at Sep 2026.
- **Arm D, as run:**
  - The outcome is daily log returns. Weights come from CausalPy `WeightedSumFitter` (Dirichlet weights on the simplex).
  - Placebo donor pools exclude NVDA. The RMSPE ratio uses Nov 9–11.
  - The Nov 9–13 secondary window is **incomplete until the Nov 13 close**, and will be re-run then.

---

## Erratum E3 (2026-11-12 06:55 UTC, i.e. 01:55 ET, **before the Nov 12 US open**)

1. **Data defect:** the cached NVDA "Nov 11 close" ($174.68) was an intraday price, pulled while the session was still open. The final close is $173.73. Arm D and the hook figures are **re-run on final closes**, and both runs are reported.
2. **New disclosure stage, declared before its data exists.** After the Nov 11 close (about 17:35 ET):
   - the NYT reported a **second affair**, with a direct report;
   - the WSJ reported that the board is **preparing to oust Huang**;
   - NVDA traded about −8.6% after hours.

   A side window, **Nov 12 alone** (the first session after these reports), is added for arm D. The **primary window (Nov 9–11) is unchanged.** The secondary window (Nov 9–13) now contains this stage by construction, so it is reported as "first admission + second affair + ouster report", never as the effect of the affair alone.
3. **If Nvidia announces Huang's departure before the Nov 17 earnings:** the announcement day is a further side window, and Nvidia becomes a code-4-type event. It is **not** added to arm C, which is fixed to the Gentry data.
4. **No other changes.** The fixed method also nets out sector-wide moves before Nov 8, e.g. the Nov 5 chip sell-off (NVDA −9.3%, AMD −17%): the synthetic twin is built from the same peers, and the event study removes factor moves.

---

## Erratum E4 (2026-11-12 ~07:45 UTC, written **before any §6 secondary analysis was run**)

1. **Aborted run, disclosed.** At 07:35 UTC a `synth --last 2026-11-12` run was started by mistake: "PST" on this machine is Philippine time,
   so the Nov 12 US session had not opened. It was stopped within 2 minutes, before any placebo finished. No output file was written, its
   pre-open price cache was deleted, and its manifest entry was removed. `e03/synth.py` now refuses `--last D` before 16:15 ET on day D.
2. **Fama–MacBeth: not run under this erratum.** It needs monthly characteristics for the whole S&P 1500, including firms that died, back to 1993.
   That's CRSP/Compustat; free fundamentals (SEC XBRL) only start in 2009. With about 0.6 code-4 events a month, a monthly cross-sectional slope on a
   "recent exit" dummy is also undefined in most months. Any later Fama–MacBeth run needs its own erratum **before** a return is computed.
3. **Bayesian hierarchical model (§6), as it will run.**
   - `ceo_left` does not exist in Gentry v7: every code-4 CEO left by definition. So pooling is **by arm only**. The groups are **disjoint**:
     C-sex (the primary 24), and C-other (usable C-all events whose `dismissal_dataset_id` is not in C-sex).
   - Arms A and B are not included, because they overlap C-sex and use a different design.
   - Model:
     - CAR_i ~ Normal(θ_i, se_i), where se_i is each event's FF5 forecast-error s.d. for [0, +1] (CAR / SCAR from the primary engine);
     - θ_i ~ StudentT(ν, μ_g, τ);
     - μ_g ~ Normal(μ, σ_arm);
     - μ ~ Normal(0, 0.05), σ_arm ~ HalfNormal(0.02), τ ~ HalfNormal(0.05), ν ~ Gamma(2, 0.1).
   - Sampling: 4 chains × 2,000 draws, seed 4. If any R-hat > 1.01, the result is reported as non-converged.
   - Reported: P(μ_C-sex > 0) as locked, its complement, μ, and the shrunk θ_i.
4. **Calendar-time portfolios (§6), as they will run.**
   - Event month m is the month of day 0. Holding months are m+1 … m+h, with h ∈ {12, 36}.
   - A firm's monthly return is its daily simple returns compounded, from the same price sources as the primary.
   - Portfolios are equal-weighted, and every month with at least one firm is used.
   - The portfolio's excess return over French RF is regressed by OLS on monthly FF5 + UMD, with Newey–West HAC (6 lags), through Sep 2026.
   - Arms: C-sex and C-all. Reported: monthly α, t, p, number of months, and mean firms per month.
5. **GARCH(1,1)-t standardized ARs (§6).**
   - Betas are the primary's FF5 OLS betas, from days [−250, −30].
   - A zero-mean GARCH(1,1) with Student-t errors (`arch`) is fitted to that model's residuals over [−250, −1], scaled by 100.
   - The 1- and 2-step variance forecasts give h₀ and h₁, and GARCH-SCAR = CAR[0, +1] / √(h₀ + h₁).
   - Test: the cross-sectional t, KP-adjusted with the primary's r̄. A fit that fails to converge is dropped and counted.
6. **Abnormal volume (§6).**
   - v = log(1 + volume). The abnormal value is v minus its mean over days [−250, −30], divided by that window's s.d.
   - Reported: the mean on day 0, on day +1, and over [0, +1], each with a one-sample t.
   - Volumes: Yahoo for Yahoo-priced firms; Tiingo `adjVolume` for Tiingo-priced firms (E1 §3).
7. **Leave-one-out (§6).** On arm B's per-case Δα (`out/arm_b_viral.json`, no new data), and on its 4 claim cases (arm A). Reported: the minimum and
   maximum of the leave-one-out means.
8. **Random placebo dates (§6).**
   - Firms: the same 24 C-sex firms. Draws: 1,000, seed 12.
   - In each draw, every firm gets one random trading day from its own history. The day must have at least 250 trading days before it,
     must not be within ±60 trading days of its true day 0, and must leave CAR[0, +1] inside the French daily data.
   - Statistic: the mean FF5 CAR[0, +1] across firms, from the same engine.
   - Reported: the share of draws ≤ the observed −1.92% (one-sided, in the C2 direction), and the two-sided share with |mean| ≥ |observed|.
9. **Multiple testing.** Holm is applied across every secondary p-value above, as §7 requires. None of them can change the verdict.

---

## Erratum E5 (2026-11-12, after code review and **still before any §6 secondary analysis was run**)

Four independent code reviews of the E4 modules led to these clarifications. All are fixed before any secondary number exists.
1. **C-other (E4 §3):** C-other excludes **every** C-sex keyword-cut id (all 34), not only the 24 usable ones. So a sex-cut event that is
   unusable on the C-sex price map can't appear in C-other. The code asserts that C-sex has n = 24 and that the arms are disjoint. R-hat is checked on **every** sampled
   variable (including the latent per-event effects), and a NaN R-hat counts as non-converged. Sampler: tune 2,000, target_accept 0.95.
   ν's Gamma(2, 0.1) is in rate form (prior mean 20). With two arms, σ_arm is prior-dominated, and this is stated next to the result.
2. **Calendar-time (E4 §4):**
   - Monthly returns come from **month-end adjusted closes**. Compounding daily returns would turn a missing-price day into a 0% return.
   - Weighting is **per event**, the calendar-time convention. A firm with two events in a window counts twice; this affects only C-all, AZTA and MTEX at h = 36.
   - p-values use the HAC normal approximation. The run asserts that the factor sample ends Sep 2026.
3. **GARCH (E4 §5):** in each case below, the event is **dropped and counted**:
   - the last residual is not day −1;
   - fewer than 120 residuals;
   - a non-converged fit;
   - a non-finite or non-positive forecast.

   Per-event α, β and ν are reported. `rescale=False`.
4. **Volume (E4 §6):** the formula is unchanged (log(1 + volume)). Zero-volume days in the estimation window and on days 0/+1 are counted and reported per event.
5. **Placebo (E4 §8):** an admissible-but-invalid placebo day (no engine result or no CAR) is **redrawn** (rejection sampling), so every draw holds all 24 firms; rejections are counted.
   The raw shares are reported **and** the (k+1)/(N+1) p-values, which are the inputs to Holm. Two limits are stated: draws are independent across firms (no calendar
   clustering in the null), and a placebo estimation window may contain the firm's true event.

---

## Erratum E6 (2026-11-12, ~23:00 PHT, before any Nov 12 price is pulled)

1. **Event.** Nvidia's board announced Huang's resignation as president, CEO and director, **effective immediately, on Thu Nov 12 2026** (CBS News, updated 09:30 ET; Axios, Nov 12). No reason was given in the company's statement.
2. **Applying E3 §3 as written.** The announcement day is a further arm-D side window. Day 0 = **Nov 12** (first session on or after the announcement). That is the **same day** as E3 §2's Nov 12 side window (the first session after the Nov 11 ouster report), so the two can't be separated. The Nov 12 window is reported as **"ouster report + resignation"**, with no effect attributed to either alone.
3. Nvidia is now a code-4-type event. Per E3 §3 it is **not** added to arm C (fixed to the Gentry data).
4. **Unchanged:** the primary window (Nov 9–11) and its result (gap −19 log pts ≈ −17%, placebo rank 1 of 27). They were computed before the announcement. The secondary window (Nov 9–13) now contains the resignation by construction and is labelled "first admission + second affair + ouster report + resignation".
5. **Descriptive hook number (not a test):** the raw market-value change shown in the hook runs **Nov 6 close → Nov 12 close** , labelled "raw, includes all news that week, including the resignation".
