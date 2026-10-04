# Findings log

One line per finding, with the script that produced it. Everything under a numbered block
was checked by running that script on 2026-10-03; numbers are on the 10,000 historical rows
unless stated. "Not checked" items are at the end. Run a script with
`.venv/Scripts/python.exe analysis/aNN_<name>.py`; figures land in `figures/analysis/`.

## 1. Each column (`a01_columns.py`)

- No missing values, no duplicate rows, ids unique. The two files share no id and together
  cover C000000 to C013999 exactly: one generated population of 14,000 split at random.
- R score: 2 decimals, 16.78 to 40.00, inside the documented 15 to 40. One candidate sits
  exactly at 40.00 (a clip); no pile-up anywhere else.
- Income: whole dollars, 13,236 to 347,143, right-skewed (log-normal, see block 3).
  Hours: whole numbers 1 to 29. Distance: 1 decimal, 0.1 to 786 km. No implausible values.
- Same distributions in both files: Kolmogorov-Smirnov p from 0.35 to 1.0 on the numeric
  columns; region, postal prefix and first generation chi-square p > 0.4. Program mix
  differs slightly (p = 0.03; Genie 22.9% vs 24.6%), which does not matter since program
  plays no part anywhere (block 5). Figure `a01_fichiers.png`.
- The committee granted 39.94% of the history: the historical rate already sits at the 40%
  budget.

## 2. Columns and region (`a02_region.py`)

- Group gaps in pooled standard deviations (Eloignee minus Centre): log distance +1.79,
  hours +1.06, log income -0.68, first generation +0.38, R score -0.22.
- Regions inside a group are interchangeable: ANOVA across the regions of a group gives
  p > 0.2 for every column and for the committee's decision. The five regions are two
  blocks, nothing finer.
- The postal prefix determines the region (18 prefixes, each in one region). Prefixes
  inside a region: 6 of 30 ANOVA tests fall below 0.05 where 1.5 are expected, so small
  prefix-level differences may exist, but on no consistent column, and the prefix adds
  nothing significant to the committee model (block 5, p = 0.09). Treated as negligible.
- Proxy ranking, AUC for guessing the group from one column: postal prefix 1.00, distance
  0.998, hours 0.81, income 0.69, first generation 0.59, R score 0.56, program 0.55.
  Figure `a02_proxys.png`.
- Distance alone is the region: "more than 60 km" classifies the group at 98.5%. All
  numeric columns together: AUC 0.999. Without distance: AUC 0.85. Removing region and
  postal code therefore removes nothing a model cannot rebuild.

## 3. Columns together (`a03_dependence.py`)

- Pooled, the columns look linked (hours and distance 0.46, income and distance -0.29).
  Inside a group every correlation vanishes: the largest is 0.036, at the noise level
  (0.032). Every dependence between columns runs through the group and nowhere else.
- Program is likewise independent of every other column inside a group; only its mix
  differs between groups (Genie 25% vs 19%).
- Shapes inside a group: R score normal (mean 27.99 vs 27.33, sd 3.0 both); log income
  normal (11.15 vs 10.84, sd 0.42 and 0.44); hours a Poisson count (mean 8.97 and 13.02,
  variance equal to the mean); first generation a coin flip (26.5% vs 44.4%); log distance
  left-skewed. Figure `a03_profils.png`.
- Consequence: there is no hidden structure to recover (no latent ability shared by R score
  and hours, no need index). An applicant is a group plus independent draws.

## 4. The committee's decision (`a04_committee.py`)

- The decision is an additive logit on four things: R score +1.38 per point (SE 0.03), log
  income +1.78 (0.09), hours +0.20 per hour (0.012), remote group -1.94 (0.10).
- In R-point equivalents: being remote costs 1.40 R points (bootstrap 95%: 1.28 to 1.54);
  one sd of income is worth +0.58; one sd of hours +0.55. Figure `a04_poids_comite.png`.
- The shape is a slope, not a cutoff, and has no interaction: squares and R x income
  (p = 0.34), different slopes per group (p = 0.56) and five regions instead of two groups
  (p = 0.75) add nothing. Log income fits better than dollars; logit better than probit.
  Figure `a04_courbe_comite.png`: the same curve, shifted.
- The penalty is constant: re-estimated on slices of R score, income, hours and first
  generation it stays between -1.6 and -2.3 with overlapping errors; the three remote
  regions get -2.00, -2.01, -1.89 against Montreal, Capitale-Nationale -0.05.
- What nothing explains: AUC 0.957, accuracy 88.8%, equal to the ceiling this logit implies
  (88.7%). The rest behaves like noise worth 1.31 R points. The production model's 88% is
  that same ceiling: no estimator can agree more with the committee, and gradient boosting
  on every column (cross-validated AUC 0.954) does not beat the four-term logit.
- Borderline applicants (probability between 10% and 90%): 36.6% of applications. The R
  score at which the committee grants one time in two is 28.11 in the centres and 29.31 in
  the remote regions, at each group's median income and hours.
- What the penalty is worth: remote grant rate 27.3% as recorded; 42.9% with the penalty
  removed (about 620 grants per 4,000 remote applications; 95%: 14.1 to 17.1 points);
  47.8% if income is ignored as well, against 48.4% in the centres. So of the 21-point
  gap, about 15.5 points are the penalty, about 5 the income preference, under 1 the rest
  (the R-score difference is offset by the hours the remote applicants work).
- With distance in the model the group coefficient reads -2.15 (an earlier figure of ours):
  distance and group are nearly the same variable, so that specification
  splits one effect in two. The four-term model is the one to quote.

## 5. What the committee ignored (`a05_ignored.py`)

- First generation (p = 0.49), program (p = 0.98), region inside the group (p = 0.75) and
  postal prefix inside the group (p = 0.09) add nothing to the committee model. Distance
  is borderline (p = 0.03 in km, 0.08 in log) and tiny.
- Raw grant rates are flat in first generation (48.3% vs 48.5% in the centres, 27.2% vs
  27.4% remote) and in program. Figure `a05_colonnes_ignorees.png`.
- First-generation applicants have the same R score, income and hours as the others in
  their group. An independent standard that counts first generation would be using a column
  the committee never looked at and that is unrelated to everything else.

## 6. The synthetic design (`a06_design.py`)

- Reconstructed generator: draw the group, draw each column independently from its
  group distribution (block 3), then the committee decides by the logit of block 4. 200
  simulations of it reproduce the actual grant rate of both groups in every R band.
- Since the columns are independent inside a group, any standard of merit or need can only
  be a function of the columns themselves (plus noise we cannot see). The candidates differ
  in who they call deserving, at 40% (figure `a06_etalons.png`):
  - R score alone: 43.4% of the centres, 35.2% of the remote regions.
  - Committee without the group penalty (R, income, hours): 42.4% vs 36.4%.
  - Committee without penalty or income (R + 0.145 per hour worked): 40.2% vs 39.6%. The
    hours the remote applicants work (4 more, worth 0.59 R points) cancel their R-score
    deficit (0.66) almost exactly.
  - Need-weighted standards flip the sign: up to 28.6% vs 57.2%.
- The committee rewarded hours worked, which favours the remote group. A bias against
  remote regions would not put that in; it reads as part of the designer's notion of merit
  or effort. This is an inference, not a measurement.
- The mild candidates label 4% to 9% of applicants differently from one another; the
  need-weighted ones 13% to 31%. Utility points depend on getting this right.

## 7. The 4,000 (`a07_candidates.py`)

- Same population: per group, every column has the same mean and distribution as in the
  history (smallest of 10 KS p-values 0.36); within-group correlations are again at noise
  level. 59.3% Centre, 40.7% Eloignee.
- The baseline on the 4,000: grant rate 39.05% (1,562 grants, in budget); 46.8% Centre vs
  27.8% Eloignee, parity gap 0.190; per region 47.1 and 46.3 against 27.0, 26.1 and 29.7.
  Figure `a07_baseline_regions.png`.
- Top 40% by R score on the 4,000 starts at 28.54 (28.48 on the history).
- Preview of the published anchor (baseline equal-opportunity gap 0.270), assuming it was
  measured on these 4,000, standard at 40%, standard error about 0.02
  (figure `a07_ancre.png`):
  - close to 0.270: R + 3 points first generation 0.245, R + hours 0.296, R with low income
    mildly favoured 0.310, R ranked within group 0.306;
  - somewhat low: committee without penalty 0.230, R alone 0.225, R + 1 point first
    generation 0.222 (2 to 3 standard errors);
  - ruled out: R + income 0.127; need-weighted standards 0.39 to 0.40.
  The gap moves a lot with the standard's rate (R alone: 0.04 at 30%, 0.245 at 50%), so the
  anchor constrains rate and definition jointly, not either one alone.

## 8. The published anchor (`a08_anchor.py`, phase 3)

The panel is now `equialgo/references.py` (R1 to R5 kept, X1 to X3 excluded). All gaps are
the baseline's equal-opportunity gap on the 4,000; standard error 0.016 to 0.022.

- Against the committee's own label (held-out history) the baseline's gap is 0.063
  (true-positive rate 84.7% Centre, 78.5% Eloignee), against the published 0.270. The 0.027
  noted in an earlier attempt is not reproduced.
- At a reference rate of exactly 40%, distance to 0.270 in standard errors: R4b (R + 3 pts
  first generation) -1.1, R3 (R + hours) +1.5, R4c (low income favoured) +1.9, R5 (R within
  group) +2.1, R2 (committee without penalty) -2.1, R4a (R + 1 pt first generation) -2.6,
  R1 (R alone) -2.8. Excluded: X1 (R + income) -9.2, X2 and X3 (need-weighted) +5.9, +6.1.
- The gap rises with the reference rate for every definition except R4b and R4c (R1: 0.144
  at 36%, 0.225 at 40%, 0.249 at 44%). Rule used: a hypothesis stays if some rate inside
  36% to 44% puts the gap within 2 standard errors of 0.270. R1 to R5 all pass: R1, R2 and
  R4a at 42% to 44%, R3 and R4c at 36% to 40%, R5 at 38%, R4b everywhere. X1, X2 and X3
  fail at every rate. Figure `a08_ancre_taux.png`.
- The same gaps on the held-out history differ by at most 0.03, so the conclusion does not
  depend on where the 0.270 was measured.
- The hours weight is a continuum the anchor cannot settle: R + w x hours gives 0.225 at
  w = 0, 0.272 at w = 0.09 and 0.296 at the committee's 0.145.
- Noise in the reference lowers the gap: R + hours with 1 R point of noise gives 0.265
  instead of 0.296, and caps the agreement any model can reach (baseline 0.903 -> 0.870).
- Estimated automated points (assumed formulas, `equialgo/evaluate.py`) on the 4,000: the
  baseline gets 9 to 17 of 35 depending on the hypothesis. The merit-only rule (`e001`,
  strength 0) gets 20.0 in its worst case (R4c) and 20.0 to 30.6 under the hypotheses it
  does not embody; with full group cutoffs the worst case is 25.3 (R4b). Standard error of
  a worst case: about 1.5 points.

## 9. Scorer feedback, first round (`a09_feedback.py`, 2026-10-03)

Six rule files were submitted (`submissions/log.csv`). The scorer returns an accuracy and a
macro F1 against the reference, nothing about equity.

- Returned accuracy: R + hours 94.63%, R within group 92.68%, committee without penalty
  92.53%, R alone 92.28%, R with low income favoured 84.03%, R + 3 pts first generation
  83.73%.
- The feedback is on a subset of unknown size (the scores are cut to two decimals and do
  not reveal it). Our predictions are computed on all 4,000, so a gap of about a point
  between predicted and returned accuracy can be sampling alone (standard error 0.5 point
  for a subset of 2,000, 0.75 for 1,000). An earlier note here claimed the feedback was on
  all 4,000; that was wrong.
- Accuracy and F1 together put the reference rate between 39% and 40.5%: the reference
  labels about 40% of applicants deserving, as assumed.
- No panel hypothesis is the reference exactly: the rule that is the hypothesis would have
  returned 100%. The need hypotheses R4b and R4c miss by 10 points of accuracy and are out.
- The returns are reproduced within 0.5 point by "top 40% of R + w x hours + noise", with w
  between about 0.145 and 0.22 and noise worth about 0.5 R point. The committee's own hours
  weight (0.145, the one file 02 uses) fits at 0.57; the best grid point is 0.18. Given the
  subset's sampling noise, these fits are not distinguishable from one another.
- With that reference, an added income, first-generation, distance or group term of 0.25 R
  point or more does not improve the fit beyond the sampling noise.
- Ceiling: the noise-free version of the fitted reference agrees with it at 95.1%. File 02
  got 94.63%, so the R + hours rule is within half a point of the most any rule can reach.
- Equal-opportunity gap of each file under the fitted reference (estimate, not returned by
  the scorer): R + hours 0.017, R within group 0.024, low income -0.028, first generation
  0.062, committee without penalty 0.086, R alone 0.106. Baseline anchor: 0.270.
- Reading: the reference is the committee's own notion of merit (R score and hours worked)
  without the regional penalty and without the income preference, plus a part we cannot
  observe.

## 10. R + hours in depth, and the program of study (`a10_hours_program.py`, `e003`)

- Hours weight supported by the history: 0.145 R point per hour (se 0.008, 95% interval
  0.130 to 0.161). Probit gives 0.144; fitted inside each group, 0.156 (Centre) and 0.134
  (Eloignee). The effect is a straight line across hours bands. A weight of 0.18 or more
  is outside what the committee's decisions support.
- The weight that calls the same share of each group deserving is 0.145 on the 4,000 and
  0.16 on the history: the fitted weight is also the parity-neutral one.
- The weight matters little: going from 0.145 to 0.18 changes 1.2% of the decisions on the
  4,000, to 0.13 or 0.16 under 1%. At most that much accuracy is at stake.
- Proxy effect, measured: fitted without the group, the hours weight falls to 0.059, and
  to 0.024 without group and income. Hours worked are a proxy of the remote regions, so a
  model that does not see the group charges the penalty to the hours. This is why the
  committee is fitted with the group and the penalty then switched off.
- Experiment `e003` confirms it on decisions. `committee_merit` (fit with the group, keep R
  and hours) reproduces the R + hours rule exactly, parity gap 0.000. Trained on the same
  label with only R and hours as inputs, logit, gradient boosting and random forest leave a
  parity gap of 0.06 to 0.07; with income as well, 0.10 to 0.12; with every column except
  region, 0.16 to 0.21, the production model's level.
- The estimator matters far less than the feature set (T2): on a given feature set the
  three estimators end within a few points of one another on the board, while the feature
  sets are 5 to 15 points apart.
- One interaction is significant in the committee: R x hours (p = 0.009 among 8 tests; the
  smallest expected by chance is 0.11). Hours count a little less for high R scores. At the
  cutoff (R 28.5) the local weight is 0.143, so the rule is unchanged where it decides.
- Program of study plays no part in the committee's decision: p = 0.98 for the program
  itself, 0.84 to 0.98 for its interactions with R, hours and group; every program effect
  is under 0.05 R point (se 0.08). The R score does not differ between programs inside a
  group (p = 0.42 and 0.89).
- The feedback cannot test the program idea either way. None of the six files reads the
  program, so a bonus of 1 R point for any one program fits the returns as well as plain
  noise does (0.3 to 1.2 against 0.5). A program effect in the reference is possible and
  unsupported; no weight for it can be derived from the history.

## 11. fairlearn comparison methods (`e004`, 2026-10-03)

Both methods learn the committee's decision; `strength` sweeps the constraint from off to
full. Numbers are on the 4,000 at a 40% grant rate.

- Demographic-parity constraint, full strength: both reach parity. ThresholdOptimizer on
  the production forest: parity gap -0.009, 94.2% agreement with R + hours, 83.3% with the
  committee on held-out history, and the decision reads the group. ExponentiatedGradient
  around a logit without region or postal code: parity gap 0.000, 94.4% agreement with
  R + hours, 85.1% with the committee, group used to train only.
- The retained model is at parity gap 0.000 with 85.3% agreement with the committee, and
  agrees with R + hours at 100% by construction. The fairlearn models differ from it on
  about 6% of applicants: they keep the income preference and the proxy-distorted weights,
  and equalize the rates afterwards.
- True-positive-rate parity measured against the committee's label does almost nothing:
  the parity gap stays at 0.15 to 0.19 at full strength. The committee's label already
  contains the penalty, so the constraint is nearly satisfied from the start. This is the
  notebook's warning, measured.
- Worst-case estimated points over the plausible set: ThresholdOptimizer (parity) 26.6 at
  full strength and 29.3 at half strength, ExponentiatedGradient (parity) 27.0, retained
  27.4. These are within the 1.5-point noise of one another; the retained model's own
  hypothesis (R3, the leading one after the feedback) is excluded from its worst case.
- The sweeps are uneven: ThresholdOptimizer's blend reaches near parity by strength 0.5,
  and ExponentiatedGradient's bound only binds from about 0.5.

## 12. Training without the biased decisions, allocation rules (`a11_dropped_decisions.py`, `e005`, `e006`, 2026-10-03)

Decisions on the 4,000 at a 40% grant rate; the models never see region, postal code or
distance. Checked by running the analysis script; `e005` and `e006` are in the
ledger.

- What the filters remove (T4). The fitted committee says the group decided 1,204 of the
  10,000 decisions: 477 remote refusals (the penalty band) and 727 Centre grants a remote
  applicant would not have had. The soft version touches 5,810 rows for a total weight of
  1,578. The residual filters (5%, 10%, 20% most surprising to a group-blind model) remove
  rows on both sides and leave the two grant rates where they were.
- Filtering works, partly. Parity gap of a logit on R and hours: 0.059 on the committee's
  label, 0.042 with the penalty band dropped, 0.032 with all flipped rows dropped, 0.016
  with soft weights, -0.013 with the flipped rows relabelled. The hours weight it learns
  goes 0.024, 0.063, 0.086, 0.116, 0.174, against 0.145 in closed form: dropping
  under-corrects, relabelling both sides over-corrects.
- The residual filters do nothing (gap 0.059 to 0.063, hours weight 0.020 to 0.024): the
  bias is not in the rows a group-blind model finds surprising. A filter has to know the
  group to find it.
- Nothing beats the closed form (T3). The best filtered models agree with it at 97% to
  99% (soft weights 98.6%, relabelled flipped rows 99.2% with a logit) and all need the
  group to choose rows, like `committee_merit` does to fit. They are a second route to the
  same decisions, with a residual gap; an argument for the retained model, not a
  replacement.
- With income among the inputs the gap stays higher (0.05 to 0.08 for the dropped or
  weighted sets with a logit): the income preference is not a flipped decision, so the
  filters leave it in.
- Estimators (T2): on the same target and features the three differ on 2% to 6% of
  decisions. Boosting and forest are less stable under relabelling (gap -0.05 on R and
  hours, where the logit gives -0.013).
- Weight grid (T1): a rule R + a x log income + b x hours + c x first generation
  reproduces every logit at 98% or more, the forests at 96% or more, boosting at 95% or
  more. The trained models are noisy versions of rules; the first-generation weight is 0
  in every best match.
- Region quotas (T5): at full strength they differ from group cutoffs on 1.1% of decisions
  on the R score and 0.65% on the corrected committee. With group cutoffs the five regions
  already sit between 37.9% and 42.2%.
- Group cutoffs on top of the corrected committee change nothing visible (parity gap
  0.0002 before and after): the correction is already at parity. Estimated worst case
  27.4 to 27.6 points with either rule, the retained model's level.
- Rate sweep of the retained spec (panel labels at 40%): worst case 25.3 at 38%, 26.5 at
  39%, 28.3 at 41%, 29.0 at 42%. The rise above 40% comes from the assumed formulas (more
  grants raise the true-positive rate of both groups) and is within about one standard
  error per step; the reference rate is 39% to 40.5% (block 9), so 40% stands.

## 13. Scorer feedback, second round (`a09_feedback.py`, files 07 to 13, 2026-10-03)

Seven more files, all at 40%: four models trained without the biased decisions, the
corrected committee with region quotas, and the two fairlearn methods under parity.

- No file beats 02 (94.63%). Region quotas (11) returned the same score to the second
  decimal; relabelled logit (08) and dropped forest (09) 94.48%, weighted logit (07)
  94.18%, dropped-band boosting (10) 93.93%. All within sampling noise of 02 and of one
  another, as expected from files that differ from it on 0.65% to 1.8% of decisions.
- The fairlearn parity files (12, 13) returned 92.43% and 92.33%: about two points under
  the corrected committee. Reaching parity is not enough; which applicants are granted
  inside each group matters, and the corrected committee picks them better.
- The fit of the reference is unchanged with thirteen files: top 40% of R + 0.145 to 0.18
  x hours plus noise worth 0.5 R point (rms error 0.53 and 0.42 points; ceiling 95.1%).
  The noise-free R3 label now ranks last among the plausible hypotheses in
  `submit.py --fit`, because it predicts 98% to 99% for the near-copies of file 02 where
  the ceiling is 95%: that table ignores the noise and should not be read as evidence
  against R3.
- Extra terms on top of the fitted reference: none improves the fit beyond noise (best: a
  remote-group term of -0.5 R point, 0.27 against 0.42).
- Reference rate from accuracy and F1 together: 39.2% to 40.5%, as before.
- No change to the retained spec follows from this round.
- Third round, file 14 (R + 0.18 x hours, the weight the fit preferred; it differs from
  file 02 on 50 decisions): 94.33%, against 94.63% for 0.145. The fit predicted 95.1%.
  The feedback does not support moving the weight; 0.145 and 0.18 now fit the fourteen
  files equally (rms error 0.53 and 0.46). The weight stays at the history's 0.145.

- Fourth round, files 15 and 16. R + 1.40 for the remote group (the
  committee's penalty handed back; parity gap -0.098, the decision reads the group):
  90.98%, F1 90.60. R + 0.059 x hours (the weight of the group-blind fit; parity gap
  +0.045): 93.78%, F1 93.51. The fitted reference predicted 91.6% and 93.3%.
- The hours weight now has four points on the scorer: 0 gives 92.28%, 0.059 gives 93.78%,
  0.145 gives 94.63%, 0.18 gives 94.33%. The score rises with the weight up to the
  history's value and is flat after it. The fit over sixteen files is unchanged: weight
  0.145 to 0.18 (rms error 0.57 and 0.47), noise 0.5 R point, ceiling 95.1%, reference
  rate 39% to 41%.
- A flat bonus to the remote group is worse than every rule that does not read the group
  except the two need rules: 1.40 is the size of the penalty, not of the R-score
  difference (0.66), so on the R score alone it over-corrects (estimated equal-opportunity
  gap -0.078 under the fitted reference, against +0.017 for file 02 and +0.075 for file 16).

## 14. Which individual decisions are biased (`a12_individual.py`, 2026-10-03)

The production forest (trained on the 10,000) on the 4,000, cutoff 0.5.

- Counterfactual flip. A remote applicant given Montreal, a Montreal prefix and the Centre
  median distance (17 km), everything else unchanged: the remote grant rate goes from 27.8%
  to 39.4%, and 190 refusals become grants (11.7% of remote applicants, 16.2% of remote
  refusals); none goes the other way. The mirror test takes 292 grants from the centres
  (46.8% to 34.5%).
- The forest spreads the penalty over the three location columns: region alone flips 79,
  distance alone 82, postal prefix alone 17, region and prefix 104, all three 190. Hiding
  the region leaves more than half of the penalty in place, carried by distance.
- The flipped applicants are borderline on merit, not weak: R score 28.0 to 29.5 (median
  28.7) against 25.5 for the other remote refusals; same income, hours and distance.
- Merit benchmark, standard at 40%. Under R + hours the forest refuses 201 of 651
  deserving remote applicants (30.9%) against 12 of 949 in the centres (1.3%), and grants
  173 non-deserving Centre applicants against 2 remote. Under R alone: 132 of 584 (22.6%)
  against 1 of 1,016. The flip test recovers 160 of the 201 and 131 of the 132: the two
  methods name the same people.
- SHAP (mean contribution to the grant probability, in points; the per-column gaps add up
  to the 19.9-point gap in mean probability): region -7.3, distance -6.9, postal prefix
  -1.7, so location -15.9; R score -3.4; income -2.0; hours +1.5; program and first
  generation 0. Location costs a remote applicant 9.6 points on average (up to 20.2) and
  gives a Centre applicant 6.2. Figure `a12_shap_groupes.png`.
- Same split as the committee's (block 4: 15.6 penalty, 4.9 income, 0.6 rest): the forest
  learned the penalty and stored it in the location columns.

## 15. Ensembles of models trained on cleaned decisions (`a13_ensembles.py`, `e007`, 2026-10-03)

Fifteen models on the R score and hours only (logit, gradient boosting, random forest, each
on the five cleaned training sets of `targets.DEBIASED`), then rank averages of them, at
40% on the 4,000. "The rule" is the retained R + 0.145 x hours. Checked by running the
analysis script; `e007` is in the ledger.

- Single non-linear models already exist (block 12): a forest or boosting on a cleaned set
  makes the same decision as the rule on 95.4% to 98.3% of applicants, parity gap -0.051
  to +0.023.
- Averaging moves toward the rule. The ensemble of all fifteen agrees with it on 99.0% (38
  decisions differ, parity gap -0.003), against 97.6% for its members on average. The
  three-estimator ensembles reach 98.0% to 98.8% on four cleaned sets; on
  `relabel_flipped` the ensemble inherits the over-correction of its two tree members
  (96.7%, gap -0.045).
- The best single model is still the relabelled logit (99.2%, 32 decisions): no ensemble
  is closer to the rule than it is.
- The disagreements are at the cutoff: median distance 0.06 R point from it, against 2.21
  over all applicants; 19 applicants swapped in each direction, spread over both groups.
  There is no profile the ensemble finds and the rule misses.
- A forest's distance from the rule is not seed noise: two seeds differ on 0.3% of
  decisions, each differs from the rule on 2.0%, and averaging five seeds leaves 2.0%. It
  comes from the cleaned training set (how much of the penalty the filter removed) and
  from the steps of the trees.
- Estimated worst case over the other hypotheses (R + hours left out, since every model
  here aims at it): rule 27.4, full ensemble 27.4, range 22.6 to 30.9. The highest are the
  under-corrected logits (parity gap 0.03 to 0.04), which lean toward "R score alone"; the
  scorer's feedback ranks that standard below R + hours (92.28% against 94.63%).
- Why nothing more can be expected: with two independent inputs and an additive committee
  (block 4), there is no interaction or curve for a flexible model to find. Figure
  `a13_ensembles.png`.

## 16. Two behaviours: a different rule for the close calls (`a15_close_calls.py`, `e008`, 2026-10-03)

A close call is an applicant within a band of the cutoff of
R + 0.145 x hours. The `close_*` scorers keep the retained rule outside the band and rank
the applicants inside it by a second behaviour, for the same number of grants (spec key
`band`, in R points). Seven second behaviours, three bands (0.25, 0.5, 1.0). Checked by
running the analysis script; `e008` is in the ledger.

- Size of the band on the 4,000: 252 applicants within 0.25 R point (6.3%), 508 within 0.5
  (12.7%), 983 within 1.0 (24.6%), in the same group mix as the whole file.
- The close calls are where the rule and the fitted reference (R + hours plus noise worth
  0.5 R point) disagree: 81% of the rule's errors are within 0.5 R point and 98% within
  1.0. Inside 0.5 the rule agrees with that reference at 69%, outside at 98.9%. So the
  idea aims at the right applicants.
- The committee did not decide its own close calls differently. Its logit refitted on the
  decisions within 2 R points of its own cutoff gives the same weights as on the rest
  (penalty 1.40 against 1.37 R points, hours 0.146 against 0.140, noise 1.33 against
  1.30). Within 1 point the penalty reads 1.28 against 1.60, a difference of about two
  standard errors that the 2-point split does not confirm. First generation, program and
  distance stay insignificant among the close calls (largest |z| 0.91 for the last two).
  The history holds no second behaviour to learn.
- Against the fitted reference every variant is below the rule (95.1%): 94.5% to 94.7% at
  band 0.25, 93.0% to 93.9% at 0.5, 88.0% to 93.0% at 1.0 (noise of the comparison: 0.25
  point). This is by construction: that reference is R + hours plus noise, so nothing else
  can inform the band. It measures the cost if the reference is what we think it is, not
  whether the reference has a second behaviour.
- What each second behaviour does to the groups (parity gap, 0 for the rule): R score
  alone +0.03 to +0.07 (back toward the centres), first generation -0.01 to -0.04, lowest
  income -0.02 to -0.07, remote group first -0.05 to -0.22. The trained models (boosting
  on the whole history, logit or boosting on the past close calls only, soft weights, no
  location column) +0.01 to +0.05: inside a narrow band R + hours barely varies, so they
  rank on what is left of the committee's income preference.
- Decisions changed: about 100 to 120 at band 0.25, 160 to 240 at 0.5, 210 to 470 at 1.0.
- Estimated worst case over the other hypotheses (R + hours left out): rule 27.4; the R
  score and the trained variants 26.9 to 30.4, the need and group variants 10.6 to 27.1.
  The variants above the rule gain by leaning toward "R score alone", like the
  under-corrected logits of block 15, and the feedback ranks that standard below R + hours.
  The gain is at most two standard errors of the estimate.
- Not a replacement for the retained rule. Whether the reference itself treats close calls
  differently can only be told by the scorer; the files that would separate it best are a
  narrow band (0.25 or 0.5) with first generation or with the R score alone.
- The other way round: a cutoff on the R score alone, the hours adapt
  the applicants near it (`r_then_hours`: R + 0.145 x hours inside the band;
  `r_then_hours_alone`: most hours first). `r_then_hours` is a path from R alone (band 0,
  parity gap 0.070) to R + hours: 74 decisions differ from the retained rule at band 0.5
  (gap 0.025), 24 at 1.0, none from 2.0 on. So "R cutoff, then hours" with a band of 2 R
  points is the retained rule. Hours alone in the band over-corrects as the band widens:
  gap 0.015 at 0.5, -0.035 at 1.0, -0.17 at 2.0, -0.26 at 3.0 (688 to 986 decisions
  changed, worst case 5.6 to 8.3 points).
- Submission files 17 to 21 (band 0.5; written, not yet scored): R + hours then first
  generation, then R alone, then the logit of the past close calls; R then R + hours; R
  then hours alone. They differ from file 02 on 74 to 176 decisions. Under the fitted
  reference each should return 0.6 to 1.4 points below file 02; with about a point of
  sampling noise, only a file clearly above 94.63% would say the reference has a second
  behaviour.
- Governance material: the band is the natural definition of "cases sent to human review"
  (6% to 13% of applicants hold 51% to 81% of the likely errors). Figure
  `a15_close_calls.png`.

## 17. Board review and monitor demo (2026-10-03)

- `e005` to `e008` are in the ledger. The retained rule's estimated worst case is 27.4
  (under R1). Above it: `r_then_hours` at band 0.5 (31.5, se 0.9), merit with group
  cutoffs at strength 0.75 (30.9), the logit on `drop_flipped` (30.9), and about sixty
  runs down to 27.5. Each gains under R1 by leaning toward the R score alone and keeps a
  parity gap of 0.02 to 0.05. The scorer returned 92.28% for R alone against 94.63% for
  R + hours, and no hedge that was submitted beat file 02. Kept: file 02.
- Monitor demo (`monitoring.py`): four replayed periods stay green on all seven
  indicators; a period decided at `debias` 0.5 reads a penalty of 0.81 R point (red),
  a group gap of 0.12 (red), a blind hours weight of 0.110 against 0.141 to 0.150 (amber)
  and an audited equal-opportunity gap of 0.16 on 600 files (amber). The penalty read on
  decisions is close to the half penalty that was put in (0.70).

## 18. Who gets a grant (`a16_who_gets_grant.py`, 2026-10-03)

Production model against the retained model on the 4,000; the committee on a sample of
4,000 historical rows for the first figure.

- The retained model grants exactly when R + 0.145 x hours reaches 30.07, the same line
  for both groups. The production model's frontier sits left of that line in the centres
  and right of it in the remote regions; the committee's is the same shift, blurred.
  Figure `a16_carte_octrois.png`.
- 388 decisions change hands. 213 applicants gain a grant (201 remote; mean R 28.60,
  income 51k, 14.7 hours, 46% first generation); 175 lose one (173 Centre; mean R 28.61,
  income 93k, 7.5 hours, 26% first generation). Same R score on both sides.
- By profile (`a16_taux_par_profil.png`): the rate by R band barely moves. By hours it
  turns from flat or falling (36% under 5 hours, 35% at 20 or more) to rising (23% to
  63%). By income quarter it goes from 31%, 36%, 41%, 48% to 41%, 38%, 40%, 41%. First
  generation 36% to 40%. Programs within 35% to 42% before, 38% to 42% after.
- Among grantees (`a16_profil_boursiers.png`): remote share 29% under production, 41%
  after, equal to their share of applicants (41%); income under the median 43% to 49%;
  first generation 31% to 34%; working 15 hours or more 14% to 20% (16% of applicants):
  the one profile the retained model over-represents, by design of the rule.

## 19. A cap on the hours counted (`a17_capped_hours.py`, `e009`, 2026-10-03)

R + 0.145 x hours, the hours counted up to a cap only
(`rule_r_hours_capped`, spec key `hours_cap`).

- The committee kept rewarding hours past 10, 12 and 15 a week at the same weight (0.14 to
  0.16 R point per hour past the cap, se 0.01 to 0.025; never different from the weight
  below it, |z| under 1.7). Past 20 hours the history cannot say: 1% of rows, weight -0.04,
  se 0.10. Dropping the hours past the cap costs 15 points of log-likelihood at 15, 87 at
  10, nothing at 20. The history gives no reason for a cap under 20.
- On the 4,000 a cap reopens the gap, because the remote group works the long weeks (74%
  above 10 hours against 29% in the centres): parity gap 0.054 at a cap of 10, 0.031 at 12,
  0.017 at 15, 0.001 at 20. Decisions that differ from the retained rule: 162, 112, 46, 4.
  Figure `a17_plafond_heures.png`.
- Under a cap of 10, 81 applicants lose a grant (79% remote, 16.7 hours, mean R 27.96) and
  81 gain one (15% remote, 7.8 hours, mean R 28.80).
- Against the fitted reference (uncapped by construction, so this cannot favour a cap):
  93.8% at 10, 94.8% at 15, 95.0% at 20, 95.1% without. Worst case over the other
  hypotheses: 29.3, 28.1, 27.3 points against 27.4 without a cap; the gain at 10 and 12
  (29.7) is the hedge toward "R score alone" already seen on the board, within noise.
- Files 22 (cap 10) and 23 (cap 15) are written, not yet scored. No file for a cap of 20:
  four decisions away from file 02, its score would be about those four applicants. Only a
  return clearly above 94.63% would mean something. Not a replacement as it stands.

## 20. Program of study: a test for the scorer (`a18_program.py`, `e010`, 2026-10-03)

An idea from block 10, taken off hold: does the reference read the program? Two rules,
`rule_r_hours_program` (spec keys `program`, `program_bonus`: R + 0.145 x hours plus a bonus
for one program) and `rule_r_hours_within_program` (the same rate in every program).

- The history cannot answer. The committee's program effects are all under 0.05 R point
  (95% intervals within -0.16 to +0.22), and the committee is not the reference.
- The sixteen scored files do not point to a program. With the fitted reference (R + hours,
  noise 0.5) the returns are off by 0.62 point rms (sd 0.14 over noise draws); adding 1 R
  point for or against one program gives 0.62 to 0.97. No program fits better than none;
  Sante either way and "against Arts et lettres" fit as well. An effect of 1 point on top
  of the noise would also put file 02 near 93.8% where it returned 94.63%.
- On the 4,000 a bonus of 0.5 R point moves 74 to 92 decisions (rate of the favoured program
  from about 40% to 43% to 48%), a bonus of 1 moves 148 to 206. Parity gap stays within
  0.015. Worst case over the other hypotheses: 26.8 to 27.5 points at 0.5, 25.8 to 26.5 at
  1, against 27.4 for the retained rule. The retained rule's rates per program already run
  from 38.4% (Arts et lettres) to 42.2% (Sante), from the R scores drawn.
- What a "+0.5 for one program" file would return, minus file 02 on the same reference:
  -0.4 point if the program plays no part, +0.4 if the reference favours that program by
  0.5, +0.9 to +1.2 if by 1, -0.6 to -0.7 if it favours another program by 1, -1.2 to -1.5
  if it disfavours this one by 1. Sd 0.15 to 0.26 over the reference's noise; the subset
  sampling comes on top. Figure `a18_programme.png`. A file with a bonus of 1 separates
  wider (-1.2 to -1.9 against +1.3 to +1.8) but is a worse model if the program plays no
  part.
- Reading rule for files 24 to 28 (one per program, bonus 0.5, written, not yet scored):
  the five are read together and against file 02's 94.63%. A program the reference favours
  returns above 94.63% while the four others return below; all five slightly below
  (about 94.2%) means no program effect of 0.5 point or more. One file a few tenths above
  the others is within noise.
- "The same rate in every program" differs from the retained rule on 46 decisions, and its
  score would move by 0.2 point between the two readings (sd 0.14): no file.
- Nothing here changes the retained rule. A bonus for a program would need a returned score
  clearly above file 02 and a reason the institution can state; the history offers none.

## Material for the deliverables (governance, monitor, pitch)

`gouvernance.md`, `monitoring.py` and `presentation.md` are written (2026-10-03). What
they draw on, by block above:

- **Choices the institution has to own** (`gouvernance.md` item 1, last slide).
  - Fairness definition: equal opportunity against a standard of merit; under "R score and
    hours worked" it coincides with parity, since each group has 40% deserving (blocks 6, 8).
  - Region is used to fit the committee model and isolate the penalty, never to decide
    (block 10). The fairlearn ThresholdOptimizer alternative decides with the group
    (block 11).
  - Hours worked count as merit. They may reflect effort or financial constraint; at equal
    hours the centres keep a higher rate, and applicants who work few hours gain least from
    the correction (audit report section 7).
  - The committee favoured higher income, an indirect bias worth 4.9 of the 21 points; the
    corrected model drops it (block 4, audit report section 3).
  - Who bears the cost: with a fixed budget, the centres lose about 7 points of grant rate,
    mostly high-income and low-hours applicants (audit report section 7).
- **Monitoring indicators** (`gouvernance.md` item 3, `monitoring.py`).
  - Grant rate per group and per region, with intervals; budget rate.
  - The penalty itself: the group coefficient, in R points, of the committee logit refitted
    on each batch of decisions (1.40 in the history). "How well the inputs predict region"
    is saturated (AUC 0.999) and cannot alert (block 2, idea T10).
  - The hours weight fitted without the group: it falls from 0.145 to 0.059 when a proxy
    absorbs a penalty, so a drop signals bias re-entering through a neutral-looking column
    (block 10).
  - Equal-opportunity gap on an independently re-reviewed sample: against the system's own
    decisions the gap reads 0.063 where the independent standard says 0.270 (block 8).
  - Subgroup rates (first generation, income, hours, program) crossed with group.
- **Pitch storyline** (slides 1 to 6).
  - Same file, different region, different answer: 48.4% against 27.3%, and the gap holds
    at equal R score (audit report section 2).
  - Decomposition: 15.6 points penalty, 4.9 income preference, 0.6 everything else (block 4).
  - Removing region does not work, and why: proxies give the group away, and a hidden
    penalty moves into the hours weight (blocks 2, 10; audit report section 4).
  - The model learned the committee: 88% accurate means 88% faithful (audit report
    section 5).
  - The correction: refit the committee, remove the penalty and the income preference.
    Parity gap 0.000 inside the budget; 94.63% on the scorer's subset (blocks 9, 10).
  - The Pareto front with the fairlearn methods as comparison (block 11).
- **Limits to state**: synthetic data, unknown reference, feedback on a subset, hours as a
  value choice, the R x hours interaction left out, program of study untested (audit report
  section 8, blocks 9 and 10).

## Not checked

- Whether the 0.270 was measured on the 4,000, on a subset, or on held-out history (the
  gaps are close on both, block 8), and at what rate the reference labels applicants
  deserving.
- Whether the automated points follow the formulas we assume.
- Whether the reference contains noise independent of the columns. If it does, no model
  reaches perfect agreement and all gaps above shrink.
- What removing each column does to the production model's gap (audit section 4): needs
  retraining, belongs to the audit report.

## Figures the audit report will reuse

| Figure | Report section | Slide |
|---|---|---|
| `figures/taux_par_region.png` (exists) | 2 | 1 |
| `figures/taux_par_cote_r.png` (exists) | 2 | 2 |
| `analysis/a03_profils.png` | 1 | |
| `analysis/a04_courbe_comite.png` | 2, 3 | 2 (alternative) |
| `analysis/a04_poids_comite.png` | 3 | |
| `analysis/a05_colonnes_ignorees.png` | 3 | |
| `analysis/a02_proxys.png` | 4 | 3 |
| `analysis/a07_baseline_regions.png` | 5 | 4 |
| `analysis/a06_etalons.png` | 6 | backup |
| `analysis/a07_ancre.png` | 6, 8 | backup |
| `analysis/a08_ancre_taux.png` | 6, 8 | backup |

Since the audit report was written (2026-10-03), its figures come from `equialgo/audit.py`
and are saved at the root of `figures/`: `taux_par_region.png`, `taux_par_cote_r.png`,
`poids_comite.png`, `proxys.png`, `production_par_region.png`, `avant_apres.png`.
| `analysis/a01_fichiers.png` | 1 (annex) | |

The functions behind the reused ones move to `equialgo/audit.py` when their report section
is written.
