# Pre-registration — stage 1 (written 2026-09-25T14:34:38Z, before running dev3)
Data allowed: bars closing before 2019-01-01 only.
Validation: 3 expanding folds — validate 2016, 2017, 2018; train on everything final before each year.
Objective per fold: mean over {1h,4h} of MCC(state vs oracle) − 0.1·max(0, flips_per_true_flip − 2). Score = mean of 3 folds.
Candidates (60 Optuna trials each, TPE seed 42, same search space as v2 incl. smoothing):
  A = v2 features (159)
  B = v2 + Bollinger/ATR (192)
  C = universal single-asset set: B minus btc_*, rel_*, br_*, volume features
Decision rule:
  crypto model = candidate with highest score = mean of its top-5 trials (not the single best, to limit best-of-60 luck);
  if C is within 0.005 of the best, use C for everything (simpler, and it is the only one usable on stocks/FX).
  stocks/FX model = C (the others need crypto-only inputs).
After this, the chosen config is frozen in FROZEN_v3.json and the 2019–2026 walk-forward is run ONCE.
