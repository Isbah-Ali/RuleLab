# RuleLab deployment decisions

All numbers below come from `python evaluate.py --decide` (validation split only)
or from the printed `python evaluate.py --eval` table (prompt 02).
No test data was loaded for any decision.

## 1. UNSAFE verdict

- **Value:** collateral_rate > 2% on the evaluated split.
- **How chosen:** product rule (fixed in the approved architecture); deployment is
  rejected outright once more than 2% of labeled-normal traffic would be blocked.
- **Number it came from:** every B2 cap printed val collateral 0.0000%
  (caps 0.1/0.5/1.0/2.0%: fit_col=0.0000% val_col=0.0000%), i.e. 0 of the
  validation normal rows blocked at any cap — all sit below the 2% UNSAFE line.

## 2. CAUTION verdict

- **Value:** collateral_rate > 0.5% OR drift > DRIFT_TOL.
- **How chosen:** product rule; 0.5% is the "review before deploy" band between
  SAFE and UNSAFE. Exactly 0.5% is not above 0.5%, so it stays SAFE on collateral.
- **Number it came from:** B0 (flag == S0) printed val collateral 0.49%
  (eval table, all caps) — just under the 0.5% CAUTION threshold — while every
  B2 cap printed 0.0000%.

## 3. LOW EVIDENCE verdict

- **Value:** attacks_blocked < 20.
- **How chosen:** product rule; a rule that blocks fewer than 20 attacks gives
  too little signal to judge, regardless of how clean its collateral looks.
- **Number it came from:** the smallest B2 validation attacks_blocked across the
  four caps was 10012 (cap 0.1%); the largest was 10448 (cap 2.0%). All four
  printed verdict=SAFE, so the 20-attack floor never bound in this run.

## 4. SAFE verdict

- **Value:** collateral_rate <= 0.5%, drift <= DRIFT_TOL, attacks_blocked >= 20.
- **How chosen:** product rule; the only remaining branch after UNSAFE, CAUTION
  and LOW EVIDENCE are excluded.
- **Number it came from:** all four B2 caps printed verdict=SAFE
  (val_col=0.0000%, drift=0.0000pp, val_attacks_blocked 10012-10448).

## 5. DRIFT_TOL

- **Value:** 0.0025 (0.25 percentage points).
- **How chosen:** max (val minus fit) collateral drift of the B2 rules across the
  4 grid caps, rounded up to the next 0.25 pp, minimum 0.25 pp.
- **Number it came from:** --decide printed max B2 val-fit drift = 0.0000 pp
  across 4 caps (fit_col=0.0000% and val_col=0.0000% at every cap), so the
  0.25 pp minimum applied: DRIFT_TOL = 0.0025.

## 6. RECOMMENDED_CAP

- **Value:** 0.02 (2% of fit normal).
- **How chosen:** among grid caps whose B2 validation verdict is SAFE, the cap
  with the highest validation macro-recall (families with n>=30); if none were
  SAFE, the smallest cap would be used.
- **Number it came from:** --decide printed val macro-recall per cap:
  0.0787 (0.1%), 0.0824 (0.5%), 0.0824 (1.0%), 0.0977 (2.0%), all SAFE.
  Highest = 0.0977 at cap 2.0%.

## 7. Minimum 20 attacks (miner min_tp)

- **Value:** each mined rule must block at least 20 fit attacks; rules below that
  are discarded by the miner, mirroring the LOW EVIDENCE display threshold.
- **How chosen:** pinned in the approved architecture (MIN_TP = 20 in config.py).
- **Number it came from:** --decide fit_attacks_blocked for B2:
  23247 (0.1%), 24221 (0.5%), 24221 (1.0%), 24384 (2.0%) — every produced rule set
  cleared the 20-attack floor by a wide margin.

## 8. n<30 display rule

- **Value:** percentages are hidden and "n<30" is shown when the family has
  fewer than 30 rows (fmt_pct).
- **How chosen:** product rule (Context: "Show no % when n < 30"); small-n recall
  percentages would be noise.
- **Number it came from:** MIN_FAMILY_N = 30 (config, used by macro-recall);
  prompt 01 printed test labels under 30 rows, e.g. xsnoop n=4, worm n=2,
  sqlattack n=2, udpstorm n=2.

## 9. Hard cap stop (miner)

- **Value:** the miner stops when the next rule would push cumulative fit
  collateral above the cap (cap of 0.2% = 0.002 is accepted exactly, 0.0021 is not).
- **How chosen:** pinned in the approved architecture; the cap is a hard stop,
  not a soft penalty.
- **Number it came from:** --eval printed V4 PASS ("B2 fit collateral <= cap at
  every cap") and --decide fit collateral was 0.0000% at all four caps
  (fit_normal_blocked = 0 everywhere), i.e. well inside every cap.
