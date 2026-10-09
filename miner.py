import numpy as np

from config import CATEGORICAL, MAX_CONDS, MAX_RULES, MIN_TP, NUM_QUANTILES, NUMERIC
from rules import Condition


def candidate_conditions(fit_df):
    cands = []
    for col in CATEGORICAL:
        for v in sorted(fit_df[col].unique().tolist()):
            cands.append(Condition(col, "==", v))
    quantiles = fit_df[NUMERIC].quantile(NUM_QUANTILES)
    for col in NUMERIC:
        seen = set()
        for q in NUM_QUANTILES:
            v = float(quantiles.loc[q, col])
            if v in seen:
                continue
            seen.add(v)
            cands.append(Condition(col, ">=", v))
            cands.append(Condition(col, "<=", v))
    return cands


def _cond_mask(cols, cond):
    arr = cols[cond.feature]
    if cond.op == "==":
        return arr == cond.value
    if cond.op == ">=":
        return arr >= cond.value
    return arr <= cond.value


def _best_rule(cols, cands, cand_masks, rem, is_att, lam, max_conds, min_tp):
    rem_att = rem & is_att
    rem_nor = rem & ~is_att
    cur_mask = np.ones(len(rem), dtype=bool)
    cur_tp = int(np.count_nonzero(rem_att))
    cur_fp = int(np.count_nonzero(rem_nor))
    cur_score = cur_tp - lam * cur_fp
    chosen = []
    chosen_set = set()
    for _ in range(max_conds):
        best = None
        for i, cond in enumerate(cands):
            if cond in chosen_set:
                continue
            mask_r = cur_mask & cand_masks[i]
            tp = int(np.count_nonzero(mask_r & rem_att))
            fp = int(np.count_nonzero(mask_r & rem_nor))
            score = tp - lam * fp
            key = (-score, cond.feature, cond.op, cond.value)
            if best is None or key < best[0]:
                best = (key, cond, cand_masks[i], tp, fp, score)
        if best is None:
            break
        _, cond, cand_mask, tp, fp, score = best
        if chosen and score <= cur_score:
            break
        if not chosen and tp < min_tp:
            return None
        chosen.append(cond)
        chosen_set.add(cond)
        cur_mask &= cand_mask
        cur_tp, cur_fp, cur_score = tp, fp, score
    if not chosen or cur_tp < min_tp:
        return None
    return tuple(chosen), cur_mask, cur_tp, cur_fp


def fits_cap(cum_fp, fp, cap, normal_total):
    return cum_fp + fp <= cap * normal_total


def mine(fit_df, cap, max_rules=MAX_RULES, max_conds=MAX_CONDS, min_tp=MIN_TP):
    assert fit_df.attrs.get("split") == "fit", "mine() only accepts fit rows"
    cands = candidate_conditions(fit_df)
    cols = {c.feature: fit_df[c.feature].to_numpy() for c in cands}
    cand_masks = [_cond_mask(cols, c) for c in cands]
    is_att = fit_df["is_attack"].to_numpy()
    attacks_total = int(np.count_nonzero(is_att))
    normal_total = int(len(fit_df) - attacks_total)
    lam = attacks_total / (cap * normal_total)
    rem = np.ones(len(fit_df), dtype=bool)
    rules = []
    cum_fp = 0
    for _ in range(max_rules):
        best = _best_rule(cols, cands, cand_masks, rem, is_att, lam, max_conds, min_tp)
        if best is None:
            break
        rule, rule_mask, tp, fp = best
        if not fits_cap(cum_fp, fp, cap, normal_total):
            break
        rules.append(rule)
        cum_fp += fp
        rem &= ~rule_mask
    return rules
