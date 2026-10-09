from dataclasses import dataclass

import numpy as np

import config as _cfg
from config import CATEGORICAL, NUMERIC

VALID_OPS = ("==", ">=", "<=")
_CATEGORICAL_SET = frozenset(CATEGORICAL)
_NUMERIC_SET = frozenset(NUMERIC)

COLLATERAL_CAUTION = 0.005
COLLATERAL_UNSAFE = 0.02
MIN_EVIDENCE = 20
MIN_FMT_N = 30
_FALLBACK_DRIFT_TOL = 0.0025


@dataclass(frozen=True, slots=True)
class Condition:
    feature: str
    op: str
    value: object

    def __post_init__(self):
        if self.op not in VALID_OPS:
            raise ValueError("invalid op: %r" % (self.op,))
        if self.feature in _CATEGORICAL_SET:
            if self.op != "==":
                raise ValueError("categorical features only support '=='")
        elif self.feature in _NUMERIC_SET:
            if self.op == "==":
                raise ValueError("numeric features only support '>=' or '<='")
            float(self.value)
        else:
            raise ValueError("unknown feature: %r" % (self.feature,))

    def mask(self, df):
        arr = df[self.feature].to_numpy()
        if self.op == "==":
            return arr == self.value
        if self.op == ">=":
            return arr >= self.value
        return arr <= self.value

    def text(self):
        v = self.value
        if isinstance(v, float):
            v = "%.6g" % v
        return "%s %s %s" % (self.feature, self.op, v)


Rule = tuple


def build_mask(df, rule):
    if not rule:
        return np.zeros(len(df), dtype=bool)
    mask = rule[0].mask(df)
    for cond in rule[1:]:
        mask &= cond.mask(df)
    return mask


def _metrics(df, mask):
    is_att = df["is_attack"].to_numpy()
    labels = df["label"].to_numpy()
    attacks_total = int(np.count_nonzero(is_att))
    normal_total = int(len(df) - attacks_total)
    attacks_blocked = int(np.count_nonzero(mask & is_att))
    normal_blocked = int(np.count_nonzero(mask & ~is_att))
    blocked_total = attacks_blocked + normal_blocked
    precision = attacks_blocked / blocked_total if blocked_total else 1.0
    recall = attacks_blocked / attacks_total if attacks_total else 0.0
    collateral_rate = normal_blocked / normal_total if normal_total else 0.0
    per_family = {}
    for lab in np.unique(labels):
        sel = labels == lab
        per_family[str(lab)] = [
            int(np.count_nonzero(mask & sel)),
            int(np.count_nonzero(sel)),
        ]
    return {
        "attacks_blocked": attacks_blocked,
        "attacks_total": attacks_total,
        "normal_blocked": normal_blocked,
        "normal_total": normal_total,
        "collateral_rate": collateral_rate,
        "precision": precision,
        "recall": recall,
        "per_family": per_family,
    }


def evaluate(df, rule):
    return _metrics(df, build_mask(df, rule))


def metrics_from_mask(df, mask):
    return _metrics(df, mask)


def evaluate_rules(df, rules):
    if not rules:
        return _metrics(df, np.zeros(len(df), dtype=bool))
    mask = build_mask(df, rules[0])
    for r in rules[1:]:
        mask |= build_mask(df, r)
    return _metrics(df, mask)


def rules_mask(df, rules):
    if not rules:
        return np.zeros(len(df), dtype=bool)
    mask = build_mask(df, rules[0])
    for r in rules[1:]:
        mask |= build_mask(df, r)
    return mask


def rule_to_text(rule):
    if not rule:
        return "no conditions (blocks nothing)"
    return " AND ".join(c.text() for c in rule)


def verdict(collateral_rate, drift=0.0, attacks_blocked=0, drift_tol=None):
    tol = drift_tol if drift_tol is not None else _cfg.DRIFT_TOL
    if tol is None:
        tol = _FALLBACK_DRIFT_TOL
    if collateral_rate > COLLATERAL_UNSAFE:
        return "UNSAFE"
    if collateral_rate > COLLATERAL_CAUTION or drift > tol:
        return "CAUTION"
    if attacks_blocked < MIN_EVIDENCE:
        return "LOW EVIDENCE"
    return "SAFE"


def fmt_pct(k, n):
    if n < MIN_FMT_N:
        return "n<30"
    return "%.1f%%" % (100.0 * k / n)
