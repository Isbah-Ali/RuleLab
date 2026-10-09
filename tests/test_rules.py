import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import DRIFT_TOL
from miner import fits_cap
from rules import Condition, build_mask, evaluate, fmt_pct, verdict


def mini_df():
    return pd.DataFrame(
        {
            "protocol_type": ["tcp", "tcp", "icmp"],
            "service": ["http", "http", "smtp"],
            "flag": ["SF", "S0", "SF"],
            "src_bytes": [100.0, 50.0, 0.0],
            "dst_bytes": [0.0, 200.0, 10.0],
            "label": ["normal", "neptune", "smurf"],
            "is_attack": [False, True, True],
        }
    )


class TestConditions(unittest.TestCase):
    def test_ge_blocks_exact_boundary(self):
        df = mini_df()
        mask = build_mask(df, (Condition("src_bytes", ">=", 100.0),))
        self.assertEqual(mask.tolist(), [True, False, False])

    def test_ge_excludes_below_boundary(self):
        df = mini_df()
        mask = build_mask(df, (Condition("src_bytes", ">=", 50.0),))
        self.assertEqual(mask.tolist(), [True, True, False])

    def test_le_blocks_exact_boundary(self):
        df = mini_df()
        mask = build_mask(df, (Condition("dst_bytes", "<=", 0.0),))
        self.assertEqual(mask.tolist(), [True, False, False])

    def test_eq_unseen_category_matches_zero_rows(self):
        df = mini_df()
        m = evaluate(df, (Condition("service", "==", "no_such_svc"),))
        self.assertEqual(m["attacks_blocked"], 0)
        self.assertEqual(m["normal_blocked"], 0)

    def test_empty_rule_blocks_nothing(self):
        df = mini_df()
        mask = build_mask(df, ())
        self.assertEqual(int(mask.sum()), 0)
        m = evaluate(df, ())
        self.assertEqual(m["attacks_blocked"], 0)
        self.assertEqual(m["normal_blocked"], 0)


class TestVerdict(unittest.TestCase):
    def test_collateral_exactly_half_pct_is_safe(self):
        self.assertEqual(verdict(0.005, drift=0.0, attacks_blocked=100), "SAFE")

    def test_collateral_just_above_half_pct_is_caution(self):
        self.assertEqual(verdict(0.0050001, drift=0.0, attacks_blocked=100), "CAUTION")

    def test_collateral_exactly_two_pct_is_caution(self):
        self.assertEqual(verdict(0.02, drift=0.0, attacks_blocked=100), "CAUTION")

    def test_collateral_just_above_two_pct_is_unsafe(self):
        self.assertEqual(verdict(0.0200001, drift=0.0, attacks_blocked=100), "UNSAFE")

    def test_attacks_19_is_low_evidence(self):
        self.assertEqual(verdict(0.0, drift=0.0, attacks_blocked=19), "LOW EVIDENCE")

    def test_attacks_20_is_not_low_evidence(self):
        self.assertEqual(verdict(0.0, drift=0.0, attacks_blocked=20), "SAFE")

    def test_drift_exactly_tol_is_safe(self):
        self.assertEqual(
            verdict(0.001, drift=DRIFT_TOL, attacks_blocked=100), "SAFE"
        )

    def test_drift_just_above_tol_is_caution(self):
        self.assertEqual(
            verdict(0.001, drift=DRIFT_TOL + 1e-9, attacks_blocked=100), "CAUTION"
        )

    def test_unsafe_precedes_low_evidence(self):
        self.assertEqual(verdict(0.03, drift=0.0, attacks_blocked=5), "UNSAFE")


class TestFmtPct(unittest.TestCase):
    def test_n_29_shows_n_lt_30(self):
        self.assertEqual(fmt_pct(5, 29), "n<30")

    def test_n_30_shows_percentage(self):
        self.assertEqual(fmt_pct(15, 30), "50.0%")


class TestHardCap(unittest.TestCase):
    def test_collateral_exactly_equal_to_cap_passes(self):
        self.assertTrue(fits_cap(50, 50, 0.002, 50000))

    def test_collateral_above_cap_fails(self):
        self.assertFalse(fits_cap(51, 50, 0.002, 50000))


if __name__ == "__main__":
    unittest.main()
