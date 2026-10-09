import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests

import llm

FAKE_KEY = "FAKE_TEST_KEY_DO_NOT_LEAK_12345"


def make_context():
    return {
        "rule_text": "flag == S0",
        "cap_pct": 1.0,
        "datasets": {
            "validation": {
                "name": "Validation",
                "attacks_blocked": 100,
                "attacks_total": 1000,
                "normal_blocked": 5,
                "normal_total": 5000,
                "recall_pct": 10.0,
                "collateral_pct": 0.1,
                "precision_pct": 95.24,
                "verdict": "SAFE",
                "cap_drift_pp": 0.0,
            },
            "kddtest": {
                "name": "KDDTest+",
                "attacks_blocked": 50,
                "attacks_total": 500,
                "normal_blocked": 2,
                "normal_total": 2000,
                "recall_pct": 10.0,
                "collateral_pct": 0.1,
                "precision_pct": 96.15,
                "verdict": "SAFE",
                "cap_drift_pp": 0.05,
            },
        },
    }


GOOD_TEXT = "Rule blocks 100 of 1000 attacks (10.0% recall) on Validation."


def make_resp(status_code, text=GOOD_TEXT):
    r = mock.Mock()
    r.status_code = status_code
    r.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": text}]}}]
    }
    return r


class TestLLM(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.usage = str(Path(self._tmp.name) / "usage.json")
        self.addCleanup(self._tmp.cleanup)
        self.sleep_patch = mock.patch("llm.time.sleep")
        self.sleep_mock = self.sleep_patch.start()
        self.addCleanup(self.sleep_patch.stop)
        self.gap_patch = mock.patch.object(llm.explain, "_last_call_ts", None, create=True)
        self.gap_patch.start()
        self.addCleanup(self.gap_patch.stop)

    def test_missing_key_template_no_http(self):
        with mock.patch("llm.requests.post") as post:
            text, source, reason = llm.explain(
                make_context(), api_key="", usage_path=self.usage, min_gap=0
            )
            self.assertEqual(source, "template")
            self.assertEqual(reason, "missing API key")
            self.assertIn("flag == S0", text)
            post.assert_not_called()

    def test_403_both_models_goes_to_template(self):
        with mock.patch("llm.requests.post", return_value=make_resp(403)) as post:
            text, source, reason = llm.explain(
                make_context(), api_key=FAKE_KEY, usage_path=self.usage, min_gap=0
            )
            self.assertEqual(post.call_count, 2)
            self.assertEqual(source, "template")
            self.assertEqual(reason, "http 403")
            self.assertIn("Deterministic template", text)

    def test_429_then_200_uses_retry_on_primary(self):
        with mock.patch(
            "llm.requests.post", side_effect=[make_resp(429), make_resp(200)]
        ) as post:
            text, source, reason = llm.explain(
                make_context(), api_key=FAKE_KEY, usage_path=self.usage, min_gap=0
            )
            self.assertEqual(post.call_count, 2)
            self.assertEqual(source, "primary")
            self.assertEqual(reason, "ok")
            self.assertEqual(text, GOOD_TEXT)

    def test_429_twice_then_backup_200(self):
        with mock.patch(
            "llm.requests.post",
            side_effect=[make_resp(429), make_resp(429), make_resp(200)],
        ) as post:
            text, source, reason = llm.explain(
                make_context(), api_key=FAKE_KEY, usage_path=self.usage, min_gap=0
            )
            self.assertEqual(post.call_count, 3)
            self.assertEqual(source, "backup")
            self.assertEqual(text, GOOD_TEXT)

    def test_ungrounded_number_falls_back_to_template(self):
        bad = "This rule blocks 999999 attacks with perfect safety."
        with mock.patch("llm.requests.post", return_value=make_resp(200, bad)):
            text, source, reason = llm.explain(
                make_context(), api_key=FAKE_KEY, usage_path=self.usage, min_gap=0
            )
            self.assertEqual(source, "template")
            self.assertEqual(reason, "grounding failed")
            self.assertIn("Deterministic template", text)

    def test_cache_hit_makes_zero_requests(self):
        with mock.patch("llm.requests.post", return_value=make_resp(200)) as post:
            cache = {}
            first = llm.explain(
                make_context(), cache=cache, api_key=FAKE_KEY,
                usage_path=self.usage, min_gap=0,
            )
            calls_after_first = post.call_count
            second = llm.explain(
                make_context(), cache=cache, api_key=FAKE_KEY,
                usage_path=self.usage, min_gap=0,
            )
            self.assertEqual(first, second)
            self.assertEqual(calls_after_first, 1)
            self.assertEqual(post.call_count, 1)

    def test_key_absent_from_exception_output(self):
        err = requests.ConnectionError("boom with key %s inside" % FAKE_KEY)
        with mock.patch("llm.requests.post", side_effect=err):
            text, source, reason = llm.explain(
                make_context(), api_key=FAKE_KEY, usage_path=self.usage, min_gap=0
            )
            blob = text + source + reason
            self.assertNotIn(FAKE_KEY, blob)
            self.assertIn("[redacted]", reason)
            self.assertEqual(source, "template")

    def test_key_absent_from_success_output(self):
        with mock.patch("llm.requests.post", return_value=make_resp(200)):
            text, source, reason = llm.explain(
                make_context(), api_key=FAKE_KEY, usage_path=self.usage, min_gap=0
            )
            blob = text + source + reason
            self.assertNotIn(FAKE_KEY, blob)

    def test_timeout_retries_once_then_backup(self):
        with mock.patch(
            "llm.requests.post",
            side_effect=[
                requests.Timeout("t1"),
                requests.Timeout("t2"),
                make_resp(200),
            ],
        ) as post:
            text, source, reason = llm.explain(
                make_context(), api_key=FAKE_KEY, usage_path=self.usage, min_gap=0
            )
            self.assertEqual(post.call_count, 3)
            self.assertEqual(source, "backup")

    def test_daily_limit_template_only(self):
        with open(self.usage, "w", encoding="utf-8") as f:
            json.dump({"date": llm.date.today().isoformat(), "count": 450}, f)
        with mock.patch("llm.requests.post") as post:
            text, source, reason = llm.explain(
                make_context(), api_key=FAKE_KEY, usage_path=self.usage, min_gap=0
            )
            self.assertEqual(source, "template")
            self.assertEqual(reason, "daily limit reached")
            post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
