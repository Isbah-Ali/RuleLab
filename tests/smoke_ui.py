import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.getLogger("streamlit").setLevel(logging.ERROR)

from streamlit.testing.v1 import AppTest

import config


def find_by_key(elements, key):
    for el in elements:
        if el.key == key:
            return el
    raise AssertionError("no element with key %r" % key)


def find_by_label(elements, label):
    for el in elements:
        if el.label == label:
            return el
    raise AssertionError("no element with label %r" % label)


def labels(elements, label):
    return [el for el in elements if el.label == label]


def main():
    results = []

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    results.append(("p04_c1_default_no_exception", not at.exception))
    if at.exception:
        print("EXCEPTION (default):", at.exception)
        return 1

    att_cols = labels(at.metric, "Attacks blocked")
    nor_cols = labels(at.metric, "Normal blocked")
    results.append(("p05_c1_both_columns", len(att_cols) == 2 and len(nor_cols) == 2))
    print("both columns: attacks_metrics=%d normal_metrics=%d" % (len(att_cols), len(nor_cols)))

    tables = at.session_state["builder_tables"]
    small = [r for r in tables["val"] if r["recall"] == "n<30"]
    no_pct = all("%" not in r["recall"] for r in tables["val"] if r["total"] < 30)
    results.append(("p05_c6_n_lt_30_rows", len(small) > 0 and no_pct))
    print("n<30 rows in val table: %d (example: %r)" % (len(small), small[0] if small else None))

    at2 = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at2.run()
    find_by_key(at2.checkbox, "n0_use").set_value(True)
    find_by_key(at2.selectbox, "n0_feat").set_value("src_bytes")
    at2.run()
    if at2.exception:
        print("EXCEPTION (enable cond1):", at2.exception)
        return 1
    sl = find_by_key(at2.select_slider, "n0_val_src_bytes")
    opts = list(sl.options)
    before = (
        labels(at2.metric, "Attacks blocked")[0].value,
        labels(at2.metric, "Normal blocked")[0].value,
    )
    sl.set_value(opts[-1])
    at2.run()
    if at2.exception:
        print("EXCEPTION (slider move):", at2.exception)
        return 1
    after = (
        labels(at2.metric, "Attacks blocked")[0].value,
        labels(at2.metric, "Normal blocked")[0].value,
    )
    both_changed = before[0] != after[0] and before[1] != after[1]
    results.append(("p05_c2_slider_move_changes_both", both_changed))
    print("slider move: src_bytes >= %r -> %r ; before=%r after=%r" % (opts[len(opts) // 2], opts[-1], before, after))

    deltas = [m.delta for m in at2.metric if m.delta not in (None, "")]
    results.append(("p05_c5_delta_appears", len(deltas) > 0))
    print("deltas after move: %r" % (deltas[:6],))

    at3 = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at3.run()
    find_by_key(at3.button, "preset_s0").click()
    at3.run()
    if at3.exception:
        print("EXCEPTION (preset_s0):", at3.exception)
        return 1
    with open(config.RESULTS_DIR / "eval.json", encoding="utf-8") as f:
        ev = json.load(f)
    b0 = next(e for e in ev["entries"] if e["method"] == "B0" and e["cap"] == 0.01)
    def fmt(m):
        return "%s / %s" % (format(m["attacks_blocked"], ","), format(m["attacks_total"], ","))
    ui_val = labels(at3.metric, "Attacks blocked")[0].value
    ui_test = labels(at3.metric, "Attacks blocked")[1].value
    exp_val = fmt(b0["val"])
    exp_test = fmt(b0["test"])
    results.append(("p05_c4_preset_val_matches_b0", ui_val == exp_val))
    results.append(("p05_c4_preset_test_matches_b0", ui_test == exp_test))
    print("preset flag=S0: ui_val=%r exp=%r ui_test=%r exp=%r" % (ui_val, exp_val, ui_test, exp_test))

    at4 = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at4.run()
    at4.radio(key="dataset").set_value("KDDTest+")
    at4.run()
    results.append(("p04_c1_kddtest_no_exception", not at4.exception))

    find_by_key(at4.button, "mine_btn").click()
    at4.run()
    results.append(("p04_c4_mine_no_exception", not at4.exception))
    load_btns = [b for b in at4.button if b.key.startswith("lb")]
    results.append(("p04_c4_rules_at_most_5", 1 <= len(load_btns) <= 5))

    with open(config.RESULTS_DIR / "drill.json", encoding="utf-8") as f:
        drill = json.load(f)
    fams = sorted(drill["families"].keys())
    at5 = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at5.run()
    drill_ok = True
    for fam in fams[:2]:
        find_by_key(at5.selectbox, "drill_family").set_value(fam)
        at5.run()
        if at5.exception:
            drill_ok = False
            print("EXCEPTION (drill %s):" % fam, at5.exception)
    results.append(("p06_c6_two_families_render", drill_ok))
    print("drill tab rendered families: %r" % (fams[:2],))

    find_by_key(at5.selectbox, "drill_family").set_value("neptune")
    at5.run()
    find_by_key(at5.button, "drill_rerun").click()
    at5.run()
    results.append(("p06_c6_rerun_no_exception", not at5.exception))
    if at5.exception:
        print("EXCEPTION (drill rerun):", at5.exception)
        return 1
    live = at5.session_state["drill_live"]
    jc = drill["families"]["neptune"]["by_cap"]["0.01"]
    match = (
        live["family"] == "neptune"
        and abs(live["seen_recall_val"] - jc["seen_recall_val"]) < 1e-12
        and abs(live["unseen_recall_val"] - jc["unseen_recall_val"]) < 1e-12
    )
    results.append(("p06_c6_rerun_matches_json", match))
    print(
        "live rerun neptune @1%%: seen=%.6f unseen=%.6f | json seen=%.6f unseen=%.6f match=%s runtime=%.1fs"
        % (
            live["seen_recall_val"],
            live["unseen_recall_val"],
            jc["seen_recall_val"],
            jc["unseen_recall_val"],
            match,
            live["runtime_s"],
        )
    )

    spec = importlib.util.spec_from_file_location("rulelab_app_bench", ROOT / "app.py")
    app_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_mod)
    k1 = ("protocol_type", "==", "tcp")
    k2 = next(k for k in app_mod.MASKS if k[0] == "count" and k[1] == ">=")
    k3 = next(k for k in app_mod.MASKS if k[0] == "dst_host_srv_count" and k[1] == ">=")
    conds3 = (k1, k2, k3)
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        app_mod.metrics_via_masks("val", app_mod.val_df, conds3)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    med = times[24]
    p95 = times[47]
    results.append(("p05_c3_median_under_100ms", med < 100.0))
    print("benchmark 50x 3-cond rule on full val: median=%.2f ms p95=%.2f ms rule=%r" % (med, p95, conds3))

    with mock.patch("requests.post") as mock_post:
        at6 = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
        at6.run()
        results.append(("p07_c5_no_llm_on_load", mock_post.call_count == 0))
        results.append(("p07_c2_default_no_exception", not at6.exception))
        with mock.patch.object(config, "GEMINI_API_KEY", ""):
            find_by_key(at6.button, "explain_btn").click()
            at6.run()
        er = at6.session_state["explain_result"]
        results.append(("p07_c2_template_missing_key", er["source"] == "template"))
        results.append(("p07_c2_reason_missing_key", er["reason"] == "missing API key"))
        results.append(("p07_c2_no_http_call", mock_post.call_count == 0))
        print("explain (no key): source=%r reason=%r http_calls=%d" % (er["source"], er["reason"], mock_post.call_count))

    print("")
    ok = True
    for name, passed in results:
        print("%s: %s" % (name, "PASS" if passed else "FAIL"))
        ok = ok and passed
    print("smoke_all_pass: %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
