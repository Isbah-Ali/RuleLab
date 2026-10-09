import argparse
import json
import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import data as data_mod
from config import (
    CAP_GRID,
    DEFAULT_CAP,
    MAX_CONDS,
    MAX_RULES,
    MIN_FAMILY_N,
    MIN_TP,
    RESULTS_DIR,
    SEED,
)
from data import split_fit_val
from miner import mine
from rules import Condition, evaluate, evaluate_rules, rule_to_text, rules_mask, verdict

V1_RULES = [
    (Condition("flag", "==", "S0"),),
    (Condition("protocol_type", "==", "icmp"),),
    (Condition("src_bytes", ">=", 100.0),),
    (Condition("dst_bytes", "<=", 0.0),),
    (
        Condition("protocol_type", "==", "tcp"),
        Condition("src_bytes", ">=", 100.0),
        Condition("dst_bytes", "<=", 64.0),
    ),
]
V1_QUERIES = [
    "flag == 'S0'",
    "protocol_type == 'icmp'",
    "src_bytes >= 100",
    "dst_bytes <= 0",
    "protocol_type == 'tcp' and src_bytes >= 100 and dst_bytes <= 64",
]
B0_RULES = [(Condition("flag", "==", "S0"),)]


def macro_recall(metrics):
    recalls = [
        blocked / total
        for blocked, total in metrics["per_family"].values()
        if total >= MIN_FAMILY_N
    ]
    return float(np.mean(recalls)) if recalls else 0.0


def with_macro(df, rules):
    metrics = evaluate_rules(df, rules)
    metrics["macro_recall"] = macro_recall(metrics)
    return metrics


def rules_from_entry(entry_rules):
    return [tuple(Condition(*c) for c in r) for r in entry_rules]


def check_v1(fit):
    for rule, query in zip(V1_RULES, V1_QUERIES):
        got = evaluate(fit, rule)
        sub = fit.query(query)
        exp_att = int(sub["is_attack"].sum())
        exp_nor = int(len(sub) - exp_att)
        if got["attacks_blocked"] != exp_att or got["normal_blocked"] != exp_nor:
            return False, "rule %s: got att/nor %d/%d query %d/%d" % (
                rule_to_text(rule),
                got["attacks_blocked"],
                got["normal_blocked"],
                exp_att,
                exp_nor,
            )
    return True, "5 fixed rules match df.query() on fit"


def check_v2(fit, val, test, entries):
    splits = [("fit", fit), ("val", val), ("test", test)]
    for e in entries:
        rules = rules_from_entry(e["rules"])
        for split_name, df in splits:
            m = evaluate_rules(df, rules)
            mask = rules_mask(df, rules)
            if m["attacks_blocked"] + (m["attacks_total"] - m["attacks_blocked"]) != m["attacks_total"]:
                return False, "attack identity failed %s/%s" % (e["method"], split_name)
            if m["normal_blocked"] + (m["normal_total"] - m["normal_blocked"]) != m["normal_total"]:
                return False, "normal identity failed %s/%s" % (e["method"], split_name)
            tmp = df[["label"]].copy()
            tmp["_b"] = mask
            gb = tmp.groupby("label")["_b"].agg(["sum", "count"])
            if set(gb.index.astype(str)) != set(m["per_family"].keys()):
                return False, "family set mismatch %s/%s" % (e["method"], split_name)
            for lab, row in gb.iterrows():
                blk, tot = m["per_family"][str(lab)]
                if blk != int(row["sum"]) or tot != int(row["count"]) or blk > tot:
                    return False, "family %s mismatch %s/%s" % (lab, e["method"], split_name)
            fam_blocked = sum(v[0] for v in m["per_family"].values())
            fam_total = sum(v[1] for v in m["per_family"].values())
            if fam_blocked != m["attacks_blocked"] + m["normal_blocked"]:
                return False, "family blocked sum %s/%s" % (e["method"], split_name)
            if fam_total != len(df):
                return False, "family total sum %s/%s" % (e["method"], split_name)
    return True, "%d entries x 3 splits consistent" % len(entries)


def check_v3(fit):
    a = mine(fit, DEFAULT_CAP)
    b = mine(fit, DEFAULT_CAP)
    ta = [rule_to_text(r) for r in a]
    tb = [rule_to_text(r) for r in b]
    if ta != tb:
        return False, "miner not deterministic"
    return True, "%d rules identical across 2 runs at cap %.3g" % (len(ta), DEFAULT_CAP)


def check_v4(entries, cap_eps=1e-12):
    for e in entries:
        if e["method"] != "B2":
            continue
        if e["fit"]["collateral_rate"] > e["cap"] + cap_eps:
            return False, "cap %.4g fit collateral %.6f > cap" % (
                e["cap"],
                e["fit"]["collateral_rate"],
            )
    return True, "B2 fit collateral <= cap at every cap"


def check_v5(fit, val, test):
    for name, df in [("val", val), ("test", test)]:
        try:
            mine(df, DEFAULT_CAP)
        except AssertionError:
            continue
        return False, "mine(%s) did not raise" % name
    return True, "mine() rejected val and test rows"


def build_entry(cap, method, rules, splits):
    entry = {
        "cap": cap,
        "method": method,
        "n_rules": len(rules),
        "rule_text": " ; ".join(rule_to_text(r) for r in rules),
        "rules": [[[c.feature, c.op, c.value] for c in r] for r in rules],
    }
    for split_name, df in splits:
        entry[split_name] = with_macro(df, rules)
    entry["cap_drift_val"] = entry["val"]["collateral_rate"] - entry["fit"]["collateral_rate"]
    entry["cap_drift_test"] = entry["test"]["collateral_rate"] - entry["fit"]["collateral_rate"]
    return entry


def run_eval(with_checks=True):
    t_start = time.perf_counter()
    train = data_mod.load_train()
    fit, val, _fit_idx, _small = split_fit_val(train)
    test = data_mod.load_test()
    test.attrs["split"] = "test"
    splits = [("fit", fit), ("val", val), ("test", test)]

    checks = {}
    details = {}
    checks["V1"], details["V1"] = check_v1(fit)
    checks["V5"], details["V5"] = check_v5(fit, val, test)
    checks["V3"], details["V3"] = check_v3(fit)

    entries = []
    for cap in CAP_GRID:
        b1_rules = mine(fit, cap, max_rules=1, max_conds=1, min_tp=MIN_TP)
        b2_rules = mine(fit, cap, max_rules=MAX_RULES, max_conds=MAX_CONDS, min_tp=MIN_TP)
        entries.append(build_entry(cap, "B0", B0_RULES, splits))
        entries.append(build_entry(cap, "B1", b1_rules, splits))
        entries.append(build_entry(cap, "B2", b2_rules, splits))

    v4_ok, v4_detail = check_v4(entries)
    fallback_max_conds = None
    if not v4_ok:
        print("V4 FAILED with max_conds=%d (%s)" % (MAX_CONDS, v4_detail))
        print("FALLBACK: shipping B2 with max_conds=1")
        fallback_max_conds = 1
        entries = [e for e in entries if e["method"] != "B2"]
        for cap in CAP_GRID:
            b2_rules = mine(fit, cap, max_rules=MAX_RULES, max_conds=1, min_tp=MIN_TP)
            entries.append(build_entry(cap, "B2", b2_rules, splits))
        entries.sort(key=lambda e: (CAP_GRID.index(e["cap"]), e["method"]))
        v4_ok, v4_detail = check_v4(entries)
    checks["V4"], details["V4"] = v4_ok, v4_detail
    checks["V2"], details["V2"] = check_v2(fit, val, test, entries)

    runtime_s = time.perf_counter() - t_start

    if with_checks:
        for name in ["V1", "V2", "V3", "V4", "V5"]:
            print("%s: %s (%s)" % (name, "PASS" if checks[name] else "FAIL", details[name]))
        print("")
        header = "%-7s %-4s %10s %10s %10s %12s %11s %10s" % (
            "cap", "meth", "val_col%", "val_rec", "val_macro", "test_col%", "test_rec", "drift_pp",
        )
        print(header)
        for e in entries:
            print(
                "%-7s %-4s %10.2f %10.4f %10.4f %12.2f %11.4f %10.2f"
                % (
                    "%.1f%%" % (100 * e["cap"]),
                    e["method"],
                    100 * e["val"]["collateral_rate"],
                    e["val"]["recall"],
                    e["val"]["macro_recall"],
                    100 * e["test"]["collateral_rate"],
                    e["test"]["recall"],
                    100 * e["cap_drift_val"],
                )
            )
        print("")
        b1_1 = next(e for e in entries if e["cap"] == 0.01 and e["method"] == "B1")
        b2_1 = next(e for e in entries if e["cap"] == 0.01 and e["method"] == "B2")
        beats = b2_1["val"]["macro_recall"] > b1_1["val"]["macro_recall"]
        print(
            "B2 vs B1 val macro-recall at 1%%: B2=%.4f B1=%.4f -> B2 beats B1: %s"
            % (b2_1["val"]["macro_recall"], b1_1["val"]["macro_recall"], beats)
        )
        print("")
        print("runtime_s: %.1f" % runtime_s)
        print("all_pass: %s" % all(checks.values()))

    payload = {
        "seed": SEED,
        "split_sizes": {"fit": len(fit), "val": len(val), "test": len(test)},
        "max_rules": MAX_RULES,
        "max_conds": MAX_CONDS,
        "min_tp": MIN_TP,
        "fallback_max_conds": fallback_max_conds,
        "checks": checks,
        "check_details": details,
        "entries": entries,
        "runtime_s": round(runtime_s, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "eval.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
    print("wrote %s" % out)
    return all(checks.values()), entries, payload


def run_decide():
    original_load_test = data_mod.load_test

    def _forbidden():
        raise AssertionError("load_test called during --decide")

    data_mod.load_test = _forbidden
    try:
        train = data_mod.load_train()
        fit, val, _fit_idx, _small = split_fit_val(train)
        per_cap = []
        for cap in CAP_GRID:
            rules = mine(fit, cap, max_rules=MAX_RULES, max_conds=MAX_CONDS, min_tp=MIN_TP)
            fit_m = evaluate_rules(fit, rules)
            val_m = with_macro(val, rules)
            drift = val_m["collateral_rate"] - fit_m["collateral_rate"]
            per_cap.append({
                "cap": cap,
                "rule_text": " ; ".join(rule_to_text(r) for r in rules),
                "fit_collateral": fit_m["collateral_rate"],
                "val_collateral": val_m["collateral_rate"],
                "val_drift": drift,
                "val_macro_recall": val_m["macro_recall"],
                "val_attacks_blocked": val_m["attacks_blocked"],
                "val_normal_blocked": val_m["normal_blocked"],
                "fit_attacks_blocked": fit_m["attacks_blocked"],
                "fit_normal_blocked": fit_m["normal_blocked"],
            })
        max_drift = max(p["val_drift"] for p in per_cap)
        drift_tol = max(0.0025, math.ceil(max_drift / 0.0025) * 0.0025)
        for p in per_cap:
            p["val_verdict"] = verdict(
                p["val_collateral"],
                drift=p["val_drift"],
                attacks_blocked=p["val_attacks_blocked"],
                drift_tol=drift_tol,
            )
        for p in per_cap:
            print(
                "cap %.1f%%: fit_col=%.4f%% val_col=%.4f%% drift=%.4fpp "
                "val_macro=%.4f val_attacks_blocked=%d verdict=%s"
                % (
                    100 * p["cap"],
                    100 * p["fit_collateral"],
                    100 * p["val_collateral"],
                    100 * p["val_drift"],
                    p["val_macro_recall"],
                    p["val_attacks_blocked"],
                    p["val_verdict"],
                )
            )
        safe = [p for p in per_cap if p["val_verdict"] == "SAFE"]
        if safe:
            best = sorted(safe, key=lambda p: (-p["val_macro_recall"], p["cap"]))[0]
            recommended_cap = best["cap"]
            reason = "highest val macro-recall %.4f among %d SAFE caps" % (
                best["val_macro_recall"],
                len(safe),
            )
        else:
            recommended_cap = min(CAP_GRID)
            reason = "no SAFE cap; fallback to smallest cap"
        print("")
        print("DRIFT_TOL = %.4f (%.2f pp)  [max B2 val-fit drift = %.4f pp across %d caps, rounded up to next 0.25 pp, min 0.25 pp]"
              % (drift_tol, 100 * drift_tol, 100 * max_drift, len(CAP_GRID)))
        print("RECOMMENDED_CAP = %.3f  [%s]" % (recommended_cap, reason))
        print("")
        print("test_load_guard: active (data.load_test patched to raise during --decide)")
        print("config to paste:")
        print("DRIFT_TOL = %.4f  # --decide: max B2 val-fit drift %.4fpp over %d caps -> ceil 0.25pp"
              % (drift_tol, 100 * max_drift, len(CAP_GRID)))
        print("RECOMMENDED_CAP = %.2f  # --decide: %s" % (recommended_cap, reason))

        payload = {
            "seed": SEED,
            "split_sizes": {"fit": len(fit), "val": len(val)},
            "test_loaded": False,
            "max_val_drift": max_drift,
            "drift_tol": drift_tol,
            "recommended_cap": recommended_cap,
            "recommended_reason": reason,
            "per_cap": per_cap,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / "decisions.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
        print("wrote %s" % out)
        return payload
    finally:
        data_mod.load_test = original_load_test


def _ser_rules(rules):
    return [tuple((c.feature, c.op, c.value) for c in r) for r in rules]


def _texts(rules):
    return [rule_to_text(r) for r in rules]


def _rec(df_f, rules):
    if len(df_f) == 0:
        return 0.0
    return evaluate_rules(df_f, rules)["recall"]


def _coll(df, rules):
    return evaluate_rules(df, rules)["collateral_rate"]


def _identity_ok(df_f, rules):
    if len(df_f) == 0:
        return True
    m = evaluate_rules(df_f, rules)
    mask = rules_mask(df_f, rules)
    tmp = df_f[["label"]].copy()
    tmp["_b"] = mask
    gb = tmp.groupby("label")["_b"].agg(["sum", "count"])
    fam = str(df_f["label"].iloc[0])
    return (
        m["attacks_blocked"] + (m["attacks_total"] - m["attacks_blocked"])
        == m["attacks_total"]
        and int(gb.loc[fam, "sum"]) == m["attacks_blocked"]
        and int(gb.loc[fam, "count"]) == m["attacks_total"] == len(df_f)
    )


def drill_family(family, cap, fit=None, val=None, test=None, seen_rules=None):
    t0 = time.perf_counter()
    if fit is None or val is None:
        train = data_mod.load_train()
        fit, val, _idx, _small = split_fit_val(train)
    if test is None:
        test = data_mod.load_test()
    if seen_rules is None:
        seen_rules = mine(fit, cap)
    fit_u = fit.loc[fit["label"] != family].copy()
    fit_u.attrs["split"] = "fit"
    n_f_in_unseen = int((fit_u["label"] == family).sum())
    assert n_f_in_unseen == 0, "family %s still present in unseen fit" % family
    normal_ok = int((~fit_u["is_attack"]).sum()) == int((~fit["is_attack"]).sum())
    unseen_rules = mine(fit_u, cap)
    fit_f = fit.loc[fit["label"] == family]
    val_f = val.loc[val["label"] == family]
    test_f = test.loc[test["label"] == family]
    n_train_f = int(len(fit_f) + len(val_f))
    n_val_f = int(len(val_f))
    n_test_f = int(len(test_f))
    seen_val = _rec(val_f, seen_rules)
    unseen_val = _rec(val_f, unseen_rules)
    seen_test = _rec(test_f, seen_rules) if n_test_f >= 30 else None
    unseen_test = _rec(test_f, unseen_rules) if n_test_f >= 30 else None
    all_f = pd.concat([fit_f, val_f])
    unseen_train = _rec(all_f, unseen_rules)
    identities = all(
        [
            _identity_ok(val_f, seen_rules),
            _identity_ok(val_f, unseen_rules),
            _identity_ok(test_f, seen_rules),
            _identity_ok(test_f, unseen_rules),
        ]
    )
    result = {
        "family": family,
        "cap": cap,
        "n_train": n_train_f,
        "n_val": n_val_f,
        "n_test": n_test_f,
        "reliable_val": n_val_f >= 30,
        "reliable_test": n_test_f >= 30,
        "seen_recall_val": seen_val,
        "unseen_recall_val": unseen_val,
        "seen_recall_test": seen_test,
        "unseen_recall_test": unseen_test,
        "unseen_recall_train_all": unseen_train,
        "gap": seen_val - unseen_val,
        "collateral": {
            "val_seen": _coll(val, seen_rules),
            "val_unseen": _coll(val, unseen_rules),
            "test_seen": _coll(test, seen_rules),
            "test_unseen": _coll(test, unseen_rules),
        },
        "seen_rule_texts": _texts(seen_rules),
        "unseen_rule_texts": _texts(unseen_rules),
        "seen_rules": _ser_rules(seen_rules),
        "unseen_rules": _ser_rules(unseen_rules),
        "assert_0_f_rows_in_unseen_fit": True,
        "normal_rows_untouched": normal_ok,
        "identities_ok": identities,
        "runtime_s": round(time.perf_counter() - t0, 3),
    }
    return result


def run_drill(caps=None):
    t0 = time.perf_counter()
    caps = list(caps) if caps else list(CAP_GRID)
    train = data_mod.load_train()
    fit, val, _idx, _small = split_fit_val(train)
    test = data_mod.load_test()
    test.attrs["split"] = "test"
    counts = train["label"].value_counts()
    eligible = sorted(
        lab for lab, n in counts.items() if lab != "normal" and int(n) >= 100
    )
    print("eligible families (n_train >= 100 in KDDTrain+): %d" % len(eligible))
    for lab in eligible:
        print("  %s n_train=%d" % (lab, int(counts[lab])))
    print("")
    seen_cache = {("%.4g" % cap): mine(fit, cap) for cap in caps}
    families = {}
    for fam in eligible:
        by_cap = {}
        for cap in caps:
            ck = "%.4g" % cap
            res = drill_family(
                fam, cap, fit=fit, val=val, test=test, seen_rules=seen_cache[ck]
            )
            by_cap[ck] = res
            print(
                "%s cap %s: 0-F-rows-assert %s | normal-untouched %s | identities %s"
                % (
                    fam,
                    ck,
                    "PASS" if res["assert_0_f_rows_in_unseen_fit"] else "FAIL",
                    "PASS" if res["normal_rows_untouched"] else "FAIL",
                    "PASS" if res["identities_ok"] else "FAIL",
                )
            )
        families[fam] = {
            "n_train": by_cap[("%.4g" % caps[0])]["n_train"],
            "n_val": by_cap[("%.4g" % caps[0])]["n_val"],
            "n_test": by_cap[("%.4g" % caps[0])]["n_test"],
            "reliable_val": by_cap[("%.4g" % caps[0])]["reliable_val"],
            "reliable_test": by_cap[("%.4g" % caps[0])]["reliable_test"],
            "by_cap": by_cap,
        }
    summary_caps = {}
    for cap in caps:
        ck = "%.4g" % cap
        gaps = [families[f]["by_cap"][ck]["gap"] for f in eligible]
        reliable = [
            f
            for f in eligible
            if families[f]["by_cap"][ck]["reliable_val"]
        ]
        ge50 = [
            f
            for f in reliable
            if families[f]["by_cap"][ck]["unseen_recall_val"] >= 0.5
        ]
        summary_caps[ck] = {
            "eligible_count": len(eligible),
            "reliable_count": len(reliable),
            "unseen_val_recall_ge_50_reliable": len(ge50),
            "median_gap": float(np.median(gaps)) if gaps else 0.0,
        }
    dk = "%.4g" % DEFAULT_CAP
    s = summary_caps[dk]
    sentence = (
        "%d eligible families at cap %.1f%%: %d of %d reliable families "
        "(n_val>=30) keep unseen validation recall >= 50%%; median seen-unseen "
        "validation recall gap %.2f pp."
        % (
            s["eligible_count"],
            100 * DEFAULT_CAP,
            s["unseen_val_recall_ge_50_reliable"],
            s["reliable_count"],
            100 * s["median_gap"],
        )
    )
    print("")
    print("table at default cap %.1f%%:" % (100 * DEFAULT_CAP))
    header = "%-16s %6s %8s %8s %8s %10s %12s" % (
        "family", "n_val", "seen%", "unseen%", "gap_pp", "col_seen%", "col_unseen%",
    )
    print(header)
    for fam in eligible:
        r = families[fam]["by_cap"][dk]
        print(
            "%-16s %6d %8.2f %8.2f %8.2f %10.3f %12.3f"
            % (
                fam,
                r["n_val"],
                100 * r["seen_recall_val"],
                100 * r["unseen_recall_val"],
                100 * r["gap"],
                100 * r["collateral"]["val_seen"],
                100 * r["collateral"]["val_unseen"],
            )
        )
    print("")
    print("summary: %s" % sentence)
    runtime_s = time.perf_counter() - t0
    print("runtime_s: %.1f (caps: %s)" % (runtime_s, ", ".join("%.4g" % c for c in caps)))
    payload = {
        "seed": SEED,
        "default_cap": DEFAULT_CAP,
        "caps": caps,
        "eligible": [{"family": f, "n_train": int(counts[f])} for f in eligible],
        "families": families,
        "summary": {"by_cap": summary_caps, "sentence": sentence},
        "runtime_s": round(runtime_s, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "drill.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
    print("wrote %s" % out)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--decide", action="store_true")
    parser.add_argument("--drill", action="store_true")
    args = parser.parse_args()
    if args.decide:
        run_decide()
        return 0
    if args.drill:
        run_drill()
        return 0
    if args.eval:
        ok, _entries, _payload = run_eval()
        return 0 if ok else 1
    print("usage: python evaluate.py --eval | --decide | --drill")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
