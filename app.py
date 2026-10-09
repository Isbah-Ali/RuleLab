import json

import numpy as np
import pandas as pd
import streamlit as st

import config
import data as data_mod
from data import split_fit_val
from evaluate import drill_family
from llm import explain
from miner import candidate_conditions, mine
from rules import Condition, fmt_pct, metrics_from_mask, rule_to_text, verdict

st.set_page_config(page_title="RuleLab", layout="wide")

SER_FLAG_S0 = ((("flag", "==", "S0"),),)


@st.cache_data
def load_data():
    train = data_mod.load_train()
    fit, val, _idx, _small = split_fit_val(train)
    test = data_mod.load_test()
    test.attrs["split"] = "test"
    return fit, val, test


@st.cache_data
def load_json_file(name):
    path = config.RESULTS_DIR / name
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def mine_cached(cap, _fit):
    rules = mine(_fit, cap)
    return [tuple((c.feature, c.op, c.value) for c in r) for r in rules]


@st.cache_data
def candidates_for(_fit):
    return candidate_conditions(_fit)


@st.cache_resource
def build_mask_cache(_fit, _val, _test, _cands):
    cache = {}
    for ds_name, df in (("fit", _fit), ("val", _val), ("test", _test)):
        for cond in _cands:
            key = (cond.feature, cond.op, cond.value)
            entry = cache.setdefault(key, {})
            entry[ds_name] = cond.mask(df)
    return cache


def rules_from_serialized(serialized):
    return [tuple(Condition(*c) for c in r) for r in serialized]


def metrics_via_masks(ds_key, ds_df, conds):
    if not conds:
        mask = np.zeros(len(ds_df), dtype=bool)
    else:
        k0 = (conds[0][0], conds[0][1], conds[0][2])
        mask = MASKS[k0][ds_key].copy()
        for c in conds[1:]:
            mask &= MASKS[(c[0], c[1], c[2])][ds_key]
    return metrics_from_mask(ds_df, mask)


def _load_builder(conditions):
    ss = st.session_state
    ss["cat_protocol"] = "any"
    ss["cat_service"] = "any"
    ss["cat_flag"] = "any"
    for i in range(3):
        ss["n%d_use" % i] = False
    ni = 0
    for feat, op, val in conditions:
        if feat == "protocol_type":
            ss["cat_protocol"] = val
        elif feat == "service":
            ss["cat_service"] = val
        elif feat == "flag":
            ss["cat_flag"] = val
        elif ni < 3:
            ss["n%d_use" % ni] = True
            ss["n%d_feat" % ni] = feat
            ss["n%d_op" % ni] = op
            opts = NUM_OPTIONS[feat]
            v = float(val)
            if v not in opts:
                v = min(opts, key=lambda o: abs(o - v))
            ss["n%d_val_%s" % (ni, feat)] = v
            ni += 1


fit_df, val_df, test_df = load_data()
CANDS = candidates_for(fit_df)
MASKS = build_mask_cache(fit_df, val_df, test_df, CANDS)
NUM_OPTIONS = {}
for _c in CANDS:
    if _c.op != "==":
        NUM_OPTIONS.setdefault(_c.feature, [])
        if _c.value not in NUM_OPTIONS[_c.feature]:
            NUM_OPTIONS[_c.feature].append(_c.value)
for _col in NUM_OPTIONS:
    NUM_OPTIONS[_col] = sorted(NUM_OPTIONS[_col])


def inject_css():
    st.markdown(
        """
<style>
html, body, [class*="css"] {
  font-family: "Segoe UI", system-ui, sans-serif;
}
.stApp {
  max-width: 1400px;
}
[data-testid="stMetric"] {
  background: #1E293B;
  border: 1px solid #334155;
  border-radius: 12px;
  padding: 16px;
}
[data-testid="stMetricLabel"] {
  color: #94A3B8;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
[data-testid="stMetricValue"] {
  color: #E2E8F0;
  font-size: 36px;
  font-weight: 700;
}
div:has(> .mk-attacks) + div [data-testid="stMetric"] {
  border-left: 4px solid #22C55E;
}
div:has(> .mk-attacks) + div [data-testid="stMetricValue"] {
  color: #22C55E;
}
div:has(> .mk-normal) + div [data-testid="stMetric"] {
  border-left: 4px solid #F43F5E;
}
div:has(> .mk-normal) + div [data-testid="stMetricValue"] {
  color: #F43F5E;
}
div:has(> .mk-precision) + div [data-testid="stMetric"] {
  border-left: 4px solid #38BDF8;
}
.rl-banner {
  background: rgba(245,158,11,0.12);
  border: 1px solid #F59E0B;
  border-radius: 8px;
  color: #F59E0B;
  font-size: 14px;
  padding: 10px 14px;
  margin-bottom: 8px;
}
.rl-pill {
  display: inline-block;
  border-radius: 999px;
  padding: 4px 14px;
  font-weight: 700;
  color: #0F172A;
  font-size: 14px;
  margin-bottom: 4px;
}
.rl-pill-safe { background: #22C55E; }
.rl-pill-caution { background: #F59E0B; }
.rl-pill-unsafe { background: #F43F5E; }
.rl-pill-low { background: #94A3B8; }
.rl-gauge {
  background: #334155;
  border-radius: 8px;
  height: 10px;
  overflow: hidden;
  margin: 4px 0 8px 0;
}
.rl-gauge > div {
  background: #38BDF8;
  height: 100%;
}
.rl-gauge-full > div {
  background: #F43F5E;
}
.rl-badge {
  display: inline-block;
  border-radius: 999px;
  padding: 2px 10px;
  font-size: 12px;
  font-weight: 600;
  border: 1px solid;
  margin-right: 8px;
}
.rl-badge-primary { color: #38BDF8; border-color: #38BDF8; }
.rl-badge-backup { color: #94A3B8; border-color: #94A3B8; }
.rl-badge-template { color: #F59E0B; border-color: #F59E0B; }
[role="tablist"] button[aria-selected="true"] {
  border-bottom: 2px solid #38BDF8;
  color: #38BDF8;
}
</style>
""",
        unsafe_allow_html=True,
    )


inject_css()
st.markdown(
    '<div class="rl-banner">Data is a 1999 simulation. Collateral counts '
    "labeled-normal rows only. Not real business cost.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")
    dataset = st.radio(
        "Dataset", ["Validation", "KDDTest+"], key="dataset", horizontal=True
    )
    cap_pct = st.slider(
        "Fit cap (% of fit normal)", 0.1, 5.0, 1.0, 0.1, key="cap_pct"
    )

cap = round(cap_pct / 100.0, 6)
eval_df = val_df if dataset == "Validation" else test_df

tab_builder, tab_miner, tab_drill, tab_eval = st.tabs(
    ["Builder", "Miner", "Drill", "Evaluation"]
)

with tab_builder:
    st.subheader("Live collateral meter")
    pr1, pr2 = st.columns(2)
    with pr1:
        st.button(
            "Hand rule: flag = S0",
            key="preset_s0",
            on_click=_load_builder,
            args=(SER_FLAG_S0[0],),
        )
    with pr2:
        if st.button("Best mined rule #1 at sidebar cap", key="preset_mined"):
            ser_m = mine_cached(cap, fit_df)
            st.session_state["miner_result"] = {"cap": cap, "rules": ser_m}
            if ser_m:
                _load_builder(ser_m[0])
                st.success("Loaded mined rule #1 into the Builder.")
            else:
                st.warning("No rules mined at this cap.")

    b_in, b_out = st.columns([1, 2], gap="large")
    with b_in:
        st.markdown("Categorical conditions")
        protocol_opts = ["any"] + sorted(fit_df["protocol_type"].unique().tolist())
        service_opts = ["any"] + sorted(fit_df["service"].unique().tolist())
        flag_opts = ["any"] + sorted(fit_df["flag"].unique().tolist())
        proto = st.selectbox("protocol_type", protocol_opts, key="cat_protocol")
        service = st.selectbox("service", service_opts, key="cat_service")
        flag = st.selectbox("flag", flag_opts, key="cat_flag")
        st.markdown("Numeric conditions (up to 3, fit quantiles)")
        numeric_conds = []
        for i in range(3):
            use = st.checkbox("Use condition %d" % (i + 1), key="n%d_use" % i)
            feat = st.selectbox(
                "Feature %d" % (i + 1), config.NUMERIC, key="n%d_feat" % i
            )
            op = st.selectbox(
                "Operator %d" % (i + 1), [">=", "<="], key="n%d_op" % i
            )
            opts = NUM_OPTIONS[feat]
            val_sl = st.select_slider(
                "Value %d" % (i + 1),
                options=opts,
                value=opts[len(opts) // 2],
                key="n%d_val_%s" % (i, feat),
            )
            if use:
                numeric_conds.append(Condition(feat, op, float(val_sl)))

    conds = []
    if proto != "any":
        conds.append(Condition("protocol_type", "==", proto))
    if service != "any":
        conds.append(Condition("service", "==", service))
    if flag != "any":
        conds.append(Condition("flag", "==", flag))
    conds.extend(numeric_conds)

    ser_conds = tuple((c.feature, c.op, c.value) for c in conds)
    st.markdown("Rule: **%s**" % rule_to_text(tuple(conds)))

    prev_all = st.session_state.get("builder_prev") or {}
    now_all = {}
    table_hooks = {}
    builder_metrics = {}
    with b_out:
        ds_cols = st.columns(2)
    ds_defs = [("Validation", "val", val_df), ("KDDTest+", "test", test_df)]
    for (ds_name, ds_key, ds_df), col in zip(ds_defs, ds_cols):
        with col:
            st.markdown("**%s**" % ds_name)
            m = metrics_via_masks(ds_key, ds_df, ser_conds)
            fit_m = metrics_via_masks("fit", fit_df, ser_conds)
            drift = m["collateral_rate"] - fit_m["collateral_rate"]
            v = verdict(
                m["collateral_rate"],
                drift=drift,
                attacks_blocked=m["attacks_blocked"],
            )
            used = min(m["collateral_rate"] / cap, 1.0) if cap > 0 else 0.0
            prev = prev_all.get(ds_key)
            d_att = None
            d_nor = None
            d_pre = None
            d_ver = None
            if prev is not None:
                if m["attacks_blocked"] != prev["att"]:
                    d_att = m["attacks_blocked"] - prev["att"]
                if m["normal_blocked"] != prev["nor"]:
                    d_nor = m["normal_blocked"] - prev["nor"]
                if abs(m["precision"] - prev["prec"]) > 1e-12:
                    d_pre = round(100 * (m["precision"] - prev["prec"]), 1)
                if v != prev["ver"]:
                    d_ver = "was %s" % prev["ver"]
            st.markdown('<span class="mk mk-attacks"></span>', unsafe_allow_html=True)
            st.metric(
                "Attacks blocked",
                "%s / %s" % (f"{m['attacks_blocked']:,}", f"{m['attacks_total']:,}"),
                delta=d_att,
            )
            st.caption("recall %.1f%%" % (100 * m["recall"]))
            st.markdown('<span class="mk mk-normal"></span>', unsafe_allow_html=True)
            st.metric(
                "Normal blocked",
                "%s (%.2f%%)" % (f"{m['normal_blocked']:,}", 100 * m["collateral_rate"]),
                delta=d_nor,
                delta_color="inverse",
            )
            st.markdown('<span class="mk mk-precision"></span>', unsafe_allow_html=True)
            st.metric(
                "Precision", "%.1f%%" % (100 * m["precision"]), delta=d_pre
            )
            pill_cls = {
                "SAFE": "rl-pill-safe",
                "CAUTION": "rl-pill-caution",
                "UNSAFE": "rl-pill-unsafe",
                "LOW EVIDENCE": "rl-pill-low",
            }.get(v, "rl-pill-low")
            st.markdown(
                '<div class="rl-pill %s">%s</div>' % (pill_cls, v),
                unsafe_allow_html=True,
            )
            st.metric("Verdict", v, delta=d_ver)
            gauge_cls = "rl-gauge rl-gauge-full" if used >= 1.0 else "rl-gauge"
            st.markdown(
                '<div class="%s"><div style="width:%.2f%%;"></div></div>'
                '<div style="color:#94A3B8;font-size:12px;">%.1f%% of cap used</div>'
                % (gauge_cls, 100 * used, 100 * used),
                unsafe_allow_html=True,
            )
            st.metric("Cap drift", "%+.2f pp" % (100 * drift))
            rows = []
            for lab, (blocked, total) in sorted(
                m["per_family"].items(), key=lambda kv: -kv[1][1]
            ):
                rows.append(
                    {
                        "family": lab,
                        "blocked": blocked,
                        "total": total,
                        "recall": fmt_pct(blocked, total),
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch")
            table_hooks[ds_key] = rows
            builder_metrics[ds_key] = {"m": m, "drift": drift, "verdict": v}
            now_all[ds_key] = {
                "att": m["attacks_blocked"],
                "nor": m["normal_blocked"],
                "prec": m["precision"],
                "ver": v,
            }
    st.session_state["builder_prev"] = now_all
    st.session_state["builder_tables"] = table_hooks

    def _explain_context():
        ctx = {
            "rule_text": rule_to_text(tuple(conds)),
            "cap_pct": round(cap_pct, 3),
            "datasets": {},
        }
        for ctx_key, ds_key, name in (
            ("validation", "val", "Validation"),
            ("kddtest", "test", "KDDTest+"),
        ):
            bm = builder_metrics[ds_key]
            m = bm["m"]
            ctx["datasets"][ctx_key] = {
                "name": name,
                "attacks_blocked": m["attacks_blocked"],
                "attacks_total": m["attacks_total"],
                "normal_blocked": m["normal_blocked"],
                "normal_total": m["normal_total"],
                "recall_pct": round(100 * m["recall"], 2),
                "collateral_pct": round(100 * m["collateral_rate"], 2),
                "precision_pct": round(100 * m["precision"], 2),
                "verdict": bm["verdict"],
                "cap_drift_pp": round(100 * bm["drift"], 2),
            }
        return ctx

    if st.button("Explain this rule", key="explain_btn"):
        ctx = _explain_context()
        llm_cache = st.session_state.setdefault("llm_cache", {})
        text, source, reason = explain(ctx, cache=llm_cache)
        st.session_state["explain_result"] = {
            "text": text,
            "source": source,
            "reason": reason,
        }
    er = st.session_state.get("explain_result")
    if er is not None:
        st.markdown("**Explain this rule**")
        badge_cls = {
            "primary": "rl-badge-primary",
            "backup": "rl-badge-backup",
        }.get(er["source"], "rl-badge-template")
        st.markdown(
            '<span class="rl-badge %s">source: %s</span>'
            '<span style="color:#94A3B8;font-size:12px;">reason: %s</span>'
            % (badge_cls, er["source"], er["reason"]),
            unsafe_allow_html=True,
        )
        st.write(er["text"])

with tab_miner:
    st.subheader("Greedy miner (fit split only)")
    st.caption("Cap = %.1f%% of fit normal" % cap_pct)
    col_a, _col_b = st.columns([1, 3])
    with col_a:
        mine_clicked = st.button("Mine rules at cap", key="mine_btn")
    if mine_clicked:
        with st.spinner("Mining..."):
            ser = mine_cached(cap, fit_df)
        st.session_state["miner_result"] = {"cap": cap, "rules": ser}
    stored = st.session_state.get("miner_result")
    if stored is None:
        st.info("Press 'Mine rules at cap' to mine rules on the fit split.")
    else:
        st.caption(
            "Mined at cap %.1f%% -- showing metrics on %s"
            % (100 * stored["cap"], dataset)
        )
        ser = stored["rules"]
        if not ser:
            st.info("No rules found at this cap.")
        for i, r in enumerate(ser):
            rule_i = rules_from_serialized([r])[0]
            m_i = metrics_via_masks(
                "val" if dataset == "Validation" else "test", eval_df, r
            )
            st.markdown("**Rule %d:** %s" % (i + 1, rule_to_text(rule_i)))
            st.write(
                "collateral: %.3f%% | attacks blocked: %d / %d | precision: %.1f%%"
                % (
                    100 * m_i["collateral_rate"],
                    m_i["attacks_blocked"],
                    m_i["attacks_total"],
                    100 * m_i["precision"],
                )
            )
            st.button(
                "Load into Builder",
                key="lb%d" % i,
                on_click=_load_builder,
                args=(r,),
            )
        st.info("Loaded rules appear in the Builder tab.")

with tab_drill:
    st.subheader("Zero-day rule drill")
    drill = load_json_file("drill.json")
    if drill is None:
        st.info("Run `python evaluate.py --drill` to produce results/drill.json.")
    else:
        fams = sorted(drill["families"].keys())
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            fam = st.selectbox("Attack family", fams, key="drill_family")
        cap_keys = ["%.4g" % c for c in drill["caps"]]
        with dcol2:
            cap_key = st.selectbox(
                "Cap",
                cap_keys,
                index=cap_keys.index("%.4g" % drill["default_cap"]),
                key="drill_cap",
            )
        st.info(drill["summary"]["sentence"])
        info = drill["families"][fam]
        bc = info["by_cap"][cap_key]
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("n_train", f"{info['n_train']:,}")
        n2.metric("n_val", f"{info['n_val']:,}")
        n3.metric("n_test", f"{info['n_test']:,}")
        n4.metric(
            "Reliable (n>=30)",
            "val %s / test %s"
            % ("yes" if info["reliable_val"] else "no", "yes" if info["reliable_test"] else "no"),
        )
        chart_data = pd.DataFrame(
            {
                "Seen recall": [bc["seen_recall_val"], bc["seen_recall_test"]],
                "Unseen recall": [bc["unseen_recall_val"], bc["unseen_recall_test"]],
            },
            index=["Validation", "Test"],
        )
        if not info["reliable_test"]:
            chart_data = chart_data.drop(index=["Test"])
            st.caption("Test recall hidden (n_test < 30).")
        try:
            import altair as alt

            long_df = (
                chart_data.reset_index()
                .melt(id_vars="index", var_name="Series", value_name="Recall")
                .rename(columns={"index": "Split"})
            )
            chart = (
                alt.Chart(long_df)
                .mark_bar()
                .encode(
                    x=alt.X("Split:N", title="Dataset"),
                    y=alt.Y("Recall:Q", title="Recall"),
                    color=alt.Color(
                        "Series:N",
                        scale=alt.Scale(
                            domain=["Seen recall", "Unseen recall"],
                            range=["#38BDF8", "#F59E0B"],
                        ),
                    ),
                    xOffset="Series:N",
                )
                .properties(height=300)
            )
            st.altair_chart(chart, width="stretch")
        except ImportError:
            st.bar_chart(chart_data)
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Gap (seen - unseen, val)", "%+.2f pp" % (100 * bc["gap"]))
        g2.metric("Collateral val seen", "%.3f%%" % (100 * bc["collateral"]["val_seen"]))
        g3.metric("Collateral val unseen", "%.3f%%" % (100 * bc["collateral"]["val_unseen"]))
        if bc["seen_recall_test"] is not None:
            g4.metric(
                "Recall test (seen / unseen)",
                "%.2f%% / %.2f%%" % (100 * bc["seen_recall_test"], 100 * bc["unseen_recall_test"]),
            )
        else:
            g4.metric("Recall test (seen / unseen)", "n<30")
        st.markdown("Seen rules (mined with %s):" % fam)
        for t in bc["seen_rule_texts"]:
            st.code(t)
        st.markdown("Unseen rules (mined without %s):" % fam)
        for t in bc["unseen_rule_texts"]:
            st.code(t)
        st.caption(
            "unseen recall on all KDDTrain+ %s rows: %.2f%% (information only)"
            % (fam, 100 * bc["unseen_recall_train_all"])
        )
        if st.button("Re-run live", key="drill_rerun"):
            with st.spinner("Mining seen and unseen rules..."):
                live = drill_family(
                    fam, cap, fit=fit_df, val=val_df, test=test_df
                )
            st.session_state["drill_live"] = live
        live = st.session_state.get("drill_live")
        if live is not None and live["family"] == fam:
            st.markdown(
                "Live re-run (cap %.1f%%, %.1f s): seen val recall %.2f%%, unseen %.2f%%, gap %+.2f pp"
                % (
                    100 * live["cap"],
                    live["runtime_s"],
                    100 * live["seen_recall_val"],
                    100 * live["unseen_recall_val"],
                    100 * live["gap"],
                )
            )
            json_cap = "%.4g" % live["cap"]
            if json_cap in info["by_cap"]:
                j = info["by_cap"][json_cap]
                match = (
                    abs(live["seen_recall_val"] - j["seen_recall_val"]) < 1e-12
                    and abs(live["unseen_recall_val"] - j["unseen_recall_val"]) < 1e-12
                )
                st.caption(
                    "Matches drill.json at cap %s: %s" % (json_cap, "yes" if match else "NO")
                )

with tab_eval:
    st.subheader("Held-out evaluation")
    ev = load_json_file("eval.json")
    if ev is None:
        st.info("Run `python evaluate.py --eval` to produce results/eval.json.")
    else:
        c1, c2, c3 = st.columns(3)
        sizes = ev["split_sizes"]
        c1.metric("Fit rows", f"{sizes['fit']:,}")
        c2.metric("Validation rows", f"{sizes['val']:,}")
        c3.metric("Test rows", f"{sizes['test']:,}")
        d1, d2 = st.columns(2)
        d1.metric(
            "RECOMMENDED_CAP",
            "%.1f%%" % (100 * config.RECOMMENDED_CAP),
        )
        d2.metric(
            "DRIFT_TOL",
            "%.2f pp" % (100 * config.DRIFT_TOL),
        )
        rows = []
        for e in ev["entries"]:
            rows.append(
                {
                    "cap": "%.1f%%" % (100 * e["cap"]),
                    "method": e["method"],
                    "val collateral %": round(100 * e["val"]["collateral_rate"], 4),
                    "val recall": round(e["val"]["recall"], 4),
                    "val macro-recall": round(e["val"]["macro_recall"], 4),
                    "test collateral %": round(100 * e["test"]["collateral_rate"], 4),
                    "val drift pp": round(100 * e["cap_drift_val"], 4),
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch")
