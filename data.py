import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd

from config import (
    CATEGORICAL,
    COLUMNS,
    DIFFICULTY_COL,
    EXPECTED_FIELDS,
    LABEL_COL,
    NUMERIC,
    SEED,
    SPLIT,
    TEST_PATH,
    TRAIN_PATH,
)


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _raw_lines(path):
    with open(path, "rb") as f:
        raw = f.read()
    return raw.splitlines()


def _field_counts(lines):
    counts = {}
    for ln in lines:
        c = ln.count(b",") + 1
        counts[c] = counts.get(c, 0) + 1
    return counts


def _load(path):
    t0 = time.perf_counter()
    lines = _raw_lines(path)
    counts = _field_counts(lines)
    if set(counts) != {EXPECTED_FIELDS}:
        raise SystemExit(
            "STOP: field count is not %d in %s: %s"
            % (EXPECTED_FIELDS, path.name, counts)
        )
    first = lines[0].decode("ascii", errors="replace").split(",")
    if not _is_number(first[0]):
        raise SystemExit(
            "STOP: header line detected in %s: %s" % (path.name, first[:5])
        )
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        dtype=str,
        keep_default_na=False,
    )
    if len(df.columns) != EXPECTED_FIELDS:
        raise SystemExit(
            "STOP: parsed column count %d != %d in %s"
            % (len(df.columns), EXPECTED_FIELDS, path.name)
        )
    if pd.to_numeric(df[LABEL_COL], errors="coerce").notna().all():
        raise SystemExit("STOP: label column in %s looks numeric" % path.name)
    if not pd.to_numeric(df[DIFFICULTY_COL], errors="coerce").notna().all():
        raise SystemExit("STOP: difficulty column in %s is not numeric" % path.name)

    raw_label = df[LABEL_COL]
    trailing_dot = bool(raw_label.str.endswith(".").any())
    df[LABEL_COL] = (
        raw_label.str.strip().str.lower().str.replace(r"\.$", "", regex=True)
    )
    df["is_attack"] = df[LABEL_COL] != "normal"
    df[DIFFICULTY_COL] = pd.to_numeric(df[DIFFICULTY_COL], errors="raise").astype(int)
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="raise")
    elapsed = time.perf_counter() - t0
    return df, len(lines), counts, trailing_dot, elapsed


def load_train():
    df, *_ = _load(TRAIN_PATH)
    return df


def load_test():
    df, *_ = _load(TEST_PATH)
    return df


def split_fit_val(df, seed=SEED, split=SPLIT):
    rng = np.random.default_rng(seed)
    fit_idx = []
    val_idx = []
    small_labels = []
    for label in sorted(df[LABEL_COL].unique()):
        idx = np.array(df.index[df[LABEL_COL] == label].tolist())
        if len(idx) < 2:
            fit_idx.extend(idx.tolist())
            small_labels.append(label)
            continue
        rng.shuffle(idx)
        n_val = int(round(len(idx) * split))
        if n_val < 1:
            n_val = 1
        if n_val >= len(idx):
            n_val = len(idx) - 1
        val_idx.extend(idx[:n_val].tolist())
        fit_idx.extend(idx[n_val:].tolist())
    fit_idx = np.array(sorted(fit_idx))
    val_idx = np.array(sorted(val_idx))
    fit = df.loc[fit_idx].copy()
    val = df.loc[val_idx].copy()
    fit.attrs["split"] = "fit"
    val.attrs["split"] = "val"
    return fit, val, fit_idx, small_labels


def _print_file_report(path, name):
    df, n_lines, counts, trailing_dot, elapsed = _load(path)
    null_cells = int((df.astype(str) == "").sum().sum())
    labels = df[LABEL_COL].value_counts()
    print("=== %s ===" % path)
    print("file: %s" % name)
    print("bytes: %d" % os.path.getsize(path))
    print("lines: %d" % n_lines)
    print("rows: %d" % len(df))
    print("field_count: %s" % counts)
    print("distinct_labels: %d" % df[LABEL_COL].nunique())
    print("top10_labels:")
    for lab, cnt in labels.head(10).items():
        print("  %s %d" % (lab, cnt))
    print("trailing_dot_found: %s" % trailing_dot)
    print(
        "protocol_type values: %s"
        % sorted(df["protocol_type"].dropna().unique().tolist())
    )
    print("flag values: %s" % sorted(df["flag"].dropna().unique().tolist()))
    print("service distinct count: %d" % df["service"].nunique())
    print(
        "difficulty min/max: %s / %s"
        % (int(df[DIFFICULTY_COL].min()), int(df[DIFFICULTY_COL].max()))
    )
    print("null_cells: %d" % null_cells)
    print(
        "load_time_s: %.2f %s"
        % (elapsed, "(SLOW >20s)" if elapsed > 20 else "(ok)")
    )
    return df


def schema_report():
    train = _print_file_report(TRAIN_PATH, "KDDTrain+.txt")
    print("")
    test = _print_file_report(TEST_PATH, "KDDTest+.txt")
    print("")

    train_labels = set(train[LABEL_COL].unique())
    test_labels = test[LABEL_COL].value_counts()
    missing = [(lab, int(c)) for lab, c in test_labels.items() if lab not in train_labels]
    print("=== labels in test but not in train ===")
    if not missing:
        print("(none)")
    for lab, cnt in sorted(missing, key=lambda x: -x[1]):
        print("  %s %d" % (lab, cnt))
    print("")

    fit, val, fit_idx, small = split_fit_val(train)
    fit_hash = hashlib.sha256(
        ",".join(str(i) for i in fit_idx).encode("ascii")
    ).hexdigest()
    print("=== split (seed %d, %.0f/%.0f stratified by label) ==="
          % (SEED, 100 * (1 - SPLIT), 100 * SPLIT))
    print("fit rows: %d" % len(fit))
    print("val rows: %d" % len(val))
    print("fit index sha256: %s" % fit_hash)
    print("small labels (<2 rows) wholly in fit: %s" % small)
    print("fit is_attack count: %d" % int(fit["is_attack"].sum()))
    print("val is_attack count: %d" % int(val["is_attack"].sum()))
    print("categorical: %s" % CATEGORICAL)
    print("numeric feature count: %d" % len(NUMERIC))
    print("features = %d (label/difficulty excluded)" % (len(NUMERIC) + len(CATEGORICAL)))


if __name__ == "__main__":
    schema_report()
