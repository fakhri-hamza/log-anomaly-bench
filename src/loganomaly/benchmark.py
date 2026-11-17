"""Compare anomaly detectors on the same windows, with the same chronological split.

Methods (so far):
  severity_rule        no labels. Flag a window if it contains a FATAL or FAILURE line.
  unseen_template_rule no labels. Flag a window if it contains a template never seen in training.
  known_bad_rule       uses training labels. Learn which templates were mostly anomalous in
                       training, flag test windows that contain one of them.
  isolation_forest     no labels. Isolation Forest (Liu et al., 2008) on log-scaled template
                       counts. Contamination is set to the training anomaly rate (a common,
                       slightly optimistic convention; stated so results are not over-read).
  random_forest        uses training labels. Supervised Random Forest on the same features.
  markov_next_template uses training labels (normal lines only). Order-1 transition counts;
                       flag a window if a line is not among the 9 most likely next templates.
  deeplog_lstm         uses training labels (normal lines only). DeepLog-style LSTM over the
                       last 10 templates, same top-9 rule. See sequence.py.

Metrics: precision, recall, F1 on test windows; plus fit and predict time.

Usage:
    python -m loganomaly.benchmark --dataset BGL --window 100
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

from loganomaly import RESULTS_DIR
from loganomaly.features import chronological_split, load_events, make_windows
from loganomaly.sequence import deeplog_lstm, markov_next_template

SEVERE_LEVELS = {"FATAL", "FAILURE"}


def per_window_any(windows, line_mask: np.ndarray) -> np.ndarray:
    """1 if any line of the window satisfies line_mask."""
    hits = np.bincount(windows.window_of_line, weights=line_mask.astype(float),
                       minlength=len(windows.labels))
    return (hits > 0).astype(int)


def severity_rule(w, split):
    flags = per_window_any(w, w.events["level"].isin(SEVERE_LEVELS).to_numpy())
    return flags[split:]


def unseen_template_rule(w, split):
    train_lines = w.window_of_line < split
    seen = set(w.events.loc[train_lines, "template_id"].unique())
    unseen = ~w.events["template_id"].isin(seen).to_numpy()
    return per_window_any(w, unseen)[split:]


def known_bad_rule(w, split, min_rate=0.5):
    train = w.events[w.window_of_line < split]
    rate = train.groupby("template_id")["is_anomaly"].mean()
    bad = set(rate[rate >= min_rate].index)
    return per_window_any(w, w.events["template_id"].isin(bad).to_numpy())[split:]


def isolation_forest(w, split):
    X = w.counts.copy()
    X.data = np.log1p(X.data)                       # dampen very frequent templates
    contamination = float(np.clip(w.labels[:split].mean(), 0.001, 0.5))
    model = IsolationForest(n_estimators=200, contamination=contamination,
                            random_state=42, n_jobs=-1)
    model.fit(X[:split])
    return (model.predict(X[split:]) == -1).astype(int)


def random_forest(w, split):
    from sklearn.ensemble import RandomForestClassifier
    X = w.counts.copy()
    X.data = np.log1p(X.data)
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                   random_state=42, n_jobs=-1)
    model.fit(X[:split], w.labels[:split])
    return model.predict(X[split:])


METHODS = {
    "severity_rule": ("rule", "none", severity_rule),
    "unseen_template_rule": ("rule", "none", unseen_template_rule),
    "known_bad_rule": ("rule", "train labels", known_bad_rule),
    "isolation_forest": ("ML", "none", isolation_forest),
    "random_forest": ("ML", "train labels", random_forest),
    "markov_next_template": ("sequence", "train labels", markov_next_template),
    "deeplog_lstm": ("sequence", "train labels", deeplog_lstm),
}


def run(name: str, window: int) -> pd.DataFrame:
    t0 = time.perf_counter()
    events = load_events(name)
    w = make_windows(events, window)
    split = chronological_split(len(w.labels))
    y_test = w.labels[split:]
    print(f"{len(events):,} lines -> {len(w.labels):,} windows of {window} lines "
          f"({w.counts.shape[1]:,} templates), built in {time.perf_counter() - t0:.0f}s")
    print(f"train: {split:,} windows ({w.labels[:split].mean():.1%} anomalous) | "
          f"test: {len(y_test):,} windows ({y_test.mean():.1%} anomalous)\n")

    rows = []
    for method, (family, supervision, fn) in METHODS.items():
        t = time.perf_counter()
        pred = fn(w, split)
        secs = time.perf_counter() - t
        rows.append({
            "method": method, "family": family, "labels_used": supervision,
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "flagged_windows": int(pred.sum()),
            "seconds": secs,
        })
        r = rows[-1]
        print(f"{method:22s} P={r['precision']:.3f}  R={r['recall']:.3f}  "
              f"F1={r['f1']:.3f}  ({secs:.1f}s)")

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{name}_w{window}_results.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"\nSaved {out}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="BGL")
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()
    run(args.dataset, args.window)
