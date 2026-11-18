"""Do per-node sequences help the sequence detectors? And how stable is the LSTM across seeds?

v0.2 limitation: BGL interleaves thousands of nodes in one file, so "the previous line" is
usually another machine's line. Here each line's history is the previous lines of *its own node*
(BGL `node` column), with the same split, windows, models and top-g rule as before.

Reports, at g = 9 (fixed in advance):
  unknown flagged  the DeepLog rule (templates unseen in training are always suspicious)
  unknown ignored  only the sequence signal on known templates

Usage:  python -m loganomaly.pernode --dataset BGL --seeds 42 1 2
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from loganomaly import RESULTS_DIR
from loganomaly.features import chronological_split, load_events, make_windows
from loganomaly.sequence import UNKNOWN, _setup, flags_from_ranks, sequence_ranks


def main(name: str, window: int, seeds: list[int], g: int = 9) -> pd.DataFrame:
    w = make_windows(load_events(name, with_node=True), window)
    split = chronological_split(len(w.labels))
    y = w.labels[split:]
    enc, test_line0, _ = _setup(w, split, 10)
    unknown = enc.codes[test_line0:] == UNKNOWN
    print(f"{w.events['node'].nunique():,} nodes; test lines with unseen templates: {unknown.mean():.1%}")

    runs = [("markov", s, None) for s in ("global", "node")]
    runs += [("lstm", s, seed) for s in ("global", "node") for seed in seeds]
    rows = []
    for model, streams, seed in runs:
        t = time.perf_counter()
        kw = {"seed": seed} if seed is not None else {}
        ranks = sequence_ranks(w, split, model, streams=streams, **kw)
        secs = time.perf_counter() - t
        for policy, r in (("flagged", ranks), ("ignored", np.where(unknown, 0, ranks))):
            pred = flags_from_ranks(w, split, r, g)
            rows.append({"model": model, "streams": streams, "seed": seed, "unknown_templates": policy,
                         "precision": precision_score(y, pred, zero_division=0),
                         "recall": recall_score(y, pred, zero_division=0),
                         "f1": f1_score(y, pred, zero_division=0), "seconds": secs})
            r_ = rows[-1]
            print(f"{model:6s} {streams:6s} seed={seed!s:4s} unknown {policy:7s} "
                  f"P={r_['precision']:.3f} R={r_['recall']:.3f} F1={r_['f1']:.3f} ({secs:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"{name}_w{window}_pernode.csv", index=False, float_format="%.4f")
    summary = (df.groupby(["model", "streams", "unknown_templates"])["f1"]
                 .agg(["mean", "std", "min", "max", "count"]).round(3))
    summary.to_csv(RESULTS_DIR / f"{name}_w{window}_pernode_summary.csv")
    print("\nF1 at g=9 (mean / std over seeds; Markov is deterministic)")
    print(summary.to_string())
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="BGL")
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    a = p.parse_args()
    main(a.dataset, a.window, a.seeds)
