"""Turn a stream of parsed log events into windows and feature vectors.

Window = a block of `size` consecutive log lines (non-overlapping).
A window is labelled anomalous if it contains at least one anomalous line
(the standard convention in log anomaly detection benchmarks).

Feature vector = how many times each template appears in the window
("bag of templates"). Sparse, because a window only uses a few templates.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from loganomaly import PROCESSED_DIR


@dataclass
class Windows:
    counts: sparse.csr_matrix        # (n_windows, n_templates) template counts
    labels: np.ndarray               # (n_windows,) 1 = anomalous window
    template_ids: np.ndarray         # column -> template id
    window_of_line: np.ndarray       # line -> window index
    events: pd.DataFrame             # the parsed events (one row per line)


def load_events(name: str, with_node: bool = False) -> pd.DataFrame:
    cols = ["timestamp", "level", "is_anomaly", "template_id"] + (["node"] if with_node else [])
    return pd.read_csv(PROCESSED_DIR / f"{name}_events.csv", usecols=cols,
                       dtype={"level": "category", "node": "category"})


def make_windows(events: pd.DataFrame, size: int = 100) -> Windows:
    n = len(events)
    window_of_line = np.arange(n) // size
    n_windows = window_of_line[-1] + 1

    template_ids, col = np.unique(events["template_id"].to_numpy(), return_inverse=True)
    counts = sparse.csr_matrix(
        (np.ones(n, dtype=np.float32), (window_of_line, col)),
        shape=(n_windows, len(template_ids)),
    )  # duplicate (row, col) pairs are summed -> counts
    labels = np.bincount(window_of_line, weights=events["is_anomaly"].to_numpy(),
                         minlength=n_windows) > 0
    return Windows(counts, labels.astype(int), template_ids, window_of_line, events)


def make_session_windows(events: pd.DataFrame) -> Windows:
    """One window per session (HDFS: per block), sessions ordered by their first line.

    Lines are re-ordered so each session is contiguous (original order kept inside a session).
    Every downstream method then works unchanged, and a sequence model sees each block's own
    sequence. A session is anomalous if its lines are labelled anomalous (HDFS labels whole blocks).
    """
    first = events.groupby("node", observed=True, sort=False).cumcount() == 0
    session_rank = pd.Series(np.cumsum(first.to_numpy()) - 1, index=events.index)
    rank_of_node = session_rank[first].set_axis(events.loc[first, "node"].to_numpy())
    events = events.assign(_w=events["node"].map(rank_of_node).astype(np.int64))
    events = events.sort_values("_w", kind="stable").reset_index(drop=True)
    window_of_line = events.pop("_w").to_numpy()
    n_windows = int(window_of_line[-1]) + 1
    template_ids, col = np.unique(events["template_id"].to_numpy(), return_inverse=True)
    counts = sparse.csr_matrix((np.ones(len(events), dtype=np.float32), (window_of_line, col)),
                               shape=(n_windows, len(template_ids)))
    labels = np.bincount(window_of_line, weights=events["is_anomaly"].to_numpy(), minlength=n_windows) > 0
    return Windows(counts, labels.astype(int), template_ids, window_of_line, events)


def chronological_split(n_windows: int, train_frac: float = 0.8) -> int:
    """Index of the first test window. Train on the past, test on the future (no shuffling)."""
    return int(n_windows * train_frac)
