import numpy as np
import pandas as pd

from loganomaly.features import make_windows
from loganomaly.sequence import (UNKNOWN, MarkovNext, encode, flags_from_ranks, history_matrix,
                                 sequence_ranks)


def test_encode_uses_training_lines_only():
    enc = encode(np.array([7, 3, 7, 9, 3, 42]), train_lines=3)
    assert enc.vocab_size == 3                                # UNKNOWN + {3, 7}
    assert enc.codes.tolist() == [2, 1, 2, UNKNOWN, 1, UNKNOWN]


def test_history_matrix_pads_the_start():
    codes = np.array([5, 6, 7, 8])
    X = history_matrix(codes, np.array([1, 3]), history=2)
    assert X.tolist() == [[UNKNOWN, 5], [6, 7]]


def test_markov_ranks_follow_transition_counts():
    m = MarkovNext().fit(prev=np.array([1, 1, 1, 2]), nxt=np.array([2, 2, 3, 1]), vocab_size=4)
    ranks = m.rank_of(np.array([1, 1, 2]), np.array([2, 3, 3]))
    assert ranks[0] == 0                       # 1 -> 2 is the most common transition
    assert ranks[1] == 1                       # 1 -> 3 is second
    assert ranks[2] == np.iinfo(np.int64).max  # 2 -> 3 never seen


def _cyclic_events(n_lines, anomaly_at=None):
    tids = np.tile([10, 20, 30, 40], n_lines // 4)
    anomalies = np.zeros(n_lines, dtype=int)
    if anomaly_at is not None:
        tids[anomaly_at] = 10                  # breaks the 10-20-30-40 cycle
        anomalies[anomaly_at] = 1
    return pd.DataFrame({"timestamp": range(n_lines), "level": "INFO",
                         "is_anomaly": anomalies, "template_id": tids})


def test_sequence_models_flag_only_the_broken_window():
    # 400 lines, windows of 40: train = windows 0-7, test = windows 8-9.
    # Line 362 (test window 9) should be 30 but is 10, an out-of-order template.
    w = make_windows(_cyclic_events(400, anomaly_at=362), size=40)
    split = 8
    for model in ("markov", "lstm"):
        kwargs = {"epochs": 30} if model == "lstm" else {}
        ranks = sequence_ranks(w, split, model, history=4, **kwargs)
        assert flags_from_ranks(w, split, ranks, g=1).tolist() == [0, 1], model


def test_unknown_template_is_always_flagged():
    ev = _cyclic_events(400)
    ev.loc[350, "template_id"] = 99            # never seen in training; test window 8 (lines 320-359)
    w = make_windows(ev, size=40)
    ranks = sequence_ranks(w, 8, "markov")
    assert flags_from_ranks(w, 8, ranks, g=1000).tolist() == [1, 0]


def test_node_history_uses_only_the_same_node():
    from loganomaly.sequence import node_history_matrix
    #            line: 0  1  2  3  4  5
    codes = np.array([1, 5, 2, 6, 3, 7])
    nodes = np.array([0, 1, 0, 1, 0, 1])          # two machines, interleaved
    X = node_history_matrix(codes, nodes, np.array([4, 5, 1]), history=2)
    assert X.tolist() == [[1, 2],                 # line 4 (node 0): previous node-0 lines 0, 2
                          [5, 6],                 # line 5 (node 1): previous node-1 lines 1, 3
                          [UNKNOWN, UNKNOWN]]     # line 1 is node 1's first line
