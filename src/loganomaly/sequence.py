"""Sequence detectors: predict the next log template, flag lines that were not predicted.

Idea (DeepLog, Du et al., CCS 2017): normal system behaviour follows a small set of
template sequences. Train a model to predict the next template from the previous `history`
templates. At test time a line is suspicious if its real template is not among the model's
top-`g` predictions. A window is flagged if it holds at least one suspicious line, so the
result is directly comparable with the window-level methods in `benchmark.py`.

Two models share this logic:
  markov  order-1 transition counts (what usually follows template A?). A cheap baseline
          that tells us whether the LSTM learns anything beyond the previous line.
  lstm    DeepLog-style 2-layer LSTM over the last `history` templates.

Choices, stated so results are not over-read:
  * Trained on the chronological training part only, on samples whose target line is
    labelled normal (as in DeepLog). That uses training labels.
  * Templates never seen in training map to an "unknown" id and are always suspicious.
  * One global stream of lines (BGL interleaves many nodes). Per-node streams are a
    possible next step.
  * g = 9 as in the DeepLog paper, fixed in advance (not tuned on the test set).
"""
from dataclasses import dataclass

import numpy as np

UNKNOWN = 0  # code for templates not seen in training


@dataclass
class Encoded:
    codes: np.ndarray       # (n_lines,) template code, 0 = unknown
    vocab_size: int         # number of codes including UNKNOWN


def encode(template_ids: np.ndarray, train_lines: int) -> Encoded:
    """Map template ids to dense codes 1..V using the training lines only."""
    known = np.unique(template_ids[:train_lines])
    pos = np.searchsorted(known, template_ids)
    pos = np.clip(pos, 0, len(known) - 1)
    hit = known[pos] == template_ids
    codes = np.where(hit, pos + 1, UNKNOWN).astype(np.int64)
    return Encoded(codes, len(known) + 1)


def history_matrix(codes: np.ndarray, targets: np.ndarray, history: int) -> np.ndarray:
    """Rows = the `history` codes before each target index (padded with UNKNOWN at the start)."""
    padded = np.concatenate([np.full(history, UNKNOWN, dtype=codes.dtype), codes])
    offsets = np.arange(history)
    return padded[targets[:, None] + offsets[None, :]]


def windows_from_lines(line_flags: np.ndarray, window_of_line: np.ndarray, n_windows: int) -> np.ndarray:
    hits = np.bincount(window_of_line, weights=line_flags.astype(float), minlength=n_windows)
    return (hits > 0).astype(int)


# --------------------------------------------------------------------------- Markov
class MarkovNext:
    """Order-1 next-template model from transition counts."""

    def fit(self, prev: np.ndarray, nxt: np.ndarray, vocab_size: int) -> "MarkovNext":
        self.counts = np.zeros((vocab_size, vocab_size), dtype=np.int64)
        np.add.at(self.counts, (prev, nxt), 1)
        return self

    def rank_of(self, prev: np.ndarray, nxt: np.ndarray) -> np.ndarray:
        """Rank (0 = most likely) of the real next template; unseen transitions get a huge rank."""
        rows = self.counts[prev]
        true = rows[np.arange(len(nxt)), nxt]
        rank = (rows > true[:, None]).sum(axis=1)
        return np.where(true > 0, rank, np.iinfo(np.int64).max)


# --------------------------------------------------------------------------- LSTM
def _torch():
    import torch
    return torch


def build_lstm(vocab_size: int, embed: int = 32, hidden: int = 64, layers: int = 2):
    torch = _torch()
    nn = torch.nn

    class DeepLog(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed)
            self.lstm = nn.LSTM(embed, hidden, num_layers=layers, batch_first=True)
            self.out = nn.Linear(hidden, vocab_size)

        def forward(self, x):
            h, _ = self.lstm(self.embed(x))
            return self.out(h[:, -1])

    return DeepLog()


def train_lstm(X: np.ndarray, y: np.ndarray, vocab_size: int, epochs: int = 3,
               batch: int = 1024, lr: float = 2e-3, seed: int = 42, verbose: bool = True):
    torch = _torch()
    torch.manual_seed(seed)
    model = build_lstm(vocab_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(Xt), generator=gen)
        total = 0.0
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if verbose:
            print(f"    epoch {epoch + 1}/{epochs}: loss {total / len(Xt):.3f}")
    return model


def lstm_rank_of(model, X: np.ndarray, y: np.ndarray, batch: int = 8192) -> np.ndarray:
    """Rank (0 = most likely) of the real next template under the LSTM."""
    torch = _torch()
    model.eval()
    ranks = np.empty(len(y), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(y), batch):
            logits = model(torch.from_numpy(X[start:start + batch]))
            true = logits[torch.arange(len(logits)), torch.from_numpy(y[start:start + batch])]
            ranks[start:start + batch] = (logits > true[:, None]).sum(dim=1).numpy()
    return ranks


# --------------------------------------------------------------------------- detectors
def _setup(w, split, history):
    """Codes, the first test line, and the normal training targets."""
    test_line0 = int(np.searchsorted(w.window_of_line, split))
    enc = encode(w.events["template_id"].to_numpy(), test_line0)
    normal = w.events["is_anomaly"].to_numpy() == 0
    train_targets = np.arange(history, test_line0)
    train_targets = train_targets[normal[train_targets]]
    return enc, test_line0, train_targets


def sequence_ranks(w, split, model: str = "lstm", history: int = 10,
                   max_train: int = 500_000, epochs: int = 3, seed: int = 42) -> np.ndarray:
    """Rank of each test line's real template (unknown templates -> huge rank)."""
    enc, test_line0, train_targets = _setup(w, split, history)
    codes = enc.codes
    test_targets = np.arange(test_line0, len(codes))
    if model == "markov":
        m = MarkovNext().fit(codes[train_targets - 1], codes[train_targets], enc.vocab_size)
        ranks = m.rank_of(codes[test_targets - 1], codes[test_targets])
    elif model == "lstm":
        rng = np.random.default_rng(seed)
        if len(train_targets) > max_train:
            train_targets = np.sort(rng.choice(train_targets, max_train, replace=False))
        net = train_lstm(history_matrix(codes, train_targets, history), codes[train_targets],
                         enc.vocab_size, epochs=epochs, seed=seed)
        ranks = lstm_rank_of(net, history_matrix(codes, test_targets, history), codes[test_targets])
    else:
        raise ValueError(model)
    ranks = np.where(codes[test_targets] == UNKNOWN, np.iinfo(np.int64).max, ranks)
    return ranks


def flags_from_ranks(w, split, ranks: np.ndarray, g: int = 9) -> np.ndarray:
    """Test-window flags: a window is flagged if any of its lines ranked outside the top g."""
    test_line0 = len(w.window_of_line) - len(ranks)
    flags = windows_from_lines(ranks >= g, w.window_of_line[test_line0:] - split,
                               len(w.labels) - split)
    return flags


def markov_next_template(w, split, g: int = 9):
    return flags_from_ranks(w, split, sequence_ranks(w, split, "markov"), g)


def deeplog_lstm(w, split, g: int = 9):
    return flags_from_ranks(w, split, sequence_ranks(w, split, "lstm"), g)


# --------------------------------------------------------------------------- top-g sweep
def sweep(name: str, window: int, gs=(1, 3, 5, 9, 15, 25)):
    """F1 for each model and g, with unknown templates flagged (DeepLog) or ignored.

    'ignored' isolates the sequence signal: it shows what the models learn from the order
    of *known* templates, without the help of the new-template signal.
    This is an analysis of the test period, not a way to pick g (g = 9 is fixed in advance).
    """
    import pandas as pd
    from sklearn.metrics import f1_score, precision_score, recall_score

    from loganomaly import RESULTS_DIR
    from loganomaly.features import chronological_split, load_events, make_windows

    w = make_windows(load_events(name), window)
    split = chronological_split(len(w.labels))
    y = w.labels[split:]
    enc, test_line0, _ = _setup(w, split, 10)
    unknown = enc.codes[test_line0:] == UNKNOWN
    print(f"test lines: {len(unknown):,}, using a template unseen in training: {unknown.mean():.1%}")
    rows = []
    for model in ("markov", "lstm"):
        ranks = sequence_ranks(w, split, model)
        for policy, r in (("flagged", ranks), ("ignored", np.where(unknown, 0, ranks))):
            for g in gs:
                pred = flags_from_ranks(w, split, r, g)
                rows.append({"model": model, "unknown_templates": policy, "g": g,
                             "precision": precision_score(y, pred, zero_division=0),
                             "recall": recall_score(y, pred, zero_division=0),
                             "f1": f1_score(y, pred, zero_division=0)})
                print(f"{model:6s} unknown {policy:7s} g={g:2d}  P={rows[-1]['precision']:.3f}  "
                      f"R={rows[-1]['recall']:.3f}  F1={rows[-1]['f1']:.3f}")
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / f"{name}_w{window}_sequence_topg.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"Saved {out}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Top-g sweep of the sequence detectors")
    parser.add_argument("--dataset", default="BGL")
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()
    sweep(args.dataset, args.window)
