"""Parse raw BGL logs into structured events + Drain templates.

A raw BGL line looks like:
    - 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.675872 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected
    ^label        ^unix ts  ^date      ^node                ^full time                ^node again           ^type ^comp ^level ^message

Label: "-" means normal; anything else (e.g. "KERNDTLB") is an alert category -> anomaly.

Drain (He et al., ICWS 2017) groups messages that differ only in variable parts:
    "generating core.2275"  and  "generating core.862"   ->   "generating core.<NUM>"
Each group is a *template*. Models then work on template IDs instead of free text.

The file is streamed line by line, so the full 4.7M-line dataset fits in memory.

Usage:
    python -m loganomaly.parse --dataset BGL_2k
    python -m loganomaly.parse --dataset BGL
"""
import argparse
import time

import pandas as pd
from drain3 import TemplateMiner
from drain3.masking import MaskingInstruction
from drain3.template_miner_config import TemplateMinerConfig

from loganomaly import PROCESSED_DIR, RAW_DIR

KEEP_TEXT_BELOW = 100_000   # keep the raw message column only for small datasets


def make_miner() -> TemplateMiner:
    config = TemplateMinerConfig()
    # Replace obvious variables before clustering, so templates are cleaner.
    config.masking_instructions = [
        MaskingInstruction(r"0x[0-9a-fA-F]+", "HEX"),
        MaskingInstruction(r"\b\d+\b", "NUM"),
    ]
    config.drain_sim_th = 0.5   # similarity needed to join an existing template
    config.drain_depth = 4      # depth of Drain's prefix tree
    config.profiling_enabled = False
    return TemplateMiner(config=config)


def parse(name: str) -> None:
    raw = RAW_DIR / f"{name}.log"
    if not raw.exists():
        raise SystemExit(f"{raw} not found. Run: python -m loganomaly.download --dataset {name}")

    miner = make_miner()
    cols = {"timestamp": [], "node": [], "component": [], "level": [],
            "is_anomaly": [], "template_id": [], "content": []}
    start = time.perf_counter()
    with open(raw, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            parts = line.rstrip("\n").split(" ", 9)
            if len(parts) < 10:                      # malformed line: pad missing fields
                parts += [""] * (10 - len(parts))
            label, ts, _date, node, _time, _node2, _type, comp, level, content = parts
            cols["timestamp"].append(int(ts) if ts.isdigit() else -1)
            cols["node"].append(node)
            cols["component"].append(comp)
            cols["level"].append(level)
            cols["is_anomaly"].append(0 if label == "-" else 1)
            cols["template_id"].append(miner.add_log_message(content)["cluster_id"])
            cols["content"].append(content)
            if i % 500_000 == 0:
                rate = i / (time.perf_counter() - start)
                print(f"  {i:,} lines  ({rate:,.0f} lines/s)", flush=True)
    elapsed = time.perf_counter() - start

    df = pd.DataFrame(cols)
    if len(df) >= KEEP_TEXT_BELOW:
        df = df.drop(columns="content")
    templates = {c.cluster_id: c.get_template() for c in miner.drain.clusters}

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"{name}_events.csv"
    df.to_csv(out, index=False)

    summary = (df.groupby("template_id")
                 .agg(count=("is_anomaly", "size"), anomalies=("is_anomaly", "sum"))
                 .reset_index())
    summary["template"] = summary["template_id"].map(templates)
    summary = summary.sort_values("count", ascending=False)
    summary.to_csv(PROCESSED_DIR / f"{name}_templates.csv", index=False)

    print(f"Lines:      {len(df):,}  (parsed in {elapsed:.0f}s, {len(df) / elapsed:,.0f} lines/s)")
    print(f"Anomalies:  {df['is_anomaly'].sum():,} ({df['is_anomaly'].mean():.1%})")
    print(f"Templates:  {len(templates):,}")
    print(f"Saved:      {out}")
    print("\nTop 10 templates:")
    with pd.option_context("display.max_colwidth", 70, "display.width", 140):
        print(summary.head(10)[["template_id", "count", "anomalies", "template"]].to_string(index=False))




# --------------------------------------------------------------------------- HDFS (v0.3)
BLOCK_RE = __import__("re").compile(r"blk_-?\d+")


def parse_hdfs(name: str = "HDFS") -> None:
    """Parse HDFS_v1 (Xu et al., SOSP 2009): 11M lines from a 200-node Hadoop cluster.

    A raw line:  081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating
                 ^date  ^time  ^pid ^level ^component                 ^message
    Labels are per *block* (anomaly_label.csv), not per line. Each line is assigned to the
    first block id it mentions; the `node` column holds that block id, and is_anomaly is the
    block's label, so a block's lines form one session.
    """
    import csv

    raw = RAW_DIR / f"{name}.log"
    labels_csv = RAW_DIR / f"{name}_anomaly_label.csv"
    with open(labels_csv, encoding="utf-8") as f:
        label = {r["BlockId"]: int(r["Label"] == "Anomaly") for r in csv.DictReader(f)}

    miner = make_miner()
    cols = {"timestamp": [], "node": [], "component": [], "level": [], "is_anomaly": [], "template_id": []}
    start = time.perf_counter()
    skipped = 0
    with open(raw, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            parts = line.rstrip("\n").split(" ", 5)
            m = BLOCK_RE.search(line)
            if len(parts) < 6 or m is None or m.group(0) not in label:
                skipped += 1
                continue
            date, hms, _pid, level, comp, content = parts
            blk = m.group(0)
            cols["timestamp"].append(int(date + hms))           # yymmddHHMMSS, sortable
            cols["node"].append(blk)
            cols["component"].append(comp.rstrip(":"))
            cols["level"].append(level)
            cols["is_anomaly"].append(label[blk])
            cols["template_id"].append(miner.add_log_message(content)["cluster_id"])
            if i % 1_000_000 == 0:
                print(f"  {i:,} lines  ({i / (time.perf_counter() - start):,.0f} lines/s)", flush=True)
    elapsed = time.perf_counter() - start
    df = pd.DataFrame(cols)
    templates = {c.cluster_id: c.get_template() for c in miner.drain.clusters}
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / f"{name}_events.csv", index=False)
    summary = (df.groupby("template_id").agg(count=("is_anomaly", "size"), anomalies=("is_anomaly", "sum"))
                 .reset_index())
    summary["template"] = summary["template_id"].map(templates)
    summary.sort_values("count", ascending=False).to_csv(PROCESSED_DIR / f"{name}_templates.csv", index=False)
    blocks = df.groupby("node", observed=True)["is_anomaly"].first()
    print(f"Lines: {len(df):,} kept, {skipped:,} without a labelled block (parsed in {elapsed:.0f}s)")
    print(f"Blocks: {len(blocks):,}, anomalous: {blocks.sum():,} ({blocks.mean():.2%})")
    print(f"Templates: {len(templates)}  levels: {df['level'].value_counts().to_dict()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="BGL_2k")
    ds = parser.parse_args().dataset
    parse_hdfs(ds) if ds.startswith("HDFS") else parse(ds)
