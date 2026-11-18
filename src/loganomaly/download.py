"""Download Loghub log datasets.

Usage:
    python -m loganomaly.download --dataset BGL_2k   # 2,000-line sample (GitHub), for quick tests
    python -m loganomaly.download --dataset BGL      # full dataset, ~4.7M lines (Zenodo, 57 MB zip)
    python -m loganomaly.download --dataset HDFS     # HDFS_v1, ~11M lines + block labels (Zenodo, 187 MB zip)
"""
import argparse
import shutil
import urllib.request
import zipfile

from loganomaly import RAW_DIR

# name -> (url, None if the URL is the log itself, or {file inside the zip: saved as})
DATASETS = {
    "BGL_2k": ("https://raw.githubusercontent.com/logpai/loghub/master/BGL/BGL_2k.log", None),
    "BGL": ("https://zenodo.org/records/8196385/files/BGL.zip?download=1", {"BGL.log": "BGL.log"}),
    "HDFS": ("https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1",
             {"HDFS.log": "HDFS.log", "preprocessed/anomaly_label.csv": "HDFS_anomaly_label.csv"}),
}


def _progress(blocks, block_size, total):
    if total > 0:
        done = min(blocks * block_size / total, 1)
        print(f"\r  {done:6.1%} of {total / 1e6:.0f} MB", end="", flush=True)


def download(name: str) -> None:
    if name not in DATASETS:
        raise SystemExit(f"Unknown dataset {name!r}. Choose from: {', '.join(DATASETS)}")
    url, members = DATASETS[name]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / f"{name}.log"
    if target.exists():
        print(f"Already downloaded: {target}")
        return

    if members is None:
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, target, _progress)
    else:
        archive = RAW_DIR / f"{name}.zip"
        if not archive.exists():                        # a previous, interrupted run may have it
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, archive, _progress)
        print("\n  extracting ...")
        with zipfile.ZipFile(archive) as zf:
            for member, saved_as in members.items():
                with zf.open(member) as src, open(RAW_DIR / saved_as, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        archive.unlink()
    n_lines = sum(1 for _ in target.open(encoding="utf-8", errors="replace"))
    print(f"\nSaved {n_lines:,} lines to {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="BGL_2k", choices=list(DATASETS))
    download(parser.parse_args().dataset)
