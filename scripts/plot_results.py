#!/usr/bin/env python
"""Plot per-epoch CSV logs written by ``train.py``.

Give it one or more CSV files (or a directory of them) and it overlays a chosen
metric across runs, labelling each line by its file name.

Examples
--------
    python scripts/plot_results.py outputs/*.csv --metric eval_acc_top1
    python scripts/plot_results.py outputs --metric effective_ratio -o used.png
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd


def collect_csvs(paths: list[str]) -> list[str]:
    files: list[str] = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(sorted(glob.glob(os.path.join(path, "*.csv"))))
        else:
            files.extend(sorted(glob.glob(path)))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="CSV files, globs, or a directory")
    parser.add_argument("--metric", default="eval_acc_top1",
                        help="column to plot against 'epoch'")
    parser.add_argument("--output", "-o", default=None,
                        help="save to this image instead of showing a window")
    args = parser.parse_args()

    files = collect_csvs(args.paths)
    if not files:
        raise SystemExit("No CSV files matched.")

    plt.figure(figsize=(10, 6))
    for path in files:
        df = pd.read_csv(path)
        if args.metric not in df.columns:
            print(f"skip {path}: no column '{args.metric}'")
            continue
        label = os.path.splitext(os.path.basename(path))[0]
        plt.plot(df["epoch"], df[args.metric], label=label)

    plt.xlabel("epoch")
    plt.ylabel(args.metric)
    plt.title(f"{args.metric} across runs")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"saved {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
