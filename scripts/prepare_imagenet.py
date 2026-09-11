#!/usr/bin/env python
"""Convert the HuggingFace ``ILSVRC/imagenet-1k`` parquet shards into the
``ImageFolder`` layout that :mod:`redundancy.datasets.imagenet` expects.

The HF distribution ships ImageNet as parquet (``data/train-*.parquet``,
``data/validation-*.parquet``) with an ``image`` struct column holding the
already-encoded image bytes and an integer ``label``. This script streams each
shard and writes the bytes straight to disk -- no decode/re-encode, so the
images are bit-identical to what the parquet holds and the conversion is I/O
bound rather than CPU bound.

Class directories are named ``class_0000 .. class_0999``. ``ImageFolder`` sorts
directory names alphabetically to build ``class_to_idx``, so this naming makes
the resulting folder index exactly equal to the parquet ``label`` -- which is
the standard ILSVRC class ordering (the same one you get from sorted synsets).

Usage::

    python scripts/prepare_imagenet.py \\
        --parquet-dir "$PARQUET_DIR/data" \\
        --out-dir "$DATA_ROOT" \\
        --split train --workers 16

``PARQUET_DIR`` and ``DATA_ROOT`` come from ``.env`` at the repository root;
``scripts/imagenet_convert.sh`` loads them for you.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow.parquet as pq

# The split name inside the output directory. HF calls it "validation"; the
# training code (and every ImageNet convention) calls it "val".
OUT_SPLIT = {"train": "train", "validation": "val"}


def convert_shard(path: str, out_root: str) -> tuple[str, int]:
    """Write every row of one parquet shard as a file under ``out_root``.

    Returns ``(shard_name, rows_written)``. Existing files are skipped, which
    makes the whole script safely resumable after an interrupted run.
    """
    shard = os.path.basename(path)
    stem = shard.replace(".parquet", "")
    written = 0

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=256, columns=["image", "label"]):
        images = batch.column("image").to_pylist()
        labels = batch.column("label").to_pylist()
        for i, (img, label) in enumerate(zip(images, labels)):
            class_dir = os.path.join(out_root, f"class_{label:04d}")
            os.makedirs(class_dir, exist_ok=True)
            # Keep the original extension when the parquet records one; the
            # handful of non-JPEG files in ImageNet stay loadable either way.
            src_name = (img.get("path") or "") if isinstance(img, dict) else ""
            ext = os.path.splitext(src_name)[1] or ".JPEG"
            dest = os.path.join(class_dir, f"{stem}_{written:06d}{ext}")
            if not os.path.exists(dest):
                # Write to a temp name then rename, so an interrupted run never
                # leaves a truncated image that a later pass would skip.
                tmp = dest + ".tmp"
                with open(tmp, "wb") as fh:
                    fh.write(img["bytes"])
                os.replace(tmp, dest)
            written += 1
    return shard, written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet-dir", required=True,
                    help="directory holding the downloaded *.parquet shards")
    ap.add_argument("--out-dir", required=True,
                    help="ImageFolder root; train/ and val/ are created inside")
    ap.add_argument("--split", default="train", choices=["train", "validation"])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    shards = sorted(
        os.path.join(args.parquet_dir, f)
        for f in os.listdir(args.parquet_dir)
        if f.startswith(f"{args.split}-") and f.endswith(".parquet")
    )
    if not shards:
        raise SystemExit(f"No '{args.split}-*.parquet' shards in {args.parquet_dir}")

    out_root = os.path.join(args.out_dir, OUT_SPLIT[args.split])
    os.makedirs(out_root, exist_ok=True)
    print(f"{args.split}: {len(shards)} shards -> {out_root} ({args.workers} workers)",
          flush=True)

    total = 0
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert_shard, s, out_root): s for s in shards}
        for fut in as_completed(futures):
            shard, n = fut.result()
            total += n
            done += 1
            print(f"[{done}/{len(shards)}] {shard}: {n} images (total {total})",
                  flush=True)

    print(f"Done: {total} images written to {out_root}", flush=True)


if __name__ == "__main__":
    main()
